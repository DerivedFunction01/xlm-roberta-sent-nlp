# %% [markdown]
# # Multilingual Language Detection via Sentence-NER (Token Classification)
# Fine-tunes XLM-RoBERTa to tag each token with its source language (BIO scheme),
# enabling transparent, evidence-based language identification.

# %%
# --- Environment Setup ---
# pip install evaluate pysbd faker seqeval
# %%
import random
import json
import multiprocessing as mp
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
)

from pathlib import Path

SEED = 42
MODEL_CHECKPOINT = "xlm-roberta-base"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_workers(split: int = 1):
    return mp.cpu_count() // split

def load_tokenized_dataset_splits(cache_dir: str):
    """Load a cached train/eval tokenized split without helper modules."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None
    try:
        from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required to load the tokenized cache") from exc

    try:
        return load_from_disk(str(cache_path))
    except Exception:
        split_names = ["train", "eval"]
        loaded_splits = {}
        for split_name in split_names:
            split_dir = cache_path / split_name
            arrow_files = sorted(split_dir.glob("*.arrow"))
            if not arrow_files:
                return None
            split_parts = [Dataset.from_file(str(path)) for path in arrow_files]
            loaded_splits[split_name] = (
                split_parts[0] if len(split_parts) == 1 else concatenate_datasets(split_parts)
            )
        return DatasetDict(loaded_splits)

_ALL_LANGS = []
with open("all_langs.json", encoding="utf-8") as f:
    data = json.load(f)
    # read the keys only
    _ALL_LANGS = list(data.keys())
# %%
from huggingface_hub import login

if Path("hf_token").exists():
    with open("hf_token") as f:
        token = f.read().strip()
    login(token=token)
    print("Logged in to Hugging Face Hub")
# %%
# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
# %%
# --- Project Imports ---
from language import ALL_LANGS, LANG_TO_GROUP
from paths import PATHS
from source_pools import load_language_sentences_from_parquet
from neutral_sources import build_neutral_sources
from multilabel_converter import convert_and_save_multilabel_dataset
from synthetic_build import build_synthetic_dataset, create_pure_synthetic_doc
from tokenization_cache import build_tokenized_dataset

_ALL_LANGS = ALL_LANGS
# %%
# Build BIO label map  (O=0, B-XX=odd, I-XX=even starting at 2)
label2id = {"O": 0}
id2label = {0: "O"}
for idx, lang in enumerate(_ALL_LANGS):
    b_id = 2 * idx + 1
    i_id = 2 * idx + 2
    label2id[f"B-{lang.upper()}"] = b_id
    label2id[f"I-{lang.upper()}"] = i_id
    id2label[b_id] = f"B-{lang.upper()}"
    id2label[i_id] = f"I-{lang.upper()}"

NUM_LABELS = len(label2id)
print(f"Total labels: {NUM_LABELS}")
print(f"Total languages: {len(_ALL_LANGS)}")
print("Sample:", dict(list(id2label.items())[:7]))

# %%
# --- Data Loading ---
wiki_english_seed_sentences = load_language_sentences_from_parquet(PATHS["wiki"]["cache_dir"], "en")
ft_english_seed_sentences = load_language_sentences_from_parquet(PATHS["finetrans"]["cache_dir"], "en")
print(f"Wiki English seed sentences: {len(wiki_english_seed_sentences):,}")
print(f"FineTranslations English seed sentences: {len(ft_english_seed_sentences):,}")
# %%
neutral_sources = build_neutral_sources(
    english_seed_sentences=(
        wiki_english_seed_sentences
        + ft_english_seed_sentences
    ),
    seed=SEED,
)
sample_o_span = neutral_sources.sample_o_span
sample_code_span = neutral_sources.sample_code_span

# %%
# --- Tokenized Dataset Load (Token Classification) ---
cached_tokenized = load_tokenized_dataset_splits("./sentences_cache/tokenized_dataset")
if cached_tokenized is not None:
    train_dataset = cached_tokenized["train"]
    eval_dataset = cached_tokenized["eval"]
    print("Loaded tokenized dataset cache")
else:
    # --- Synthetic Dataset Build ---
    synthetic_dataset = build_synthetic_dataset(
        seed=SEED,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        sample_o_span=sample_o_span,
        sample_code_span=sample_code_span,
    )

    train_dataset, eval_dataset = build_tokenized_dataset(
        synthetic_dataset,
        seed=SEED,
        model_checkpoint=MODEL_CHECKPOINT,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        max_length=512,
    )

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# %%
# --- Tokenized Dataset Load (MultiLabel)
multilabel_train_dataset = None
multilabel_eval_dataset = None
cache_multilabel = load_tokenized_dataset_splits(PATHS["multilabel_dataset"]["cache_dir"])
if cache_multilabel is not None:
    multilabel_train_dataset = cache_multilabel["train"]
    multilabel_eval_dataset = cache_multilabel["eval"]
    print("Loaded multilabel dataset cache")
    if "input_ids" not in multilabel_train_dataset.column_names or "input_ids" not in multilabel_eval_dataset.column_names:
        print("Existing multilabel cache is missing tokenized features; rebuilding cache...")
        convert_and_save_multilabel_dataset()
        cache_multilabel = load_tokenized_dataset_splits(PATHS["multilabel_dataset"]["cache_dir"])
        if cache_multilabel is not None:
            multilabel_train_dataset = cache_multilabel["train"]
            multilabel_eval_dataset = cache_multilabel["eval"]
else:
    print("Building multilabel dataset cache from tokenized dataset...")
    convert_and_save_multilabel_dataset()
    cache_multilabel = load_tokenized_dataset_splits(PATHS["multilabel_dataset"]["cache_dir"])
    if cache_multilabel is not None:
        multilabel_train_dataset = cache_multilabel["train"]
        multilabel_eval_dataset = cache_multilabel["eval"]
if multilabel_train_dataset is None or multilabel_eval_dataset is None:
    raise RuntimeError("Failed to load or build the multilabel dataset cache")
print(f"Multilabel Train: {len(multilabel_train_dataset)} | Eval: {len(multilabel_eval_dataset)}")

# %%
# --- Model Setup ---
from transformers import (
    TrainingArguments,
    Trainer
)
import evaluate

def make_training_args(
    output_dir: str,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    eval_steps: int,
    save_steps: int,
    epochs: int,
    gradient_accumulation_steps: int,
):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps, 
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=100,  # Less noise in the console
        save_total_limit=2,  # Essential for 500k runs
        report_to="tensorboard",
        dataloader_num_workers=mp.cpu_count() // 2,
        push_to_hub=True,
    )


def make_trainer(
    *,
    model,
    train_dataset,
    eval_dataset,
    data_collator,
    compute_metrics,
    output_dir: str,
    epochs: int = 3,
    eval_steps: int = 100,
    save_steps: int = 100,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 2,

):
    return Trainer(
        model=model,
        args=make_training_args(
            output_dir,
            train_batch_size=train_batch_size,
            eval_steps=eval_steps,
            save_steps=save_steps,
            epochs=epochs,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


# %%
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
# --- Model ---
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

# %%
# --- Training ---
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_preds = [
        [id2label[pred] for pred, lbl in zip(preds, lbls) if lbl != -100]
        for preds, lbls in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[lbl] for lbl in lbls if lbl != -100]
        for lbls in labels
    ]
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"], # type: ignore
        "recall":    results["overall_recall"], # type: ignore
        "f1":        results["overall_f1"], # type: ignore
        "accuracy":  results["overall_accuracy"], # type: ignore
    }


data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = make_trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    output_dir="./lang-ner-xlmr",
)

print("Starting fine-tuning …")
trainer.train()
trainer.save_model("./lang-ner-xlmr-final")
trainer.save_state()
tokenizer.save_pretrained("./lang-ner-xlmr-final")
trainer.push_to_hub()
print("Model saved to ./lang-ner-xlmr-final")

# %%
# --- Multilabel Classification
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(_ALL_LANGS),
    problem_type="multi_label_classification",
    id2label={i: lang.upper() for i, lang in enumerate(_ALL_LANGS)},
    label2id={lang.upper(): i for i, lang in enumerate(_ALL_LANGS)},
)

def compute_multilabel_metrics(p):
    logits, labels = p
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    tp = np.logical_and(predictions == 1, labels == 1).sum()
    fp = np.logical_and(predictions == 1, labels == 0).sum()
    fn = np.logical_and(predictions == 0, labels == 1).sum()
    exact_match = (predictions == labels).all(axis=1).mean()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(exact_match),
    }

class CustomMultiLabelDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.float32)
        features = [{k: v for k, v in feature.items() if k != "labels"} for feature in features]
        batch = super().__call__(features)
        batch["labels"] = labels
        return batch

multilabel_data_collator = CustomMultiLabelDataCollatorWithPadding(tokenizer)

multilabel_trainer = make_trainer(
    model=model,
    train_dataset=multilabel_train_dataset,
    eval_dataset=multilabel_eval_dataset,
    data_collator=multilabel_data_collator,
    compute_metrics=compute_multilabel_metrics,
    output_dir="./lang-ner-xlmr-multilabel",
)

print("Starting multilabel fine-tuning …")
multilabel_trainer.train()
multilabel_trainer.save_model("./lang-ner-xlmr-multilabel-final")
multilabel_trainer.save_state()
tokenizer.save_pretrained("./lang-ner-xlmr-multilabel-final")
multilabel_trainer.push_to_hub()
print("Multilabel model saved to ./lang-ner-xlmr-multilabel-final")
