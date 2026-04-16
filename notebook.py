# %% [markdown]
# # Multilingual Language Detection via Sentence-NER (Token Classification)
# Fine-tunes XLM-RoBERTa to tag each token with its source language (BIO scheme),
# enabling transparent, evidence-based language identification.

# %%
# --- Environment Setup ---
# pip install evaluate seqeval --quiet
# %%
import json
import random
import multiprocessing as mp
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
)

from pathlib import Path
from datasets import load_from_disk

SEED = 42
MODEL_CHECKPOINT = "xlm-roberta-base"
BASE_DIR = Path(".")
LANGUAGE_ALIASES_PATH = BASE_DIR / "language_aliases.json"
TOKENIZED_CACHE_DIR = BASE_DIR / "sentences_cache" / "tokenized_dataset"
MULTILABEL_CACHE_DIR = BASE_DIR / "sentences_cache" / "multilabel_dataset"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_workers(split: int = 1):
    return mp.cpu_count() // split


def load_all_langs() -> list[str]:
    with LANGUAGE_ALIASES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object in {LANGUAGE_ALIASES_PATH}")
    return list(data.keys())


def load_dataset_cache(cache_dir: Path):
    if not cache_dir.exists():
        return None
    return load_from_disk(str(cache_dir))


def load_tokenized_dataset_cache():
    if not TOKENIZED_CACHE_DIR.exists():
        return None
    return load_dataset_cache(TOKENIZED_CACHE_DIR)


def load_multilabel_dataset_cache():
    if not MULTILABEL_CACHE_DIR.exists():
        return None
    return load_dataset_cache(MULTILABEL_CACHE_DIR)

from huggingface_hub import login

if (BASE_DIR / "hf_token").exists():
    with (BASE_DIR / "hf_token").open() as f:
        token = f.read().strip()
    login(token=token)
    print("Logged in to Hugging Face Hub")
# %%
# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
# %%
# --- Language List ---
_ALL_LANGS = load_all_langs()
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
cached_tokenized = load_tokenized_dataset_cache()
if cached_tokenized is not None:
    train_dataset = cached_tokenized["train"]
    eval_dataset = cached_tokenized["eval"]
    print("Loaded tokenized dataset cache")
else:
    raise RuntimeError(
        "Tokenized cache not found. Run `python fetch.py` first to prepare the datasets."
    )

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
first_example = train_dataset[0]
input_ids = first_example["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids)
labels = first_example["labels"]
# Convert via id2label
labels = [id2label[label] for label in labels if label != -100]
print(f"Tokens: {tokens}")
print(f"Labels: {labels}")

# %%
# --- Tokenized Dataset Load (MultiLabel)
multilabel_train_dataset = None
multilabel_eval_dataset = None
cache_multilabel = load_multilabel_dataset_cache()
if cache_multilabel is not None:
    multilabel_train_dataset = cache_multilabel["train"]
    multilabel_eval_dataset = cache_multilabel["eval"]
    print("Loaded multilabel dataset cache")
else:
    raise RuntimeError(
        "Multilabel cache not found. Run `python fetch.py` first to prepare the datasets."
    )
if multilabel_train_dataset is None or multilabel_eval_dataset is None:
    raise RuntimeError("Failed to load the cached multilabel dataset")
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
    epochs: int = 2,
    eval_steps: int = 2500,
    save_steps: int = 2500,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 18,

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
print("Ready NER fine-tuning …")
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
# %%
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

trainer = make_trainer(
    model=model,
    train_dataset=multilabel_train_dataset,
    eval_dataset=multilabel_eval_dataset,
    data_collator=multilabel_data_collator,
    compute_metrics=compute_multilabel_metrics,
    output_dir="./lang-ner-xlmr-multilabel",
)

print("Ready multilabel fine-tuning …")
# %%
trainer.train()
trainer.save_model()
trainer.save_state()
trainer.push_to_hub()

# %%
