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
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)

from pathlib import Path
from huggingface_hub import login

SEED = 42
MODEL_CHECKPOINT = "xlm-roberta-base"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_workers(split: int = 1):
    return mp.cpu_count() // split

lang_sentences: dict[str, list[str]] | None = None
smol_sentences: dict[str, list[str]] | None = None
ft_sentences: dict[str, list[str]] | None = None
# %%
# --- Project Imports ---
from language import ALL_LANGS, LANG_TO_GROUP, LANGUAGE_GROUPS, LANGUAGE_GROUP_WEIGHTS
from wiki_sources import load_wiki_sentences
from smol_sources import load_smol_sentences
from finetranslations_sources import load_finetranslations_sentences
from neutral_sources import build_neutral_sources
from synthetic_build import build_synthetic_dataset
from tokenization_cache import build_tokenized_dataset
# %%
# Build BIO label map  (O=0, B-XX=odd, I-XX=even starting at 2)
label2id = {"O": 0}
id2label = {0: "O"}
for idx, lang in enumerate(ALL_LANGS):
    b_id = 2 * idx + 1
    i_id = 2 * idx + 2
    label2id[f"B-{lang.upper()}"] = b_id
    label2id[f"I-{lang.upper()}"] = i_id
    id2label[b_id] = f"B-{lang.upper()}"
    id2label[i_id] = f"I-{lang.upper()}"

NUM_LABELS = len(label2id)
print(f"Total labels: {NUM_LABELS}")
print(f"Total languages: {len(ALL_LANGS)}")
print("Sample:", dict(list(id2label.items())[:7]))
# %%
if Path("hf_token").exists():
    with open("hf_token") as f:
        token = f.read().strip()
    login(token=token)
    print("Logged in to Hugging Face Hub")
# %%
# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
# %%
# --- Data Loading ---
lang_sentences = load_wiki_sentences(
    ALL_LANGS,
    lang_to_group=LANG_TO_GROUP,
    seed=SEED,
    max_workers=get_workers(2),
)
#%%
smol_sentences = load_smol_sentences(lang_to_group=LANG_TO_GROUP, seed=SEED)
if smol_sentences is not None:
    total_smol_sentences = sum(len(v) for v in smol_sentences.values())
    print(
        f"\nSMOL kept separate for pool split: "
        f"{len(smol_sentences)} languages | {total_smol_sentences} sentences"
    )
#%%
ft_sentences = None
try:
    ft_sentences = load_finetranslations_sentences(
        lang_to_group=LANG_TO_GROUP,
        seed=SEED,
        max_workers=get_workers(4),
    )
    if ft_sentences is not None:
        total_ft_sentences = sum(len(v) for v in ft_sentences.values())
        print(
            f"\nFineTranslations kept separate for pool split: "
            f"{len(ft_sentences)} languages | {total_ft_sentences} sentences"
        )
except Exception as exc:
    print(f"\nFineTranslations augmentation skipped: {exc}")
#%%
neutral_sources = build_neutral_sources(
    english_seed_sentences=(
        lang_sentences.get("en", [])
        + (ft_sentences.get("en", []) if ft_sentences else [])
    ),
    seed=SEED,
)
sample_o_span = neutral_sources.sample_o_span
sample_code_span = neutral_sources.sample_code_span

# %%
# --- Synthetic Dataset Build ---
synthetic_dataset = build_synthetic_dataset(
    seed=SEED,
    tokenizer=tokenizer,
    coverage_sentence_map=lang_sentences,
    smol_sentence_map=smol_sentences,
    ft_sentence_map=ft_sentences,
    all_langs=ALL_LANGS,
    lang_to_group=LANG_TO_GROUP,
    language_groups=LANGUAGE_GROUPS, # type: ignore
    language_group_weights=LANGUAGE_GROUP_WEIGHTS,
    label2id=label2id,
    id2label=id2label,
    sample_o_span=sample_o_span,
    sample_code_span=sample_code_span,
    generation_workers=get_workers(4),
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

training_args = TrainingArguments(
    output_dir="./lang-ner-xlmr",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Effectively batch size 32
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=2500,  # Evaluate less frequently for speed
    save_steps=2500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    logging_steps=100,  # Less noise in the console
    save_total_limit=2,  # Essential for 500k runs
    report_to="tensorboard",
    dataloader_num_workers=mp.cpu_count() // 2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # type: ignore
    eval_dataset=eval_dataset,  # type: ignore
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning …")
trainer.train()
trainer.save_model("./lang-ner-xlmr-final")
tokenizer.save_pretrained("./lang-ner-xlmr-final")
trainer.push_to_hub()
print("Model saved to ./lang-ner-xlmr-final")

# %%
# --- Transparency Validation ---
# Feed a mixed-language sentence to the NER pipeline and visualise the evidence.

ner_pipeline = pipeline(
    "ner",
    model="./lang-ner-xlmr-final",
    tokenizer="./lang-ner-xlmr-final",
    aggregation_strategy="simple",   # merges consecutive same-label tokens
    device=0 if torch.cuda.is_available() else -1,
)

DEMO_SENTENCES = [
    # English + French
    "The committee approved the proposal. Le comité a approuvé la proposition avec quelques modifications.",
    # English + Spanish
    "I really enjoyed the conference yesterday. Fue una experiencia increíble para todos los participantes.",
    # English + German + Russian
    "Hello, my name is Anna. Ich komme aus Deutschland. Я живу в Берлине уже пять лет.",
]

def display_transparency(text: str):
    """Print a token-level language attribution report."""
    results = ner_pipeline(text)
    print(f"\nInput : {text}")
    print("-" * 70)
    print(f"{'Span':<35} {'Label':<12} {'Confidence':>10}")
    print("-" * 70)
    for entity in results:
        word  = entity["word"].replace("▁", " ").strip()
        label = entity["entity_group"]
        score = entity["score"]
        bar   = "█" * int(score * 20)
        print(f"{word:<35} {label:<12} {score:>6.2%}  {bar}")
    print()


print("\n=== TRANSPARENCY VALIDATION ===")
for sentence in DEMO_SENTENCES:
    display_transparency(sentence)

# %%
# --- Save Label Map for Later Use ---
with open("./lang-ner-xlmr-final/label_map.json", "w") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
print("Label map saved.")
