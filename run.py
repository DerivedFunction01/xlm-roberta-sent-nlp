# %% [markdown]
# # Multilingual Language Detection via Sentence-NER (Token Classification)
# Fine-tunes XLM-RoBERTa to tag each token with its source language (BIO scheme),
# enabling transparent, evidence-based language identification.

# %%
# --- Environment Setup ---
# pip install transformers datasets tokenizers evaluate pysbd torch accelerate

import random
import re
import json
from collections import defaultdict

import torch
import numpy as np
import evaluate
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_CHECKPOINT = "xlm-roberta-base"
MAX_LENGTH = 512
ARTICLES_PER_LANG = 500   # increase for a larger dataset
EXAMPLES_TARGET = 10_000  # synthetic mixed-language training examples to generate

# %%
# --- Language Configuration ---
# Script groups and their ISO codes.
LANGUAGE_GROUPS = {
    "Latin":    ["en", "es", "fr", "de", "it", "pt", "nl", "vi", "tr"],
    "Cyrillic": ["ru", "bg", "uk", "sr"],
    "EastAsian":["zh", "ja", "ko"],
    "Indic":    ["hi", "ur", "bn", "ta", "te"],
    "Arabic":   ["ar", "fa"],
}

ALL_LANGS = [lang for langs in LANGUAGE_GROUPS.values() for lang in langs]

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
print("Sample:", dict(list(id2label.items())[:7]))

# %%
# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# %%
# --- Data Extraction ---
# Pull ~ARTICLES_PER_LANG sentences per language from the Wikipedia streaming dataset.

def extract_sentences_from_wiki(lang: str, n_articles: int = ARTICLES_PER_LANG) -> list[str]:
    """Stream Wikipedia articles and split them into non-trivial sentences."""
    try:
        import pysbd
        segmenter = pysbd.Segmenter(language="en", clean=True)  # fallback lang
        use_pysbd = True
    except ImportError:
        use_pysbd = False

    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    sentences = []
    for article in dataset.take(n_articles):
        text = article.get("text", "")
        if use_pysbd:
            sents = segmenter.segment(text)
        else:
            # Simple fallback splitter
            sents = re.split(r"(?<=[.!?])\s+", text)
        for s in sents:
            s = s.strip()
            if 30 < len(s) < 400:   # filter very short/long sentences
                sentences.append(s)
    return sentences


print("Extracting sentences … (this may take a few minutes with streaming)")
lang_sentences: dict[str, list[str]] = {}
for lang in ALL_LANGS:
    lang_sentences[lang] = extract_sentences_from_wiki(lang)
    print(f"  {lang}: {len(lang_sentences[lang])} sentences")

# %%
# --- Synthetic Document Mixer ---

def bio_label_tokens(tokens: list[str], lang: str, is_first: bool) -> list[int]:
    """Assign BIO labels to a token sequence for a given language."""
    labels = []
    for j, _ in enumerate(tokens):
        if j == 0 and is_first:
            labels.append(label2id[f"B-{lang.upper()}"])
        elif j == 0:
            labels.append(label2id[f"B-{lang.upper()}"])
        else:
            labels.append(label2id[f"I-{lang.upper()}"])
    return labels


def augment_boundary(tokens: list[str], strip_punct: bool) -> list[str]:
    """Optionally remove sentence-final punctuation to simulate no-boundary code-switching."""
    if strip_punct and tokens:
        tokens = [t for t in tokens if t not in [".", "!", "?", "▁.", "▁!", "▁?"]]
    return tokens


def swap_random_tokens(tokens: list[str], labels: list[int], swap_rate: float = 0.02):
    """Randomly swap tokens between positions to simulate within-sentence code-switching."""
    n = len(tokens)
    n_swaps = max(1, int(n * swap_rate))
    for _ in range(n_swaps):
        i, j = random.sample(range(n), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
        labels[i], labels[j] = labels[j], labels[i]
    return tokens, labels


def create_synthetic_doc(
    pool: dict[str, list[str]],
    n_segments: int = 4,
    strip_punct_prob: float = 0.5,
    swap_prob: float = 0.3,
) -> dict:
    """
    Build one mixed-language training example.
    Returns a dict with 'tokens' and 'ner_tags' (label IDs).
    """
    chosen_langs = random.sample(ALL_LANGS, k=min(n_segments, len(ALL_LANGS)))
    all_tokens, all_labels = [], []
    total_tokens = 0

    for lang in chosen_langs:
        if total_tokens >= MAX_LENGTH - 20:
            break
        sents = pool.get(lang, [])
        if not sents:
            continue
        sent = random.choice(sents)
        tokens = tokenizer.tokenize(sent)
        if not tokens:
            continue

        strip = random.random() < strip_punct_prob
        tokens = augment_boundary(tokens, strip_punct=strip)

        labels = bio_label_tokens(tokens, lang, is_first=(len(all_tokens) == 0))

        if random.random() < swap_prob:
            tokens, labels = swap_random_tokens(tokens[:], labels[:])

        # Trim to fit within MAX_LENGTH
        remaining = MAX_LENGTH - 2 - total_tokens  # reserve [CLS] and [SEP]
        tokens = tokens[:remaining]
        labels = labels[:remaining]

        all_tokens.extend(tokens)
        all_labels.extend(labels)
        total_tokens += len(tokens)

    return {"tokens": all_tokens, "ner_tags": all_labels}


print("Generating synthetic mixed-language documents …")
raw_examples = [
    create_synthetic_doc(lang_sentences)
    for _ in range(EXAMPLES_TARGET)
]
print(f"Generated {len(raw_examples)} examples")
print("Sample tokens:", raw_examples[0]["tokens"][:12])
print("Sample labels:", [id2label[l] for l in raw_examples[0]["ner_tags"][:12]])

# %%
# --- Label Alignment (sub-token → word-level) ---
# XLM-R uses SentencePiece; the tokenizer produces sub-tokens.
# We already work at the tokenizer sub-token level above, so alignment is 1:1.
# Below we convert token lists → input IDs and add special-token labels (-100).

def tokenize_and_align(example: dict) -> dict:
    """
    Re-encode the pre-tokenized token list and propagate labels.
    Special tokens ([CLS], [SEP]) receive label -100 (ignored by loss).
    """
    encoding = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    word_ids = encoding.word_ids()
    labels = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            labels.append(example["ner_tags"][word_id])
        else:
            # Continuation sub-token → use I- variant
            orig_label = example["ner_tags"][word_id]
            lang_tag = id2label[orig_label]
            if lang_tag.startswith("B-"):
                i_tag = "I-" + lang_tag[2:]
                labels.append(label2id.get(i_tag, orig_label))
            else:
                labels.append(orig_label)
        prev_word_id = word_id

    encoding["labels"] = labels
    return encoding


hf_dataset = Dataset.from_list(raw_examples)
hf_dataset = hf_dataset.map(tokenize_and_align, batched=False, remove_columns=["tokens", "ner_tags"])

# Train / validation split (90 / 10)
split = hf_dataset.train_test_split(test_size=0.1, seed=SEED)
train_dataset = split["train"]
eval_dataset  = split["test"]
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
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }


data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./lang-ner-xlmr",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning …")
trainer.train()
trainer.save_model("./lang-ner-xlmr-final")
tokenizer.save_pretrained("./lang-ner-xlmr-final")
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
