# %% [markdown]
# # Multilingual Language Detection via Sentence-NER (Token Classification)
# Fine-tunes XLM-RoBERTa to tag each token with its source language (BIO scheme),
# enabling transparent, evidence-based language identification.

# %%
# --- Environment Setup ---
# pip install evaluate pysbd faker seqeval
# %%
import random
import re
import json
import gc
import multiprocessing as mp
from collections import defaultdict, deque
import string
from faker import Faker
import torch
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)

import os
import pandas as pd
from pathlib import Path
from huggingface_hub import login
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_CHECKPOINT = "xlm-roberta-base"
MAX_LENGTH = 512
ARTICLES_PER_LANG = 500   # increase for a larger dataset
EXAMPLES_TARGET = 500_000  # synthetic mixed-language training examples to generate
RESERVE_FRACTION = 0.15   # fraction of each language's sentences kept for guaranteed coverage
MIN_RESERVED_SENTENCES = 4
MAX_RESERVED_SENTENCES = 1000
MIN_COVERAGE_DOCS_PER_LANG = 2
MAX_COVERAGE_DOCS_PER_LANG = 5
SYNTHETIC_CACHE = "./sentences_cache/synthetic_examples.parquet"
SYNTHETIC_CACHE_META = "./sentences_cache/synthetic_examples.meta.json"
CACHE_DIR = "./sentences_cache/tokenized_dataset"
CACHE_META = "./sentences_cache/tokenized_dataset.meta.json"
CACHE_VERSION = 1
TOKENIZED_CACHE_VERSION = 1

# %%
# --- Language Configuration ---
# Script groups and their ISO codes.
LANGUAGE_GROUPS = {
    "Latin": [
        "en", "es", "fr", "de", "it", "pt", "nl", "vi", "tr", "la",
        "id", "ms", "af", "sq", "is", "no", "sv", "da", "fi", "hu", "pl", "cs", "ro",
    ],
    "Cyrillic":     ["ru", "bg", "uk", "sr", "be", "kk", "mk", "mn"],
    "EastAsian":    ["zh", "ja", "ko"],
    "Indic":        ["hi", "ur", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "as", "or"],
    "ArabicScript": ["ar", "fa", "ps", "sd", "ug"],
    "OtherScripts": ["el", "he", "hy", "ka", "am", "km", "lo", "my", "th"],
}

ALL_LANGS = [lang for langs in LANGUAGE_GROUPS.values() for lang in langs]
LANG_TO_GROUP = {lang: group for group, langs in LANGUAGE_GROUPS.items() for lang in langs}

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
if Path("hf_token").exists():
    with open("hf_token") as f:
        token = f.read().strip()
    login(token=token)
    print("Logged in to Hugging Face Hub")
# %%
# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
# %%
# --- Data Extraction ---
# Pull ~ARTICLES_PER_LANG sentences per language from the Wikipedia streaming dataset.

SENTENCES_DIR = "./sentences_cache"
os.makedirs(SENTENCES_DIR, exist_ok=True)


def parquet_path(lang: str) -> str:
    return os.path.join(SENTENCES_DIR, f"{lang}.parquet")


MIN_ARTICLE_CHARS = 3_000  # skip stubs
WIKI_MARKUP = re.compile(r"\[\[.*?\]\]|\{\{.*?\}\}|==.*?==", flags=re.DOTALL)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Languages natively supported by pysbd (da and el added from MajorEconomies).
# Native support in pysbd as of 2026
PYSBD_SUPPORTED = {
    "en", "hi", "mr", "bg", "es", "ru", "ar", "am", "hy", "fa",
    "ur", "pl", "zh", "nl", "da", "fr", "it", "el", "my", "ja", "de", "kk",
}

PYSBD_FALLBACKS = {
    # Cyrillic
    "uk": "ru", "be": "ru", "sr": "ru", "mk": "ru", "mn": "ru",
    # Latin / Romance
    "pt": "es", "ro": "fr", "la": "it", "sq": "it",
    # Latin / Northern & Central
    "sv": "da", "no": "da", "is": "da",
    "fi": "en", "hu": "en", "cs": "pl",
    # SE Asia Latin
    "vi": "en", "id": "en", "ms": "en", "af": "nl", "tr": "en",
    # RTL / Semitic
    "he": "ar", "ps": "fa", "ug": "ar",
    # Indic
    "bn": "hi", "ta": "hi", "te": "hi", "gu": "hi", "kn": "hi",
    "ml": "hi", "pa": "hi", "as": "hi", "or": "hi", "sd": "hi",
    # Others
    "ka": "en", "km": "zh", "ko": "zh", "lo": "zh", "th": "zh",
}

# Script-aware sentence length bounds (min_chars, max_chars).
_SENT_BOUNDS: dict[str, tuple[int, int]] = {
    "zh": (8,  180), "ja": (10, 180),
    "ko": (15, 220), "th": (15, 250), "km": (15, 250), "lo": (15, 250), "my": (15, 250),
    "ar": (25, 450), "fa": (25, 450), "he": (25, 400), "ur": (25, 450),
    "hi": (30, 500), "bn": (30, 500), "ta": (30, 500), "te": (30, 500), "am": (25, 400),
    "fi": (20, 450), "hu": (20, 450), "tr": (20, 450), "vi": (15, 300),
    "de": (40, 600), "ru": (35, 650), "uk": (35, 650), "el": (35, 650),
    "hy": (30, 500), "ka": (25, 450),
}
_DEFAULT_BOUNDS = (30, 600)


def _is_valid_sentence(s: str, lang: str) -> bool:
    mn, mx = _SENT_BOUNDS.get(lang, _DEFAULT_BOUNDS)
    return mn < len(s) < mx


def clean_and_halve(text: str):
    """Return the first half of a long article with wiki markup stripped, or None if too short."""
    if len(text) < MIN_ARTICLE_CHARS:
        return None
    text = WIKI_MARKUP.sub("", text)
    return text[: len(text) // 2].strip()


# ---------------------------------------------------------------------------
# Per-process segmenter cache.
# ProcessPoolExecutor forks a fresh Python interpreter per worker — pysbd
# Segmenter objects cannot be pickled, so we initialise them lazily inside
# each worker process and cache them in a module-level dict.
# ---------------------------------------------------------------------------
_PROC_SEGMENTERS: dict[str, object] = {}


def _get_segmenter(lang: str):
    """Return a cached pysbd Segmenter for this process, or None for regex fallback."""
    if lang in _PROC_SEGMENTERS:
        return _PROC_SEGMENTERS[lang]
    try:
        import pysbd as _pysbd
        proxy = PYSBD_FALLBACKS.get(lang, lang if lang in PYSBD_SUPPORTED else None)
        seg = _pysbd.Segmenter(language=proxy, clean=True) if proxy else None
    except (ImportError, ValueError):
        seg = None
    _PROC_SEGMENTERS[lang] = seg
    return seg


def extract_sentences_from_wiki(lang: str, n_articles: int = ARTICLES_PER_LANG) -> list[str]:
    """Stream Wikipedia articles, keep long ones, take first half, and split into sentences."""
    segmenter = _get_segmenter(lang)
    fetch_target = n_articles * 20

    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        split="train",
        streaming=True,
    )

    sentences = []
    seen = 0
    for article in dataset.take(fetch_target):
        text = clean_and_halve(article.get("text", ""))
        if text is None:
            continue

        sents = segmenter.segment(text) if segmenter else SENT_SPLIT.split(text) # type: ignore
        for s in sents:
            s = s.strip()
            if _is_valid_sentence(s, lang):
                sentences.append(s)

        seen += 1
        if seen >= n_articles:
            break

    return sentences


def load_or_extract(lang: str) -> tuple[str, list[str]]:
    """Return cached sentences from parquet if available, otherwise extract and save."""
    path = parquet_path(lang)
    if os.path.exists(path):
        return lang, pd.read_parquet(path)["sentence"].tolist()
    sentences = extract_sentences_from_wiki(lang)
    pd.DataFrame({"sentence": sentences}).to_parquet(path, index=False)
    return lang, sentences

# ProcessPoolExecutor saturates CPU cores for segmentation work.
# Workers == min(cpu_count, n_langs) — no point exceeding either.
MAX_WORKERS = min(mp.cpu_count(), len(ALL_LANGS))

print(f"Extracting sentences \u2192 cached under \'{SENTENCES_DIR}/'")
print(f"(Workers: {MAX_WORKERS} processes | cached languages skip extraction)\n")

lang_sentences: dict[str, list[str]] = {}
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(load_or_extract, lang): lang for lang in ALL_LANGS}
    for future in tqdm(as_completed(futures), total=len(ALL_LANGS), desc="Languages"):
        lang, sentences = future.result()
        lang_sentences[lang] = sentences
        tqdm.write(f"  {lang}: {len(sentences)} sentences  \u2192  {parquet_path(lang)}")

# %%
# --- Neutral (O-label) Corpus ---
# Three complementary O-label sources, all inserted as label-0 spans:
#   1. im2latex-100k     — real LaTeX formulas from arXiv papers
#   2. math_gen          — procedurally generated math expressions (14 domains)
#   3. symbol_noise      — random streams of unicode symbols, emoji, punctuation
#
# Together they teach the model that equations, markup, and noise tokens
# do not belong to any language.

# ── 1. im2latex ────────────────────────────────────────────────────────────────
LATEX_CACHE     = "./sentences_cache/latex_formulas.parquet"
LATEX_MIN_CHARS = 8
LATEX_MAX_CHARS = 300

_LATEX_WRAP = re.compile(
    r"^\s*\$+|\$+\s*$|^\\\[|\\\]$|^\\begin\{.*?\}|\\end\{.*?\}$"
)

def _clean_formula(f: str) -> str:
    return _LATEX_WRAP.sub("", f).strip()

def load_latex_formulas() -> list[str]:
    if os.path.exists(LATEX_CACHE):
        return pd.read_parquet(LATEX_CACHE)["formula"].tolist()
    print("Downloading im2latex-100k ...")
    ds = load_dataset("yuntian-deng/im2latex-100k", split="train")
    formulas = []
    for row in ds:
        _f = row["formula"] if isinstance(row, dict) else ""
        assert isinstance(_f, str)
        f = _clean_formula(_f)
        if LATEX_MIN_CHARS <= len(f) <= LATEX_MAX_CHARS:
            formulas.append(f)
    pd.DataFrame({"formula": formulas}).to_parquet(LATEX_CACHE, index=False)
    print(f"  Cached {len(formulas)} usable formulas → {LATEX_CACHE}")
    return formulas

latex_formulas: list[str] = load_latex_formulas()
print(f"im2latex:    {len(latex_formulas):>6} formulas")

# ── 2. math_gen ────────────────────────────────────────────────────────────────
# Import the procedural generator living alongside this file.
import importlib.util as _ilu, pathlib as _pl
_mg_path = _pl.Path(__file__).parent / "math_gen.py"
_spec    = _ilu.spec_from_file_location("math_gen", _mg_path)
_mg      = _ilu.module_from_spec(_spec) # type: ignore
_spec.loader.exec_module(_mg) # type: ignore
generate_synthetic_math = _mg.generate_synthetic_math

SYNTH_MATH_N = 50_000   # pre-generate a pool; cheap and avoids per-doc overhead
synth_math_pool: list[str] = [generate_synthetic_math() for _ in range(SYNTH_MATH_N)]
print(f"math_gen:    {len(synth_math_pool):>6} expressions")

# ── 3. Symbol / emoji noise ────────────────────────────────────────────────────
# Random streams of unicode block symbols, emoji, punctuation, and arrows.
# Mimics garbled text, copy-paste artifacts, and non-linguistic tokens.

fake = Faker()

def generate_symbol_noise(min_len: int = 3, max_len: int = 20) -> str:
    """Return a random string of symbols/emoji with optional spacing using Faker."""
    n = random.randint(min_len, max_len)

    # We mix emojis and standard punctuation
    parts = []
    for _ in range(n):
        # 50/50 chance to pick an emoji or a symbol
        if random.random() < 0.5:
            parts.append(fake.emoji())
        else:
            parts.append(random.choice(string.punctuation))
    spaced = []
    for i, ch in enumerate(parts):
        spaced.append(ch)
        if i < len(parts) - 1 and random.random() < 0.3:
            spaced.append(" ")

    return "".join(spaced)


# Pre-generate a noise pool as well
NOISE_N = 30_000
noise_pool: list[str] = [generate_symbol_noise() for _ in range(NOISE_N)]
print(f"symbol noise:{len(noise_pool):>6} strings")

# ── Combined O-label pool ──────────────────────────────────────────────────────
# Weighted so real LaTeX and synthetic math dominate over pure noise.
_O_SOURCES  = [latex_formulas, synth_math_pool, noise_pool]
_O_WEIGHTS  = [0.45,           0.40,            0.15]

def sample_o_span() -> str:
    """Draw one O-label span from the combined pool."""
    pool = random.choices(_O_SOURCES, weights=_O_WEIGHTS, k=1)[0]
    return random.choice(pool)

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
    if n < 2:
        return tokens, labels
    n_swaps = max(1, int(n * swap_rate))
    for _ in range(n_swaps):
        i, j = random.sample(range(n), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
        labels[i], labels[j] = labels[j], labels[i]
    return tokens, labels


def build_sentence_pools(
    sentence_map: dict[str, list[str]],
) -> tuple[dict[str, deque[str]], dict[str, deque[str]]]:
    """
    Split each language into a reserved coverage pool and a main sampling pool.

    Both pools are shuffled first, then consumed without replacement. Once a pool
    is empty, callers can fall back to the original sentence list.
    """
    reserved: dict[str, deque[str]] = {}
    main: dict[str, deque[str]] = {}

    for lang, sentences in sentence_map.items():
        if not sentences:
            continue
        shuffled = sentences[:]
        random.shuffle(shuffled)
        reserve_target = int(round(len(shuffled) * RESERVE_FRACTION))
        reserve_n = min(
            len(shuffled),
            max(
                MIN_RESERVED_SENTENCES,
                min(reserve_target, MAX_RESERVED_SENTENCES),
            ),
        )
        reserved[lang] = deque(shuffled[:reserve_n])
        main[lang] = deque(shuffled[reserve_n:])

    return reserved, main


def draw_sentence(
    lang: str,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
    source_pool: dict[str, list[str]] | None = None,
) -> str | None:
    """
    Draw a sentence without replacement from the preferred pool, then fallback
    pool, then finally the original list if we have exhausted our bags.
    """
    if primary_pool.get(lang):
        return primary_pool[lang].popleft()
    if fallback_pool and fallback_pool.get(lang):
        return fallback_pool[lang].popleft()
    if source_pool and source_pool.get(lang):
        return random.choice(source_pool[lang])
    return None


def chunk_list(items: list, n_chunks: int) -> list[list]:
    """Split a list into roughly equal contiguous chunks."""
    if n_chunks <= 1:
        return [items]
    chunk_size = (len(items) + n_chunks - 1) // n_chunks
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def partition_sentence_pools(
    pools: dict[str, deque[str]],
    n_workers: int,
) -> list[dict[str, deque[str]]]:
    """Round-robin partition each language pool across workers."""
    worker_pools: list[dict[str, deque[str]]] = [dict() for _ in range(n_workers)]
    for lang, dq in pools.items():
        items = list(dq)
        for worker_idx in range(n_workers):
            shard = items[worker_idx::n_workers]
            if shard:
                worker_pools[worker_idx][lang] = deque(shard)
    return worker_pools


def generate_synthetic_examples_chunk(
    worker_idx: int,
    jobs: list[tuple[str, str | None]],
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None,
    source_pool: dict[str, list[str]],
) -> list[dict]:
    """
    Generate a chunk of synthetic examples in one worker.

    Each worker gets a deterministic seed offset and its own pool shard so that
    sentence reuse stays low until the local shard is exhausted.
    """
    seed = SEED + (worker_idx * 10_000)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    examples: list[dict] = []
    worker_desc = f"Worker {worker_idx}"
    for kind, lang in tqdm(jobs, desc=worker_desc, position=worker_idx, leave=False):
        if kind == "coverage":
            examples.append(
                create_synthetic_doc(
                    primary_pool,
                    fallback_pool=fallback_pool,
                    source_pool=source_pool,
                    required_langs=[lang] if lang else None,
                )
            )
        else:
            examples.append(
                create_synthetic_doc(
                    primary_pool,
                    fallback_pool=fallback_pool,
                    source_pool=source_pool,
                )
            )
    return examples


def save_synthetic_examples_cache(
    coverage_examples: list[dict],
    random_examples: list[dict],
) -> None:
    """Persist synthetic examples to Parquet plus a small metadata sidecar."""
    rows = []
    for kind, examples in (("coverage", coverage_examples), ("random", random_examples)):
        for example in examples:
            rows.append(
                {
                    "kind": kind,
                    "tokens": json.dumps(example["tokens"]),
                    "ner_tags": json.dumps(example["ner_tags"]),
                }
            )

    pd.DataFrame(rows).to_parquet(SYNTHETIC_CACHE, index=False)
    with open(SYNTHETIC_CACHE_META, "w") as f:
        json.dump(
            {
                "cache_version": CACHE_VERSION,
                "examples_target": EXAMPLES_TARGET,
                "reserve_fraction": RESERVE_FRACTION,
                "min_reserved_sentences": MIN_RESERVED_SENTENCES,
                "max_reserved_sentences": MAX_RESERVED_SENTENCES,
                "min_coverage_docs_per_lang": MIN_COVERAGE_DOCS_PER_LANG,
                "max_coverage_docs_per_lang": MAX_COVERAGE_DOCS_PER_LANG,
            },
            f,
            indent=2,
        )


def load_synthetic_examples_cache() -> tuple[list[dict], list[dict]] | None:
    """Load cached synthetic examples if the cache metadata matches the current config."""
    if not (os.path.exists(SYNTHETIC_CACHE) and os.path.exists(SYNTHETIC_CACHE_META)):
        return None

    with open(SYNTHETIC_CACHE_META) as f:
        meta = json.load(f)

    expected_meta = {
        "cache_version": CACHE_VERSION,
        "examples_target": EXAMPLES_TARGET,
        "reserve_fraction": RESERVE_FRACTION,
        "min_reserved_sentences": MIN_RESERVED_SENTENCES,
        "max_reserved_sentences": MAX_RESERVED_SENTENCES,
        "min_coverage_docs_per_lang": MIN_COVERAGE_DOCS_PER_LANG,
        "max_coverage_docs_per_lang": MAX_COVERAGE_DOCS_PER_LANG,
    }
    if meta != expected_meta:
        return None

    df = pd.read_parquet(SYNTHETIC_CACHE)
    coverage_examples: list[dict] = []
    random_examples: list[dict] = []
    for row in df.itertuples(index=False):
        example = {
            "tokens": json.loads(row.tokens), # type: ignore
            "ner_tags": json.loads(row.ner_tags), # type: ignore
        }
        if row.kind == "coverage":
            coverage_examples.append(example)
        else:
            random_examples.append(example)
    return coverage_examples, random_examples


def save_tokenized_dataset_cache(train_dataset, eval_dataset) -> None:
    """Persist the tokenized train/eval split to disk."""
    synthetic_meta = None
    if os.path.exists(SYNTHETIC_CACHE_META):
        with open(SYNTHETIC_CACHE_META) as f:
            synthetic_meta = json.load(f)

    DatasetDict({"train": train_dataset, "eval": eval_dataset}).save_to_disk(CACHE_DIR)
    with open(CACHE_META, "w") as f:
        json.dump(
            {
                "cache_version": TOKENIZED_CACHE_VERSION,
                "model_checkpoint": MODEL_CHECKPOINT,
                "max_length": MAX_LENGTH,
                "seed": SEED,
                "synthetic_cache_meta": synthetic_meta,
            },
            f,
            indent=2,
        )


def load_tokenized_dataset_cache():
    """Load tokenized train/eval split if the metadata matches the current config."""
    if not (os.path.exists(CACHE_DIR) and os.path.exists(CACHE_META)):
        return None

    with open(CACHE_META) as f:
        meta = json.load(f)

    synthetic_meta = None
    if os.path.exists(SYNTHETIC_CACHE_META):
        with open(SYNTHETIC_CACHE_META) as f:
            synthetic_meta = json.load(f)

    expected_meta = {
        "cache_version": TOKENIZED_CACHE_VERSION,
        "model_checkpoint": MODEL_CHECKPOINT,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "synthetic_cache_meta": synthetic_meta,
    }
    if meta != expected_meta:
        return None

    return load_from_disk(CACHE_DIR)


def create_synthetic_doc(
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
    source_pool: dict[str, list[str]] | None = None,
    latex_pool: list[str] | None = None,
    required_langs: list[str] | None = None,
    o_inject_prob: float = 0.4,   # P(inserting at least one O-label span)
    n_segments: int = 4,
    strip_punct_prob: float = 0.5,
    swap_prob: float = 0.3,
) -> dict:
    """
    Sampling strategy:
      - Groups are weighted by their "global footprint" so major languages
        (Latin, EastAsian) appear more often than tail groups (MajorEconomies).
      - Within each selected group, one language is chosen uniformly.
      - `required_langs` can force coverage so every language appears at least
        once in the overall synthetic corpus.
    """
    # Weight per group — tuned to reflect real-world multilingual text distribution.
    GROUP_WEIGHTS = {
        # Latin is large (23 langs) but many are minor — keep dominant feel
        "Latin":        4.0,
        "EastAsian":    2.5,
        "Cyrillic":     1.5,
        "Indic":        1.5,
        "ArabicScript": 1.5,
        "OtherScripts": 1.0,
    }
    group_names  = list(LANGUAGE_GROUPS.keys())
    weights      = [GROUP_WEIGHTS.get(g, 1.0) for g in group_names]
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    chosen_langs: list[str] = []
    seen_langs: set[str] = set()

    # Always keep any requested languages. This is the coverage guarantee.
    for lang in required_langs or []:
        if (primary_pool.get(lang) or (fallback_pool and fallback_pool.get(lang)) or (source_pool and source_pool.get(lang))) and lang not in seen_langs:
            chosen_langs.append(lang)
            seen_langs.add(lang)

    # Sample remaining segments from groups, with replacement so a group can repeat.
    n_remaining_segments = max(0, n_segments - len(chosen_langs))
    selected_groups = random.choices(group_names, weights=norm_weights, k=n_remaining_segments)
    # Pick one language uniformly within each selected group.
    for group in selected_groups:
        lang = random.choice(LANGUAGE_GROUPS[group])
        if lang not in seen_langs:
            chosen_langs.append(lang)
            seen_langs.add(lang)

    all_tokens, all_labels = [], []
    total_tokens = 0

    for lang in chosen_langs:
        if total_tokens >= MAX_LENGTH - 20:
            break
        sent = draw_sentence(lang, primary_pool, fallback_pool, source_pool)
        if sent is None:
            continue
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

    # --- O-label span injection ---
    # Sample from the combined pool (LaTeX / synthetic math / symbol noise).
    # Allow 1-3 injections per doc so the model sees O spans in varied positions.
    if random.random() < o_inject_prob and total_tokens < MAX_LENGTH - 20:
        n_injections = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(n_injections):
            if total_tokens >= MAX_LENGTH - 10:
                break
            span = sample_o_span()
            o_tokens = tokenizer.tokenize(span)
            remaining = MAX_LENGTH - 2 - len(all_tokens)
            o_tokens = o_tokens[:min(remaining, 40)]  # cap single span at 40 tokens
            if o_tokens:
                insert_pos = random.randint(0, len(all_tokens))
                all_tokens = all_tokens[:insert_pos] + o_tokens + all_tokens[insert_pos:]
                all_labels = all_labels[:insert_pos] + [0] * len(o_tokens) + all_labels[insert_pos:]
                total_tokens += len(o_tokens)

    return {"tokens": all_tokens, "ner_tags": all_labels}


print("Generating synthetic mixed-language documents …")
cached_examples = load_synthetic_examples_cache()
if cached_examples is not None:
    coverage_examples, random_examples = cached_examples
    print(f"Loaded cached synthetic examples from {SYNTHETIC_CACHE}")
else:
    # Guarantee representation by reserving a fraction of each language's sentences,
    # then building several guaranteed documents from that reserved pool.
    reserved_sentence_pools, main_sentence_pools = build_sentence_pools(lang_sentences)
    reserved_total = sum(len(pool) for pool in reserved_sentence_pools.values())
    main_total = sum(len(pool) for pool in main_sentence_pools.values())
    print(
        f"Reserved sentence bags: {reserved_total} total | "
        f"Main sentence bags: {main_total} total"
    )

    missing_coverage_langs = [lang for lang in ALL_LANGS if not lang_sentences.get(lang)]
    if missing_coverage_langs:
        print("WARNING: no extracted sentences for:", ", ".join(missing_coverage_langs))

    coverage_plan = []
    for lang in ALL_LANGS:
        if not lang_sentences.get(lang):
            continue
        reserved_n = len(reserved_sentence_pools.get(lang, []))
        coverage_docs_for_lang = max(
            1,
            min(MAX_COVERAGE_DOCS_PER_LANG, max(MIN_COVERAGE_DOCS_PER_LANG, (reserved_n + 3) // 4)),
        )
        coverage_plan.extend([lang] * coverage_docs_for_lang)

    random_job_count = max(0, EXAMPLES_TARGET - len(coverage_plan))
    generation_jobs: list[tuple[str, str | None]] = [("coverage", lang) for lang in coverage_plan]
    generation_jobs.extend([("random", None)] * random_job_count)

    generation_workers = min(
        mp.cpu_count(),
        max(1, len(generation_jobs)),
    )
    generation_workers = min(generation_workers, len(generation_jobs) or 1)

    coverage_examples = []
    random_examples = []

    if generation_workers == 1:
        for kind, lang in tqdm(generation_jobs, desc="Synthetic docs"):
            if kind == "coverage":
                coverage_examples.append(
                    create_synthetic_doc(
                        reserved_sentence_pools,
                        fallback_pool=main_sentence_pools,
                        source_pool=lang_sentences,
                        required_langs=[lang] if lang else None,
                    )
                )
            else:
                random_examples.append(
                    create_synthetic_doc(
                        main_sentence_pools,
                        fallback_pool=reserved_sentence_pools,
                        source_pool=lang_sentences,
                    )
                )
    else:
        job_chunks = chunk_list(generation_jobs, generation_workers)
        reserved_worker_pools = partition_sentence_pools(reserved_sentence_pools, generation_workers)
        main_worker_pools = partition_sentence_pools(main_sentence_pools, generation_workers)
        coverage_examples = [None] * len(coverage_plan)
        random_examples = [None] * random_job_count

        with ProcessPoolExecutor(max_workers=generation_workers) as pool:
            future_to_jobs = {}
            job_start = 0
            for worker_idx, jobs in enumerate(job_chunks):
                if not jobs:
                    continue
                future = pool.submit(
                    generate_synthetic_examples_chunk,
                    worker_idx,
                    jobs,
                    reserved_worker_pools[worker_idx],
                    main_worker_pools[worker_idx],
                    lang_sentences,
                )
                future_to_jobs[future] = (job_start, jobs)
                job_start += len(jobs)

            for future in tqdm(as_completed(future_to_jobs), total=len(future_to_jobs), desc="Synthetic docs"):
                chunk_examples = future.result()
                job_start, jobs = future_to_jobs[future]
                for offset, (example, job) in enumerate(zip(chunk_examples, jobs)):
                    kind, _ = job
                    if kind == "coverage":
                        coverage_examples[job_start + offset] = example
                    else:
                        random_examples[job_start + offset - len(coverage_plan)] = example

        coverage_examples = [ex for ex in coverage_examples if ex is not None]
        random_examples = [ex for ex in random_examples if ex is not None]

    save_synthetic_examples_cache(coverage_examples, random_examples) # type: ignore

raw_examples = coverage_examples + random_examples

print(f"Generated {len(raw_examples)} examples")
print("Sample tokens:", raw_examples[0]["tokens"][:12])
print("Sample labels:", [id2label[l] for l in raw_examples[0]["ner_tags"][:12]])


def print_sampling_stats(examples: list[dict], top_n: int = 12) -> None:
    """Print per-language and per-group coverage stats for the synthetic corpus."""
    example_lang_counts = defaultdict(int)
    token_lang_counts = defaultdict(int)
    group_example_counts = defaultdict(int)
    group_token_counts = defaultdict(int)

    for example in examples:
        langs_in_example: set[str] = set()
        for tag_id in example["ner_tags"]:
            if tag_id == 0:
                continue
            lang = id2label[tag_id][2:].lower()
            token_lang_counts[lang] += 1
            group = LANG_TO_GROUP.get(lang, "Unknown")
            group_token_counts[group] += 1
            langs_in_example.add(lang)

        for lang in langs_in_example:
            example_lang_counts[lang] += 1
            group = LANG_TO_GROUP.get(lang, "Unknown")
            group_example_counts[group] += 1

    missing_langs = [lang for lang in ALL_LANGS if example_lang_counts[lang] == 0]

    print("\nSampling stats")
    print("-" * 72)
    print(f"Examples: {len(examples)}")
    print(f"Languages covered: {len(ALL_LANGS) - len(missing_langs)}/{len(ALL_LANGS)}")
    if missing_langs:
        print("Missing languages:", ", ".join(missing_langs))
    else:
        print("Missing languages: none")

    print("\nPer-language coverage (examples containing the language):")
    for lang, count in sorted(example_lang_counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]:
        print(f"  {lang:<3}  {count:>5}")

    print("\nPer-group coverage (examples / tokens):")
    for group in LANGUAGE_GROUPS:
        print(
            f"  {group:<12} "
            f"{group_example_counts[group]:>5} examples | "
            f"{group_token_counts[group]:>7} tokens"
        )

    top_tokens = sorted(token_lang_counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    print("\nTop token languages:")
    for lang, count in top_tokens:
        print(f"  {lang:<3}  {count:>7}")


print_sampling_stats(raw_examples) # type: ignore

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


coverage_dataset = Dataset.from_list(coverage_examples) # type: ignore
random_dataset = Dataset.from_list(random_examples) # type: ignore
cached_tokenized = load_tokenized_dataset_cache()
if cached_tokenized is not None:
    train_dataset = cached_tokenized["train"]
    eval_dataset = cached_tokenized["eval"]
    print(f"Loaded tokenized dataset cache from {CACHE_DIR}")
else:
    map_workers = max(1, mp.cpu_count() // 2)
    print(f"Tokenizing dataset with num_proc={map_workers}")
    coverage_dataset = coverage_dataset.map(
        tokenize_and_align,
        batched=False,
        remove_columns=["tokens", "ner_tags"],
        num_proc=map_workers,
    )
    random_dataset = random_dataset.map(
        tokenize_and_align,
        batched=False,
        remove_columns=["tokens", "ner_tags"],
        num_proc=map_workers,
    )

    # Train / validation split (90 / 10).
    # Keep the coverage set in train so every language is guaranteed to appear there.
    split = random_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = concatenate_datasets([coverage_dataset, split["train"]])
    eval_dataset = split["test"]
    save_tokenized_dataset_cache(train_dataset, eval_dataset)

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")


def release_generation_memory() -> None:
    """Drop large one-off generation artifacts after the dataset is ready."""
    for name in [
        "coverage_examples",
        "random_examples",
        "raw_examples",
        "coverage_dataset",
        "random_dataset",
        "reserved_sentence_pools",
        "main_sentence_pools",
        "lang_sentences",
        "coverage_plan",
        "generation_jobs",
        "job_chunks",
        "reserved_worker_pools",
        "main_worker_pools",
        "latex_formulas",
        "synth_math_pool",
        "noise_pool",
        "_O_SOURCES",
    ]:
        globals()[name] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


release_generation_memory()

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
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="steps",       # Evaluate every X steps
    save_strategy="steps",       # Save every X steps
    save_steps=500,              # Number of update steps between two evaluations/saves
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=3,
    report_to="tensorboard",
    seed=SEED,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
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
