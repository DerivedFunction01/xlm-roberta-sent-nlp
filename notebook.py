# %% [markdown]
# # Multilingual Language Detection via Sentence-NER (Token Classification)
# Fine-tunes XLM-RoBERTa to tag each token with its source language (BIO scheme),
# enabling transparent, evidence-based language identification.

# %%
# --- Environment Setup ---
# pip install evaluate pysbd faker seqeval
# %%
import random
import codecs
import re
import json
import gc
import multiprocessing as mp
import unicodedata
import traceback
from collections import defaultdict, deque
import string
from faker import Faker
import torch
import numpy as np
import evaluate
from datasets import (
    load_dataset,
    get_dataset_config_names,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)

import os
import glob
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
ARTICLES_PER_LANG = 10_000   # increase for a larger dataset
EXAMPLES_TARGET = 2_000_000  # synthetic mixed-language training examples to generate
RESERVE_FRACTION = 0.15   # fraction of each language's sentences kept for guaranteed coverage
MIN_RESERVED_SENTENCES = 4
MAX_RESERVED_SENTENCES = 20_000
MIN_COVERAGE_DOCS_PER_LANG = 2
MAX_COVERAGE_DOCS_PER_LANG = 5
MAX_WIKI_INDEX = ARTICLES_PER_LANG * 10
MAX_WIKI_SENTENCES = 200_000
MAX_WIKI_SENTENCES_BY_LANG = {
    "en": 300_000,
}
WIKI_ROLLING_STATS_WINDOW = 250
SENTENCES_DIR = "./sentences_cache"
os.makedirs(SENTENCES_DIR, exist_ok=True)
WIKI_TEMP_DIR = os.path.join(SENTENCES_DIR, "_wiki_tmp")
os.makedirs(WIKI_TEMP_DIR, exist_ok=True)
WIKI_SEGMENTATION_DEBUG_DIR = os.path.join(WIKI_TEMP_DIR, "segmentation_debug")
os.makedirs(WIKI_SEGMENTATION_DEBUG_DIR, exist_ok=True)
SYNTHETIC_CACHE = f"{SENTENCES_DIR}/synthetic_examples.parquet"
SYNTHETIC_CACHE_META = f"{SENTENCES_DIR}/synthetic_examples.meta.json"
SYNTHETIC_TEMP_DIR = os.path.join(SENTENCES_DIR, "_synthetic_tmp")
os.makedirs(SYNTHETIC_TEMP_DIR, exist_ok=True)
CACHE_DIR = f"{SENTENCES_DIR}/tokenized_dataset"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_META = f"{CACHE_DIR}/tokenized_dataset.meta.json"
CACHE_DEBUG_PARQUET = f"{CACHE_DIR}/tokenized_debug.parquet"
CACHE_DEBUG_META = f"{CACHE_DIR}/tokenized_debug.meta.json"
CACHE_VERSION = 2
TOKENIZED_CACHE_VERSION = 2
USE_SYNTHETIC_CACHE = True
FORCE_REBUILD_SYNTHETIC_CACHE = False
USE_TOKENIZED_CACHE = True
FORCE_REBUILD_TOKENIZED_CACHE = False
GENERATION_WORKERS = mp.cpu_count()
TOKENIZE_NUM_PROC = max(1, mp.cpu_count() // 2)

# Optional notebook-state placeholders.
# These let later cells run even if the generation cell was skipped.
lang_sentences: dict[str, list[str]] | None = None
smol_sentences: dict[str, list[str]] | None = None
reserved_sentence_pools: dict[str, deque[str]] | None = None
main_sentence_pools: dict[str, deque[str]] | None = None
coverage_plan: list[str] | None = None
generation_jobs: list[tuple[str, str | None]] | None = None
job_chunks: list[list[tuple[str, str | None]]] | None = None
reserved_worker_pools: list[dict[str, deque[str]]] | None = None
main_worker_pools: list[dict[str, deque[str]]] | None = None
coverage_examples: list[dict] | None = None
random_examples: list[dict] | None = None
raw_examples: list[dict] | None = None

# %%
# --- Language Configuration ---
# Script groups and their ISO codes.
# English gets its own tier so it does not have to compete with the rest of the Latin bucket.
LANGUAGE_GROUPS = {
    "English":      ["en"],
    "LatinCore":    ["es", "fr", "de", "it", "pt", "nl"],
    "LatinTier2":   ["vi", "tr", "la", "id", "ms", "af", "sq", "is", "no", "sv", "da", "fi", "hu", "pl", "cs", "ro"],
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
# --- Data Extraction ---
# Pull ~ARTICLES_PER_LANG sentences per language from the Wikipedia streaming dataset.

def parquet_path(lang: str) -> str:
    return os.path.join(SENTENCES_DIR, f"{lang}.parquet")


MIN_ARTICLE_CHARS_BY_GROUP: dict[str, int] = {
    "English": 2_000,
    "LatinCore": 2_000,
    "LatinTier2": 2_000,
    "Cyrillic": 2_000,
    "EastAsian": 1_200,
    "Indic": 2_000,
    "ArabicScript": 2_000,
    "OtherScripts": 2_000,
}
MIN_ARTICLE_CHARS_DEFAULT = 3_000
WIKI_MARKUP = re.compile(r"\[\[.*?\]\]|\{\{.*?\}\}|==.*?==", flags=re.DOTALL)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WIKI_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
BRACKET_NOTES = re.compile(r"\s*[\(\[【（][^\)\]】）]{0,60}[\)\]】）]\s*")
WIKI_ASCII_WORDS = re.compile(r"[A-Za-z]+")
WIKI_SPACES = re.compile(r"\s{2,}")
WIKI_WRITE_BATCH_SIZE = 2048
WIKI_PUNCT_REPEAT = re.compile(r"([,.;:!?…،。！？])\1+")
WIKI_TRAILING_ORPHAN_LETTER = re.compile(r"[\s,.;:!?…،。！？]+([^\W\d_])$")
WIKI_LEADING_ORPHAN_LETTER = re.compile(r"^[\"'“”‘’«»‹›\s,.;:!?…،。！？]+([^\W\d_])\s+")
WIKI_BLOCKED_MARKERS = ("http",)
WIKI_BLOCKED_CHARS = {"=", "<", ">", "|"}
WIKI_OPENING_QUOTES = {"\"", "'", "“", "”", "‘", "’", "«", "»", "‹", "›"}
LENGTH_PRIORITY_SCAN_LIMIT = int(MAX_WIKI_INDEX // 1.5)
LENGTH_PRIORITY_SENTENCE_CAP_BY_LANG = {
    "vi": 50_000,
    "sv": 50_000,
    "lo": 20_000,
    "sd": 20_000,
    "am": 20_000,
    "km": 20_000,
    "ug": 20_000,
    "my": 20_000,
}
LENGTH_PRIORITY_LANGS = set(LENGTH_PRIORITY_SENTENCE_CAP_BY_LANG)
LENGTH_PRIORITY_SENTENCE_CAP = 25_000

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
    "hy": (30, 500), "ka": (25, 450), "en": (24, 600),
}
_DEFAULT_BOUNDS = (30, 600)
WIKI_NON_CONTENT = re.compile(r"[\W_]+", flags=re.UNICODE)
WIKI_DIGITS = re.compile(r"\d")
MAX_DIGIT_RATIO = 0.10
WIKI_WORDS = re.compile(r"\b\w+\b", flags=re.UNICODE)
LATIN_GROUPS = {"English", "LatinCore", "LatinTier2"}
MIN_LATIN_WORDS = 4


def _non_punct_char_count(s: str) -> int:
    """Count visible characters after stripping punctuation and whitespace."""
    return len(WIKI_NON_CONTENT.sub("", s))


def _digit_count(s: str) -> int:
    """Count digit characters in a sentence."""
    return len(WIKI_DIGITS.findall(s))


def _word_count(s: str) -> int:
    """Count Unicode word-like tokens in a sentence."""
    return len(WIKI_WORDS.findall(s))


def _strip_bracket_notes(text: str) -> str:
    """Remove short parenthetical/bracketed notes that are usually noise."""
    return BRACKET_NOTES.sub(" ", text)


def _collapse_spaces(text: str) -> str:
    """Collapse repeated whitespace to a single space."""
    return WIKI_SPACES.sub(" ", text)


def _strip_leading_punct(sentence: str) -> str:
    """Strip leading punctuation unless the sentence starts with a quote."""
    sentence = sentence.lstrip()
    if not sentence or sentence[0] in WIKI_OPENING_QUOTES:
        return sentence

    idx = 0
    while idx < len(sentence):
        ch = sentence[idx]
        if ch.isspace():
            idx += 1
            continue
        if unicodedata.category(ch).startswith("P"):
            idx += 1
            continue
        break
    return sentence[idx:].lstrip()


def _collapse_repeated_punct(sentence: str) -> str:
    """Collapse repeated punctuation runs like ',,' or '..' to a single char."""
    return WIKI_PUNCT_REPEAT.sub(r"\1", sentence)


def _strip_trailing_orphan_letter(sentence: str, lang: str) -> str:
    """Drop a trailing orphan letter after punctuation for non-Latin scripts."""
    if LANG_TO_GROUP.get(lang) in LATIN_GROUPS:
        return sentence
    sentence = WIKI_TRAILING_ORPHAN_LETTER.sub("", sentence)
    return sentence.rstrip()


def _strip_leading_orphan_letter(sentence: str, lang: str) -> str:
    """Drop a leading orphan letter after quote/punctuation."""
    return WIKI_LEADING_ORPHAN_LETTER.sub("", sentence).lstrip()


def _has_blocked_artifact(sentence: str) -> bool:
    """Return True for obvious markup / URL artifacts that should be dropped."""
    lower = sentence.lower()
    return any(marker in lower for marker in WIKI_BLOCKED_MARKERS) or any(
        ch in sentence for ch in WIKI_BLOCKED_CHARS
    )


def _post_clean_wiki_sentence(sentence: str, lang: str) -> str:
    """Normalize a wiki sentence after extraction and before synthetic sampling."""
    sentence = clean_wiki_sentence(sentence, lang)
    sentence = _strip_leading_punct(sentence)
    sentence = _strip_leading_orphan_letter(sentence, lang)
    sentence = _collapse_repeated_punct(sentence)
    sentence = _strip_trailing_orphan_letter(sentence, lang)
    sentence = _collapse_spaces(sentence)
    return sentence.strip()


def _is_post_clean_wiki_sentence_valid(sentence: str, lang: str) -> bool:
    """Re-run sentence validity after cleanup and punctuation normalization."""
    return bool(sentence) and _is_valid_sentence(sentence, lang)


def post_clean_wiki_sentences(sentences: list[str], lang: str) -> list[str]:
    """Apply the final wiki cleanup pass, including dedupe and artifact filtering."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
        if _has_blocked_artifact(sentence):
            continue
        sentence = _post_clean_wiki_sentence(sentence, lang)
        if not sentence or _has_blocked_artifact(sentence):
            continue
        if sentence in seen:
            continue
        if _is_post_clean_wiki_sentence_valid(sentence, lang):
            seen.add(sentence)
            cleaned.append(sentence)
    return cleaned


WIKI_CLEANUP_META = os.path.join(SENTENCES_DIR, "wiki_cleanup.meta.json")


def _load_cleanup_meta() -> dict[str, dict]:
    """Load the global wiki cleanup sidecar metadata."""
    if not os.path.exists(WIKI_CLEANUP_META):
        return {}
    try:
        with open(WIKI_CLEANUP_META, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_cleanup_meta(meta: dict[str, dict]) -> None:
    """Atomically write the global wiki cleanup sidecar metadata."""
    _write_json_atomic(WIKI_CLEANUP_META, meta)


def _cleanup_fingerprint(path: str) -> dict[str, int]:
    """Return a lightweight fingerprint for a parquet file."""
    stat = os.stat(path)
    return {
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }

def finalize_wiki_sentence_cache(sentence_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Apply the final wiki cleanup pass after wiki + SMOL loading,
    then rewrite the cached parquet files so later runs reuse the cleaned version.
    """

    cleaned_map: dict[str, list[str]] = {}
    total_before = 0
    total_after = 0
    changed_langs = 0
    cleanup_meta = _load_cleanup_meta()

    langs = sorted(sentence_map.keys())

    # Global progress bar
    with tqdm(total=len(langs), desc="Languages", unit="lang") as pbar_langs:
        for lang in langs:
            sentences = sentence_map[lang]
            path = parquet_path(lang)
            fingerprint = _cleanup_fingerprint(path) if os.path.exists(path) else {}
            meta_entry = cleanup_meta.get(lang, {})
            already_cleaned = (
                bool(meta_entry.get("cleaned"))
                and meta_entry.get("path") == path
                and meta_entry.get("input_fingerprint") == fingerprint
            )

            # Per-language progress bar (closes automatically)
            with tqdm(
                total=len(sentences),
                desc=f"{lang} cleanup",
                unit="sent",
                leave=False
            ) as pbar_sent:

                if already_cleaned:
                    cleaned = sentences
                else:
                    # Run cleaning
                    cleaned = post_clean_wiki_sentences(sentences, lang)

                # Advance per-language bar to completion
                pbar_sent.update(len(sentences))

            # Metrics
            before = len(sentences)
            after = len(cleaned)
            total_before += before
            total_after += after

            if cleaned != sentences:
                changed_langs += 1
                _write_sentence_parquet(path, cleaned)

            cleaned_map[lang] = cleaned

            cleanup_meta[lang] = {
                "path": path,
                "cleaned": True,
                "input_fingerprint": fingerprint,
                "input_sentence_count": before,
                "cleaned_sentence_count": after,
                "updated": cleaned != sentences,
            }

            # Advance global bar
            pbar_langs.update(1)

    _write_cleanup_meta(cleanup_meta)
    print(
        f"\nWiki post-clean: {changed_langs} languages updated | "
        f"{total_before:,} -> {total_after:,} sentences"
    )

    return cleaned_map


def _is_valid_sentence(s: str, lang: str) -> bool:
    mn, mx = _SENT_BOUNDS.get(lang, _DEFAULT_BOUNDS)
    visible = _non_punct_char_count(s)
    if not (mn < visible < mx):
        return False
    if LANG_TO_GROUP.get(lang) in LATIN_GROUPS and _word_count(s) < MIN_LATIN_WORDS:
        return False
    digits = _digit_count(s)
    return digits <= visible * MAX_DIGIT_RATIO


def _article_min_chars(lang: str) -> int:
    """Return the minimum article length for a language."""
    group = LANG_TO_GROUP.get(lang)
    return MIN_ARTICLE_CHARS_BY_GROUP.get(group, MIN_ARTICLE_CHARS_DEFAULT)  # type: ignore


def _split_wiki_paragraphs(text: str, lang: str) -> list[str] | None:
    """Split an article into cleaned paragraphs, or None if too short."""
    if len(text) < _article_min_chars(lang):
        return None
    text = WIKI_MARKUP.sub("", text)
    paragraphs = [
        p.strip()
        for p in WIKI_PARAGRAPH_SPLIT.split(text)
        if p.strip()
    ]
    return paragraphs or None


def prepare_wiki_paragraphs(text: str, lang: str) -> list[str] | None:
    """Return a language-aware paragraph slice, or None if the article is too short."""
    paragraphs = _split_wiki_paragraphs(text, lang)
    if paragraphs is None:
        return None

    fraction = 0.75 if lang == "en" else 0.50
    cutoff = max(1, int(len(paragraphs) * fraction))
    selected = paragraphs[:cutoff]
    selected.sort(key=len, reverse=True)
    return selected


def _log_segmentation_failure(
    lang: str,
    article_idx: int,
    paragraph_idx: int,
    paragraph: str,
    exc: Exception,
    article_title: str = "",
) -> None:
    """Write a debug record for a paragraph that makes pysbd fail."""
    snippet = paragraph[:800].replace("\n", " ")
    log_path = os.path.join(WIKI_SEGMENTATION_DEBUG_DIR, f"{lang}.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"lang={lang} article_idx={article_idx} paragraph_idx={paragraph_idx}\n"
            f"title={article_title!r}\n"
            f"error={type(exc).__name__}: {exc}\n"
            f"snippet={snippet}\n"
            f"traceback={traceback.format_exc()}\n"
            f"{'-' * 100}\n"
        )
    print(
        f"  pysbd failed for lang={lang} article={article_idx} paragraph={paragraph_idx} "
        f"-> {log_path}"
    )


def _sanitize_paragraph_for_pysbd(paragraph: str) -> str:
    """Remove backslash-heavy markup that can trip pysbd's regex cleaner."""
    if "\\" not in paragraph:
        return paragraph
    return paragraph.replace("\\", " ")


def _extract_article_sentences(
    article_text: str,
    lang: str,
    segmenter,
    article_idx: int,
    article_title: str = "",
) -> list[str]:
    """Extract cleaned sentences from one article text."""
    paragraphs = prepare_wiki_paragraphs(article_text, lang)
    if paragraphs is None:
        return []

    article_batch: list[str] = []
    for paragraph_idx, paragraph in enumerate(paragraphs):
        safe_paragraph = _sanitize_paragraph_for_pysbd(paragraph)
        try:
            sents = segmenter.segment(safe_paragraph) if segmenter else SENT_SPLIT.split(safe_paragraph) # type: ignore
        except re.error as exc:
            _log_segmentation_failure(
                lang,
                article_idx,
                paragraph_idx,
                paragraph,
                exc,
                article_title=article_title,
            )
            sents = SENT_SPLIT.split(safe_paragraph)
        for s in sents:
            s = clean_wiki_sentence(s, lang)
            if _is_valid_sentence(s, lang):
                article_batch.append(s)

    return article_batch


def _collect_priority_articles(dataset, lang: str, scan_limit: int) -> list[tuple[int, int, dict]]:
    """Collect a bounded pool of long articles and sort them by length descending."""
    min_chars = _article_min_chars(lang)
    candidates: list[tuple[int, int, dict]] = []
    for article_idx, article in enumerate(dataset.take(scan_limit)):
        article_text = article.get("text", "")
        if len(article_text) < min_chars:
            continue
        candidates.append((len(article_text), article_idx, article))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def _strip_ascii_for_lang(lang: str) -> bool:
    """Return True when we should scrub ASCII words from a language's text."""
    return LANG_TO_GROUP.get(lang) not in LATIN_GROUPS


def clean_wiki_sentence(sentence: str, lang: str) -> str:
    """Remove parenthetical text and extra whitespace from a sentence."""
    if "\\" in sentence:
        sentence = sentence.replace("\\", "")
    sentence = _strip_bracket_notes(sentence)
    if _strip_ascii_for_lang(lang):
        sentence = WIKI_ASCII_WORDS.sub("", sentence)
    sentence = _collapse_spaces(sentence)
    return sentence.strip()


def temp_parquet_path(lang: str) -> str:
    """Return the temporary parquet path used while extracting one language."""
    return os.path.join(WIKI_TEMP_DIR, f"{lang}.parquet")


def temp_meta_path(lang: str) -> str:
    """Return the JSON sidecar used to resume a partially written temp parquet."""
    return os.path.join(WIKI_TEMP_DIR, f"{lang}.meta.json")


def max_wiki_sentences_for_lang(lang: str) -> int:
    """Return the sentence cap for a language, with per-language overrides."""
    return MAX_WIKI_SENTENCES_BY_LANG.get(lang, MAX_WIKI_SENTENCES)


def max_length_priority_sentences_for_lang(lang: str) -> int:
    """Return a smaller cap for the length-priority extraction path."""
    return min(
        max_wiki_sentences_for_lang(lang),
        LENGTH_PRIORITY_SENTENCE_CAP_BY_LANG.get(lang, LENGTH_PRIORITY_SENTENCE_CAP),
    )


def _rolling_avg(values: deque[int]) -> float:
    """Return the mean of the values in a rolling window."""
    return round(sum(values) / len(values), 1) if values else 0.0


def _wiki_checkpoint_payload(
    lang: str,
    next_article_idx: int,
    accepted_articles: int,
    miss_streak: int,
    n_articles: int,
    committed_sentences: list[str],
    article_lengths_window: deque[int],
    sentence_lengths_window: deque[int],
) -> dict:
    """Build the temp meta payload with checkpoint and tuning stats."""
    return {
        "lang": lang,
        "next_article_idx": next_article_idx,
        "accepted_articles": accepted_articles,
        "miss_streak": miss_streak,
        "final_target_articles": n_articles,
        "committed_sentence_count": len(committed_sentences),
        "rolling_avg_article_chars": _rolling_avg(article_lengths_window),
        "rolling_avg_sentence_chars": _rolling_avg(sentence_lengths_window),
        "rolling_article_lengths": list(article_lengths_window),
        "seed": SEED,
    }


def _write_sentence_parquet(path: str, sentences: list[str]) -> None:
    """Write a sentence list to parquet, preferring pyarrow for compact output."""
    if not sentences:
        pd.DataFrame({"sentence": []}).to_parquet(path, index=False)
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pd.DataFrame({"sentence": sentences}).to_parquet(path, index=False)
        return
    table = pa.table({"sentence": pa.array(sentences, type=pa.string())})
    pq.write_table(table, path)


def _write_sentence_batch(writer, pa, batch: list[str]) -> None:
    """Write a small batch of extracted sentences to an open parquet writer."""
    if not batch:
        return
    table = pa.table({"sentence": pa.array(batch, type=pa.string())})
    writer.write_table(table)


def _write_json_atomic(path: str, payload: dict) -> None:
    """Write JSON through a temporary file so checkpoint updates stay atomic-ish."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _synthetic_worker_temp_path(worker_idx: int) -> str:
    """Return the temp parquet path for a synthetic-doc worker."""
    return os.path.join(SYNTHETIC_TEMP_DIR, f"worker_{worker_idx}.parquet")


def _write_synthetic_examples_parquet(path: str, coverage_examples: list[dict], random_examples: list[dict]) -> None:
    """Write synthetic examples to parquet with kind/original_text/tokens/ner_tags columns."""
    rows = []
    for kind, examples in (("coverage", coverage_examples), ("random", random_examples)):
        for example in examples:
            rows.append(
                {
                    "kind": kind,
                    "original_text": example.get("original_text", ""),
                    "tokens": json.dumps(example["tokens"]),
                    "ner_tags": json.dumps(example["ner_tags"]),
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _read_synthetic_examples_parquet(path: str) -> tuple[list[dict], list[dict]]:
    """Read a synthetic-example parquet shard back into coverage/random lists."""
    df = pd.read_parquet(path)
    coverage_examples: list[dict] = []
    random_examples: list[dict] = []
    for row in df.itertuples(index=False):
        example = {
            "original_text": getattr(row, "original_text", ""),
            "tokens": json.loads(row.tokens), # type: ignore
            "ner_tags": json.loads(row.ner_tags), # type: ignore
        }
        if not example["original_text"]:
            example["original_text"] = " ".join(example["tokens"])
        if row.kind == "coverage":
            coverage_examples.append(example)
        else:
            random_examples.append(example)
    return coverage_examples, random_examples


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


def extract_sentences_from_wiki(lang: str, n_articles: int = ARTICLES_PER_LANG) -> str:
    """Stream Wikipedia articles and write the cleaned sentences to a temp parquet."""
    segmenter = _get_segmenter(lang)
    fetch_target = n_articles * 20
    final_path = parquet_path(lang)
    temp_path = temp_parquet_path(lang)
    meta_path = temp_meta_path(lang)

    if lang in LENGTH_PRIORITY_LANGS:
        scan_limit = min(fetch_target, LENGTH_PRIORITY_SCAN_LIMIT)
        sentence_cap = max_length_priority_sentences_for_lang(lang)
        print(
            f"  Length-priority mode enabled for {lang} "
            f"(scan_limit={scan_limit}, sentence_cap={sentence_cap}, "
            f"min_chars={_article_min_chars(lang)})"
        )
        dataset = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{lang}",
            split="train",
            streaming=True,
        )
        dataset = dataset.shuffle(buffer_size=1000, seed=SEED)
        priority_articles = _collect_priority_articles(dataset, lang, scan_limit)

        committed_sentences: list[str] = []
        sentence_lengths_window: deque[int] = deque(maxlen=WIKI_ROLLING_STATS_WINDOW)
        article_lengths_window: deque[int] = deque(maxlen=WIKI_ROLLING_STATS_WINDOW)
        accepted_articles = 0

        with tqdm(
            total=sentence_cap,
            desc=f"{lang} sentences",
            unit="sentence",
            leave=False,
            dynamic_ncols=True,
        ) as bar:
            for article_len, article_idx, article in priority_articles:
                if len(committed_sentences) >= sentence_cap or accepted_articles >= n_articles:
                    break

                article_text = article.get("text", "")
                article_title = article.get("title", "")
                article_lengths_window.append(article_len)

                article_batch = _extract_article_sentences(
                    article_text,
                    lang,
                    segmenter,
                    article_idx,
                    article_title=article_title,
                )
                if not article_batch:
                    continue

                remaining_sentences = sentence_cap - len(committed_sentences)
                if remaining_sentences <= 0:
                    break
                if len(article_batch) > remaining_sentences:
                    article_batch = article_batch[:remaining_sentences]

                committed_sentences.extend(article_batch)
                sentence_lengths_window.extend(len(s) for s in article_batch)
                accepted_articles += 1
                bar.update(len(article_batch))
                bar.set_postfix_str(
                    f"acc {accepted_articles}/{n_articles} | "
                    f"sent {len(committed_sentences)} | "
                    f"avgA {_rolling_avg(article_lengths_window):.0f} | "
                    f"avgS {_rolling_avg(sentence_lengths_window):.0f}"
                )

        committed_sentences = post_clean_wiki_sentences(committed_sentences, lang)
        _write_sentence_parquet(final_path, committed_sentences)
        return final_path

    committed_sentences: list[str] = []
    sentence_cap = max_wiki_sentences_for_lang(lang)
    sentence_lengths_window: deque[int] = deque(
        (len(s) for s in committed_sentences[-WIKI_ROLLING_STATS_WINDOW:]),
        maxlen=WIKI_ROLLING_STATS_WINDOW,
    )
    article_lengths_window: deque[int] = deque(maxlen=WIKI_ROLLING_STATS_WINDOW)
    next_article_idx = 0
    accepted_articles = 0
    miss_streak = 0
    completed_cleanly = False
    if os.path.exists(temp_path) and os.path.exists(meta_path):
        try:
            committed_sentences = pd.read_parquet(temp_path)["sentence"].tolist()
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if "accepted_articles" not in meta or "next_article_idx" not in meta:
                raise ValueError("stale checkpoint metadata")
            next_article_idx = int(meta["next_article_idx"])
            accepted_articles = int(meta["accepted_articles"])
            miss_streak = int(meta.get("miss_streak", 0))
            sentence_lengths_window.extend(
                len(s) for s in committed_sentences[-WIKI_ROLLING_STATS_WINDOW:]
            )
            article_lengths_window.extend(int(v) for v in meta.get("rolling_article_lengths", []))
            print(
                f"  Resuming {lang} from stream article {next_article_idx} "
                f"with {accepted_articles} accepted articles"
            )
        except Exception:
            committed_sentences = []
            next_article_idx = 0
            accepted_articles = 0
            miss_streak = 0
            for path in (temp_path, meta_path):
                if os.path.exists(path):
                    os.remove(path)
    elif os.path.exists(temp_path) or os.path.exists(meta_path):
        for path in (temp_path, meta_path):
            if os.path.exists(path):
                os.remove(path)

    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(buffer_size=1000, seed=SEED)

    def _update_progress(bar, scanned_articles: int) -> None:
        """Show both scan progress and productive yield in the tqdm footer."""
        bar.set_postfix_str(
            f"acc {accepted_articles}/{n_articles} | "
            f"sent {len(committed_sentences)} | "
            f"yield {accepted_articles / max(1, scanned_articles):.1%} | "
            f"avgA {_rolling_avg(article_lengths_window):.0f} | "
            f"avgS {_rolling_avg(sentence_lengths_window):.0f} | "
            f"miss {miss_streak}"
        )

    try:
        with tqdm(
            total=sentence_cap,
            initial=len(committed_sentences),
            desc=f"{lang} sentences",
            unit="sentence",
            leave=False,
            dynamic_ncols=True,
        ) as bar:
            for article_idx, article in enumerate(dataset.take(fetch_target)):
                if article_idx < next_article_idx:
                    _update_progress(bar, article_idx + 1)
                    continue
                if article_idx >= MAX_WIKI_INDEX:
                    print(
                        f"  Stopping {lang} at article index {article_idx} "
                        f"(MAX_WIKI_INDEX={MAX_WIKI_INDEX})"
                    )
                    break

                article_text = article.get("text", "")
                article_title = article.get("title", "")
                article_lengths_window.append(len(article_text))

                article_batch = _extract_article_sentences(
                    article_text,
                    lang,
                    segmenter,
                    article_idx,
                    article_title=article_title,
                )
                if not article_batch:
                    next_article_idx = article_idx + 1
                    miss_streak += 1
                    _write_sentence_parquet(temp_path, committed_sentences)
                    _write_json_atomic(
                        meta_path,
                        _wiki_checkpoint_payload(
                            lang,
                            next_article_idx,
                            accepted_articles,
                            miss_streak,
                            n_articles,
                            committed_sentences,
                            article_lengths_window,
                            sentence_lengths_window,
                        ),
                    )
                    _update_progress(bar, article_idx + 1)
                    continue

                remaining_sentences = sentence_cap - len(committed_sentences)
                if remaining_sentences <= 0:
                    print(
                        f"  Stopping {lang} after reaching sentence cap="
                        f"{sentence_cap}"
                    )
                    break

                if len(article_batch) > remaining_sentences:
                    article_batch = article_batch[:remaining_sentences]

                committed_sentences.extend(article_batch)
                sentence_lengths_window.extend(len(s) for s in article_batch)
                accepted_articles += 1
                miss_streak = 0
                _write_sentence_parquet(temp_path, committed_sentences)
                _write_json_atomic(
                    meta_path,
                    _wiki_checkpoint_payload(
                        lang,
                        article_idx + 1,
                        accepted_articles,
                        miss_streak,
                        n_articles,
                        committed_sentences,
                        article_lengths_window,
                        sentence_lengths_window,
                    ),
                )
                next_article_idx = article_idx + 1
                bar.update(len(article_batch))
                _update_progress(bar, article_idx + 1)
                if len(committed_sentences) >= sentence_cap:
                    print(
                        f"  Stopping {lang} after reaching sentence cap="
                        f"{sentence_cap}"
                    )
                    break
                if accepted_articles >= n_articles:
                    break

        committed_sentences = post_clean_wiki_sentences(committed_sentences, lang)
        _write_sentence_parquet(final_path, committed_sentences)
        completed_cleanly = True
    finally:
        if completed_cleanly:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)

    return final_path


def load_or_extract(lang: str) -> tuple[str, str]:
    """Return the parquet path for a language, extracting it if needed."""
    path = parquet_path(lang)
    if os.path.exists(path):
        cached = pd.read_parquet(path)["sentence"].tolist()
        cleaned = post_clean_wiki_sentences(cached, lang)
        if cleaned != cached:
            _write_sentence_parquet(path, cleaned)
        return lang, path
    extracted_path = extract_sentences_from_wiki(lang)
    return lang, extracted_path


# ---------------------------------------------------------------------------
# SMOL augmentation
# ---------------------------------------------------------------------------
# The loader lives in this notebook so the full data pipeline is self-contained.
# Most SMOL config tags already match our pipeline ISO codes, so we only list
# the aliases that need collapsing.
SMOL_CODE_MAP: dict[str, str] = {
    "pt-PT": "pt",
    "yue": "zh",
    "ar-MA": "ar",
    "arz": "ar",
    "aeb": "ar",
    "ayl": "ar",
    "apd": "ar",
}

MAX_SENTENCES_PER_LANG = 5_000
UNCAPPED_LANGS: set[str] = set()

SMOL_CACHE_DIR = SENTENCES_DIR
SMOL_CACHE_FILE = os.path.join(SMOL_CACHE_DIR, "smol_sentences.json")
os.makedirs(SMOL_CACHE_DIR, exist_ok=True)

def _smol_parse_config(config: str) -> tuple[str, str] | None:
    try:
        _, pair = config.split("__", 1)
        sl, tl = pair.split("_", 1)
        return sl, tl
    except ValueError:
        return None


def _smol_add_sentences(bucket: list[str], raw_sentences: object, lang: str) -> None:
    """Normalize and append a sequence of SMOL sentences into a bucket."""
    if isinstance(raw_sentences, str):
        raw_sentences = [raw_sentences]
    if not isinstance(raw_sentences, list):
        return

    for sent in raw_sentences:
        if not isinstance(sent, str):
            continue
        sent = _strip_bracket_notes(sent)
        sent = _collapse_spaces(sent)
        if _is_valid_sentence(sent, lang):
            bucket.append(sent)


def _load_smoldoc(accumulator: dict[str, list[str]]) -> tuple[set[str], set[str]]:
    configs = [c for c in get_dataset_config_names("google/smol") if c.startswith("smoldoc__")]
    source_langs_seen: set[str] = set()
    target_langs_seen: set[str] = set()
    for config in tqdm(configs, desc="SmolDoc subsets"):
        parsed = _smol_parse_config(config)
        if parsed is None:
            continue
        sl, tl = parsed
        mapped = SMOL_CODE_MAP.get(tl, tl)
        if mapped not in LANG_TO_GROUP:
            continue

        ds = load_dataset("google/smol", config, split="train", trust_remote_code=True)
        target_bucket = accumulator.setdefault(mapped, [])
        target_langs_seen.add(mapped)
        if sl in LANG_TO_GROUP and sl not in source_langs_seen:
            source_bucket = accumulator.setdefault(sl, [])
            for row in ds:
                _smol_add_sentences(source_bucket, row.get("srcs") or row.get("src"), sl) # type: ignore
            source_langs_seen.add(sl)

        for row in ds:
            _smol_add_sentences(target_bucket, row.get("trgs"), mapped)  # type: ignore

    return source_langs_seen, target_langs_seen


def _load_smolsent(accumulator: dict[str, list[str]]) -> tuple[set[str], set[str]]:
    configs = [c for c in get_dataset_config_names("google/smol") if c.startswith("smolsent__")]
    source_langs_seen: set[str] = set()
    target_langs_seen: set[str] = set()
    for config in tqdm(configs, desc="SmolSent subsets"):
        parsed = _smol_parse_config(config)
        if parsed is None:
            continue
        sl, tl = parsed
        mapped = SMOL_CODE_MAP.get(tl, tl)
        if mapped not in LANG_TO_GROUP:
            continue

        ds = load_dataset("google/smol", config, split="train", trust_remote_code=True)
        target_bucket = accumulator.setdefault(mapped, [])
        target_langs_seen.add(mapped)
        if sl in LANG_TO_GROUP and sl not in source_langs_seen:
            source_bucket = accumulator.setdefault(sl, [])
            for row in ds:
                _smol_add_sentences(source_bucket, row.get("srcs") or row.get("src"), sl) # type: ignore
            source_langs_seen.add(sl)

        seg = _get_segmenter(mapped)

        for row in ds:
            trg = row.get("trg") or "" # type: ignore
            if not isinstance(trg, str):
                continue
            sents = seg.segment(trg) if seg else [trg]  # type: ignore
            _smol_add_sentences(target_bucket, sents, mapped)

    return source_langs_seen, target_langs_seen


def load_smol_sentences(force_rebuild: bool = False, seed: int = 42) -> dict[str, list[str]]:
    """Load, dedupe, shuffle, and cache google/smol sentences by ISO code."""
    if not force_rebuild and os.path.exists(SMOL_CACHE_FILE):
        print(f"Loading SMOL cache from {SMOL_CACHE_FILE}")
        with open(SMOL_CACHE_FILE, encoding="utf-8") as f:
            cached: dict[str, list[str]] = json.load(f)
        print(f"  {len(cached)} languages | {sum(len(v) for v in cached.values()):,} sentences total")
        return cached

    accumulator: dict[str, list[str]] = {}

    print("Loading SmolDoc ...")
    smoldoc_src_langs, smoldoc_target_langs = _load_smoldoc(accumulator)

    print("Loading SmolSent ...")
    smolsent_src_langs, smolsent_target_langs = _load_smolsent(accumulator)

    print(f"SMOL src languages loaded once: {len(smoldoc_src_langs | smolsent_src_langs)}")
    print(f"SMOL target languages loaded: {len(smoldoc_target_langs | smolsent_target_langs)}")

    rng = random.Random(seed)
    result: dict[str, list[str]] = {}
    for lang, sents in sorted(accumulator.items()):
        seen: set[str] = set()
        deduped: list[str] = []
        for sent in sents:
            if sent not in seen:
                seen.add(sent)
                deduped.append(sent)
        rng.shuffle(deduped)
        cap = None if lang in UNCAPPED_LANGS else MAX_SENTENCES_PER_LANG
        result[lang] = deduped if cap is None else deduped[:cap]

    with open(SMOL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in result.values())
    print(f"\nSMOL sentences cached -> {SMOL_CACHE_FILE}")
    print(f"  {len(result)} languages | {total:,} sentences total")
    for lang in sorted(result):
        print(f"  {lang:<6}  {len(result[lang]):>5} sentences")

    return result

# ProcessPoolExecutor saturates CPU cores for segmentation work.
# Workers == min(cpu_count, n_langs) — no point exceeding either.
MAX_WORKERS = min(mp.cpu_count() // 2, len(ALL_LANGS))

print(f"Extracting sentences \u2192 cached under \'{SENTENCES_DIR}/'")
print(f"(Workers: {MAX_WORKERS} processes | cached languages skip extraction)\n")

lang_sentences = {}
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(load_or_extract, lang): lang for lang in ALL_LANGS}
    for future in tqdm(as_completed(futures), total=len(ALL_LANGS), desc="Languages"):
        lang, cache_path = future.result()
        sentences = pd.read_parquet(cache_path)["sentence"].tolist()
        lang_sentences[lang] = sentences
        tqdm.write(f"  {lang}: {len(sentences)} sentences  \u2192  {cache_path}")

# Optional SMOL augmentation from google/smol (SmolDoc + SmolSent).
USE_SMOL_AUGMENTATION = True
SMOL_FORCE_REBUILD = False
if USE_SMOL_AUGMENTATION:
    try:
        smol_sentences = load_smol_sentences(force_rebuild=SMOL_FORCE_REBUILD)
        total_smol_sentences = sum(len(v) for v in smol_sentences.values())
        print(
            f"\nSMOL kept separate for pool split: "
            f"{len(smol_sentences)} languages | {total_smol_sentences} sentences"
        )
    except Exception as exc:
        print(f"\nSMOL augmentation skipped: {exc}")

lang_sentences = finalize_wiki_sentence_cache(lang_sentences)

# %%
# --- Neutral (O-label) Corpus ---
# Four complementary O-label sources, all inserted as label-0 spans:
#   1. im2latex-100k     — real LaTeX formulas from arXiv papers
#   2. math_gen          — procedurally generated math expressions (14 domains)
#   3. symbol_noise      — random streams of unicode symbols, emoji, punctuation
#   4. gibberish          — ROT13 English / Faker text that should still be O
#
# Together they teach the model that equations, markup, noise, and gibberish
# do not belong to any language.

# Keep the cache helpers local to this cell so the noise pools can be rebuilt
# without needing to execute the wiki extraction cell first.
def _write_text_parquet(path: str, column_name: str, values: list[str]) -> None:
    """Write a text list to parquet under a named column."""
    if not values:
        pd.DataFrame({column_name: []}).to_parquet(path, index=False)
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pd.DataFrame({column_name: values}).to_parquet(path, index=False)
        return
    table = pa.table({column_name: pa.array(values, type=pa.string())})
    pq.write_table(table, path)


def _load_or_build_text_pool(
    path: str,
    column_name: str,
    builder,
    label: str,
) -> list[str]:
    """Load a cached text pool from parquet or build and cache it."""
    if os.path.exists(path):
        return pd.read_parquet(path)[column_name].tolist()
    values = builder()
    _write_text_parquet(path, column_name, values)
    print(f"  Cached {len(values)} {label} → {path}")
    return values

# ── 1. im2latex ────────────────────────────────────────────────────────────────
LATEX_CACHE     = f"{SENTENCES_DIR}/latex_formulas.parquet"
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
SYNTH_MATH_CACHE = f"{SENTENCES_DIR}/synth_math_pool.parquet"
synth_math_pool: list[str] = _load_or_build_text_pool(
    SYNTH_MATH_CACHE,
    "expression",
    lambda: [generate_synthetic_math() for _ in range(SYNTH_MATH_N)],
    "synthetic math expressions",
)
print(f"math_gen:    {len(synth_math_pool):>6} expressions")

# ── 3. code_noise ──────────────────────────────────────────────────────────────
# HTML tags, code fragments, logs, and configs with content removed.
_code_path = _pl.Path(__file__).parent / "code_noise.py"
_code_spec = _ilu.spec_from_file_location("code_noise", _code_path)
_code = _ilu.module_from_spec(_code_spec) # type: ignore
_code_spec.loader.exec_module(_code) # type: ignore
generate_html_artifact = _code.generate_html_artifact
generate_css_artifact = _code.generate_css_artifact
generate_code_artifact = _code.generate_code_artifact

HTML_NOISE_N = 30_000
HTML_NOISE_CACHE = f"{SENTENCES_DIR}/html_noise_pool.parquet"
html_noise_pool: list[str] = _load_or_build_text_pool(
    HTML_NOISE_CACHE,
    "snippet",
    lambda: [generate_html_artifact() for _ in range(HTML_NOISE_N)],
    "HTML noise snippets",
)
print(f"html noise:  {len(html_noise_pool):>6} snippets")

CSS_NOISE_N = 30_000
CSS_NOISE_CACHE = f"{SENTENCES_DIR}/css_noise_pool.parquet"
css_noise_pool: list[str] = _load_or_build_text_pool(
    CSS_NOISE_CACHE,
    "snippet",
    lambda: [generate_css_artifact() for _ in range(CSS_NOISE_N)],
    "CSS noise snippets",
)
print(f"css noise:   {len(css_noise_pool):>6} snippets")

CODE_NOISE_N = 30_000
CODE_NOISE_CACHE = f"{SENTENCES_DIR}/code_noise_pool.parquet"
code_noise_pool: list[str] = _load_or_build_text_pool(
    CODE_NOISE_CACHE,
    "snippet",
    lambda: [generate_code_artifact() for _ in range(CODE_NOISE_N)],
    "code noise snippets",
)
print(f"code noise:  {len(code_noise_pool):>6} snippets")

# ── 4. Symbol / emoji noise ────────────────────────────────────────────────────
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
NOISE_CACHE = f"{SENTENCES_DIR}/symbol_noise_pool.parquet"
noise_pool: list[str] = _load_or_build_text_pool(
    NOISE_CACHE,
    "snippet",
    lambda: [generate_symbol_noise() for _ in range(NOISE_N)],
    "symbol noise strings",
)
print(f"symbol noise:{len(noise_pool):>6} strings")

english_seed_sentences = (lang_sentences or {}).get("en", [])


def generate_gibberish_text() -> str:
    """Return rot13-transformed English or Faker-based filler text."""
    if english_seed_sentences and random.random() < 0.7:
        base = random.choice(english_seed_sentences)
    else:
        base = fake.text(max_nb_chars=random.randint(80, 240))
    gib = codecs.decode(base, "rot_13")
    return _collapse_spaces(gib).strip()


GIBBERISH_N = 30_000
GIBBERISH_CACHE = f"{SENTENCES_DIR}/gibberish_pool.parquet"
gibberish_pool: list[str] = _load_or_build_text_pool(
    GIBBERISH_CACHE,
    "snippet",
    lambda: [generate_gibberish_text() for _ in range(GIBBERISH_N)],
    "gibberish strings",
)
print(f"gibberish:  {len(gibberish_pool):>6} strings")

# ── Combined O-label pool ──────────────────────────────────────────────────────
# Weighted so real LaTeX and synthetic math dominate, while HTML, CSS, code, and
# gibberish remain visible but secondary.
_O_SOURCES  = [latex_formulas, synth_math_pool, html_noise_pool, css_noise_pool, code_noise_pool, noise_pool, gibberish_pool]
_O_WEIGHTS  = [0.28,           0.22,            0.14,           0.08,          0.10,         0.09,       0.09]

def sample_o_span() -> str:
    """Draw one O-label span from the combined pool."""
    pool = random.choices(_O_SOURCES, weights=_O_WEIGHTS, k=1)[0]
    return random.choice(pool)


def sample_code_span() -> str:
    """Draw one code artifact span."""
    return random.choice(code_noise_pool)

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
    reserve_fraction: float = RESERVE_FRACTION,
    min_reserved: int = MIN_RESERVED_SENTENCES,
    max_reserved: int = MAX_RESERVED_SENTENCES,
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
        reserve_target = int(round(len(shuffled) * reserve_fraction))
        reserve_n = min(
            len(shuffled),
            max(
                min_reserved,
                min(reserve_target, max_reserved),
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
    allow_source_reuse: bool = False,
) -> str | None:
    """
    Draw a sentence without replacement from the preferred pool, then fallback
    pool. Source reuse is optional and disabled by default so depleted languages
    stop being oversampled.
    """
    if primary_pool.get(lang):
        return primary_pool[lang].popleft()
    if fallback_pool and fallback_pool.get(lang):
        return fallback_pool[lang].popleft()
    if allow_source_reuse and source_pool and source_pool.get(lang):
        return random.choice(source_pool[lang])
    return None


def remaining_sentence_count(
    lang: str,
    primary_pool: dict[str, deque[str]],
    fallback_pool: dict[str, deque[str]] | None = None,
) -> int:
    """Return how many unused sentences remain for a language in the active pools."""
    total = len(primary_pool.get(lang, ()))
    if fallback_pool is not None:
        total += len(fallback_pool.get(lang, ()))
    return total


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
) -> str:
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
    coverage_examples = [ex for idx, ex in enumerate(examples) if jobs[idx][0] == "coverage"]
    random_examples = [ex for idx, ex in enumerate(examples) if jobs[idx][0] == "random"]
    temp_path = _synthetic_worker_temp_path(worker_idx)
    _write_synthetic_examples_parquet(temp_path, coverage_examples, random_examples)
    _write_json_atomic(
        temp_path.replace(".parquet", ".meta.json"),
        {
            "worker_idx": worker_idx,
            "seed": seed,
            "job_count": len(jobs),
            "coverage_count": len(coverage_examples),
            "random_count": len(random_examples),
        },
    )
    return temp_path


def save_synthetic_examples_cache(
    coverage_examples: list[dict],
    random_examples: list[dict],
) -> None:
    """Persist synthetic examples to Parquet plus a small metadata sidecar."""
    _write_synthetic_examples_parquet(SYNTHETIC_CACHE, coverage_examples, random_examples)
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


def save_token_debug_cache(coverage_examples: list[dict], random_examples: list[dict]) -> None:
    """Persist a flattened token-level debug parquet with original text and token ids."""
    rows = []
    for examples in (coverage_examples, random_examples):
        for example in examples:
            original_text = example.get("original_text", "")
            tokens = example["tokens"]
            ids = example["ner_tags"]
            for token, tag_id in zip(tokens, ids):
                rows.append(
                    {
                        "original_text": original_text,
                        "token": token,
                        "id": tag_id,
                    }
                )

    pd.DataFrame(rows).to_parquet(CACHE_DEBUG_PARQUET, index=False)
    with open(CACHE_DEBUG_META, "w") as f:
        json.dump(
            {
                "cache_version": TOKENIZED_CACHE_VERSION,
                "model_checkpoint": MODEL_CHECKPOINT,
                "max_length": MAX_LENGTH,
                "seed": SEED,
                "row_count": len(rows),
            },
            f,
            indent=2,
        )


def ensure_token_debug_cache() -> None:
    """Rebuild the token debug cache from synthetic examples if it is missing."""
    if os.path.exists(CACHE_DEBUG_PARQUET) and os.path.exists(CACHE_DEBUG_META):
        return
    cached_examples = load_synthetic_examples_cache() if (USE_SYNTHETIC_CACHE and not FORCE_REBUILD_SYNTHETIC_CACHE) else None
    if cached_examples is None:
        return
    coverage_examples, random_examples = cached_examples
    save_token_debug_cache(coverage_examples, random_examples)


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

    coverage_examples, random_examples = _read_synthetic_examples_parquet(SYNTHETIC_CACHE)
    return coverage_examples, random_examples


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
      - Groups are weighted by their "global footprint" so major language buckets
        appear more often than tail groups.
      - Languages are chosen from what is still available in the pools, so
        depleted languages naturally stop appearing.
      - `required_langs` can force coverage so every language appears at least
        once in the overall synthetic corpus.
    """
    # Weight per group — tuned to reflect real-world multilingual text distribution.
    GROUP_WEIGHTS = {
        # Give English a dedicated lane so it is not diluted by the broader Latin set.
        "English":      5.0,
        "LatinCore":    3.0,
        "LatinTier2":   1.5,
        "EastAsian":    2.5,
        "Cyrillic":     1.5,
        "Indic":        1.5,
        "ArabicScript": 1.5,
        "OtherScripts": 1.0,
    }
    chosen_langs: list[str] = []
    seen_langs: set[str] = set()

    def _candidate_langs() -> list[str]:
        candidates = [
            lang for lang in ALL_LANGS
            if remaining_sentence_count(lang, primary_pool, fallback_pool) > 0
            and lang not in seen_langs
        ]
        return candidates

    def _sample_language(candidates: list[str]) -> str | None:
        if not candidates:
            return None
        weights = [
            GROUP_WEIGHTS.get(LANG_TO_GROUP.get(lang, ""), 1.0)
            * remaining_sentence_count(lang, primary_pool, fallback_pool)
            for lang in candidates
        ]
        return random.choices(candidates, weights=weights, k=1)[0]

    # Always keep any requested languages. This is the coverage guarantee.
    for lang in required_langs or []:
        if remaining_sentence_count(lang, primary_pool, fallback_pool) > 0 and lang not in seen_langs:
            chosen_langs.append(lang)
            seen_langs.add(lang)

    # Sample remaining segments from the languages that still have usable sentence supply.
    n_remaining_segments = max(0, n_segments - len(chosen_langs))
    for _ in range(n_remaining_segments):
        lang = _sample_language(_candidate_langs())
        if lang is None:
            break
        if lang not in seen_langs:
            chosen_langs.append(lang)
            seen_langs.add(lang)

    all_tokens, all_labels = [], []
    original_text_parts: list[str] = []
    total_tokens = 0

    for lang in chosen_langs:
        if total_tokens >= MAX_LENGTH - 20:
            break
        sent = draw_sentence(lang, primary_pool, fallback_pool, source_pool, allow_source_reuse=False)
        if sent is None:
            continue
        original_text_parts.append(sent)
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
    if len(chosen_langs) == 1 and total_tokens < MAX_LENGTH - 20:
        span = sample_code_span()
        original_text_parts.append(span)
        code_tokens = tokenizer.tokenize(span)
        remaining = MAX_LENGTH - 2 - len(all_tokens)
        code_tokens = code_tokens[:min(remaining, 120)]  # code snippets can be longer than other O spans
        if code_tokens:
            insert_pos = random.randint(0, len(all_tokens))
            all_tokens = all_tokens[:insert_pos] + code_tokens + all_tokens[insert_pos:]
            all_labels = all_labels[:insert_pos] + [0] * len(code_tokens) + all_labels[insert_pos:]
            total_tokens += len(code_tokens)

    if random.random() < o_inject_prob and total_tokens < MAX_LENGTH - 20:
        n_injections = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(n_injections):
            if total_tokens >= MAX_LENGTH - 10:
                break
            span = sample_o_span()
            original_text_parts.append(span)
            o_tokens = tokenizer.tokenize(span)
            remaining = MAX_LENGTH - 2 - len(all_tokens)
            o_tokens = o_tokens[:min(remaining, 40)]  # cap single span at 40 tokens
            if o_tokens:
                insert_pos = random.randint(0, len(all_tokens))
                all_tokens = all_tokens[:insert_pos] + o_tokens + all_tokens[insert_pos:]
                all_labels = all_labels[:insert_pos] + [0] * len(o_tokens) + all_labels[insert_pos:]
                total_tokens += len(o_tokens)

    return {"original_text": " ".join(original_text_parts).strip(), "tokens": all_tokens, "ner_tags": all_labels}


print("Generating synthetic mixed-language documents …")
cached_examples = load_synthetic_examples_cache() if (USE_SYNTHETIC_CACHE and not FORCE_REBUILD_SYNTHETIC_CACHE) else None
if cached_examples is not None:
    coverage_examples, random_examples = cached_examples
    print(f"Loaded cached synthetic examples from {SYNTHETIC_CACHE}")
else:
    # Guarantee representation by reserving a fraction of each language's sentences,
    # then building several guaranteed documents from that reserved pool.
    reserved_sentence_pools, main_sentence_pools = build_sentence_pools(lang_sentences)
    reserved_total = sum(len(pool) for pool in reserved_sentence_pools.values())
    main_total = sum(len(pool) for pool in main_sentence_pools.values())

    if smol_sentences:
        smol_reserved_sentence_pools, smol_main_sentence_pools = build_sentence_pools(
            smol_sentences,
            reserve_fraction=0.5,
            min_reserved=1,
            max_reserved=MAX_RESERVED_SENTENCES,
        )
        for lang, pool in smol_reserved_sentence_pools.items():
            reserved_sentence_pools.setdefault(lang, deque()).extend(pool)
        for lang, pool in smol_main_sentence_pools.items():
            main_sentence_pools.setdefault(lang, deque()).extend(pool)
        smol_reserved_total = sum(len(pool) for pool in smol_reserved_sentence_pools.values())
        smol_main_total = sum(len(pool) for pool in smol_main_sentence_pools.values())
        print(
            f"SMOL split 50/50 -> reserved: {smol_reserved_total} | main: {smol_main_total}"
        )

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
    generation_jobs = [("coverage", lang) for lang in coverage_plan]
    generation_jobs.extend([("random", None)] * random_job_count)

    generation_workers = min(
        GENERATION_WORKERS,
        max(1, len(generation_jobs)),
    )
    generation_workers = min(generation_workers, len(generation_jobs) or 1)

    coverage_examples = []
    random_examples = []

    if generation_workers == 1:
        temp_path = generate_synthetic_examples_chunk(
            0,
            generation_jobs,
            reserved_sentence_pools,
            main_sentence_pools,
            lang_sentences,
        )
        coverage_examples, random_examples = _read_synthetic_examples_parquet(temp_path)
        for path in (temp_path, temp_path.replace(".parquet", ".meta.json")):
            if os.path.exists(path):
                os.remove(path)
    else:
        job_chunks = chunk_list(generation_jobs, generation_workers)
        reserved_worker_pools = partition_sentence_pools(reserved_sentence_pools, generation_workers)
        main_worker_pools = partition_sentence_pools(main_sentence_pools, generation_workers)
        coverage_examples = []
        random_examples = []

        with ProcessPoolExecutor(max_workers=generation_workers) as pool:
            future_to_jobs = {}
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
                future_to_jobs[future] = worker_idx

            for future in tqdm(as_completed(future_to_jobs), total=len(future_to_jobs), desc="Synthetic docs"):
                temp_path = future.result()
                chunk_coverage, chunk_random = _read_synthetic_examples_parquet(temp_path)
                coverage_examples.extend(chunk_coverage)
                random_examples.extend(chunk_random)
                for path in (temp_path, temp_path.replace(".parquet", ".meta.json")):
                    if os.path.exists(path):
                        os.remove(path)

    if USE_SYNTHETIC_CACHE:
        save_synthetic_examples_cache(coverage_examples, random_examples) # type: ignore

raw_examples = (coverage_examples or []) + (random_examples or [])

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


def release_wikipedia_generation_memory() -> None:
    """Drop the Wikipedia extraction and sentence-pool artifacts early."""
    for name in [
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
        "html_noise_pool",
        "css_noise_pool",
        "code_noise_pool",
        "noise_pool",
        "gibberish_pool",
        "_O_SOURCES",
    ]:
        globals()[name] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


release_wikipedia_generation_memory()

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
    encoding["original_text"] = example.get("original_text", " ".join(example["tokens"]))
    return encoding


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

    try:
        return load_from_disk(CACHE_DIR)
    except Exception:
        # Some Colab zips/extractions are happier if we reconstruct the split
        # datasets directly from their Arrow shards.
        split_names = ["train", "eval"]
        loaded_splits = {}
        for split_name in split_names:
            split_dir = os.path.join(CACHE_DIR, split_name)
            arrow_files = sorted(glob.glob(os.path.join(split_dir, "*.arrow")))
            if not arrow_files:
                return None
            split_parts = [Dataset.from_file(path) for path in arrow_files]
            loaded_splits[split_name] = (
                split_parts[0] if len(split_parts) == 1 else concatenate_datasets(split_parts)
            )
        return DatasetDict(loaded_splits)


cached_tokenized = load_tokenized_dataset_cache() if (USE_TOKENIZED_CACHE and not FORCE_REBUILD_TOKENIZED_CACHE) else None
if cached_tokenized is not None:
    train_dataset = cached_tokenized["train"]
    eval_dataset = cached_tokenized["eval"]
    print(f"Loaded tokenized dataset cache from {CACHE_DIR}")
    ensure_token_debug_cache()
else:
    if coverage_examples is None or random_examples is None:
        cached_examples = load_synthetic_examples_cache() if (USE_SYNTHETIC_CACHE and not FORCE_REBUILD_SYNTHETIC_CACHE) else None
        if cached_examples is not None:
            coverage_examples, random_examples = cached_examples
            print(f"Loaded cached synthetic examples from {SYNTHETIC_CACHE}")
        else:
            raise RuntimeError(
                "Synthetic examples are not available in memory or cache. "
                "Run the generation cell first, or enable USE_SYNTHETIC_CACHE."
            )

    coverage_dataset = Dataset.from_list(coverage_examples) # type: ignore
    random_dataset = Dataset.from_list(random_examples) # type: ignore

    save_token_debug_cache(coverage_examples, random_examples)  # type: ignore

    coverage_dataset = coverage_dataset.map(
        tokenize_and_align,
        batched=False,
        remove_columns=["tokens", "ner_tags"],
        num_proc=TOKENIZE_NUM_PROC,
    )
    random_dataset = random_dataset.map(
        tokenize_and_align,
        batched=False,
        remove_columns=["tokens", "ner_tags"],
        num_proc=TOKENIZE_NUM_PROC,
    )

    # Train / validation split (90 / 10).
    # Keep the coverage set in train so every language is guaranteed to appear there.
    split = random_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = concatenate_datasets([coverage_dataset, split["train"]])
    eval_dataset = split["test"]
    if USE_TOKENIZED_CACHE:
        save_tokenized_dataset_cache(train_dataset, eval_dataset)

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")


def release_generation_memory() -> None:
    """Drop synthetic-example artifacts that are no longer needed after tokenization."""
    for name in [
        "coverage_examples",
        "random_examples",
        "raw_examples",
        "coverage_dataset",
        "random_dataset",
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
