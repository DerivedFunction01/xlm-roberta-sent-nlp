from __future__ import annotations

import hashlib
import re
import unicodedata
import traceback
from pathlib import Path

from language import LATIN_GROUPS, LANGUAGE_GROUP_MIN_CHARS
WIKI_MARKUP = re.compile(r"\[\[.*?\]\]|\{\{.*?\}\}|==.*?==", flags=re.DOTALL)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WIKI_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
BRACKET_NOTES = re.compile(r"\s*[\(\[【（][^\)\]】）]{0,60}[\)\]】）]\s*")
WIKI_ASCII_WORDS = re.compile(r"[A-Za-z]+")
WIKI_SPACES = re.compile(r"\s{2,}")
WIKI_PUNCT_REPEAT = re.compile(r"([,.;:!?…،。！？])\1+")
WIKI_TRAILING_ORPHAN_LETTER = re.compile(r"[\s,.;:!?…،。！？]+([^\W\d_])$")
WIKI_LEADING_ORPHAN_LETTER = re.compile(r"^[\"'“”‘’«»‹›\s,.;:!?…،。！？]+([^\W\d_])\s+")
WIKI_BLOCKED_MARKERS = ("http",)
WIKI_BLOCKED_CHARS = {"=", "<", ">", "|"}
WIKI_OPENING_QUOTES = {"\"", "'", "“", "”", "‘", "’", "«", "»", "‹", "›"}
WIKI_NON_CONTENT = re.compile(r"[\W_]+", flags=re.UNICODE)
WIKI_DIGITS = re.compile(r"\d")
WIKI_WORDS = re.compile(r"\b\w+\b", flags=re.UNICODE)
MAX_DIGIT_RATIO = 0.10
MIN_LATIN_WORDS = 4
POOL_TERMINAL_PUNCT_CHOICES = (".", ":", ";", "!", "?")
POOL_WRAPPER_PAIRS = (
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
    ("【", "】"),
    ("（", "）"),
    ("「", "」"),
    ("『", "』"),
    ("«", "»"),
    ("‹", "›"),
    ("“", "”"),
    ("‘", "’"),
)
POOL_SAME_CHAR_WRAPPERS = ('"', "'")
POOL_LEADING_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"(?:\d{1,4}|[ivxlcdm]{1,8}|[a-zA-Z])(?:[.)]|[\]\)])\s+|"
    r"[\-\*\u2022\u00b7\u2023\u2043\u2219\u2013\u2014]\s+"
    r")",
    flags=re.IGNORECASE,
)
POOL_TRAILING_PUNCT_RE = re.compile(r"^(?P<body>.*?)(?P<punct>[.!?;:…]+)$", flags=re.UNICODE)

PYSBD_SUPPORTED = {
    "en", "hi", "mr", "bg", "es", "ru", "ar", "am", "hy", "fa",
    "ur", "pl", "zh", "nl", "da", "fr", "it", "el", "my", "ja", "de", "kk",
}
PYSBD_FALLBACKS = {
    "uk": "ru", "be": "ru", "sr": "ru", "mk": "ru", "mn": "ru",
    "pt": "es", "ro": "fr", "la": "it", "sq": "it",
    "sv": "da", "no": "da", "is": "da",
    "fi": "en", "hu": "en", "cs": "pl",
    "vi": "en", "id": "en", "ms": "en", "af": "nl", "tr": "en",
    "he": "ar", "ps": "fa", "ug": "ar",
    "bn": "hi", "ta": "hi", "te": "hi", "gu": "hi", "kn": "hi",
    "ml": "hi", "pa": "hi", "as": "hi", "or": "hi", "sd": "hi",
    "ka": "en", "km": "zh", "ko": "zh", "lo": "zh", "th": "zh",
}

SENT_BOUNDS: dict[str, tuple[int, int]] = {
    "zh": (8, 180), "ja": (10, 180),
    "ko": (15, 220), "th": (15, 250), "km": (15, 250), "lo": (15, 250), "my": (15, 250),
    "ar": (25, 450), "fa": (25, 450), "he": (25, 400), "ur": (25, 450),
    "hi": (30, 500), "bn": (30, 500), "ta": (30, 500), "te": (30, 500), "am": (25, 400),
    "fi": (20, 450), "hu": (20, 450), "tr": (20, 450), "vi": (15, 300),
    "de": (40, 600), "ru": (35, 650), "uk": (35, 650), "el": (35, 650),
    "hy": (30, 500), "ka": (25, 450), "en": (24, 600),
}
DEFAULT_BOUNDS = (30, 600)
_PROC_SEGMENTERS: dict[str, object] = {}


def _non_punct_char_count(s: str) -> int:
    return len(WIKI_NON_CONTENT.sub("", s))


def _digit_count(s: str) -> int:
    return len(WIKI_DIGITS.findall(s))


def _word_count(s: str) -> int:
    return len(WIKI_WORDS.findall(s))


def _strip_bracket_notes(text: str) -> str:
    return BRACKET_NOTES.sub(" ", text)


def _collapse_spaces(text: str) -> str:
    return WIKI_SPACES.sub(" ", text)


def _strip_leading_punct(sentence: str) -> str:
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
    return WIKI_PUNCT_REPEAT.sub(r"\1", sentence)


def _strip_trailing_orphan_letter(sentence: str, lang_to_group: dict[str, str], lang: str) -> str:
    if lang_to_group.get(lang) in LATIN_GROUPS:
        return sentence
    return WIKI_TRAILING_ORPHAN_LETTER.sub("", sentence).rstrip()


def _strip_leading_orphan_letter(sentence: str) -> str:
    return WIKI_LEADING_ORPHAN_LETTER.sub("", sentence).lstrip()


def _has_blocked_artifact(sentence: str) -> bool:
    lower = sentence.lower()
    return any(marker in lower for marker in WIKI_BLOCKED_MARKERS) or any(ch in sentence for ch in WIKI_BLOCKED_CHARS)


def _strip_ascii_for_lang(lang: str, lang_to_group: dict[str, str]) -> bool:
    return lang_to_group.get(lang) not in LATIN_GROUPS


def clean_sentence(sentence: str, lang: str, lang_to_group: dict[str, str]) -> str:
    if "\\" in sentence:
        sentence = sentence.replace("\\", "")
    sentence = _strip_bracket_notes(sentence)
    if _strip_ascii_for_lang(lang, lang_to_group):
        sentence = WIKI_ASCII_WORDS.sub("", sentence)
    sentence = _collapse_spaces(sentence)
    return sentence.strip()


def _stable_pool_choice(seed_text: str, choices: tuple[str, ...]) -> str:
    digest = hashlib.blake2b(seed_text.encode("utf-8"), digest_size=2).digest()
    return choices[int.from_bytes(digest, "big") % len(choices)]


def _strip_outer_pool_wrappers(text: str) -> str:
    stripped = text.strip()
    if len(stripped) < 2:
        return stripped

    changed = True
    while changed:
        changed = False
        stripped = stripped.strip()
        if len(stripped) < 2:
            break
        for open_ch, close_ch in POOL_WRAPPER_PAIRS:
            if stripped.startswith(open_ch) and stripped.endswith(close_ch):
                inner = stripped[len(open_ch) : len(stripped) - len(close_ch)].strip()
                if inner:
                    stripped = inner
                    changed = True
                    break
        else:
            if stripped[0] in POOL_SAME_CHAR_WRAPPERS and stripped[-1] == stripped[0]:
                inner = stripped[1:-1].strip()
                if inner:
                    stripped = inner
                    changed = True
    return stripped


def normalize_sentence_for_pool(
    sentence: str,
    *,
    lang: str = "",
    seed: int = 0,
) -> str:
    """Normalize a cached pool sentence for synthetic sampling."""
    if not isinstance(sentence, str):
        return ""

    text = _collapse_spaces(sentence).strip()
    if not text:
        return ""

    text = _strip_outer_pool_wrappers(text)
    text = POOL_LEADING_MARKER_RE.sub("", text).strip()
    text = _collapse_repeated_punct(text)
    text = _collapse_spaces(text).strip()
    if not text:
        return ""

    match = POOL_TRAILING_PUNCT_RE.match(text)
    if match:
        body = match.group("body").rstrip()
        if body:
            choice_seed = f"{seed}\0{lang}\0{body}"
            text = f"{body}{_stable_pool_choice(choice_seed, POOL_TERMINAL_PUNCT_CHOICES)}"

    text = _collapse_repeated_punct(text)
    text = _collapse_spaces(text).strip()
    return text


def _is_valid_sentence(
    s: str,
    lang: str,
    lang_to_group: dict[str, str],
    sent_bounds: dict[str, tuple[int, int]] | None = None,
    default_bounds: tuple[int, int] = DEFAULT_BOUNDS,
) -> bool:
    sent_bounds = sent_bounds or SENT_BOUNDS
    mn, mx = sent_bounds.get(lang, default_bounds)
    visible = _non_punct_char_count(s)
    if not (mn < visible < mx):
        return False
    if lang_to_group.get(lang) in LATIN_GROUPS and _word_count(s) < MIN_LATIN_WORDS:
        return False
    digits = _digit_count(s)
    return digits <= visible * MAX_DIGIT_RATIO


def post_clean_sentences(
    sentences: list[str],
    lang: str,
    lang_to_group: dict[str, str],
    sent_bounds: dict[str, tuple[int, int]] | None = None,
    default_bounds: tuple[int, int] = DEFAULT_BOUNDS,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
        if _has_blocked_artifact(sentence):
            continue
        sentence = clean_sentence(sentence, lang, lang_to_group)
        sentence = _strip_leading_punct(sentence)
        sentence = _strip_leading_orphan_letter(sentence)
        sentence = _collapse_repeated_punct(sentence)
        sentence = _strip_trailing_orphan_letter(sentence, lang_to_group, lang)
        sentence = _collapse_spaces(sentence)
        sentence = sentence.strip()
        if not sentence or _has_blocked_artifact(sentence):
            continue
        if sentence in seen:
            continue
        if _is_valid_sentence(sentence, lang, lang_to_group, sent_bounds, default_bounds):
            seen.add(sentence)
            cleaned.append(sentence)
    return cleaned


def _get_segmenter(lang: str):
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


def sanitize_paragraph_for_pysbd(paragraph: str) -> str:
    if "\\" not in paragraph:
        return paragraph
    return paragraph.replace("\\", " ")


def _article_min_chars(lang: str, lang_to_group: dict[str, str]) -> int:
    group = lang_to_group.get(lang)
    assert group is not None
    return LANGUAGE_GROUP_MIN_CHARS.get(group, 3_000)


def _split_paragraphs(text: str, lang: str, lang_to_group: dict[str, str]) -> list[str] | None:
    if len(text) < _article_min_chars(lang, lang_to_group):
        return None
    text = WIKI_MARKUP.sub("", text)
    paragraphs = [p.strip() for p in WIKI_PARAGRAPH_SPLIT.split(text) if p.strip()]
    return paragraphs or None


def log_segmentation_failure(
    log_path: str | Path,
    *,
    lang: str,
    article_idx: int,
    paragraph_idx: int,
    paragraph: str,
    exc: Exception,
    article_title: str = "",
) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    snippet = paragraph[:800].replace("\n", " ")
    with log_path.open("a", encoding="utf-8") as f:
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
