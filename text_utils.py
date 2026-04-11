from __future__ import annotations

import hashlib
import re
import unicodedata
import traceback
from functools import lru_cache
from pathlib import Path

from language import ENGLISH_STOP_WORDS, LATIN_GROUPS, LANGUAGE_GROUPS, LANGUAGE_GROUP_MIN_CHARS, canonical_lang
try:
    from nltk.corpus import words as nltk_words
except Exception:  # pragma: no cover - optional dependency
    nltk_words = None

WIKI_MARKUP = re.compile(r"\[\[.*?\]\]|\{\{.*?\}\}|==.*?==", flags=re.DOTALL)
SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
WIKI_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
BRACKET_NOTES = re.compile(r"\s*[\(\[【（][^\)\]】）]{0,60}[\)\]】）]\s*")
WIKI_ASCII_WORDS = re.compile(r"[A-Za-z]+")
WIKI_SPACES = re.compile(r"\s+")
WIKI_PUNCT_REPEAT = re.compile(r"([,.;:!?…،。！？])\1+")
WIKI_TRAILING_ORPHAN_LETTER = re.compile(r"[\s,.;:!?…،。！？]+([^\W\d_])$")
WIKI_LEADING_ORPHAN_LETTER = re.compile(r"^[\"'“”‘’«»‹›\s,.;:!?…،。！？]+([^\W\d_])\s+")
WIKI_BLOCKED_MARKERS = ("http",)
WIKI_BLOCKED_CHARS = {"=", "<", ">", "|"}
WIKI_OPENING_QUOTES = {"\"", "'", "“", "”", "‘", "’", "«", "»", "‹", "›"}
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]{0,80}>")
WIKI_NON_CONTENT = re.compile(r"[\W_]+", flags=re.UNICODE)
WIKI_DIGITS = re.compile(r"\d")
WIKI_WORDS = re.compile(r"\b\w+\b", flags=re.UNICODE)
MAX_DIGIT_RATIO = 0.10
MIN_LATIN_WORDS = 4
ENGLISH_FILTER_POLICY = {
    "min_alpha_words": 4,
    "min_local_hits": 3,
    "ascii_ratio_floor": 0.70,
    "latin": {
        "corpus_hits_min": 4,
        "corpus_ratio_min": 0.60,
        "ascii_ratio_min": 0.80,
    },
    "nonlatin": {
        "corpus_hits_min": 3,
        "corpus_ratio_min": 0.30,
        "ascii_ratio_min": 0.50,
    },
    "finetrans": {
        "max_english_ratio": 0.35,
        "strict_language_score_cutoff": 0.98,
        "min_tokens": 4,
    },
}
ENGLISH_STOP_WORD_SET = {word.lower() for word in ENGLISH_STOP_WORDS}
NLTK_ENGLISH_SECONDARY_LIMIT = 50_000
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

PYSBD_FALLBACK_GROUPS: dict[str, list[str]] = {}


def _assign_pysbd_fallbacks(proxy: str, *langs: str) -> None:
    PYSBD_FALLBACK_GROUPS[proxy] = list(langs)


_assign_pysbd_fallbacks("ru", "uk", "be", "sr", "mk", "mn", "tt", "ky", "tg", "ba", "ce")
_assign_pysbd_fallbacks("es", "pt", "eu", "ca", "gl")
_assign_pysbd_fallbacks("fr", "ro", "oc", "br", "mg")
_assign_pysbd_fallbacks("it", "la", "sq", "rm", "mt")
_assign_pysbd_fallbacks("da", "sv", "no", "is")
_assign_pysbd_fallbacks("pl", "cs", "bs", "hr", "sl", "sk", "et", "lv", "lt")
_assign_pysbd_fallbacks("en", "fi", "hu", "vi", "id", "ms", "af", "tr", "sw", "tl", "ga", "gd", "cy", "eo", "jv", "om", "so", "su", "uz", "ku", "yo", "zu", "ny", "ka")
_assign_pysbd_fallbacks("ar", "he", "ps", "ug", "dv", "ckb")
_assign_pysbd_fallbacks("hi", "bn", "ta", "te", "gu", "kn", "ml", "pa", "as", "or", "sd", "si", "ne")
_assign_pysbd_fallbacks("zh", "km", "ko", "lo", "th", "bo")
_assign_pysbd_fallbacks("am", "ti")
_assign_pysbd_fallbacks("he", "hbo")
_assign_pysbd_fallbacks("el", "grc")
_assign_pysbd_fallbacks("nl", "af")
_assign_pysbd_fallbacks("de", "lb")

PYSBD_FALLBACKS: dict[str, str] = {
    lang: proxy
    for proxy, langs in PYSBD_FALLBACK_GROUPS.items()
    for lang in langs
}

GROUP_SENT_BOUNDS: dict[str, tuple[int, int]] = {}


def _assign_group_bounds(bounds: tuple[int, int], *groups: str) -> None:
    for group in groups:
        GROUP_SENT_BOUNDS[group] = bounds


_assign_group_bounds((24, 600), "English")
_assign_group_bounds((35, 650), "Russian", "EastSlavicCyrillic", "BalkanCyrillic", "CentralAsianCaucusCyrillic")
_assign_group_bounds((40, 600), "German")
_assign_group_bounds((30, 600), "Spanish", "French", "Portuguese", "Italian", "Dutch", "Polish")
_assign_group_bounds((20, 450), "Turkish", "WesternLatin", "CelticLatin", "AdriaticLatin", "BalticLatin", "CentralEuropeanLatin", "Norwegian", "NordicCore", "KurdishLatin")
_assign_group_bounds((10, 180), "Japanese")
_assign_group_bounds((8, 180), "Chinese")
_assign_group_bounds((15, 220), "Korean")
_assign_group_bounds((25, 450), "Arabic", "ArabicOther", "KurdishArabic", "OtherScriptsWest")
_assign_group_bounds((30, 500), "Hindi", "IndicOther")
_assign_group_bounds((15, 300), "SoutheastAsianLatin", "AfricanLatin", "PeripheralLatin")
_assign_group_bounds((15, 250), "OtherScriptsEast")

LANG_SENT_BOUNDS_OVERRIDES: dict[str, tuple[int, int]] = {
    "hy": (30, 500),
}

SENT_BOUNDS: dict[str, tuple[int, int]] = {
    lang: GROUP_SENT_BOUNDS[group]
    for group, langs in LANGUAGE_GROUPS.items()
    if group in GROUP_SENT_BOUNDS
    for lang in langs
}
SENT_BOUNDS.update(LANG_SENT_BOUNDS_OVERRIDES)
DEFAULT_BOUNDS = (30, 600)
_PROC_SEGMENTERS: dict[str, object] = {}


@lru_cache(maxsize=1)
def _nltk_english_secondary_word_set() -> set[str]:
    if nltk_words is None:
        return set()
    try:
        secondary: set[str] = set()
        for word in nltk_words.words():
            word = word.lower().strip()
            if not word or not word.isalpha() or word in ENGLISH_STOP_WORD_SET:
                continue
            if len(word) > 3: secondary.add(word)
            if len(secondary) >= NLTK_ENGLISH_SECONDARY_LIMIT:
                break
        return secondary
    except LookupError:
        return set()


@lru_cache(maxsize=262_144)
def _is_local_english_stopword(token: str) -> bool:
    word = token.lower().strip()
    return bool(word) and word in ENGLISH_STOP_WORD_SET


@lru_cache(maxsize=262_144)
def _is_broad_english_word(token: str) -> bool:
    word = token.lower().strip()
    if not word:
        return False
    secondary = _nltk_english_secondary_word_set()
    if word in secondary:
        return True
    if word.endswith("s") and word[:-1] in secondary:
        return True
    return False


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
    lang = canonical_lang(lang)
    if lang_to_group.get(lang) in LATIN_GROUPS:
        return sentence
    return WIKI_TRAILING_ORPHAN_LETTER.sub("", sentence).rstrip()


def _strip_leading_orphan_letter(sentence: str) -> str:
    return WIKI_LEADING_ORPHAN_LETTER.sub("", sentence).lstrip()


def _has_blocked_artifact(sentence: str) -> bool:
    lower = sentence.lower()
    return any(marker in lower for marker in WIKI_BLOCKED_MARKERS) or any(ch in sentence for ch in WIKI_BLOCKED_CHARS)


def _strip_ascii_for_lang(lang: str, lang_to_group: dict[str, str]) -> bool:
    lang = canonical_lang(lang)
    return lang_to_group.get(lang) not in LATIN_GROUPS


def _english_leak_stats(sentence: str) -> tuple[int, int, int]:
    words = [word.lower() for word in WIKI_WORDS.findall(sentence)]
    if not words:
        return 0, 0, 0
    local_hits = sum(_is_local_english_stopword(word) for word in words)
    ascii_words = sum(word.isascii() and word.isalpha() for word in words)
    alpha_words = sum(word.isalpha() for word in words)
    return local_hits, ascii_words, alpha_words


def _english_corpus_hits(sentence: str) -> int:
    words = [word.lower() for word in WIKI_WORDS.findall(sentence)]
    if not words:
        return 0
    broad_hits = sum(_is_broad_english_word(word) for word in words)
    return broad_hits


def _looks_like_english_sentence(sentence: str, lang: str, lang_to_group: dict[str, str]) -> bool:
    lang = canonical_lang(lang)
    if lang == "en":
        return False
    local_hits, ascii_words, alpha_words = _english_leak_stats(sentence)
    if alpha_words < ENGLISH_FILTER_POLICY["min_alpha_words"]:
        return False
    if local_hits < ENGLISH_FILTER_POLICY["min_local_hits"]:
        return False
    ascii_ratio = ascii_words / alpha_words
    if ascii_ratio < ENGLISH_FILTER_POLICY["ascii_ratio_floor"]:
        return False
    broad_hits = _english_corpus_hits(sentence)
    stop_ratio = broad_hits / alpha_words
    if lang_to_group.get(lang) in LATIN_GROUPS:
        latin_policy = ENGLISH_FILTER_POLICY["latin"]
        return (
            broad_hits >= latin_policy["corpus_hits_min"]
            and stop_ratio >= latin_policy["corpus_ratio_min"]
            and ascii_ratio >= latin_policy["ascii_ratio_min"]
        )
    nonlatin_policy = ENGLISH_FILTER_POLICY["nonlatin"]
    return (
        broad_hits >= nonlatin_policy["corpus_hits_min"]
        and stop_ratio >= nonlatin_policy["corpus_ratio_min"]
        and ascii_ratio >= nonlatin_policy["ascii_ratio_min"]
    )


def clean_sentence(sentence: str, lang: str, lang_to_group: dict[str, str]) -> str:
    lang = canonical_lang(lang)
    if "\\" in sentence:
        sentence = sentence.replace("\\", "")
    sentence = HTML_TAG_RE.sub(" ", sentence)
    sentence = _strip_bracket_notes(sentence)
    sentence = _collapse_repeated_punct(sentence)
    if _looks_like_english_sentence(sentence, lang, lang_to_group):
        return ""
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

    text = HTML_TAG_RE.sub(" ", text)
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
    lang = canonical_lang(lang)
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
    lang = canonical_lang(lang)
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
    lang = canonical_lang(lang)
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
    lang = canonical_lang(lang)
    group = lang_to_group.get(lang)
    assert group is not None
    return LANGUAGE_GROUP_MIN_CHARS.get(group, 3_000)


def _split_paragraphs(text: str, lang: str, lang_to_group: dict[str, str]) -> list[str] | None:
    lang = canonical_lang(lang)
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
