from __future__ import annotations

import re
import unicodedata


LATIN_GROUPS = {"English", "LatinCore", "LatinTier2"}
WIKI_MARKUP = re.compile(r"\[\[.*?\]\]|\{\{.*?\}\}|==.*?==", flags=re.DOTALL)
BRACKET_NOTES = re.compile(r"\s*[\(\[【（][^\)\]】）]{0,60}[\)\]】）]\s*")
WIKI_ASCII_WORDS = re.compile(r"[A-Za-z]+")
WIKI_SPACES = re.compile(r"\s{2,}")
WIKI_PUNCT_REPEAT = re.compile(r"([,.;:!?…،。！？])\1+")
WIKI_TRAILING_ORPHAN_LETTER = re.compile(r"[\s,.;:!?…،。！？]+([^\W\d_])$")
WIKI_LEADING_ORPHAN_LETTER = re.compile(r"^[\"'“”‘’«»‹›\s,.;:!?…،。！？]+([^\W\d_])\s+")
WIKI_BLOCKED_MARKERS = ("http",)
WIKI_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
WIKI_BLOCKED_CHARS = {"=", "<", ">", "|"}
WIKI_OPENING_QUOTES = {"\"", "'", "“", "”", "‘", "’", "«", "»", "‹", "›"}
WIKI_NON_CONTENT = re.compile(r"[\W_]+", flags=re.UNICODE)
WIKI_DIGITS = re.compile(r"\d")
WIKI_WORDS = re.compile(r"\b\w+\b", flags=re.UNICODE)
MAX_DIGIT_RATIO = 0.10
MIN_LATIN_WORDS = 4


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


def _is_valid_sentence(
    s: str,
    lang: str,
    lang_to_group: dict[str, str],
    sent_bounds: dict[str, tuple[int, int]] | None = None,
    default_bounds: tuple[int, int] = (30, 600),
) -> bool:
    sent_bounds = sent_bounds or {}
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
    sent_bounds: dict[str, tuple[int, int]],
    default_bounds: tuple[int, int],
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
