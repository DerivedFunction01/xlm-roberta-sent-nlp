from __future__ import annotations

import re
from typing import Any, Callable


SourceExtractor = Callable[[dict[str, Any], dict[str, Any]], list[str]]
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]{0,80}>")
_REPEATED_PUNCT_RE = re.compile(r"([,.;:!?…،。！？])\1+")


def _normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = _HTML_TAG_RE.sub(" ", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return " ".join(text.split()).strip()


def _extract_texts_from_message_list(
    row: dict[str, Any],
    field_name: str,
    *,
    role_key: str = "role",
    content_keys: tuple[str, ...] = ("text", "content", "value", "message", "utterance", "answer", "response"),
    skip_roles: tuple[str, ...] = ("system",),
) -> list[str]:
    value = row.get(field_name)
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = _normalize_text(item)
            if cleaned:
                texts.append(cleaned)
            continue
        if not isinstance(item, dict):
            continue
        role = item.get(role_key)
        if isinstance(role, str) and role.lower().strip() in skip_roles:
            continue
        for key in content_keys:
            content = item.get(key)
            if isinstance(content, str):
                cleaned = _normalize_text(content)
                if cleaned:
                    texts.append(cleaned)
                break
    return texts


def _extract_texts_from_fields(row: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    texts: list[str] = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            cleaned = _normalize_text(value)
            if cleaned:
                texts.append(cleaned)
    return texts


def _extract_french_instruct_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_message_list(row, "conversation")
    if texts:
        return texts
    return _extract_texts_from_fields(row, ("context",))


def _extract_due_chiacchiere_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_message_list(row, "messages")
    if texts:
        return texts
    return _extract_texts_from_fields(row, ("topic", "sub_topic"))


def _extract_aya_hindi_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_fields(row, ("inputs", "targets"))
    return texts


def _extract_indonesia_sft_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_message_list(row, "messages")
    if texts:
        return texts
    return _extract_texts_from_fields(row, ("qid",))


def _extract_openhermes_spanish_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_message_list(row, "conversations_spanish", role_key="from")
    if texts:
        return texts
    return _extract_texts_from_message_list(row, "messages")


def _extract_openhermes_ru_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_message_list(row, "conversations", role_key="from")
    if texts:
        return texts
    return _extract_texts_from_message_list(row, "messages")


def _extract_arabic_openhermes_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_fields(row, ("user", "gpt"))
    if texts:
        return texts
    return _extract_texts_from_message_list(row, "conversations")


def _extract_ko_openhermes_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_fields(row, ("instruction", "input", "output"))
    return texts


def _extract_german_openhermes_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_fields(row, ("instruction", "input", "output"))
    return texts


def _extract_dutch_openhermes_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_message_list(row, "messages")
    if texts:
        return texts
    texts = _extract_texts_from_message_list(row, "conversations_nl", role_key="from")
    if texts:
        return texts
    return _extract_texts_from_fields(row, ("system_prompt",))


def _extract_ptbr_enus_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    language = str(row.get("LANGUAGE", "")).strip().lower()
    if language not in {"pt", "pt-br", "ptbr", "portuguese"}:
        return []
    return _extract_texts_from_fields(row, ("INSTRUCTION", "RESPONSE"))


def _extract_chinese_qwen_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_fields(row, ("Input", "Answer_content"))
    if texts:
        return texts
    return _extract_texts_from_fields(row, ("CoT_content",))


def _extract_turkish_openhermes_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_fields(row, ("instruction", "input", "output"))
    return texts


def _extract_japanese_self_instruct_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_message_list(row, "messages")
    if texts:
        return texts
    texts = _extract_texts_from_fields(row, ("generated_instruction", "self_instruct_output"))
    return texts


def _extract_alpaca_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_fields(row, ("instruction", "input", "output"))
    if texts:
        return texts
    return []


def _extract_latin_english_parallel_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_fields(row, ("la",))
    if texts:
        return texts
    return []


def _extract_generic_texts(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    texts = _extract_texts_from_message_list(row, "messages")
    if texts:
        return texts
    for field in ("conversation", "conversations", "conversations_nl", "conversations_spanish", "instruction", "input", "output", "text", "content"):
        texts.extend(_extract_texts_from_message_list(row, field, role_key="from"))
        texts.extend(_extract_texts_from_fields(row, (field,)))
    return texts


INSTRUCTION_SOURCE_EXTRACTORS: dict[str, SourceExtractor] = {
    "french_instruct": _extract_french_instruct_texts,
    "due_chiacchiere": _extract_due_chiacchiere_texts,
    "aya_hindi": _extract_aya_hindi_texts,
    "indonesia_sft": _extract_indonesia_sft_texts,
    "openhermes_spanish": _extract_openhermes_spanish_texts,
    "openhermes_ru": _extract_openhermes_ru_texts,
    "arabic_openhermes": _extract_arabic_openhermes_texts,
    "ko_openhermes": _extract_ko_openhermes_texts,
    "german_openhermes": _extract_german_openhermes_texts,
    "dutch_openhermes": _extract_dutch_openhermes_texts,
    "ptbr_enus": _extract_ptbr_enus_texts,
    "chinese_qwen": _extract_chinese_qwen_texts,
    "turkish_openhermes": _extract_turkish_openhermes_texts,
    "japanese_self_instruct": _extract_japanese_self_instruct_texts,
    "alpaca": _extract_alpaca_texts,
    "latin_english_parallel": _extract_latin_english_parallel_texts,
    "generic": _extract_generic_texts,
}


DEFAULT_INSTRUCTION_SOURCE_SPECS = [
    {
        "name": "alpaca_en",
        "repo_id": "tatsu-lab/alpaca",
        "split": "train",
        "lang": "en",
        "extractor": "alpaca",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "latin_english_parallel_la",
        "repo_id": "grosenthal/latin_english_parallel",
        "split": "train",
        "lang": "la",
        "extractor": "latin_english_parallel",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "french_instruct",
        "repo_id": "angeluriot/french_instruct",
        "split": "train",
        "lang": "fr",
        "extractor": "french_instruct",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "due_chiacchiere",
        "repo_id": "DeepMount00/due-chiacchiere",
        "split": "train",
        "lang": "it",
        "extractor": "due_chiacchiere",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "aya_hindi_complete",
        "repo_id": "Cognitive-Lab/Aya_Hindi",
        "config_name": "complete_dataset",
        "split": "train",
        "lang": "hi",
        "extractor": "aya_hindi",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "indonesia_sft",
        "repo_id": "IndonesiaAI/sft-dataset",
        "split": "train",
        "lang": "id",
        "extractor": "indonesia_sft",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "openhermes_spanish",
        "repo_id": "Iker/OpenHermes-2.5-English-Spanish",
        "split": "train",
        "lang": "es",
        "extractor": "openhermes_spanish",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "openhermes_ru",
        "repo_id": "d0rj/OpenHermes-2.5-ru",
        "split": "train",
        "lang": "ru",
        "extractor": "openhermes_ru",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "arabic_openhermes",
        "repo_id": "2A2I/Arabic-OpenHermes-2.5",
        "split": "train",
        "lang": "ar",
        "extractor": "arabic_openhermes",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "ko_openhermes",
        "repo_id": "nlp-with-deeplearning/ko.openhermes",
        "split": "train",
        "lang": "ko",
        "extractor": "ko_openhermes",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "german_openhermes",
        "repo_id": "stefan-it/nanochat-german-openhermes",
        "split": "train",
        "lang": "de",
        "extractor": "german_openhermes",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "dutch_openhermes",
        "repo_id": "yhavinga/Openhermes-2.5-dutch-97k",
        "split": "train",
        "lang": "nl",
        "extractor": "dutch_openhermes",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "ptbr_enus_pt",
        "repo_id": "cnmoro/Instruct-PTBR-ENUS-11M",
        "split": "train",
        "lang": "pt",
        "extractor": "ptbr_enus",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "chinese_qwen",
        "repo_id": "Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k",
        "split": "train",
        "lang": "zh",
        "extractor": "chinese_qwen",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "turkish_openhermes",
        "repo_id": "umarigan/openhermes_tr",
        "split": "train",
        "lang": "tr",
        "extractor": "turkish_openhermes",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
    {
        "name": "japanese_self_instruct",
        "repo_id": "Aratako/Self-Instruct-Qwen2.5-72B-Instruct-60k",
        "split": "train",
        "lang": "ja",
        "extractor": "japanese_self_instruct",
        "trust_remote_code": False,
        "max_rows": 100_000,
    },
]
