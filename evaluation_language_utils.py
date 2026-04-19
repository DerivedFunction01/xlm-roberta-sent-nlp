from __future__ import annotations

from collections import defaultdict
from typing import Any

from language import canonical_lang

MIN_ARTIFACT_SPAN_CHARS = 4
MIN_ARTIFACT_CONFIDENCE = 0.5
ARTIFACT_SPAN_WEIGHT = 0.35


def normalize_label(label: str) -> str:
    if label.startswith(("B-", "I-")):
        label = label[2:]
    return canonical_lang(label.lower())


def is_artifact_span(span_len: int, score: float) -> bool:
    return span_len <= MIN_ARTIFACT_SPAN_CHARS and score >= MIN_ARTIFACT_CONFIDENCE


def build_lang_stats(entities: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float | int]], int]:
    """Aggregate merged entity spans into per-language coverage stats."""
    char_coverage: defaultdict[str, float] = defaultdict(float)
    conf_weighted: defaultdict[str, float] = defaultdict(float)
    entity_counts: defaultdict[str, int] = defaultdict(int)

    total_tagged_chars = 0.0
    ignored_artifacts = 0

    for entity in entities:
        label = normalize_label(entity.get("entity_group", entity.get("entity", "O")))
        if label == "o":
            continue

        start = entity.get("start")
        end = entity.get("end")
        if start is None or end is None:
            continue

        span_len = max(int(end) - int(start), 1)
        score = float(entity.get("score", 0.0))
        span_weight = ARTIFACT_SPAN_WEIGHT if is_artifact_span(span_len, score) else 1.0
        if span_weight < 1.0:
            ignored_artifacts += 1

        effective_span_len = span_len * span_weight
        char_coverage[label] += effective_span_len
        conf_weighted[label] += effective_span_len * score
        entity_counts[label] += 1
        total_tagged_chars += effective_span_len

    if total_tagged_chars == 0:
        return {}, ignored_artifacts

    stats: dict[str, dict[str, float | int]] = {}
    for lang, coverage in char_coverage.items():
        avg_confidence = conf_weighted[lang] / coverage if coverage else 0.0
        coverage_pct = coverage / total_tagged_chars
        stats[lang] = {
            "char_coverage": coverage,
            "coverage_pct": coverage_pct,
            "avg_confidence": avg_confidence,
            "entity_count": entity_counts[lang],
            "rank_score": coverage_pct * avg_confidence,
        }

    return stats, ignored_artifacts


def dominant_language_from_entities(entities: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, float | int]], int]:
    lang_stats, ignored_artifacts = build_lang_stats(entities)
    if not lang_stats:
        return "", {}, ignored_artifacts
    ranked = sorted(lang_stats.items(), key=lambda item: item[1]["rank_score"], reverse=True)
    return ranked[0][0], lang_stats, ignored_artifacts
