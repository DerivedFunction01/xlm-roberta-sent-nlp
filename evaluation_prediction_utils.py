from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from transformers import pipeline

from language import canonical_lang


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    """Yield consecutive slices of a list."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _normalize_pipeline_output(output: Any) -> list[list[dict[str, Any]]]:
    """Normalize token-classification pipeline output to a list of per-example predictions."""
    if not output:
        return []
    if isinstance(output[0], dict):
        return [output]
    return output


def predict_token_classification_texts(
    texts: list[str],
    *,
    model,
    tokenizer,
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    """Run a token-classification pipeline over a list of texts."""
    nlp = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

    predictions: list[list[dict[str, Any]]] = []
    for batch_texts in batched(texts, batch_size):
        batch_predictions = nlp(batch_texts, batch_size=len(batch_texts))
        predictions.extend(_normalize_pipeline_output(batch_predictions))
    return predictions


def predict_multilabel_texts(
    texts: list[str],
    *,
    model,
    tokenizer,
    batch_size: int,
    max_length: int = 512,
) -> list[tuple[str, dict[str, dict[str, float | int]], int]]:
    """Run a multi-label sequence-classification model over a list of texts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    id2label = {
        int(key): value
        for key, value in getattr(model.config, "id2label", {}).items()
    }

    predictions: list[tuple[str, dict[str, dict[str, float | int]], int]] = []
    for batch_texts in batched(texts, batch_size):
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        for row in probs:
            ranked_pairs = sorted(
                ((idx, float(score)) for idx, score in enumerate(row)),
                key=lambda item: item[1],
                reverse=True,
            )

            lang_stats: dict[str, dict[str, float | int]] = {}
            for idx, score in ranked_pairs:
                label = id2label.get(idx, f"label_{idx}")
                lang = canonical_lang(label.lower())
                if lang in lang_stats:
                    continue
                lang_stats[lang] = {
                    "rank_score": score,
                    "coverage_pct": score,
                    "avg_confidence": score,
                    "entity_count": 1,
                }

            if not lang_stats:
                predictions.append(("", {}, 0))
                continue

            pred_lang = next(iter(lang_stats.keys()))
            predictions.append((pred_lang, lang_stats, 0))

    return predictions
