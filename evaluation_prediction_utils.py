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


def select_multilabel_prediction(
    lang_stats: dict[str, dict[str, float | int]],
    *,
    runner_up_ratio: float = 0.9,
    true_lang: str | None = None,
) -> tuple[str, bool, list[tuple[str, dict[str, float | int]]]]:
    """Select the winning language, optionally accepting a close runner-up.

    If `true_lang` is provided and the second-ranked language matches it while
    its score is at least `runner_up_ratio` times the top score, the runner-up
    is accepted instead of the top prediction.
    """
    ranked_langs = sorted(
        lang_stats.items(),
        key=lambda item: item[1]["rank_score"],
        reverse=True,
    )
    if not ranked_langs:
        return "", False, []

    pred_lang = ranked_langs[0][0]
    accepted_runner_up = False
    if true_lang and len(ranked_langs) > 1:
        top_score = float(ranked_langs[0][1]["rank_score"])
        second_lang, second_stats = ranked_langs[1]
        second_score = float(second_stats["rank_score"])
        if second_lang == true_lang and second_score >= top_score * runner_up_ratio:
            pred_lang = second_lang
            accepted_runner_up = True

    return pred_lang, accepted_runner_up, ranked_langs
