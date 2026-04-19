#!/usr/bin/env python3
"""
Test the model on the papluca/language-identification dataset from Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

from evaluation_language_utils import dominant_language_from_entities
from evaluation_prediction_utils import (
    predict_multilabel_texts,
    predict_token_classification_texts,
    select_multilabel_prediction,
)
from evaluation_run_config import load_or_create_run_config, resolve_output_path

CONFIG_PATH = Path(project_root) / ".evaluation_config.json"
TOKENIZER_MODEL = "xlm-roberta-base"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the model on papluca/language-identification."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to the shared evaluation manifest JSON.")
    parser.add_argument("--config-id", type=str, default=None, help="Optional config id to select from the manifest.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 80)
    print("TESTING ON PAPLUCA LANGUAGE-IDENTIFICATION DATASET")
    print("=" * 80)

    config = load_or_create_run_config(config_path=args.config, run_name="papluca", config_id=args.config_id)
    config_id = str(config["id"])
    model_name = str(config["model_name"])
    task_type = str(config["task_type"])
    sample_size = int(config.get("sample_size", 2000))
    batch_size = int(config.get("batch_size", 32))
    runner_up_ratio = float(config.get("multilabel_runner_up_ratio", 0.9))
    results_dir = Path(str(config.get("results_dir", "evaluation_results/papluca")))
    results_dir.mkdir(parents=True, exist_ok=True)
    mismatch_output = resolve_output_path(
        results_dir=results_dir,
        value=config.get("mismatch_output"),
        default_name="mismatches.jsonl",
    )
    results_output = results_dir / "results.json"

    print("\n1. Loading model and tokenizer...")
    print(f"   ✓ Config id: {config_id}")
    if task_type == "token-classification":
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    print(f"   ✓ Model loaded: {model_name}")
    print(f"   ✓ Tokenizer loaded: {TOKENIZER_MODEL}")

    print("\n2. Loading papluca/language-identification dataset...")
    try:
        dataset = load_dataset("papluca/language-identification")
        print("   ✓ Dataset loaded")
        print(f"   ✓ Splits available: {list(dataset.keys())}")

        split_to_use = "test" if "test" in dataset else list(dataset.keys())[0]
        test_data = dataset[split_to_use]
        print(f"   ✓ Using '{split_to_use}' split with {len(test_data)} examples")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        raise SystemExit(1) from e

    print(f"\n3. Running inference on dataset using {task_type}...")
    print("-" * 80)

    papluca_to_iso = {
        "arabic": "ar",
        "bulgarian": "bg",
        "german": "de",
        "greek": "el",
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "hindi": "hi",
        "italian": "it",
        "japanese": "ja",
        "portuguese": "pt",
        "russian": "ru",
        "swahili": "sw",
        "turkish": "tr",
        "urdu": "ur",
    }

    correct_count = 0
    total_count = 0
    per_lang_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    results_by_lang = defaultdict(list)
    mismatches: list[dict[str, object]] = []

    sample_size = min(sample_size, len(test_data))
    print(f"Processing {sample_size} examples...\n")

    sample_data = test_data.select(range(sample_size))
    texts = [example["text"][:512] for example in sample_data] # type: ignore
    if task_type == "token-classification":
        predictions = predict_token_classification_texts(
            texts,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
    else:
        predictions = predict_multilabel_texts(
            texts,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

    for example, prediction in tqdm(
        zip(sample_data, predictions),
        total=sample_size,
        desc="Running inference",
    ):
        text = example["text"] # type: ignore
        true_lang_name = example["labels"] # type: ignore
        true_lang = papluca_to_iso.get(true_lang_name.lower(), true_lang_name.lower())
        ranked_langs: list[tuple[str, dict[str, float | int]]] = []
        accepted_runner_up = False

        if task_type == "token-classification":
            pred_lang, lang_stats, ignored_artifacts = dominant_language_from_entities(prediction) # type: ignore
            ranked_langs = sorted(
                lang_stats.items(),
                key=lambda item: item[1]["rank_score"],
                reverse=True,
            )
        else:
            pred_lang, lang_stats, ignored_artifacts = prediction
            pred_lang, accepted_runner_up, ranked_langs = select_multilabel_prediction(
                lang_stats,
                runner_up_ratio=runner_up_ratio,
                true_lang=true_lang,
            )
        if not pred_lang:
            continue

        is_correct = pred_lang == true_lang
        if is_correct:
            correct_count += 1

        total_count += 1
        per_lang_stats[true_lang]["total"] += 1
        if is_correct:
            per_lang_stats[true_lang]["correct"] += 1

        ranked_langs = sorted(
            lang_stats.items(),
            key=lambda item: item[1]["rank_score"],
            reverse=True,
        )
        results_by_lang[true_lang].append(
            {
                "text": text[:100],
                "predicted": pred_lang,
                "correct": is_correct,
                "score": lang_stats.get(pred_lang, {}).get("rank_score", 0.0),
                "ignored_artifacts": ignored_artifacts,
                "accepted_runner_up": accepted_runner_up,
                "ranked_langs": [lang for lang, _ in ranked_langs],
            }
        )
        if not is_correct:
            mismatches.append(
                {
                    "text": text,
                    "text_preview": text[:200],
                    "predicted": pred_lang,
                    "true_lang": true_lang,
                    "dataset_label": true_lang_name,
                    "score": lang_stats.get(pred_lang, {}).get("rank_score", 0.0),
                    "ignored_artifacts": ignored_artifacts,
                    "ranked_langs": [
                        {
                            "lang": lang,
                            "rank_score": float(stat["rank_score"]),
                            "coverage_pct": float(stat["coverage_pct"]),
                            "avg_confidence": float(stat["avg_confidence"]),
                            "entity_count": int(stat["entity_count"]),
                        }
                        for lang, stat in ranked_langs
                    ],
                }
            )

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    overall_accuracy = 100 * correct_count / total_count if total_count > 0 else 0
    print(f"\nOverall Accuracy: {correct_count}/{total_count} = {overall_accuracy:.1f}%")

    print("\n" + "=" * 80)
    print("PER-LANGUAGE BREAKDOWN")
    print("=" * 80)

    print(f"\n{'Language':<12} {'Correct':<10} {'Total':<10} {'Accuracy':<12}")
    print("-" * 44)
    for lang in sorted(per_lang_stats.keys()):
        stats = per_lang_stats[lang]
        correct = stats["correct"]
        total = stats["total"]
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"{lang:<12} {correct:<10} {total:<10} {accuracy:>10.1f}%")

    print("\n" + "=" * 80)
    print("SAMPLE ERRORS (First 5 per language)")
    print("=" * 80)

    for lang in sorted(results_by_lang.keys()):
        errors = [r for r in results_by_lang[lang] if not r["correct"]]
        if errors:
            print(f"\n{lang.upper()} - {len(errors)} errors out of {len(results_by_lang[lang])}")
            for error in errors[:3]:
                print(f"  Text: {error['text'][:60]}...")
                print(f"    Expected: {lang}, Got: {error['predicted']} (score: {error['score']:.3f})")

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if overall_accuracy >= 85:
        status = "✓ EXCELLENT - Model generalizes well to real data"
    elif overall_accuracy >= 70:
        status = "✓ GOOD - Model works reasonably well"
    elif overall_accuracy >= 50:
        status = "⚠ ACCEPTABLE - May need refinement"
    else:
        status = "✗ POOR - Model needs significant improvement"

    print(f"\nStatus: {status}")
    print(f"Accuracy: {overall_accuracy:.1f}%")
    print("\nLanguages covered:")
    print("  Total languages in model: 60")
    print(f"  Languages in papluca test: {len(per_lang_stats)}")
    print(
        "  Best performing: "
        f"{max((l for l in per_lang_stats if per_lang_stats[l]['total'] > 5), key=lambda l: per_lang_stats[l]['correct']/max(per_lang_stats[l]['total'], 1)) if any(per_lang_stats[l]['total'] > 5 for l in per_lang_stats) else 'N/A'}"
    )

    with mismatch_output.open("w", encoding="utf-8") as f:
        for item in mismatches:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n✓ Mismatches saved to {mismatch_output} ({len(mismatches)} rows)")

    with results_output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_accuracy": overall_accuracy,
                "correct": correct_count,
                "total": total_count,
                "per_language": {
                    lang: {
                        "correct": stats["correct"],
                        "total": stats["total"],
                        "accuracy": 100 * stats["correct"] / max(stats["total"], 1),
                    }
                    for lang, stats in per_lang_stats.items()
                },
            },
            f,
            indent=2,
        )

    print(f"\n✓ Detailed results saved to {results_output}")


if __name__ == "__main__":
    main()
