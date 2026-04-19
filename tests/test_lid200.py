#!/usr/bin/env python3
"""
Test the model on the mikaberidze/lid200 language-identification dataset.

By default this evaluates only the languages supported by the local model
and normalized through `language.canonical_lang`. You can further narrow the
benchmark with `--langs en es fr` (canonical model codes).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

# Attach the current directory (project root) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)
from language import ALL_LANGS, canonical_lang, is_dataset_label_script_compatible
from evaluation_language_utils import dominant_language_from_entities
from evaluation_prediction_utils import (
    predict_multilabel_texts,
    predict_token_classification_texts,
    select_multilabel_prediction,
)
from evaluation_run_config import load_or_create_run_config, resolve_output_path

CONFIG_PATH = Path(project_root) / "evaluation_config.json"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the model on mikaberidze/lid200."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to the evaluation JSON config.")
    return parser.parse_args()


def _resolve_lang_column(dataset) -> str:
    if "lang" in dataset.column_names:
        return "lang"
    raise KeyError("Could not find a 'lang' column in mikaberidze/lid200")


def _resolve_text_column(dataset) -> str:
    if "text" in dataset.column_names:
        return "text"
    raise KeyError("Could not find a 'text' column in mikaberidze/lid200")


def _label_name_from_example(example: dict, dataset, lang_column: str) -> str:
    """Decode the ClassLabel-backed `lang` field into its string label."""
    value = example[lang_column]
    feature = dataset.features[lang_column]
    if hasattr(feature, "int2str") and isinstance(value, int):
        return feature.int2str(value)
    if hasattr(feature, "names") and isinstance(value, int):
        return feature.names[value]
    return str(value)


def _dataset_label_to_iso3(label: str) -> str:
    """Strip the script suffix from labels like 'eng_Latn'."""
    return label.split("_", 1)[0]


def _dataset_label_to_canonical(label: str) -> str:
    return canonical_lang(_dataset_label_to_iso3(label))


def main() -> None:
    args = _parse_args()

    print("=" * 80)
    print("TESTING ON MIKABERIDZE/LID200 DATASET")
    print("=" * 80)

    config = load_or_create_run_config(config_path=args.config, run_name="lid200")
    model_name = str(config["model_name"])
    task_type = str(config["task_type"])
    langs = [str(lang).lower() for lang in config.get("langs", [])] or None
    split_name = str(config.get("split", "test"))
    batch_size = int(config.get("batch_size", 32))
    runner_up_ratio = float(config.get("multilabel_runner_up_ratio", 0.9))
    results_dir = Path(str(config.get("results_dir", "evaluation_results/lid200")))
    results_dir.mkdir(parents=True, exist_ok=True)
    mismatch_output = resolve_output_path(
        results_dir=results_dir,
        value=config.get("mismatch_output"),
        default_name="mismatches.jsonl",
    )
    results_output = results_dir / "results.json"

    # ===== LOAD MODEL & TOKENIZER =====
    print("\n1. Loading model and tokenizer...")
    if task_type == "token-classification":
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   ✓ Model loaded: {model_name}")
    print(f"   ✓ Tokenizer loaded")

    # ===== LOAD DATASET =====
    print("\n2. Loading mikaberidze/lid200 dataset...")
    dataset = load_dataset("mikaberidze/lid200")
    print(f"   ✓ Dataset loaded")
    print(f"   ✓ Splits available: {list(dataset.keys())}")

    split_to_use = split_name if split_name in dataset else list(dataset.keys())[0]
    test_data = dataset[split_to_use]
    print(f"   ✓ Using '{split_to_use}' split with {len(test_data)} examples")

    lang_column = _resolve_lang_column(test_data)
    text_column = _resolve_text_column(test_data)

    # ===== DETERMINE FILTER =====
    if langs:
        keep_langs = langs
    else:
        keep_langs = ALL_LANGS[:]

    print("\n3. Filtering languages...")
    print(f"   ✓ Keeping {len(keep_langs)} canonical languages")
    # Print one example row in test_data
    print(f"   ✓ Example dataset row: {test_data[0]}")
    filtered_test = test_data.filter(
        lambda ex: _dataset_label_to_canonical(_label_name_from_example(ex, test_data, lang_column))
        in keep_langs
    )
    print(f"   ✓ Filtered dataset size: {len(filtered_test)}")

    compatible_test = filtered_test.filter(
        lambda ex: is_dataset_label_script_compatible(
            _dataset_label_to_canonical(_label_name_from_example(ex, filtered_test, lang_column)),
            _label_name_from_example(ex, filtered_test, lang_column),
        )
    )
    dropped_incompatible = len(filtered_test) - len(compatible_test)
    filtered_test = compatible_test
    print(f"   ✓ Dropped {dropped_incompatible} Latin-script rows for non-Latin languages")

    if len(filtered_test) == 0:
        raise RuntimeError(
            "No examples matched the selected languages. "
            "Check the --langs values against the supported language list."
        )

    # ===== RUN INFERENCE =====
    print(f"\n4. Running inference on filtered dataset using {task_type}...")
    print("-" * 80)

    correct_count = 0
    total_count = 0
    per_lang_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    results_by_lang = defaultdict(list)
    mismatches: list[dict[str, object]] = []

    texts_for_inference = [example[text_column] for example in filtered_test]
    if task_type == "token-classification":
        all_predictions = predict_token_classification_texts(
            texts_for_inference,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
    else:
        all_predictions = predict_multilabel_texts(
            texts_for_inference,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

    print(f"Processing {len(filtered_test)} examples and their predictions...\n")

    for example_idx, (example, predictions) in enumerate(tqdm(
        zip(filtered_test, all_predictions),
        total=len(filtered_test),
        desc="Processing predictions",
    )):
        text = example[text_column]
        true_lang_name = _label_name_from_example(example, filtered_test, lang_column)
        true_lang = _dataset_label_to_canonical(true_lang_name)
        if true_lang not in ALL_LANGS:
            continue
        ranked_langs: list[tuple[str, dict[str, float | int]]] = []
        accepted_runner_up = False

        if task_type == "token-classification":
            pred_lang, lang_stats, ignored_artifacts = dominant_language_from_entities(predictions)
            ranked_langs = sorted(
                lang_stats.items(),
                key=lambda item: item[1]["rank_score"],
                reverse=True,
            )
        else:
            pred_lang, lang_stats, ignored_artifacts = predictions
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

        results_by_lang[true_lang].append(
            {
                "text": text[:100],
                "predicted": pred_lang,
                "correct": is_correct,
                "score": lang_stats.get(pred_lang, {}).get("rank_score", 0.0),
                "true_lang": true_lang,
                "dataset_label": true_lang_name,
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

    # ===== RESULTS =====
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
    print("SAMPLE ERRORS")
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
        status = "✓ EXCELLENT - Model generalizes well to this filtered set"
    elif overall_accuracy >= 70:
        status = "✓ GOOD - Model works reasonably well"
    elif overall_accuracy >= 50:
        status = "⚠ ACCEPTABLE - May need refinement"
    else:
        status = "✗ POOR - Model needs significant improvement"

    print(f"\nStatus: {status}")
    print(f"Accuracy: {overall_accuracy:.1f}%")
    print(f"\nLanguages covered:")
    print(f"  Total languages in model: {len(ALL_LANGS)}")
    print(f"  Filtered languages: {len(per_lang_stats)}")

    with mismatch_output.open("w", encoding="utf-8") as f:
        for item in mismatches:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n✓ Mismatches saved to {mismatch_output} ({len(mismatches)} rows)")

    if any(per_lang_stats[l]["total"] > 5 for l in per_lang_stats):
        best_lang = max(
            (l for l in per_lang_stats if per_lang_stats[l]["total"] > 5),
            key=lambda l: per_lang_stats[l]["correct"] / max(per_lang_stats[l]["total"], 1),
        )
    else:
        best_lang = "N/A"
    print(f"  Best performing: {best_lang}")

    with results_output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_accuracy": overall_accuracy,
                "correct": correct_count,
                "total": total_count,
                "filtered_langs": keep_langs,
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
            ensure_ascii=False,
        )

    print(f"\n✓ Detailed results saved to {results_output}")


if __name__ == "__main__":
    main()
