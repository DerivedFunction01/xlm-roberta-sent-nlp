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
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

from evaluation_language_utils import dominant_language_from_entities

MODEL_CHECKPOINT = "DerivedFunction/lang-ner-xlmr"
TOKENIZER_MODEL = "xlm-roberta-base"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the model on papluca/language-identification."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Maximum number of examples to evaluate.",
    )
    parser.add_argument(
        "--mismatch-output",
        type=Path,
        default=Path("papluca_mismatches.jsonl"),
        help="Path to write mismatched examples as JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 80)
    print("TESTING ON PAPLUCA LANGUAGE-IDENTIFICATION DATASET")
    print("=" * 80)

    print("\n1. Loading model and tokenizer...")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    print(f"   ✓ Model loaded: {MODEL_CHECKPOINT}")
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

    print("\n3. Setting up inference pipeline...")
    nlp = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    print("   ✓ Pipeline ready")

    print("\n4. Running inference on dataset...")
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

    sample_size = min(args.sample_size, len(test_data))
    print(f"Processing {sample_size} examples...\n")

    for example in tqdm(
        test_data.select(range(sample_size)),
        total=sample_size,
        desc="Running inference",
    ):
        text = example["text"]
        true_lang_name = example["labels"]
        true_lang = papluca_to_iso.get(true_lang_name.lower(), true_lang_name.lower())

        result = nlp(text[:512])

        pred_lang, lang_stats, ignored_artifacts = dominant_language_from_entities(result)
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

    if args.mismatch_output:
        args.mismatch_output.parent.mkdir(parents=True, exist_ok=True)
        with args.mismatch_output.open("w", encoding="utf-8") as f:
            for item in mismatches:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n✓ Mismatches saved to {args.mismatch_output} ({len(mismatches)} rows)")

    with open("papluca_results.json", "w", encoding="utf-8") as f:
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

    print(f"\n✓ Detailed results saved to papluca_results.json")


if __name__ == "__main__":
    main()
