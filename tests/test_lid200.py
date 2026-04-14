#!/usr/bin/env python3
"""
Test the model on the mikaberidze/lid200 language-identification dataset.

By default this evaluates only the languages supported by the local model
and mapped through `language.LANG_ISO2_TO_ISO3`. You can further narrow the
benchmark with `--langs en es fr` (ISO-2 codes).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
import sys
import torch # Added for GPU detection

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Attach the current directory (project root) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)
from language import ALL_LANGS, LANG_ISO2_TO_ISO3

MODEL_CHECKPOINT = "DerivedFunction/lang-ner-xlmr"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the model on mikaberidze/lid200."
    )
    parser.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help=(
            "Optional ISO-2 languages to keep, e.g. --langs en es fr. "
            "Defaults to all model languages that have a mapping in all_langs.json."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate, falling back to the first available split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference on GPU.",
    )
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


def main() -> None:
    args = _parse_args()

    print("=" * 80)
    print("TESTING ON MIKABERIDZE/LID200 DATASET")
    print("=" * 80)

    # ===== LOAD MODEL & TOKENIZER =====
    print("\n1. Loading model and tokenizer...")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    print(f"   ✓ Model loaded: {MODEL_CHECKPOINT}")
    print(f"   ✓ Tokenizer loaded")

    # ===== LOAD DATASET =====
    print("\n2. Loading mikaberidze/lid200 dataset...")
    dataset = load_dataset("mikaberidze/lid200")
    print(f"   ✓ Dataset loaded")
    print(f"   ✓ Splits available: {list(dataset.keys())}")

    split_to_use = args.split if args.split in dataset else list(dataset.keys())[0]
    test_data = dataset[split_to_use]
    print(f"   ✓ Using '{split_to_use}' split with {len(test_data)} examples")

    lang_column = _resolve_lang_column(test_data)
    text_column = _resolve_text_column(test_data)

    # ===== DETERMINE FILTER =====
    model_langs_iso2 = [lang for lang in ALL_LANGS if lang in LANG_ISO2_TO_ISO3]
    if args.langs:
        keep_langs_iso2 = [lang.lower() for lang in args.langs]
    else:
        keep_langs_iso2 = model_langs_iso2

    iso2_to_iso3 = LANG_ISO2_TO_ISO3
    iso3_to_iso2 = {iso3: iso2 for iso2, iso3 in iso2_to_iso3.items()}
    keep_langs_iso3 = {iso2_to_iso3[lang] for lang in keep_langs_iso2 if lang in iso2_to_iso3}

    print("\n3. Filtering languages...")
    print(f"   ✓ Keeping {len(keep_langs_iso2)} ISO-2 languages")
    print(f"   ✓ Corresponding ISO-3 labels: {len(keep_langs_iso3)}")
    # Print one example row in test_data
    print(f"   ✓ Example dataset row: {test_data[0]}")
    filtered_test = test_data.filter(
        lambda ex: _dataset_label_to_iso3(_label_name_from_example(ex, test_data, lang_column))
        in keep_langs_iso3
    )
    print(f"   ✓ Filtered dataset size: {len(filtered_test)}")

    if len(filtered_test) == 0:
        raise RuntimeError(
            "No examples matched the selected languages. "
            "Check the --langs values against all_langs.json."
        )

    # ===== SETUP PIPELINE =====
    print("\n4. Setting up inference pipeline...")
    nlp = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1, # Use GPU if available
    )
    print("   ✓ Pipeline ready")

    # ===== RUN INFERENCE =====
    print("\n5. Running inference on filtered dataset...")
    print("-" * 80)

    correct_count = 0
    total_count = 0
    per_lang_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    results_by_lang = defaultdict(list)

    # Prepare texts for batch inference, replicating the original text[:512] truncation
    texts_for_inference = [example[text_column] for example in filtered_test]

    # Run inference on all texts at once using the pipeline
    # The pipeline will handle batching internally for efficiency on GPU
    all_predictions = nlp(
        texts_for_inference,
        batch_size=args.batch_size,
    )

    print(f"Processing {len(filtered_test)} examples and their predictions...\n")

    for example_idx, (example, predictions) in enumerate(tqdm(
        zip(filtered_test, all_predictions),
        total=len(filtered_test),
        desc="Processing predictions",
    )):
        text = example[text_column]
        true_lang_name = _label_name_from_example(example, filtered_test, lang_column)
        true_lang_iso2 = iso3_to_iso2.get(_dataset_label_to_iso3(true_lang_name))
        if true_lang_iso2 is None:
            continue

        if not predictions:
            continue

        pred = predictions[0]
        pred_entity = pred.get("entity_group", pred.get("entity", "O"))
        if pred_entity.startswith(("B-", "I-")):
            pred_lang = pred_entity[2:].lower()
        else:
            pred_lang = pred_entity.lower()

        is_correct = pred_lang == true_lang_iso2
        if is_correct:
            correct_count += 1

        total_count += 1
        per_lang_stats[true_lang_iso2]["total"] += 1
        if is_correct:
            per_lang_stats[true_lang_iso2]["correct"] += 1

        results_by_lang[true_lang_iso2].append(
            {
                "text": text[:100],
                "predicted": pred_lang,
                "correct": is_correct,
                "score": pred.get("score", 0.0),
                "true_lang": true_lang_iso2,
                "dataset_label": true_lang_name,
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

    if any(per_lang_stats[l]["total"] > 5 for l in per_lang_stats):
        best_lang = max(
            (l for l in per_lang_stats if per_lang_stats[l]["total"] > 5),
            key=lambda l: per_lang_stats[l]["correct"] / max(per_lang_stats[l]["total"], 1),
        )
    else:
        best_lang = "N/A"
    print(f"  Best performing: {best_lang}")

    with open("lid200_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_accuracy": overall_accuracy,
                "correct": correct_count,
                "total": total_count,
                "filtered_langs_iso2": keep_langs_iso2,
                "filtered_langs_iso3": sorted(keep_langs_iso3),
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

    print(f"\n✓ Detailed results saved to lid200_results.json")


if __name__ == "__main__":
    main()
