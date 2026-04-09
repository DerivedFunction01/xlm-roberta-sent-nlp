from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
CODE_FENCE_RE = re.compile(r"```|~~~")
URL_RE = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
CODEISH_RE = re.compile(
    r"\b(def|class|import|from|return|function|const|let|var|if|else|for|while|try|catch|except|throw|switch|case|public|private|static)\b"
)
MATHISH_RE = re.compile(r"[=+\-*/^<>|~]{2,}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe instruction parquet caches for unigram and noise signals.")
    parser.add_argument(
        "--cache-dir",
        default="sentences_cache/instruction_sentences",
        help="Directory containing per-language parquet files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many top unigrams to print per language.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="How many suspicious examples to print per language.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save the full EDA summary as JSON.",
    )
    return parser.parse_args()


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def _sentence_flags(text: str) -> list[str]:
    flags: list[str] = []
    cleaned = text.strip()
    if not cleaned:
        return ["empty"]
    if len(cleaned) < 20:
        flags.append("short")
    if len(cleaned) > 1500:
        flags.append("very_long")
    if CODE_FENCE_RE.search(cleaned):
        flags.append("code_fence")
    if URL_RE.search(cleaned):
        flags.append("url")
    if CODEISH_RE.search(cleaned):
        flags.append("codeish_keyword")
    if MATHISH_RE.search(cleaned):
        flags.append("mathish")
    token_count = len(TOKEN_RE.findall(cleaned))
    if token_count < 4:
        flags.append("lt4_tokens")
    digit_ratio = sum(ch.isdigit() for ch in cleaned) / max(1, len(cleaned))
    if digit_ratio > 0.20:
        flags.append("digit_heavy")
    punct_ratio = sum((not ch.isalnum()) and (not ch.isspace()) for ch in cleaned) / max(1, len(cleaned))
    if punct_ratio > 0.35:
        flags.append("punct_heavy")
    if cleaned.isupper() and len(cleaned) <= 24:
        flags.append("all_caps_short")
    return flags


def _summarize_series(values: pd.Series) -> dict[str, Any]:
    lengths = values.str.len().tolist()
    token_counts = values.map(lambda s: len(_tokenize(s))).tolist()
    return {
        "count": len(values),
        "char_min": min(lengths) if lengths else 0,
        "char_median": median(lengths) if lengths else 0,
        "char_mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "char_max": max(lengths) if lengths else 0,
        "token_median": median(token_counts) if token_counts else 0,
        "token_mean": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
    }


def _analyze_parquet(path: Path, top_n: int, sample_limit: int) -> dict[str, Any]:
    df = pd.read_parquet(path)
    if "sentence" not in df.columns:
        return {"error": "missing sentence column"}

    sentences = df["sentence"].dropna().astype(str)
    tokens = Counter()
    flag_counts = Counter()
    suspicious_rows: list[dict[str, Any]] = []

    for idx, sentence in enumerate(sentences):
        toks = _tokenize(sentence)
        tokens.update(toks)
        flags = _sentence_flags(sentence)
        flag_counts.update(flags)
        if flags:
            suspicious_rows.append(
                {
                    "idx": idx,
                    "flags": flags,
                    "sentence": sentence[:500],
                }
            )

    summary = _summarize_series(sentences)
    top_unigrams = tokens.most_common(top_n)
    suspicious_rows.sort(key=lambda item: (len(item["flags"]), len(item["sentence"])), reverse=True)

    return {
        "summary": summary,
        "top_unigrams": top_unigrams,
        "flag_counts": dict(flag_counts),
        "suspicious_examples": suspicious_rows[:sample_limit],
    }


def main() -> None:
    args = _parse_args()
    cache_dir = Path(args.cache_dir)
    parquet_paths = sorted(cache_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {cache_dir}")

    report: dict[str, Any] = {}
    print(f"Scanning {len(parquet_paths)} parquet files under {cache_dir}\n")
    for path in tqdm(parquet_paths, desc="Parquets"):
        lang = path.stem
        print(f"=== {lang} ===")
        result = _analyze_parquet(path, args.top_n, args.sample_limit)
        report[lang] = result
        if "error" in result:
            print(f"  error: {result['error']}")
            continue
        summary = result["summary"]
        print(
            "  rows={count:,} | chars min/med/mean/max="
            "{char_min}/{char_median}/{char_mean}/{char_max} | tokens med/mean="
            "{token_median}/{token_mean}".format(**summary)
        )
        print("  top unigrams:")
        for token, count in result["top_unigrams"]:
            print(f"    - {token}: {count}")
        print("  flag counts:")
        for flag, count in sorted(result["flag_counts"].items(), key=lambda item: (-item[1], item[0])):
            print(f"    - {flag}: {count}")
        print("  suspicious examples:")
        for item in result["suspicious_examples"]:
            print(f"    - idx={item['idx']} flags={','.join(item['flags'])} :: {item['sentence']}")
        print()

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {output_path}")


if __name__ == "__main__":
    main()
