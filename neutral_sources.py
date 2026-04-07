from __future__ import annotations

import codecs
import json
import os
import random
import string
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from datasets import load_dataset
from faker import Faker

import code_noise
import math_gen
from paths import SENTENCES_DIR

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


LATEX_MIN_CHARS = 8
LATEX_MAX_CHARS = 300
SYNTH_MATH_N = 50_000
NOISE_N = 30_000
GIBBERISH_N = 30_000

_LATEX_WRAP = __import__("re").compile(r"^\s*\$+|\$+\s*$|^\\\[|\\\]$|^\\begin\{.*?\}|\\end\{.*?\}$")


def _collapse_spaces(text: str) -> str:
    return __import__("re").sub(r"\s{2,}", " ", text).strip()


def _write_text_parquet(path: str, column_name: str, values: list[str]) -> None:
    if not values:
        pd.DataFrame({column_name: []}).to_parquet(path, index=False)
        return
    if pa is None or pq is None:
        pd.DataFrame({column_name: values}).to_parquet(path, index=False)
        return
    table = pa.table({column_name: pa.array(values, type=pa.string())})
    pq.write_table(table, path)


def _load_or_build_text_pool(path: str, column_name: str, builder: Callable[[], list[str]], label: str) -> list[str]:
    if os.path.exists(path):
        return pd.read_parquet(path)[column_name].tolist()
    values = builder()
    _write_text_parquet(path, column_name, values)
    print(f"  Cached {len(values)} {label} -> {path}")
    return values


def _clean_formula(f: str) -> str:
    return _LATEX_WRAP.sub("", f).strip()


def load_latex_formulas(sentences_dir: str = SENTENCES_DIR) -> list[str]:
    cache = os.path.join(sentences_dir, "latex_formulas.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)["formula"].tolist()
    print("Downloading im2latex-100k ...")
    ds = load_dataset("yuntian-deng/im2latex-100k", split="train")
    formulas = []
    for row in ds:
        formula = row["formula"] if isinstance(row, dict) else ""
        assert isinstance(formula, str)
        f = _clean_formula(formula)
        if LATEX_MIN_CHARS <= len(f) <= LATEX_MAX_CHARS:
            formulas.append(f)
    pd.DataFrame({"formula": formulas}).to_parquet(cache, index=False)
    print(f"  Cached {len(formulas)} usable formulas -> {cache}")
    return formulas


def generate_symbol_noise(min_len: int = 3, max_len: int = 20) -> str:
    fake = Faker()
    n = random.randint(min_len, max_len)
    parts = []
    for _ in range(n):
        parts.append(fake.emoji() if random.random() < 0.5 else random.choice(string.punctuation))
    out = []
    for i, ch in enumerate(parts):
        out.append(ch)
        if i < len(parts) - 1 and random.random() < 0.3:
            out.append(" ")
    return "".join(out)


@dataclass
class NeutralSources:
    latex_formulas: list[str]
    synth_math_pool: list[str]
    html_noise_pool: list[str]
    css_noise_pool: list[str]
    code_noise_pool: list[str]
    noise_pool: list[str]
    gibberish_pool: list[str]

    def sample_o_span(self) -> str:
        pools = [
            self.latex_formulas,
            self.synth_math_pool,
            self.html_noise_pool,
            self.css_noise_pool,
            self.code_noise_pool,
            self.noise_pool,
            self.gibberish_pool,
        ]
        weights = [0.28, 0.22, 0.14, 0.08, 0.10, 0.09, 0.09]
        pool = random.choices(pools, weights=weights, k=1)[0]
        return random.choice(pool)

    def sample_code_span(self) -> str:
        return random.choice(self.code_noise_pool)


def build_neutral_sources(
    *,
    sentences_dir: str = SENTENCES_DIR,
    english_seed_sentences: list[str] | None = None,
    seed: int = 42,
) -> NeutralSources:
    english_seed_sentences = english_seed_sentences or []
    random.seed(seed)
    fake = Faker()

    latex_formulas = load_latex_formulas(sentences_dir)
    print(f"im2latex:    {len(latex_formulas):>6} formulas")

    synth_math_pool = _load_or_build_text_pool(
        os.path.join(sentences_dir, "synth_math_pool.parquet"),
        "expression",
        lambda: [math_gen.generate_synthetic_math() for _ in range(SYNTH_MATH_N)],
        "synthetic math expressions",
    )
    print(f"math_gen:    {len(synth_math_pool):>6} expressions")

    html_noise_pool = _load_or_build_text_pool(
        os.path.join(sentences_dir, "html_noise_pool.parquet"),
        "snippet",
        lambda: [code_noise.generate_html_artifact() for _ in range(NOISE_N)],
        "HTML noise snippets",
    )
    print(f"html noise:  {len(html_noise_pool):>6} snippets")

    css_noise_pool = _load_or_build_text_pool(
        os.path.join(sentences_dir, "css_noise_pool.parquet"),
        "snippet",
        lambda: [code_noise.generate_css_artifact() for _ in range(NOISE_N)],
        "CSS noise snippets",
    )
    print(f"css noise:   {len(css_noise_pool):>6} snippets")

    code_noise_pool = _load_or_build_text_pool(
        os.path.join(sentences_dir, "code_noise_pool.parquet"),
        "snippet",
        lambda: [code_noise.generate_code_artifact() for _ in range(NOISE_N)],
        "code noise snippets",
    )
    print(f"code noise:  {len(code_noise_pool):>6} snippets")

    noise_pool = _load_or_build_text_pool(
        os.path.join(sentences_dir, "symbol_noise_pool.parquet"),
        "snippet",
        lambda: [generate_symbol_noise() for _ in range(NOISE_N)],
        "symbol noise strings",
    )
    print(f"symbol noise:{len(noise_pool):>6} strings")

    def generate_gibberish_text() -> str:
        if english_seed_sentences and random.random() < 0.7:
            base = random.choice(english_seed_sentences)
        else:
            base = fake.text(max_nb_chars=random.randint(80, 240))
        gib = codecs.decode(base, "rot_13")
        return _collapse_spaces(gib).strip()

    gibberish_pool = _load_or_build_text_pool(
        os.path.join(sentences_dir, "gibberish_pool.parquet"),
        "snippet",
        lambda: [generate_gibberish_text() for _ in range(GIBBERISH_N)],
        "gibberish strings",
    )
    print(f"gibberish:  {len(gibberish_pool):>6} strings")

    return NeutralSources(
        latex_formulas=latex_formulas,
        synth_math_pool=synth_math_pool,
        html_noise_pool=html_noise_pool,
        css_noise_pool=css_noise_pool,
        code_noise_pool=code_noise_pool,
        noise_pool=noise_pool,
        gibberish_pool=gibberish_pool,
    )
