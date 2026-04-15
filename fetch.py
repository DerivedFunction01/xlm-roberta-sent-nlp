# %%
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from huggingface_hub import login
from transformers import AutoTokenizer

from fetch_sources import refresh_sources
from language import ALL_LANGS
from neutral_sources import build_neutral_sources
from multilabel_converter import build_label_maps, convert_and_save_multilabel_dataset
from paths import PATHS
from source_pools import load_language_sentences_from_parquet
from synthetic_build import build_synthetic_dataset
from tokenization_cache import build_tokenized_dataset
from tqdm.auto import tqdm

DEFAULT_MODEL_CHECKPOINT = "xlm-roberta-base"


def _maybe_login() -> None:
    token_path = Path("hf_token")
    if not token_path.exists():
        return
    with token_path.open() as f:
        token = f.read().strip()
    if token:
        login(token=token)
        print("Logged in to Hugging Face Hub")

refresh_sources()
_maybe_login()

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_CHECKPOINT)
label2id, id2label = build_label_maps(ALL_LANGS)
#%%
wiki_english_seed_sentences = load_language_sentences_from_parquet(PATHS["wiki"]["cache_dir"], "en")
ft_english_seed_sentences = load_language_sentences_from_parquet(PATHS["finetrans"]["cache_dir"], "en")
tatoeba_english_seed_sentences = load_language_sentences_from_parquet(PATHS["tatoeba"]["cache_dir"], "en")
neutral_sources = build_neutral_sources(
    english_seed_sentences=wiki_english_seed_sentences + ft_english_seed_sentences,
)
#%%
print("Building synthetic dataset cache ...")
synthetic_dataset = build_synthetic_dataset(
    tokenizer=tokenizer,
    label2id=label2id,
    id2label=id2label,
    sample_o_span=neutral_sources.sample_o_span,
    sample_code_span=neutral_sources.sample_code_span,
)
print(f"Synthetic examples: {len(synthetic_dataset):,}")
#%%
print("Building tokenized dataset cache ...")
train_dataset, eval_dataset = build_tokenized_dataset(
    synthetic_dataset,
    model_checkpoint=DEFAULT_MODEL_CHECKPOINT,
    tokenizer=tokenizer,
    label2id=label2id,
    id2label=id2label,
    max_length=512,
)
print(f"Tokenized train/eval: {len(train_dataset):,} / {len(eval_dataset):,}")

print("Building multilabel dataset cache ...")
multilabel_dataset = convert_and_save_multilabel_dataset()
for split_name, split in multilabel_dataset.items():
    print(f"  {split_name}: {len(split):,} examples")


# %%
