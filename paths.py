from __future__ import annotations

import os

from typing import Any


PATHS: dict[str, Any] = {
    "sentences_dir": "./sentences_cache",
}

PATHS["wiki"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "wiki"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "wiki", "wiki_cleanup.meta.json"),
    "temp_dir": os.path.join(PATHS["sentences_dir"], "_wiki_tmp"),
    "seg_debug_dir": os.path.join(
        PATHS["sentences_dir"], "_wiki_tmp", "segmentation_debug"
    ),
}
PATHS["smol"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "smol_sentences"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "smol_sentences", "smol_sentences.meta.json"),
}
PATHS["instruction"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "instruction_sentences"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "instruction_sentences", "instruction_sentences.meta.json"),
}
PATHS["finetrans"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "finetranslations"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "finetranslations", "finetranslations.meta.json"),
    "temp_dir": os.path.join(PATHS["sentences_dir"], "_finetrans_tmp"),
    "temp_file": os.path.join(PATHS["sentences_dir"], "_finetrans_tmp", "finetranslations_sentences.parquet"),
}
PATHS["neutral"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "neutral_sentences"),
}
PATHS["synthetic"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "synthetic_examples"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "synthetic_examples", "synthetic_examples.meta.json"),
    "temp_dir": os.path.join(PATHS["sentences_dir"], "_synthetic_tmp"),
}
PATHS["source_pools"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "source_pools"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "source_pools", "sentence_pools.meta.json"),
}
PATHS["tokenized"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "tokenized_dataset"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "tokenized_dataset", "tokenized_dataset.meta.json"),
}
PATHS["multilabel_dataset"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "multilabel_dataset"),
}
PATHS["versions"] = {
    "cache": 2,
    "tokenized": 2,
}

for path in [
    PATHS["sentences_dir"],
    PATHS["wiki"]["temp_dir"],
    PATHS["wiki"]["seg_debug_dir"],
    PATHS["smol"]["cache_dir"],
    PATHS["instruction"]["cache_dir"],
    PATHS["finetrans"]["temp_dir"],
    PATHS["finetrans"]["cache_dir"],
    PATHS["synthetic"]["cache_dir"],
    PATHS["synthetic"]["temp_dir"],
    PATHS["source_pools"]["cache_dir"],
    PATHS["tokenized"]["cache_dir"],
]:
    os.makedirs(path, exist_ok=True)
