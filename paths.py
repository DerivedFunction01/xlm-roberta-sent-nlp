from __future__ import annotations

import os

from typing import Any


PATHS: dict[str, Any] = {
    "sentences_dir": "./sentences_cache",
}

PATHS["wiki"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "wiki.parquet"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "wiki_cleanup.meta.json"),
    "temp_dir": os.path.join(PATHS["sentences_dir"], "_wiki_tmp"),
    "seg_debug_dir": os.path.join(PATHS["sentences_dir"], "_wiki_tmp", "segmentation_debug"),
}
PATHS["smol"] = {
    "cache_file": os.path.join(PATHS["sentences_dir"], "smol_sentences.json"),
}
PATHS["finetrans"] = {
    "cache_file": os.path.join(PATHS["sentences_dir"], "finetranslations_sentences.parquet"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "finetranslations_sentences.meta.json"),
    "cache_dir": os.path.join(PATHS["sentences_dir"], "finetranslations"),
    "cache_dir_meta": os.path.join(PATHS["sentences_dir"], "finetranslations", "finetranslations.meta.json"),
    "temp_dir": os.path.join(PATHS["sentences_dir"], "_finetrans_tmp"),
    "temp_file": os.path.join(PATHS["sentences_dir"], "_finetrans_tmp", "finetranslations_sentences.parquet"),
}
PATHS["synthetic"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "synthetic_examples"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "synthetic_examples", "synthetic_examples.meta.json"),
    "temp_dir": os.path.join(PATHS["sentences_dir"], "_synthetic_tmp"),
}
PATHS["tokenized"] = {
    "cache_dir": os.path.join(PATHS["sentences_dir"], "tokenized_dataset"),
    "cache_meta": os.path.join(PATHS["sentences_dir"], "tokenized_dataset", "tokenized_dataset.meta.json"),
}
PATHS["versions"] = {
    "cache": 2,
    "tokenized": 2,
}

for path in [
    PATHS["sentences_dir"],
    PATHS["wiki"]["temp_dir"],
    PATHS["wiki"]["seg_debug_dir"],
    PATHS["finetrans"]["temp_dir"],
    PATHS["finetrans"]["cache_dir"],
    PATHS["synthetic"]["cache_dir"],
    PATHS["synthetic"]["temp_dir"],
    PATHS["tokenized"]["cache_dir"],
]:
    os.makedirs(path, exist_ok=True)
