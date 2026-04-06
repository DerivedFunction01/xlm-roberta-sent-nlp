from __future__ import annotations

import os


SENTENCES_DIR = "./sentences_cache"
WIKI_TEMP_DIR = os.path.join(SENTENCES_DIR, "_wiki_tmp")
WIKI_SEGMENTATION_DEBUG_DIR = os.path.join(WIKI_TEMP_DIR, "segmentation_debug")
SMOL_CACHE_FILE = os.path.join(SENTENCES_DIR, "smol_sentences.json")
FINETRANS_CACHE_FILE = os.path.join(SENTENCES_DIR, "finetranslations_sentences.parquet")
FINETRANS_CACHE_META = os.path.join(SENTENCES_DIR, "finetranslations_sentences.meta.json")
FINETRANS_TEMP_DIR = os.path.join(SENTENCES_DIR, "_finetrans_tmp")
FINETRANS_TEMP_FILE = os.path.join(FINETRANS_TEMP_DIR, "finetranslations_sentences.parquet")
SYNTHETIC_CACHE = os.path.join(SENTENCES_DIR, "synthetic_examples")
SYNTHETIC_CACHE_META = os.path.join(SYNTHETIC_CACHE, "synthetic_examples.meta.json")
SYNTHETIC_TEMP_DIR = os.path.join(SENTENCES_DIR, "_synthetic_tmp")
CACHE_DIR = f"{SENTENCES_DIR}/tokenized_dataset"
CACHE_META = f"{CACHE_DIR}/tokenized_dataset.meta.json"
CACHE_VERSION = 2
TOKENIZED_CACHE_VERSION = 2

os.makedirs(SENTENCES_DIR, exist_ok=True)
os.makedirs(WIKI_TEMP_DIR, exist_ok=True)
os.makedirs(WIKI_SEGMENTATION_DEBUG_DIR, exist_ok=True)
os.makedirs(FINETRANS_TEMP_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_CACHE, exist_ok=True)
os.makedirs(SYNTHETIC_TEMP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
