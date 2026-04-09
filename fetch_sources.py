# %%
from __future__ import annotations

import multiprocessing as mp
from finetranslations_sources import load_finetranslations_sentences
from wiki_sources import load_wiki_sentences
from smol_sources import load_smol_sentences
max_workers = max(1, mp.cpu_count() // 3)
# %%
print("Refreshing wiki caches ...")
load_wiki_sentences(
    max_workers=max_workers,
)
# %%
print("Refreshing FineTranslations caches ...")
load_finetranslations_sentences(
    max_workers=max_workers,
)
# %%
load_smol_sentences()

# %%
