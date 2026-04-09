# %%
from __future__ import annotations

import multiprocessing as mp
from finetranslations_sources import load_finetranslations_sentences
from instruction_sources import load_instruction_sentences
from wiki_sources import load_wiki_sentences
from smol_sources import load_smol_sentences

#%%
def refresh_sources() -> None:
    #%%
    workers = max(1, mp.cpu_count() // 3)
    #%%
    print("Refreshing wiki caches ...")
    _ = load_wiki_sentences(
        max_workers=workers,
    )
    #%%
    print("Refreshing FineTranslations caches ...")
    _ = load_finetranslations_sentences(
        max_workers=workers,
    )
    #%%
    print("Refreshing instruction-source caches ...")
    _ = load_instruction_sentences(
        max_workers=workers,
    )
    #%%
    _ = load_smol_sentences()

#%%
def main() -> None:
    refresh_sources()

#%%
if __name__ == "__main__":
    main()

# %%
