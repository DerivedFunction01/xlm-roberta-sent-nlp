# %%
from __future__ import annotations

import multiprocessing as mp
from convert_tatoeba_sentences import convert_tatoeba_sentences
from finetranslations_sources import load_finetranslations_sentences
from wiki_sources import load_wiki_sentences
from smol_sources import load_smol_sentences
from huggingface_hub import login
from pathlib import Path
#%%
def _maybe_login() -> None:
    token_path = Path("hf_token")
    if not token_path.exists():
        return
    with token_path.open() as f:
        token = f.read().strip()
    if token:
        login(token=token)
        print("Logged in to Hugging Face Hub")

_maybe_login()
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
    print("Refreshing Tatoeba caches ...")
    _ = convert_tatoeba_sentences()
    #%%
    _ = load_smol_sentences()

#%%
def main() -> None:
    refresh_sources()

#%%
if __name__ == "__main__":
    main()

# %%
