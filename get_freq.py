import pandas as pd
import requests

import text_utils
from language import LANG_TO_GROUP, canonical_lang

BASE = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018"


LANG_CONFIG = {
    "en": {"cutoff": 5650, "min_freq": 5},
    "es": {"cutoff": 4600, "min_freq": 5},
    "fr": {"cutoff": 3200, "min_freq": 5},
    "de": {"cutoff": 1950, "min_freq": 5},
    "it": {"cutoff": 3300, "min_freq": 5},
    "pt": {"cutoff": 2600, "min_freq": 5},
    "da": {"cutoff": 2450, "min_freq": 5},
    "no": {"cutoff": 2000, "min_freq": 5},
    "sv": {"cutoff": 1300, "min_freq": 5},
    "pl": {"cutoff": 3950, "min_freq": 5},
    "tr": {"cutoff": 3250, "min_freq": 5},
    "fi": {"cutoff": 2800, "min_freq": 5},
    "vi": {"cutoff": 1100, "min_freq": 5},
    "id": {"cutoff": 1200, "min_freq": 5},
    "ru": {"cutoff": 2000, "min_freq": 5},
    "uk": {"cutoff": 500, "min_freq": 5},
    "ar": {"cutoff": 1950, "min_freq": 5},
}

KEEP_APOSTROPHE_START = False
STRIP_MID_APOSTROPHE = False


def _is_int_token(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _infer_freq_column(sample_rows: list[list[str]]) -> int:
    first_col_hits = 0
    second_col_hits = 0
    for parts in sample_rows:
        if len(parts) < 2:
            continue
        if _is_int_token(parts[0]):
            first_col_hits += 1
        if _is_int_token(parts[1]):
            second_col_hits += 1
    if first_col_hits > second_col_hits:
        return 0
    return 1


def _parse_word_freq(parts: list[str], freq_col: int) -> tuple[str, int] | None:
    if len(parts) < 2:
        return None
    if freq_col not in (0, 1):
        raise ValueError(f"Unsupported frequency column: {freq_col}")
    other_col = 1 - freq_col
    if len(parts) <= other_col:
        return None
    word_col = other_col
    if not _is_int_token(parts[freq_col]):
        if _is_int_token(parts[other_col]):
            freq_col = other_col
            word_col = 1 - freq_col
        else:
            return None
    word = parts[word_col]
    freq = int(parts[freq_col])
    return word, freq


def fetch_wordlist(lang: str, cutoff: int, min_freq: int) -> tuple[list[dict], int]:
    url = f"{BASE}/{lang}/{lang}_50k.txt"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lang = canonical_lang(lang)
    lines = r.text.splitlines()
    sample_rows = [line.strip().split() for line in lines[: min(10, len(lines))]]
    freq_col = _infer_freq_column(sample_rows)
    rows = []
    contaminated_count = 0
    for i, line in enumerate(lines):
        if i >= cutoff:
            break
        parts = line.strip().split()
        parsed = _parse_word_freq(parts, freq_col)
        if parsed is None:
            continue
        word, freq = parsed
        if freq < min_freq:
            continue
        starts_apos = word.startswith("'")
        has_mid_apos = "'" in word and not starts_apos
        if starts_apos and not KEEP_APOSTROPHE_START:
            continue
        if has_mid_apos and STRIP_MID_APOSTROPHE:
            continue
        word = word.lower()
        if text_utils._has_script_contamination(word, lang, LANG_TO_GROUP):
            contaminated_count += 1
            continue
        rows.append(
            {
                "word": word,
                "lang": lang,
                "freq": freq,
                "rank": i + 1,
            }
        )
    return rows, contaminated_count


def main() -> None:
    dfs = []
    contaminated_total = 0
    for lang, cfg in LANG_CONFIG.items():
        print(f"Fetching {lang}...")
        rows, contaminated_count = fetch_wordlist(lang, cfg["cutoff"], cfg["min_freq"])
        contaminated_total += contaminated_count
        dfs.append(pd.DataFrame(rows))

    df = pd.concat(dfs, ignore_index=True)

    overlap = (
        df.groupby("word")["lang"].apply(lambda x: ",".join(sorted(set(x)))).reset_index()
    )
    overlap.columns = ["word", "all_langs"]
    df = df.merge(overlap, on="word")
    df["overlaps"] = df.apply(
        lambda r: ",".join(l for l in r["all_langs"].split(",") if l != r["lang"]), axis=1
    )
    df["is_overlap"] = df["overlaps"] != ""
    df[["word", "lang", "freq", "rank", "overlaps", "is_overlap"]].to_parquet(
        "word_dict.parquet", index=False
    )
    print(
        f"Done — {len(df):,} clean rows → word_dict.parquet "
        f"({contaminated_total:,} contaminated words skipped)"
    )


if __name__ == "__main__":
    main()
