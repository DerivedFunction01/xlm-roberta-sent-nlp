from __future__ import annotations
from script_types import Script

LANGUAGE_BUCKETS = {
    # ~41% of CC — intentionally capped to avoid crowding out other languages
    "English": {
        "langs": ["en"],
        "weight": 2.9,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ~6.3% of CC — was badly underweighted relative to German/French
    "Russian": {
        "langs": ["ru"],
        "weight": 1.95,
        "min_chars": 2_000,
        "script": Script.CYRILLIC,
    },
    # ~5.9% of CC
    "German": {
        "langs": ["de"],
        "weight": 1.9,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ~5.7% of CC — bumped up from 1.7 to match its actual footprint
    "Japanese": {
        "langs": ["ja"],
        "weight": 1.9,
        "min_chars": 1_200,
        "script": Script.JAPANESE,
    },
    # ~5.0% of CC — CC likely undercounts due to Great Firewall
    "Chinese": {
        "langs": ["zh"],
        "weight": 1.9,
        "min_chars": 1_200,
        "script": Script.HAN,
    },
    # ~4.6% of CC
    "French": {
        "langs": ["fr"],
        "weight": 1.9,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ~4.6% of CC
    "Spanish": {
        "langs": ["es"],
        "weight": 1.9,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ~2.5% of CC
    "Portuguese": {
        "langs": ["pt"],
        "weight": 1.7,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ~2.4% of CC
    "Italian": {
        "langs": ["it"],
        "weight": 1.6,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ~2.0% of CC — split out from CentralEuropeanLatin; rivals Italian/Portuguese
    "Polish": {
        "langs": ["pl"],
        "weight": 1.55,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ~1.8% of CC — was significantly underweighted at 1.15
    "Dutch": {
        "langs": ["nl"],
        "weight": 1.55,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ~1.2% of CC — split out from CentralEuropeanLatin; large internet population
    "Turkish": {
        "langs": ["tr"],
        "weight": 1.45,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ind ~1.1%, vie ~1.05% of CC
    "SoutheastAsianLatin": {
        "langs": ["vi", "id", "ms", "tl"],
        "weight": 1.55,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    "WesternLatin": {
        "langs": [
            "ca",
            "gl",
            "oc",
            "eu",
            "la",
        ],
        "weight": 1.2,
        "min_chars": 1_500,
        "script": Script.LATIN,
    },
    "CelticLatin": {
        "langs": ["br", "ga", "gd", "cy", "sco"],
        "weight": 1.3,
        "min_chars": 1_500,
        "script": Script.LATIN,
    },
    "AdriaticLatin": {
        "langs": [
            "bs",
            "hr",
            "sl",
            "sk",
            "sq",
        ],
        "weight": 1.4,
        "min_chars": 1_500,
        "script": Script.LATIN,
    },
    "BalticLatin": {
        "langs": ["et", "lv", "lt"],
        "weight": 1.2,
        "min_chars": 1_500,
        "script": Script.LATIN,
    },
    # ces ~1.14%, ron ~0.53%, hun ~0.52% of CC — smaller tier after splitting out pl/tr
    "CentralEuropeanLatin": {
        "langs": ["cs", "ro", "hu"],
        "weight": 1.3,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # ~0.81% of CC — was overweighted at 1.7
    "Korean": {
        "langs": ["ko"],
        "weight": 1.35,
        "min_chars": 1_200,
        "script": Script.HANGUL,
    },
    # ukr ~0.70%, bel ~0.017% of CC
    "EastSlavicCyrillic": {
        "langs": ["uk", "be"],
        "weight": 1.7,
        "min_chars": 2_000,
        "script": Script.CYRILLIC,
    },
    # ~0.65% of CC — upweighted relative to CC share given speaker population
    "Arabic": {
        "langs": ["ar"],
        "weight": 1.4,
        "min_chars": 2_000,
        "script": Script.ARABIC,
    },
    "Norwegian": {
        "langs": ["no"],
        "weight": 1.0,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # sv ~0.7%, dan ~0.51%, fin ~0.37%, isl ~0.04%, afr ~0.01%
    # combined ~2.0% of CC — was drastically overweighted at 6.0
    # note: Swedish Wikipedia is heavily bot-generated stubs, don't rely on article count
    "NordicCore": {
        "langs": ["sv", "da", "is", "af", "fi"],
        "weight": 2.1,
        "min_chars": 2_000,
        "script": Script.LATIN,
    },
    # bul ~0.27%, srp ~0.25%, mkd ~0.037% of CC
    "BalkanCyrillic": {
        "langs": ["bg", "sr", "mk"],
        "weight": 1.05,
        "min_chars": 2_000,
        "script": Script.CYRILLIC,
    },
    # fas ~0.20% of CC (ignore the one anomalous crawl spike)
    "ArabicOther": {
        "langs": ["fa", "ps", "sd", "ug", "ur"],
        "weight": 0.95,
        "min_chars": 2_000,
        "script": Script.ARABIC,
    },
    # Hindi is the main Devanagari-script pool here, so keep it separate.
    "Hindi": {
        "langs": ["hi"],
        "weight": 1.25,
        "min_chars": 2_000,
        "script": Script.DEVANAGARI,
    },
    # Remaining shared Devanagari-script languages keep their own pooled bucket
    # so Devanagari character noise can be sampled consistently.
    "Devanagari": {
        "langs": ["mr", "ne"],
        "weight": 0.75,
        "min_chars": 2_000,
        "script": Script.DEVANAGARI,
    },
    # Shared Bengali-Assamese script.
    "Bengali": {
        "langs": ["bn", "as"],
        "weight": 0.55,
        "min_chars": 2_000,
        "script": Script.BENGALI,
    },
    # Remaining Indic scripts that do not have another same-script partner in
    # this repo are kept together as a fallback pool.
    "IndicOther": {
        "langs": ["ta", "te", "gu", "kn", "ml", "pa", "or"],
        "weight": 0.95,
        "min_chars": 2_000,
        "script": Script.INDIC_OTHER,
    },
    # kk ~0.038%, mn ~0.016% of CC — very thin corpus, weight is already a large relative boost
    "CentralAsianCaucusCyrillic": {
        "langs": ["kk", "mn", "tt", "ky", "tg", "ba", "ce"],
        "weight": 1.1,
        "min_chars": 2_000,
        "script": Script.CYRILLIC,
    },
    # Kurdish is split by script/source:
    # - ku: Wikipedia / Latin-script Kurdish
    # - ckb: FineTranslations / Arabic-script Kurdish
    "KurdishLatin": {
        "langs": ["ku"],
        "weight": 0.45,
        "min_chars": 1_500,
        "script": Script.LATIN,
    },
    "KurdishArabic": {
        "langs": ["ckb"],
        "weight": 0.45,
        "min_chars": 2_000,
        "script": Script.ARABIC,
    },
    "AfricanLatin": {
        "langs": ["sw", "yo", "zu", "ny", "xh"],
        "weight": 1.0,
        "min_chars": 1_500,
        "script": Script.LATIN,
    },
    "PeripheralLatin": {
        "langs": ["eo", "jv", "lb", "mg", "mt", "om", "rm", "so", "su", "uz"],
        "weight": 1.0,
        "min_chars": 1_500,
        "script": Script.LATIN,
    },
    # Split the remaining non-Latin scripts into two buckets to keep
    # Greco-Semitic/Caucasus-style scripts separate from Brahmic/Tibetan ones.
    "OtherScriptsWest": {
        "langs": ["el", "hy", "ka", "am", "ti", "dv"],
        "weight": 1.0,
        "min_chars": 2_000,
        "script": Script.MIXED_WEST,
    },
    "Hebrew": {
        "langs": ["he"],
        "weight": 0.95,
        "min_chars": 2_000,
        "script": Script.HEBREW,
    },
    "Yiddish": {
        "langs": ["yi"],
        "weight": 0.85,
        "min_chars": 2_000,
        "script": Script.HEBREW,
    },
    "OtherScriptsEast": {
        "langs": ["km", "lo", "my", "th", "si", "bo"],
        "weight": 1.0,
        "min_chars": 2_000,
        "script": Script.MIXED_EAST,
    },
}

WIKI = {
    "max_wiki_index": 100_000,
    "articles_per_lang": 10_000,
    "max_wiki_sentences": 150_000,
    "paragraph_fraction_default": 0.60,
    "paragraph_fractions_by_group": {
        "English": 0.75,
    },
    "long_paragraph_streak": 2,
    "long_paragraph_bonus": 0.10,
    "cap_multipliers": {
        "en": 1.50,
        "de": 1.25,
        "fr": 1.25,
        "es": 1.25,
        "ru": 1.25,
        "zh": 1.25,
        "ja": 1.25,
        "pt": 1.25,
        "it": 1.25,
        "hi": 1.25,
        "ko": 1.25,
        "ar": 1.25,
        "no": 0.65,
        "sco": 0.75,
    },
    "source_langs": {
        "no": ("no", "nn"),
    },
    "latin_secondary_groups": {
        "AfricanLatin",
        "AdriaticLatin",
        "BalticLatin",
        "CelticLatin",
        "Scots",
        "KurdishLatin",
        "PeripheralLatin",
        "WesternLatin",
    },
    "rolling_stats_window": 250,
    "length_priority_scan_limit": 66_666,
    "length_priority_sentence_cap_by_lang": {
        "vi": 50_000,
        "sv": 50_000,
        "lo": 20_000,
        "sd": 20_000,
        "am": 20_000,
        "km": 20_000,
        "ug": 20_000,
        "my": 20_000,
    },
    "length_priority_sentence_cap": 25_000,
}

TATOEBA = {
    "max_sentences": 100_000,
    "cap_multipliers": {
        "en": 1.50,
        "de": 1.25,
        "fr": 1.25,
        "es": 1.25,
        "ru": 1.25,
        "zh": 1.25,
        "ja": 1.25,
        "pt": 1.25,
        "it": 1.25,
        "ko": 1.25,
        "ar": 1.25,
        "sco": 0.75,
    },
}

DOC_MIX = {
    "pure": {
        "fraction": 0.55,
        "pool": "reserve",
        "min_sentences": 1,
        "max_sentences": 6,
        "strip_punct_prob": 0.10,
        # Occasionally drop diacritics in Latin-script text so Spanish/Portuguese
        # examples include accentless variants too.
        "accent_strip_prob": 0.05,
        # Insert a stray Latin letter inside an existing span to reduce single-letter bias.
        "random_letter_prob": 0.02,
        # Insert a stray digit inside an existing span to model typed or OCR noise.
        "random_digit_prob": 0.01,
        "foreign_sentence_prob": 0.0,
        "sentence_uppercase_prob": 0.02,
        "sentence_lowercase_prob": 0.03,
        "format_noise_prob": 0.30,
        "paragraph_break_prob": 0.25,
        "uppercase_word_prob": 0.03,
        "lowercase_word_prob": 0.03,
        "titlecase_word_prob": 0.02,
        "merge_word_prob": 0.03,
        "split_word_prob": 0.05,
        "typo_char_prob": 0.02,
    },
    "homogeneous": {
        "fraction": 0.25,
        "pool": "main",
        "min_sentences": 2,
        "max_sentences": 6,
        "strip_punct_prob": 0.15,
        "accent_strip_prob": 0.02,
        "random_letter_prob": 0.01,
        "random_digit_prob": 0.005,
        "foreign_sentence_prob": 0.85,
        "sentence_uppercase_prob": 0.01,
        "sentence_lowercase_prob": 0.02,
        "format_noise_prob": 0.20,
        "paragraph_break_prob": 0.20,
        "uppercase_word_prob": 0.02,
        "lowercase_word_prob": 0.02,
        "titlecase_word_prob": 0.01,
        "merge_word_prob": 0.05,
        "split_word_prob": 0.01,
        "typo_char_prob": 0.01,
    },
    "spliced": {
        "fraction": 0.10,
        "pool": "main",
        "min_sentences": 3,
        "max_sentences": 5,
        "strip_punct_prob": 0.20,
        "accent_strip_prob": 0.02,
        "random_letter_prob": 0.02,
        "random_digit_prob": 0.01,
        "foreign_sentence_prob": 1.0,
        "splice_strip_next_punct_prob": 1.0,
        "splice_lowercase_next_prob": 1.0,
        "format_noise_prob": 0.15,
        "paragraph_break_prob": 0.10,
        "uppercase_word_prob": 0.02,
        "lowercase_word_prob": 0.02,
        "titlecase_word_prob": 0.01,
        "merge_word_prob": 0.01,
        "split_word_prob": 0.01,
        "typo_char_prob": 0.01,
    },
    "mixed": {
        "fraction": 0.10,
        "pool": "main",
        "min_segments": 2,
        "max_segments": 4,
        "strip_punct_prob": 0.25,
        "swap_prob": 0.06,
        "o_inject_prob": 0.06,
        "allow_repeated_langs": True,
    },
}

SMOL = {
    "use": True,
    "rebuild": False,
}

FT = {
    "use": True,
    "rebuild": False,
    "max_lang": 50_000,
    "overflow_lang": 55_000,
    "max_row": 50_000,
    "miss": 1_000,
    "include_en": True, # Langs are those that will create an english parquet. Do not add any more to this list.
    "langs": {"en", "es", "fr", "pt", "it", "nl", "de", "sv", "da"},
    "cap_multipliers": {
        "no": 1.5,
        "sco": 0.5,
    },
}
FT["every"] = len(FT["langs"])

PURE_DOC_FRACTION = DOC_MIX["pure"]["fraction"]

POOL = {
    "wiki": {
        "reserve": PURE_DOC_FRACTION,
        "min": 4,
        "max": int(WIKI["max_wiki_sentences"] * PURE_DOC_FRACTION),
    },
    "smol": {
        "reserve": 0.95,
        "min": 1,
        "max": 1_000,
    },
    "ft": {
        "reserve": PURE_DOC_FRACTION,
        "min": 1,
        "max": int(FT["max_row"] * PURE_DOC_FRACTION),
    },
    "tatoeba": {
        "reserve": PURE_DOC_FRACTION,
        "min": 1,
        "max": int(PURE_DOC_FRACTION * TATOEBA["max_sentences"]),
    },
}

RUN = {
    "len": 512,
    "target": 5_000_000, 
    "syn_cache": True,
    "syn_rebuild": False,
    "tok_cache": True,
    "tok_rebuild": False,
    "tok_skip_check": False,
    "retry": 8,
    "preview": 10_000,
}
