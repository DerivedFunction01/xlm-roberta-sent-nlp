from __future__ import annotations

LANGUAGE_BUCKETS = {
    "English": {
        "langs": ["en"],
        "weight": 2.5,
        "min_chars": 2_000,
        "latin": True,
    },
    "Spanish": {
        "langs": ["es"],
        "weight": 1.8,
        "min_chars": 2_000,
        "latin": True,
    },
    "French": {
        "langs": ["fr"],
        "weight": 1.8,
        "min_chars": 2_000,
        "latin": True,
    },
    "German": {
        "langs": ["de"],
        "weight": 1.6,
        "min_chars": 2_000,
        "latin": True,
    },
    "Portuguese": {
        "langs": ["pt"],
        "weight": 1.4,
        "min_chars": 2_000,
        "latin": True,
    },
    "Italian": {
        "langs": ["it"],
        "weight": 1.3,
        "min_chars": 2_000,
        "latin": True,
    },
    "Dutch": {
        "langs": ["nl"],
        "weight": 1.15,
        "min_chars": 2_000,
        "latin": True,
    },
    "NordicCore": {
        "langs": ["sv", "da", "no", "is", "af"],
        "weight": 1.0,
        "min_chars": 2_000,
        "latin": True,
    },
    "CentralEuropeanLatin": {
        "langs": ["pl", "cs", "ro", "hu", "tr"],
        "weight": 1.0,
        "min_chars": 2_000,
        "latin": True,
    },
    "SoutheastAsianLatin": {
        "langs": ["vi", "id", "ms", "sq", "la"],
        "weight": 0.95,
        "min_chars": 2_000,
        "latin": True,
    },
    "Russian": {
        "langs": ["ru"],
        "weight": 1.45,
        "min_chars": 2_000,
        "latin": False,
    },
    "EastSlavicCyrillic": {
        "langs": ["uk", "be"],
        "weight": 1.15,
        "min_chars": 2_000,
        "latin": False,
    },
    "BalkanCyrillic": {
        "langs": ["bg", "sr", "mk"],
        "weight": 1.0,
        "min_chars": 2_000,
        "latin": False,
    },
    "CentralAsianCyrillic": {
        "langs": ["kk", "mn"],
        "weight": 0.9,
        "min_chars": 2_000,
        "latin": False,
    },
    "Chinese": {
        "langs": ["zh"],
        "weight": 1.6,
        "min_chars": 1_200,
        "latin": False,
    },
    "Japanese": {
        "langs": ["ja"],
        "weight": 1.35,
        "min_chars": 1_200,
        "latin": False,
    },
    "Korean": {
        "langs": ["ko"],
        "weight": 1.2,
        "min_chars": 1_200,
        "latin": False,
    },
    "Arabic": {
        "langs": ["ar"],
        "weight": 1.35,
        "min_chars": 2_000,
        "latin": False,
    },
    "ArabicOther": {
        "langs": ["fa", "ps", "sd", "ug"],
        "weight": 0.9,
        "min_chars": 2_000,
        "latin": False,
    },
    "Hindi": {
        "langs": ["hi"],
        "weight": 1.25,
        "min_chars": 2_000,
        "latin": False,
    },
    "IndicOther": {
        "langs": ["ur", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "as", "or"],
        "weight": 0.9,
        "min_chars": 2_000,
        "latin": False,
    },
    "OtherScripts": {
        "langs": ["el", "he", "hy", "ka", "am", "km", "lo", "my", "th"],
        "weight": 0.8,
        "min_chars": 2_000,
        "latin": False,
    },
}

RESERVE_FRACTION = 0.15
MIN_RESERVED_SENTENCES = 4
MAX_RESERVED_SENTENCES = 20_000
SMOL_RESERVE_FRACTION = 0.50
SMOL_MIN_RESERVED_SENTENCES = 1
SMOL_MAX_RESERVED_SENTENCES = MAX_RESERVED_SENTENCES
FT_RESERVE_FRACTION = 0.10
FT_MIN_RESERVED_SENTENCES = 1
FT_MAX_RESERVED_SENTENCES = MAX_RESERVED_SENTENCES
FT_MAX_SENTENCES_PER_LANG = 30_000
FT_OVERFLOW_SENTENCES_PER_LANG = 50_000
FT_MAX_ROW_INDEX = 50_000
FT_MAX_MISS_STREAK = 1_000
FT_INCLUDE_TRANSLATED_ENGLISH = False
FT_TRANSLATED_ENGLISH_LANGS = {"en", "es", "fr", "pt", "it", "nl", "de", "sv", "da", "id", "ms"}
FT_ENGLISH_ACCEPT_EVERY = len(FT_TRANSLATED_ENGLISH_LANGS)
USE_SMOL_AUGMENTATION = True
USE_FINETRANS_AUGMENTATION = True
SMOL_FORCE_REBUILD = False
FT_FORCE_REBUILD = False
