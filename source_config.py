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

POOL = {
    "wiki": {
        "reserve": 0.15,
        "min": 4,
        "max": 20_000,
    },
    "smol": {
        "reserve": 0.95,
        "min": 1,
        "max": 20_000,
    },
    "ft": {
        "reserve": 0.15,
        "min": 1,
        "max": 20_000,
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
    "overflow_lang": 75_000,
    "max_row": 50_000,
    "miss": 1_000,
    "include_en": True,
    "langs": {"en", "es", "fr", "pt", "it", "nl", "de", "sv", "da", "id", "ms"},
}
FT["every"] = len(FT["langs"])

RUN = {
    "len": 512,
    "target": 2_500_000,  # synthetic mixed-language training examples to generate
    "cov_min": 2,
    "cov_max": 5,
    "syn_cache": True,
    "syn_rebuild": False,
    "tok_cache": True,
    "tok_rebuild": False,
    "tok_skip_check": False,
    "retry": 8,
    "preview": 2_000,
}
