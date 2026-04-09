from __future__ import annotations

from instruction_dataset_sources import DEFAULT_INSTRUCTION_SOURCE_SPECS

LANGUAGE_BUCKETS = {
    # ~41% of CC — intentionally capped to avoid crowding out other languages
    "English": {
        "langs": ["en"],
        "weight": 2.5,
        "min_chars": 2_000,
        "latin": True,
    },
    # ~6.3% of CC — was badly underweighted relative to German/French
    "Russian": {
        "langs": ["ru"],
        "weight": 1.8,
        "min_chars": 2_000,
        "latin": False,
    },
    # ~5.9% of CC
    "German": {
        "langs": ["de"],
        "weight": 1.8,
        "min_chars": 2_000,
        "latin": True,
    },
    # ~5.7% of CC — bumped up from 1.7 to match its actual footprint
    "Japanese": {
        "langs": ["ja"],
        "weight": 1.8,
        "min_chars": 1_200,
        "latin": False,
    },
    # ~5.0% of CC — CC likely undercounts due to Great Firewall
    "Chinese": {
        "langs": ["zh"],
        "weight": 1.8,
        "min_chars": 1_200,
        "latin": False,
    },
    # ~4.6% of CC
    "French": {
        "langs": ["fr"],
        "weight": 1.8,
        "min_chars": 2_000,
        "latin": True,
    },
    # ~4.6% of CC
    "Spanish": {
        "langs": ["es"],
        "weight": 1.8,
        "min_chars": 2_000,
        "latin": True,
    },
    # ~2.5% of CC
    "Portuguese": {
        "langs": ["pt"],
        "weight": 1.6,
        "min_chars": 2_000,
        "latin": True,
    },
    # ~2.4% of CC
    "Italian": {
        "langs": ["it"],
        "weight": 1.5,
        "min_chars": 2_000,
        "latin": True,
    },
    # ~2.0% of CC — split out from CentralEuropeanLatin; rivals Italian/Portuguese
    "Polish": {
        "langs": ["pl"],
        "weight": 1.5,
        "min_chars": 2_000,
        "latin": True,
    },
    # ~1.8% of CC — was significantly underweighted at 1.15
    "Dutch": {
        "langs": ["nl"],
        "weight": 1.5,
        "min_chars": 2_000,
        "latin": True,
    },
    # ~1.2% of CC — split out from CentralEuropeanLatin; large internet population
    "Turkish": {
        "langs": ["tr"],
        "weight": 1.4,
        "min_chars": 2_000,
        "latin": True,
    },
    # ind ~1.1%, vie ~1.05% of CC
    "SoutheastAsianLatin": {
        "langs": ["vi", "id", "ms", "sq", "la"],
        "weight": 1.4,
        "min_chars": 2_000,
        "latin": True,
    },
    # ces ~1.14%, ron ~0.53%, hun ~0.52% of CC — smaller tier after splitting out pl/tr
    "CentralEuropeanLatin": {
        "langs": ["cs", "ro", "hu"],
        "weight": 1.2,
        "min_chars": 2_000,
        "latin": True,
    },
    # ~0.81% of CC — was overweighted at 1.7
    "Korean": {
        "langs": ["ko"],
        "weight": 1.3,
        "min_chars": 1_200,
        "latin": False,
    },
    # ukr ~0.70%, bel ~0.017% of CC
    "EastSlavicCyrillic": {
        "langs": ["uk", "be"],
        "weight": 1.15,
        "min_chars": 2_000,
        "latin": False,
    },
    # ~0.65% of CC — upweighted relative to CC share given speaker population
    "Arabic": {
        "langs": ["ar"],
        "weight": 1.35,
        "min_chars": 2_000,
        "latin": False,
    },
    # sv ~0.7%, dan ~0.51%, nor+nno ~0.33%, fin ~0.37%, isl ~0.04%, afr ~0.01%
    # combined ~2.0% of CC — was drastically overweighted at 6.0
    # note: Swedish Wikipedia is heavily bot-generated stubs, don't rely on article count
    "NordicCore": {
        "langs": ["sv", "da", "no", "is", "af", "fi"],
        "weight": 1.8,
        "min_chars": 2_000,
        "latin": True,
    },
    # bul ~0.27%, srp ~0.25%, mkd ~0.037% of CC
    "BalkanCyrillic": {
        "langs": ["bg", "sr", "mk"],
        "weight": 1.0,
        "min_chars": 2_000,
        "latin": False,
    },
    # fas ~0.20% of CC (ignore the one anomalous crawl spike)
    "ArabicOther": {
        "langs": ["fa", "ps", "sd", "ug"],
        "weight": 0.9,
        "min_chars": 2_000,
        "latin": False,
    },
    # ~0.22% of CC — genuine web underrepresentation relative to speaker count,
    # but corpus is thin; 1.0 avoids oversampling a small pool
    "Hindi": {
        "langs": ["hi"],
        "weight": 1.0,
        "min_chars": 2_000,
        "latin": False,
    },
    # combined ~0.27% of CC — upweighted for script diversity
    "IndicOther": {
        "langs": ["ur", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "as", "or"],
        "weight": 0.9,
        "min_chars": 2_000,
        "latin": False,
    },
    # kk ~0.038%, mn ~0.016% of CC — very thin corpus, weight is already a large relative boost
    "CentralAsianCyrillic": {
        "langs": ["kk", "mn"],
        "weight": 0.9,
        "min_chars": 2_000,
        "latin": False,
    },
    "AfricanLatin": {
        "langs": ["sw", "tl", "eu"],
        "weight": 0.8,
        "min_chars": 1_500,
        "latin": True,
    },
    # el ~0.55%, he ~0.24%, th ~0.38%, hy ~0.033%, ka ~0.044% etc. — combined ~1%+
    # nudged up slightly from 0.8 given Greek and Thai have meaningful CC presence
    "OtherScripts": {
        "langs": ["el", "he", "hy", "ka", "am", "km", "lo", "my", "th", "si", "bo", "ti", "dv"],
        "weight": 0.9,
        "min_chars": 2_000,
        "latin": False,
    },
}

POOL = {
    "wiki": {
        "reserve": 0.60,
        "min": 4,
        "max": 120_000,
    },
    "instruct": {
        "reserve": 0.60,
        "min": 1,
        "max": 120_000,
    },
    "smol": {
        "reserve": 0.95,
        "min": 1,
        "max": 1_000,
    },
    "ft": {
        "reserve": 0.60,
        "min": 1,
        "max": 30_000,
    },
}

DOC_MIX = {
    "pure": {
        "fraction": 0.60,
        "pool": "reserve",
        "min_sentences": 1,
        "max_sentences": 4,
        "strip_punct_prob": 0.10,
        "format_noise_prob": 0.30,
    },
    "homogeneous": {
        "fraction": 0.30,
        "pool": "main",
        "min_sentences": 2,
        "max_sentences": 6,
        "strip_punct_prob": 0.15,
        "format_noise_prob": 0.20,
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

INSTRUCT = {
    "use": True,
    "rebuild": False,
    "max_lang": 50_000,
    "overflow_lang": 75_000,
    "sources": DEFAULT_INSTRUCTION_SOURCE_SPECS,
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
    "lang_overrides": {
        "zh": "yue",
    },
}
FT["every"] = len(FT["langs"])

RUN = {
    "len": 512,
    "target": 3_000_000,  # synthetic mixed-language training examples to generate
    "syn_cache": True,
    "syn_rebuild": False,
    "tok_cache": True,
    "tok_rebuild": False,
    "tok_skip_check": False,
    "retry": 8,
    "preview": 2_000,
}
