from __future__ import annotations


LANGUAGE_GROUPS = {
    "English":      ["en"],
    "LatinCore":    ["es", "fr", "de", "it", "pt", "nl"],
    "LatinTier2":   ["vi", "tr", "la", "id", "ms", "af", "sq", "is", "no", "sv", "da", "fi", "hu", "pl", "cs", "ro"],
    "Cyrillic":     ["ru", "bg", "uk", "sr", "be", "kk", "mk", "mn"],
    "EastAsian":    ["zh", "ja", "ko"],
    "Indic":        ["hi", "ur", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "as", "or"],
    "ArabicScript": ["ar", "fa", "ps", "sd", "ug"],
    "OtherScripts": ["el", "he", "hy", "ka", "am", "km", "lo", "my", "th"],
}

ALL_LANGS = [lang for langs in LANGUAGE_GROUPS.values() for lang in langs]
LANG_TO_GROUP = {lang: group for group, langs in LANGUAGE_GROUPS.items() for lang in langs}
