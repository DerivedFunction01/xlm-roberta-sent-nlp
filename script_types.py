from __future__ import annotations

from enum import StrEnum


class Script(StrEnum):
    LATIN = "Latn"
    CYRILLIC = "Cyrl"
    ARABIC = "Arab"
    HEBREW = "Hebr"
    DEVANAGARI = "Deva"
    BENGALI = "Beng"
    GREEK = "Grek"
    ARMENIAN = "Armn"
    GEORGIAN = "Geor"
    ETHIOPIC = "Ethi"
    THAANA = "Thaa"
    HIRAGANA = "Hira"
    KATAKANA = "Kana"
    JAPANESE = "Jpan"
    HAN = "Hani"
    HANGUL = "Hang"
    KHMER = "Khmr"
    LAO = "Laoo"
    MYANMAR = "Mymr"
    THAI = "Thai"
    SINHALA = "Sinh"
    TIBETAN = "Tibt"
    TAMIL = "Taml"
    TELUGU = "Telu"
    GUJARATI = "Gujr"
    KANNADA = "Knda"
    MALAYALAM = "Mlym"
    GURMUKHI = "Guru"
    ORIYA = "Orya"
    MIXED_WEST = "mixed_west"
    MIXED_EAST = "mixed_east"
    INDIC_OTHER = "indic_other"


SCRIPT_NAME_MARKERS: dict[Script, tuple[str, ...]] = {
    Script.LATIN: (Script.LATIN,),
    Script.CYRILLIC: (Script.CYRILLIC,),
    Script.ARABIC: (Script.ARABIC,),
    Script.HEBREW: (Script.HEBREW,),
    Script.DEVANAGARI: (Script.DEVANAGARI,),
    Script.BENGALI: (Script.BENGALI,),
    Script.JAPANESE: (Script.HIRAGANA, Script.KATAKANA, "CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"),
    Script.HAN: ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"),
    Script.HANGUL: (Script.HANGUL,),
}

LANGUAGE_SCRIPT_MARKER_OVERRIDES: dict[str, frozenset[str]] = {
    "el": frozenset({Script.GREEK}),
    "hy": frozenset({Script.ARMENIAN}),
    "ka": frozenset({Script.GEORGIAN}),
    "am": frozenset({Script.ETHIOPIC}),
    "ti": frozenset({Script.ETHIOPIC}),
    "dv": frozenset({Script.THAANA}),
    "km": frozenset({Script.KHMER}),
    "lo": frozenset({Script.LAO}),
    "my": frozenset({Script.MYANMAR}),
    "th": frozenset({Script.THAI}),
    "si": frozenset({Script.SINHALA}),
    "bo": frozenset({Script.TIBETAN}),
    "ta": frozenset({Script.TAMIL}),
    "te": frozenset({Script.TELUGU}),
    "gu": frozenset({Script.GUJARATI}),
    "kn": frozenset({Script.KANNADA}),
    "ml": frozenset({Script.MALAYALAM}),
    "pa": frozenset({Script.GURMUKHI}),
    "or": frozenset({Script.ORIYA, "ODIA"}),
}
