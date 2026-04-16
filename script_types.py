from __future__ import annotations

from enum import StrEnum


class Script(StrEnum):
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    ARABIC = "arabic"
    DEVANAGARI = "devanagari"
    BENGALI = "bengali"
    JAPANESE = "japanese"
    HAN = "han"
    HANGUL = "hangul"
    MIXED_WEST = "mixed_west"
    MIXED_EAST = "mixed_east"
    INDIC_OTHER = "indic_other"
