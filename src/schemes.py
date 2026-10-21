import re

import pandas as pd

SCHEME_CATALOGUE = [
    ("PM-KISAN", "Central", "Income support", r"pm[\s-]*kisan"),
    ("Rythu Bharosa", "State", "Investment support", r"rythu\s*bharosa"),
    ("Kisan Credit Card", "Central", "Farm credit", r"kisan credit card|\bkcc\b"),
    ("Crop Insurance", "Central", "Crop insurance", r"fasal bima|crop insurance"),
    ("Input Subsidy", "State", "Seed & fertiliser subsidy", r"input subsidy"),
    ("Minimum Support Price", "Central", "Price support", r"minimum support price|\bmsp\b"),
    ("Soil Health Card", "Central", "Soil testing", r"soil health card"),
    ("PM Krishi Sinchai Yojana", "Central", "Irrigation", r"krishi sinchai|irrigation"),
    ("e-NAM", "Central", "Market access", r"\be-?nam\b|national agricultural market"),
    ("Other / Unmatched", None, None, None),
    ("Unspecified", None, None, None),
]

_PATTERNS = [(name, re.compile(pattern, re.I)) for name, _, _, pattern in SCHEME_CATALOGUE if pattern]


def schemes_in(answer):
    if pd.isna(answer):
        return set()
    text = str(answer)
    return {name for name, pattern in _PATTERNS if pattern.search(text)}
