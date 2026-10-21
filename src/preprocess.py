import re

import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from src.config import CANONICAL_SCHEMES

NULL_TOKENS = {"nan", "none", "", "na", "n/a", "-"}

SCHEME_ABBREVIATIONS = {
    "kcc": "Kisan Credit Card",
    "msp": "Minimum Support Price",
    "minimun selling prices": "Minimum Support Price",
    "minimum selling price": "Minimum Support Price",
    "pmfby": "Crop Insurance",
    "e-nam": "e-NAM",
    "enam": "e-NAM",
    "national agricultural market": "e-NAM",
    "irrigation schemes": "PM Krishi Sinchai Yojana",
    "pmksy": "PM Krishi Sinchai Yojana",
    "pradhan mantri krishi sinchai yojana": "PM Krishi Sinchai Yojana",
}

NIL_RESPONSE_TOKENS = {
    "no", "none", "nothing", "nil", "nill", "n/a", "na", "nope",
    "no issues", "no issue", "no problem", "no problems", "no complaints",
    "no suggestions", "no major issues faced", "none significant", "nothing present",
}


def clean_text(value):
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if text.lower() in NULL_TOKENS:
        return None
    return text


def is_nil_response(text):
    if pd.isna(text):
        return False
    return text.strip().lower().rstrip(".") in NIL_RESPONSE_TOKENS


def normalize_scheme_name(raw_name, choices=CANONICAL_SCHEMES, score_cutoff=70):
    text = clean_text(raw_name)
    if text is None or is_nil_response(text):
        return None

    first_segment = re.split(r"[,/&]| and ", text)[0].strip() or text

    abbreviation = SCHEME_ABBREVIATIONS.get(first_segment.lower())
    if abbreviation:
        return abbreviation

    match = process.extractOne(
        first_segment, choices, scorer=fuzz.WRatio,
        processor=default_process, score_cutoff=score_cutoff,
    )
    return "Other / Unmatched" if match is None else match[0]


def clean_text_columns(df, text_col_map):
    df = df.copy()
    for key, col in text_col_map.items():
        df[f"{key}_clean"] = df[col].apply(clean_text)
    return df
