import numpy as np
import pandas as pd

from src.config import (
    COL_A2_AGE, COL_A7_LANDHOLDING, COL_A13_EXPERIENCE, COL_B1_AWARE, COL_B2_HEARD, COL_B3_ENROLLED,
    COL_C1_SCHEME, COL_E4_DURATION,
)
from src.preprocess import is_nil_response
from src.schemes import schemes_in


def _numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def _excel_row(index):
    return int(index) + 2


def _claims_unaware_but_named_scheme(df):
    said_no = df[COL_B1_AWARE].astype(str).str.strip().str.lower().str.startswith("no")
    named_scheme = df[COL_C1_SCHEME].notna() & ~df[COL_C1_SCHEME].apply(is_nil_response)
    return said_no & named_scheme


def numeric_plausibility_checks(df):
    age = _numeric(df, COL_A2_AGE)
    experience = _numeric(df, COL_A13_EXPERIENCE)
    land = _numeric(df, COL_A7_LANDHOLDING)
    duration = _numeric(df, COL_E4_DURATION)

    issues = []

    def add(index, field, value, issue):
        issues.append({"source_excel_row": _excel_row(index), "field": field, "value": value, "issue": issue})

    for i in df.index[age.notna() & ((age < 15) | (age > 100))]:
        add(i, "A2 Age", age[i], "Age outside plausible adult range (15-100)")

    both = age.notna() & experience.notna()
    for i in df.index[both & ((age - experience) < 10)]:
        add(i, "A2 Age vs A13 Experience", f"age={age[i]}, experience={experience[i]}",
            "Age minus farming experience < 10 years (implies starting farming before age 10)")

    for i in df.index[land.notna() & (land < 0)]:
        add(i, "A7 Landholding", land[i], "Negative landholding")

    for i in df.index[experience.notna() & (experience < 0)]:
        add(i, "A13 Farming experience", experience[i], "Negative experience")

    for i in df.index[duration.notna() & (duration <= 0)]:
        add(i, "E4 Duration", duration[i], "Zero or negative interview duration")

    return pd.DataFrame(issues, columns=["source_excel_row", "field", "value", "issue"])


def _enrolled_not_heard(df):
    heard = df[COL_B2_HEARD].apply(schemes_in)
    enrolled = df[COL_B3_ENROLLED].apply(schemes_in)
    return pd.Series([e - h for h, e in zip(heard, enrolled)], index=df.index)


def logical_consistency_checks(df):
    b1 = df[COL_B1_AWARE]
    c1 = df[COL_C1_SCHEME]
    rows = [
        {
            "source_excel_row": _excel_row(i),
            "field": "B1 vs C1",
            "value": f"B1={b1[i]!r}, C1={c1[i]!r}",
            "issue": 'Respondent said "not aware" but gave scheme-specific feedback',
        }
        for i in df.index[_claims_unaware_but_named_scheme(df)]
    ]
    missing = _enrolled_not_heard(df)
    rows += [
        {
            "source_excel_row": _excel_row(i),
            "field": "B2 vs B3",
            "value": "enrolled but not listed as heard: " + ", ".join(sorted(schemes)),
            "issue": "Scheme listed as enrolled (B3) but not as heard of (B2)",
        }
        for i, schemes in missing.items() if schemes
    ]
    return pd.DataFrame(rows, columns=["source_excel_row", "field", "value", "issue"])


def apply_quality_flags(df):
    df = df.copy()
    age = _numeric(df, COL_A2_AGE)
    experience = _numeric(df, COL_A13_EXPERIENCE)
    impossible = age.notna() & experience.notna() & ((age - experience) < 0)

    df["age_numeric_clean"] = age.mask(impossible)
    df["farming_experience_clean"] = experience.mask(impossible)
    df["age_experience_flag"] = np.where(impossible, "impossible: experience >= age", None)
    df["b1_c1_consistency_flag"] = np.where(
        _claims_unaware_but_named_scheme(df),
        "inconsistent: claimed unaware (B1=No) but named a scheme (C1)", None,
    )
    return df


def impute_numeric_fields(df):
    df = df.copy()
    age = df["age_numeric_clean"]
    df["age_was_imputed"] = age.isna()
    df["age_imputed"] = age.fillna(round(age.mean(), 2))
    return df
