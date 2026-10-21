import pandas as pd

from src.config import COL_C2_SATISFACTION, COL_C3_ASPECTS, LIKERT_5PT_MAP, SATISFACTION_MAP


def likert_to_numeric(value, mapping):
    if pd.isna(value):
        return None
    return mapping.get(str(value).strip().lower())


def likert_polarity(numeric_value):
    if pd.isna(numeric_value):
        return None
    if numeric_value > 0:
        return "Positive"
    if numeric_value < 0:
        return "Negative"
    return "Neutral"


def add_likert_numeric_columns(df):
    df = df.copy()
    df["C2_satisfaction_numeric"] = df[COL_C2_SATISFACTION].apply(lambda v: likert_to_numeric(v, SATISFACTION_MAP))
    df["C2_satisfaction_polarity"] = df["C2_satisfaction_numeric"].apply(likert_polarity)

    for key, col in COL_C3_ASPECTS.items():
        df[f"C3_{key}_numeric"] = df[col].apply(lambda v: likert_to_numeric(v, LIKERT_5PT_MAP))
        df[f"C3_{key}_polarity"] = df[f"C3_{key}_numeric"].apply(likert_polarity)
    return df


def flag_divergence(nlp_label, rating_polarity):
    if pd.isna(nlp_label) or pd.isna(rating_polarity):
        return None
    return {nlp_label, rating_polarity} == {"Positive", "Negative"}


def add_overall_divergence(df, overall_label_col):
    df = df.copy()
    df["divergence_overall"] = [
        flag_divergence(nlp, rating)
        for nlp, rating in zip(df[overall_label_col], df["C2_satisfaction_polarity"])
    ]
    return df


def add_aspect_divergence(df, aspect_label_cols):
    df = df.copy()
    for aspect, nlp_col in aspect_label_cols.items():
        df[f"divergence_{aspect}"] = [
            flag_divergence(nlp, rating)
            for nlp, rating in zip(df[nlp_col], df[f"C3_{aspect}_polarity"])
        ]
    return df


def divergence_summary(df, divergence_cols):
    rows = []
    for col in divergence_cols:
        valid = df[col].dropna()
        n_valid = len(valid)
        n_divergent = int(valid.sum()) if n_valid else 0
        rows.append({
            "comparison": col,
            "n_comparable": n_valid,
            "n_divergent": n_divergent,
            "pct_divergent": round(n_divergent / n_valid * 100, 1) if n_valid else float("nan"),
        })
    return pd.DataFrame(rows)
