import pandas as pd

from src.config import COL_A5_DISTRICT, COL_A6_EDUCATION, COL_A7_LANDHOLDING, COL_B4_SOURCE


def landholding_bucket(acres):
    if pd.isna(acres):
        return None
    try:
        acres = float(acres)
    except (TypeError, ValueError):
        return None
    if acres < 2.5:
        return "Marginal (<2.5 ac)"
    if acres < 5:
        return "Small (2.5-5 ac)"
    if acres < 10:
        return "Semi-medium (5-10 ac)"
    return "Large (10+ ac)"


def sentiment_crosstab(df, sentiment_col, group_col):
    valid = df[[sentiment_col, group_col]].dropna()
    if valid.empty:
        return pd.DataFrame()
    table = pd.crosstab(valid[group_col], valid[sentiment_col], normalize="index").mul(100).round(1)
    return table.join(valid.groupby(group_col).size().rename("n"))


def all_demographic_crosstabs(df, sentiment_col):
    df = df.copy()
    df["landholding_bucket"] = df[COL_A7_LANDHOLDING].apply(landholding_bucket)
    group_cols = {
        "district": COL_A5_DISTRICT,
        "education": COL_A6_EDUCATION,
        "landholding": "landholding_bucket",
        "info_source": COL_B4_SOURCE,
    }
    return {name: sentiment_crosstab(df, sentiment_col, col) for name, col in group_cols.items()}
