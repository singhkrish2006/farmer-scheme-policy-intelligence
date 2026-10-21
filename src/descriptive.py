from collections import Counter

import pandas as pd

from src.config import (
    ASPECT_LABELS, COL_A2_AGE, COL_A7_LANDHOLDING, COL_A9_CROPS, COL_B1_AWARE, COL_B2_HEARD, COL_B3_ENROLLED,
    COL_C3_ASPECTS,
)
from src.crosstabs import landholding_bucket
from src.schemes import schemes_in

AGE_BINS = [0, 30, 40, 50, 60, 70, 100]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]


def pct_table(series):
    counts = series.dropna().value_counts()
    total = counts.sum()
    return pd.DataFrame({
        "Category": counts.index.astype(str),
        "Count": counts.values.astype(int),
        "Percent": (counts.values / total * 100).round(1),
    })


def numeric_summary(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(values.median())}


def age_bands(df):
    return pd.cut(pd.to_numeric(df[COL_A2_AGE], errors="coerce"), bins=AGE_BINS, labels=AGE_LABELS)


def landholding_bands(df):
    return df[COL_A7_LANDHOLDING].apply(landholding_bucket)


def crop_counts(df, top_n=15):
    counter = Counter()
    for entry in df[COL_A9_CROPS].dropna():
        for crop in str(entry).split(","):
            crop = crop.strip().title()
            if crop:
                counter[crop] += 1
    n = len(df)
    return pd.DataFrame(
        [{"Category": crop, "Count": count, "Percent": round(count / n * 100, 1)} for crop, count in counter.most_common(top_n)]
    )


def awareness_gap(df):
    heard = df[COL_B2_HEARD].apply(schemes_in).apply(len)
    enrolled = df[COL_B3_ENROLLED].apply(schemes_in).apply(len)
    usable = (heard > 0) | (enrolled > 0)
    gap = usable & (heard > enrolled)
    return {
        "n_usable": int(usable.sum()),
        "n_gap": int(gap.sum()),
        "pct_gap": round(gap.sum() / usable.sum() * 100, 1),
        "avg_heard": round(heard[usable].mean(), 2),
        "avg_enrolled": round(enrolled[usable].mean(), 2),
    }


def awareness_table(df):
    return pct_table(df[COL_B1_AWARE].replace({"No (If No, skip to Section D)": "No"}))


def label_distribution(series):
    table = pct_table(series)
    order = {label: i for i, label in enumerate(SENTIMENT_ORDER)}
    return table.assign(_o=table["Category"].map(order)).sort_values("_o").drop(columns="_o").reset_index(drop=True)


def aspect_distribution(df):
    result = {}
    for aspect in COL_C3_ASPECTS:
        labels = df[f"absa_{aspect}_label"].dropna()
        result[aspect] = {"n": int(len(labels)), "table": pct_table(labels)}
    return result


def scheme_table(df):
    with_scheme = df[df["scheme_normalized"].notna()]
    total = len(with_scheme)
    rows = []
    for scheme, group in with_scheme.groupby("scheme_normalized"):
        sentiment = group["overall_vader_label"].value_counts()
        scored = sentiment.sum()
        rows.append({
            "Scheme": scheme,
            "N": len(group),
            "pct_of_feedback": round(len(group) / total * 100, 1),
            "pct_positive": round(sentiment.get("Positive", 0) / scored * 100, 1),
            "pct_neutral": round(sentiment.get("Neutral", 0) / scored * 100, 1),
            "pct_negative": round(sentiment.get("Negative", 0) / scored * 100, 1),
            "avg_satisfaction": round(pd.to_numeric(group["C2_satisfaction_numeric"], errors="coerce").mean(), 2),
            "avg_impact": round(pd.to_numeric(group["C3_impact_numeric"], errors="coerce").mean(), 2),
        })
    return pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)


def absa_negative_by_scheme(df, min_n=30):
    counts = df["scheme_normalized"].value_counts()
    rows = []
    for scheme in counts[counts >= min_n].index:
        group = df[df["scheme_normalized"] == scheme]
        row = {"Scheme": scheme, "N": len(group)}
        for aspect in COL_C3_ASPECTS:
            labels = group[f"absa_{aspect}_label"].dropna()
            row[f"{aspect}_pct_negative"] = round((labels == "Negative").sum() / len(labels) * 100, 1) if len(labels) else float("nan")
            row[f"{aspect}_n_mentions"] = int(len(labels))
        rows.append(row)
    return pd.DataFrame(rows)


def sentiment_by_group(df, group_col, group_values=None):
    values = group_values if group_values is not None else df[group_col]
    frame = pd.DataFrame({"group": values, "label": df["overall_vader_label"]}).dropna()
    table = pd.crosstab(frame["group"], frame["label"], normalize="index").mul(100).round(1)
    for label in SENTIMENT_ORDER:
        if label not in table.columns:
            table[label] = 0.0
    table["n"] = frame.groupby("group").size()
    return table[SENTIMENT_ORDER + ["n"]].sort_values("Negative", ascending=False)


def mean_ratings(df):
    rows = []
    for aspect in COL_C3_ASPECTS:
        values = pd.to_numeric(df[f"C3_{aspect}_numeric"], errors="coerce")
        rows.append({"Aspect": ASPECT_LABELS[aspect], "N": int(values.notna().sum()), "Avg rating": round(values.mean(), 2)})
    return pd.DataFrame(rows).sort_values("Avg rating", ascending=False).reset_index(drop=True)
