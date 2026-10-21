import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.absa import aspect_label, score_aspects
from src.config import (
    COL_C1_SCHEME, COL_C3_ASPECTS, DATA_FILE, MASTER_FILE, MASTER_SHEET, OUTPUT_DIR, SHEET_NAME, TEXT_COLS,
)
from src.crosstabs import all_demographic_crosstabs
from src.data_quality import apply_quality_flags, impute_numeric_fields
from src.excel_style import AUTHOR
from src.preprocess import clean_text_columns, is_nil_response, normalize_scheme_name
from src.sentiment_lexicon import build_analyzer, label_from_score, score_column, score_text
from src.triangulate import (
    add_aspect_divergence, add_likert_numeric_columns, add_overall_divergence, divergence_summary,
)

FEEDBACK_COLS = ["C4_positive_clean", "C5_problems_clean", "C6_suggestions_clean"]


def run_vader_sentiment(df, analyzer):
    df = df.copy()
    score_cols = []
    for key in TEXT_COLS:
        scores, labels = score_column(analyzer, df[f"{key}_clean"])
        df[f"{key}_vader_score"] = scores
        df[f"{key}_vader_label"] = labels
        score_cols.append(f"{key}_vader_score")

    df["overall_vader_score"] = df[score_cols].mean(axis=1, skipna=True)
    df["overall_vader_label"] = df["overall_vader_score"].apply(label_from_score)
    return df


def combine_feedback(row):
    return ". ".join(v for v in row if pd.notna(v) and not is_nil_response(v))


def run_absa(df, analyzer):
    df = df.copy()
    combined = df[FEEDBACK_COLS].apply(combine_feedback, axis=1)
    df["scheme_feedback_text"] = combined.where(combined != "", None)

    aspect_scores = df["scheme_feedback_text"].apply(lambda t: score_aspects(t, analyzer, score_text))
    for aspect in COL_C3_ASPECTS:
        df[f"absa_{aspect}_score"] = aspect_scores.apply(lambda scores: scores.get(aspect))
        df[f"absa_{aspect}_label"] = df[f"absa_{aspect}_score"].apply(aspect_label)
    return df


def run_transformer_sentiment(df, analyzer):
    from src.sentiment_transformer import score_texts

    df = df.copy()
    df["transformer_score"], df["transformer_label"] = score_texts(df["scheme_feedback_text"].tolist())
    df["vader_on_feedback_score"], df["vader_on_feedback_label"] = score_column(analyzer, df["scheme_feedback_text"])
    return df


def method_agreement(df):
    both = df[df["transformer_label"].notna() & df["vader_on_feedback_label"].notna()]
    n_agree = int((both["transformer_label"] == both["vader_on_feedback_label"]).sum())
    summary = pd.DataFrame([{
        "n_comparable": len(both),
        "n_agree": n_agree,
        "pct_agree": round(n_agree / len(both) * 100, 1) if len(both) else float("nan"),
    }])
    confusion = pd.crosstab(
        both["vader_on_feedback_label"], both["transformer_label"], rownames=["VADER"], colnames=["Transformer"],
    )
    return summary, confusion


def save_master(df, divergence, crosstabs, agreement, confusion):
    with pd.ExcelWriter(MASTER_FILE, engine="openpyxl") as writer:
        writer.book.properties.creator = AUTHOR
        df.to_excel(writer, sheet_name=MASTER_SHEET, index=False)
        divergence.to_excel(writer, sheet_name="Divergence Summary", index=False)
        for name, table in crosstabs.items():
            if not table.empty:
                table.to_excel(writer, sheet_name=f"Crosstab {name}"[:31])
        if agreement is not None:
            agreement.to_excel(writer, sheet_name="Method Agreement", index=False)
            confusion.to_excel(writer, sheet_name="Method Confusion Matrix")


def save_summaries(df, divergence):
    divergence.to_csv(OUTPUT_DIR / "divergence_summary.csv", index=False)

    label_cols = [f"{k}_vader_label" for k in TEXT_COLS] + ["overall_vader_label"]
    distribution = [
        {"column": col, "label": label, "count": int(count)}
        for col in label_cols for label, count in df[col].value_counts().items()
    ]
    pd.DataFrame(distribution).to_csv(OUTPUT_DIR / "sentiment_distribution.csv", index=False)
    df["scheme_normalized"].value_counts().to_csv(OUTPUT_DIR / "scheme_name_normalization_counts.csv")


def main(skip_transformer):
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    df = impute_numeric_fields(apply_quality_flags(df))
    df = clean_text_columns(df, TEXT_COLS)
    df["scheme_normalized"] = df[COL_C1_SCHEME].apply(normalize_scheme_name)

    analyzer = build_analyzer()
    print("Scoring VADER sentiment and aspects")
    df = run_absa(run_vader_sentiment(df, analyzer), analyzer)

    agreement = confusion = None
    if not skip_transformer:
        print("Scoring transformer sentiment")
        df = run_transformer_sentiment(df, analyzer)
        agreement, confusion = method_agreement(df)
        print(agreement.to_string(index=False))
    else:
        df["vader_on_feedback_score"], df["vader_on_feedback_label"] = score_column(analyzer, df["scheme_feedback_text"])

    df = add_likert_numeric_columns(df)
    df = add_overall_divergence(df, "overall_vader_label")
    df = add_aspect_divergence(df, {a: f"absa_{a}_label" for a in COL_C3_ASPECTS})

    divergence_cols = ["divergence_overall"] + [f"divergence_{a}" for a in COL_C3_ASPECTS]
    divergence = divergence_summary(df, divergence_cols)
    print(divergence.to_string(index=False))

    save_master(df, divergence, all_demographic_crosstabs(df, "overall_vader_label"), agreement, confusion)
    save_summaries(df, divergence)
    print(f"Saved {MASTER_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transformer", action="store_true")
    main(parser.parse_args().skip_transformer)
