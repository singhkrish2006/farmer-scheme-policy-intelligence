import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import (
    ASPECT_LABELS, COL_A3_GENDER, COL_A4_VILLAGE, COL_A5_DISTRICT, COL_A6_EDUCATION, COL_A7_LANDHOLDING,
    COL_A8_LAND_TYPE, COL_A10_OCCUPATION, COL_A11_INCOME, COL_B1_AWARE, COL_B4_SOURCE, COL_C2_SATISFACTION,
    COL_C3_ASPECTS, COL_D1_GOVT_SUPPORT, COL_D2_TREND, COL_D3_CONFIDENCE, COL_E1_LANGUAGE, COL_E2_COMFORT,
    COL_E4_DURATION, MASTER_FILE, MASTER_SHEET, OUTPUT_DIR,
)
from src.crosstabs import landholding_bucket
from src.excel_style import save_workbook

YES_NO = {True: "Yes", False: "No"}


def build_responses(df):
    fact = pd.DataFrame({
        "RespondentID": df.index + 1,
        "SourceExcelRow": df.index + 2,
        "District": df[COL_A5_DISTRICT],
        "Village": df[COL_A4_VILLAGE],
        "Age": df["age_imputed"],
        "AgeWasImputed": df["age_was_imputed"],
        "Gender": df[COL_A3_GENDER],
        "Education": df[COL_A6_EDUCATION].replace({"diploma": "Diploma"}),
        "LandholdingAcres": pd.to_numeric(df[COL_A7_LANDHOLDING], errors="coerce"),
        "LandholdingBand": df[COL_A7_LANDHOLDING].apply(landholding_bucket),
        "LandType": df[COL_A8_LAND_TYPE],
        "Occupation": df[COL_A10_OCCUPATION],
        "IncomeBand": df[COL_A11_INCOME],
        "FarmingExperienceYears": df["farming_experience_clean"],
        "Aware": df[COL_B1_AWARE].replace({"No (If No, skip to Section D)": "No"}),
        "InfoSource": df[COL_B4_SOURCE],
        "SchemeName": df["scheme_normalized"],
        "SatisfactionLabel": df[COL_C2_SATISFACTION],
        "SatisfactionScore": pd.to_numeric(df["C2_satisfaction_numeric"], errors="coerce"),
        "TextSentimentLabel": df["overall_vader_label"],
        "TextSentimentScore": df["overall_vader_score"],
        "TransformerLabel": df["transformer_label"],
        "VaderFeedbackLabel": df["vader_on_feedback_label"],
        "DivergentOverall": df["divergence_overall"].map(YES_NO),
        "GovtRating_D1": df[COL_D1_GOVT_SUPPORT],
        "Trend_D2": df[COL_D2_TREND],
        "Confidence_D3": df[COL_D3_CONFIDENCE],
        "ChallengeSentiment_D4": df["D4_challenge_vader_label"],
        "Language": df[COL_E1_LANGUAGE],
        "ComfortLevel": df[COL_E2_COMFORT],
        "InterviewDurationMin": pd.to_numeric(df[COL_E4_DURATION], errors="coerce"),
        "FeedbackText": df["scheme_feedback_text"],
    }, index=df.index)

    both_present = df["vader_on_feedback_label"].notna() & df["transformer_label"].notna()
    agree = df["vader_on_feedback_label"] == df["transformer_label"]
    fact.insert(23, "MethodsAgree", np.where(both_present, np.where(agree, "Agree", "Disagree"), None))

    flags = df["age_experience_flag"].fillna("") + df["b1_c1_consistency_flag"].fillna("")
    fact.insert(len(fact.columns) - 1, "DataQualityFlag", flags.replace("", None))
    return fact


def build_aspects(df, respondent_ids):
    rows = []
    for i, row in df.iterrows():
        for aspect, column in COL_C3_ASPECTS.items():
            rating, sentiment = row[column], row[f"absa_{aspect}_label"]
            if pd.isna(rating) and pd.isna(sentiment):
                continue
            rows.append({
                "RespondentID": respondent_ids[i],
                "Aspect": ASPECT_LABELS[aspect],
                "StructuredRatingLabel": rating,
                "StructuredRatingScore": row[f"C3_{aspect}_numeric"],
                "ABSASentimentLabel": sentiment,
                "ABSASentimentScore": row[f"absa_{aspect}_score"],
                "Divergent": YES_NO.get(row[f"divergence_{aspect}"]),
            })
    return pd.DataFrame(rows)


def main():
    df = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)
    responses = build_responses(df)
    aspects = build_aspects(df, responses["RespondentID"])

    save_workbook(OUTPUT_DIR / "PowerBI_Dataset.xlsx", {"Fact_Responses": responses, "Fact_Aspect": aspects})
    print(f"Fact_Responses {responses.shape}, Fact_Aspect {aspects.shape}")


if __name__ == "__main__":
    main()
