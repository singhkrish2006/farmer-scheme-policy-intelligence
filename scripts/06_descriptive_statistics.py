import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import (
    ASPECT_LABELS, COL_A3_GENDER, COL_A6_EDUCATION, COL_A7_LANDHOLDING, COL_A8_LAND_TYPE, COL_A10_OCCUPATION,
    COL_A11_INCOME, COL_B2_HEARD, COL_B3_ENROLLED, COL_B4_SOURCE, COL_C2_SATISFACTION, COL_C3_ASPECTS,
    COL_D1_GOVT_SUPPORT, COL_D2_TREND, COL_D3_CONFIDENCE, COL_E1_LANGUAGE, COL_E2_COMFORT, COL_E4_DURATION,
    MASTER_FILE, MASTER_SHEET, OUTPUT_DIR,
)
from src.descriptive import (
    absa_negative_by_scheme, age_bands, awareness_gap, awareness_table, crop_counts, label_distribution,
    landholding_bands, mean_ratings, numeric_summary, pct_table, scheme_table, sentiment_by_group,
)
from src.excel_style import save_workbook
from src.schemes import schemes_in

RESULTS_FILE = OUTPUT_DIR / "descriptive_statistics.xlsx"
NUMBERS_FILE = OUTPUT_DIR / "report_numbers.json"
TOPIC_FILE = OUTPUT_DIR / "topic_modeling_results.xlsx"
VALIDATION_FILE = OUTPUT_DIR / "validation_results.xlsx"


def awareness_by_scheme(df):
    heard = df[COL_B2_HEARD].apply(schemes_in)
    enrolled = df[COL_B3_ENROLLED].apply(schemes_in)
    rows = []
    for scheme in sorted(set().union(*heard)):
        n_heard = int(sum(scheme in h for h in heard))
        n_both = int(sum(scheme in h and scheme in e for h, e in zip(heard, enrolled)))
        rows.append({
            "Scheme": scheme, "Heard of": n_heard, "Enrolled": int(sum(scheme in e for e in enrolled)),
            "Heard and enrolled": n_both, "Conversion %": round(n_both / n_heard * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("Heard of", ascending=False).reset_index(drop=True)


def aspect_sentiment_table(df):
    rows = []
    for aspect in COL_C3_ASPECTS:
        labels = df[f"absa_{aspect}_label"].dropna()
        n = len(labels)
        rows.append({
            "Aspect": ASPECT_LABELS[aspect], "Mentions": n,
            **{f"% {l}": round((labels == l).sum() / n * 100, 1) for l in ("Negative", "Neutral", "Positive")},
        })
    return pd.DataFrame(rows)


def divergence_table():
    divergence = pd.read_csv(OUTPUT_DIR / "divergence_summary.csv")
    names = {"divergence_overall": "Overall (C2 vs text sentiment)", **{f"divergence_{a}": ASPECT_LABELS[a] for a in COL_C3_ASPECTS}}
    divergence["comparison"] = divergence["comparison"].map(names)
    return divergence.sort_values("pct_divergent", ascending=False, key=lambda s: s.where(divergence["comparison"] != names["divergence_overall"], 999))


def confusion_table(df):
    both = df[df["vader_on_feedback_label"].notna() & df["transformer_label"].notna()]
    table = pd.crosstab(both["vader_on_feedback_label"], both["transformer_label"]).reindex(
        index=["Negative", "Neutral", "Positive"], columns=["Negative", "Neutral", "Positive"], fill_value=0)
    return table.rename_axis("VADER \\ Transformer").reset_index()


def group_table(df, column, values=None):
    return sentiment_by_group(df, column, values).reset_index().rename(columns={"group": "Group"})


def numbers(df, tables):
    agreement = pd.read_csv(OUTPUT_DIR / "statistical_tests_agreement.csv").set_index("statistic_name")
    chi = pd.read_csv(OUTPUT_DIR / "statistical_tests_chisquare.csv").set_index("label")
    corr = pd.read_csv(OUTPUT_DIR / "statistical_tests_correlations.csv").set_index("aspect")
    divergence = pd.read_csv(OUTPUT_DIR / "divergence_summary.csv").set_index("comparison")
    topics = pd.ExcelFile(TOPIC_FILE)
    preprocessing = topics.parse("Preprocessing Report").set_index("field")
    gap = awareness_gap(df)
    aware = tables["Awareness B1"].set_index("Category")
    overall = tables["Sentiment Overall VADER"].set_index("Category")
    both = df[df["vader_on_feedback_label"].notna() & df["transformer_label"].notna()]
    age = numeric_summary(df["age_numeric_clean"].where(df["age_numeric_clean"].between(15, 100)))
    land = numeric_summary(pd.to_numeric(df[COL_A7_LANDHOLDING], errors="coerce"))
    exp = numeric_summary(df["farming_experience_clean"])

    out = {
        "n_respondents": len(df),
        "n_scored": int(df["overall_vader_label"].notna().sum()),
        "n_with_feedback_text": int(df["scheme_feedback_text"].notna().sum()),
        "n_with_scheme": int(df["scheme_normalized"].notna().sum()),
        "age": age, "landholding": land, "experience": exp,
        "aware_yes": int(aware.loc["Yes", "Count"]), "aware_pct": float(aware.loc["Yes", "Percent"]), "aware_n": int(aware["Count"].sum()),
        "awareness_gap": {k: (float(v) if not isinstance(v, int) else v) for k, v in gap.items()},
        "sentiment_overall": {l: {"n": int(overall.loc[l, "Count"]), "pct": float(overall.loc[l, "Percent"])} for l in overall.index},
        "method_agreement": {"n": len(both), "n_agree": int((both["vader_on_feedback_label"] == both["transformer_label"]).sum())},
        "kappa": float(agreement.loc["kappa", "statistic"]),
        "mcnemar_bowker": {"chi2": float(agreement.loc["chi2", "statistic"]), "dof": int(agreement.loc["chi2", "dof"]), "n": int(agreement.loc["chi2", "n"]), "p": float(agreement.loc["chi2", "p_value"])},
        "landholding_spearman": {"rho": float(agreement.loc["spearman_rho", "statistic"]), "n": int(agreement.loc["spearman_rho", "n"]), "p": float(agreement.loc["spearman_rho", "p_value"])},
        "divergence": {k: {"n": int(r.n_comparable), "n_div": int(r.n_divergent), "pct": float(r.pct_divergent)} for k, r in divergence.iterrows()},
        "chi_square": {k: {"chi2": float(r.chi2), "dof": int(r.dof), "n": int(r.n), "p": float(r.p_value), "v": float(r.cramers_v)} for k, r in chi.iterrows()},
        "spearman_aspects": {k: {"rho": float(r.spearman_rho), "n": int(r.n), "p": float(r.p_value)} for k, r in corr.iterrows()},
        "preprocessing": {k: {c: float(v) for c, v in r.items()} for k, r in preprocessing.iterrows()},
        "n_suggestions": int(topics.parse("Suggestion Keyword Frequency").columns[2].split("n=")[1].rstrip(")")),
        "data_quality": {
            "numeric_issue_rows": int(pd.read_csv(OUTPUT_DIR / "data_quality_numeric_issues.csv")["source_excel_row"].nunique()),
            "age_experience_impossible": int(df["age_experience_flag"].notna().sum()),
            "unaware_but_rated": int(df["b1_c1_consistency_flag"].notna().sum()),
            "enrolled_not_heard": int((pd.read_csv(OUTPUT_DIR / "data_quality_logic_issues.csv")["field"] == "B2 vs B3").sum()),
            "age_imputed": int(df["age_was_imputed"].sum()),
        },
    }
    if VALIDATION_FILE.exists():
        methods = pd.read_excel(VALIDATION_FILE, sheet_name="Overall Methods")
        out["validation"] = methods.to_dict(orient="records")
    return out


def main():
    df = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)
    topics = pd.ExcelFile(TOPIC_FILE)

    tables = {
        "Demo Age Band": pct_table(age_bands(df)),
        "Demo Gender": pct_table(df[COL_A3_GENDER]),
        "Demo Education": pct_table(df[COL_A6_EDUCATION].replace({"diploma": "Diploma"})),
        "Demo Landholding Band": pct_table(landholding_bands(df)),
        "Demo Land Type": pct_table(df[COL_A8_LAND_TYPE]),
        "Demo Occupation": pct_table(df[COL_A10_OCCUPATION]),
        "Demo Income Band": pct_table(df[COL_A11_INCOME]),
        "Demo Top Crops": crop_counts(df),
        "Awareness B1": awareness_table(df),
        "Awareness Info Source": pct_table(df[COL_B4_SOURCE].where(df[COL_B4_SOURCE].astype(str).str.strip() != "")),
        "Awareness by Scheme": awareness_by_scheme(df),
        "Sentiment Overall VADER": label_distribution(df["overall_vader_label"]),
        "Sentiment Transformer": label_distribution(df["transformer_label"]),
        "Sentiment VADER Same Text": label_distribution(df["vader_on_feedback_label"]),
        "VADER vs Transformer": confusion_table(df),
        "ABSA by Aspect": aspect_sentiment_table(df),
        "Triangulation Divergence": divergence_table(),
        "Scheme Summary": scheme_table(df),
        "ABSA by Scheme": absa_negative_by_scheme(df),
        "Suggestion Themes": topics.parse("Suggestion Themes Ranked"),
        "Topics Dissatisfaction": topics.parse("Dissatisfaction Topics"),
        "Topics Satisfaction": topics.parse("Satisfaction Topics"),
        "Outlook D1 Support": pct_table(df[COL_D1_GOVT_SUPPORT]),
        "Outlook D2 Trend": pct_table(df[COL_D2_TREND]),
        "Outlook D3 Confidence": pct_table(df[COL_D3_CONFIDENCE]),
        "Ratings C2 Satisfaction": pct_table(df[COL_C2_SATISFACTION]),
        "Ratings C3 Mean": mean_ratings(df),
        "Sentiment by Income": group_table(df, COL_A11_INCOME),
        "Sentiment by Education": group_table(df, COL_A6_EDUCATION),
        "Sentiment by Landholding": group_table(df, COL_A7_LANDHOLDING, landholding_bands(df)),
        "Sentiment by Gender": group_table(df, COL_A3_GENDER),
        "Sentiment by Info Source": group_table(df, COL_B4_SOURCE),
        "Survey Language": pct_table(df[COL_E1_LANGUAGE]),
        "Survey Comfort": pct_table(df[COL_E2_COMFORT]),
    }
    duration = numeric_summary(pd.to_numeric(df[COL_E4_DURATION], errors="coerce"))
    duration_values = pd.to_numeric(df[COL_E4_DURATION], errors="coerce").dropna()
    tables["Survey Duration"] = pd.DataFrame([{**duration, "min": float(duration_values.min()), "max": float(duration_values.max())}])

    save_workbook(RESULTS_FILE, tables)
    report = numbers(df, tables)
    report["duration"] = tables["Survey Duration"].iloc[0].to_dict()
    NUMBERS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {RESULTS_FILE.name} ({len(tables)} tables) and {NUMBERS_FILE.name}")


if __name__ == "__main__":
    main()
