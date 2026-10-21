import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import (
    COL_A3_GENDER, COL_A6_EDUCATION, COL_A11_INCOME, COL_A7_LANDHOLDING, COL_B4_SOURCE, COL_C3_ASPECTS,
    DATA_FILE, MASTER_FILE, MASTER_SHEET, OUTPUT_DIR, SHEET_NAME,
)
from src.crosstabs import landholding_bucket
from src.data_quality import logical_consistency_checks, numeric_plausibility_checks
from src.descriptive import absa_negative_by_scheme
from src.excel_style import save_workbook
from src.preprocess import is_nil_response
from src.preprocess_nlp import normalize_for_topics, preprocessing_report
from src.statistical_tests import (
    aspect_correlations, chi_square_test, fmt_p, kappa_test, mcnemar_bowker, spearman_test,
)
from src.suggestions import keyword_frequency_table, suggestion_texts, theme_table
from src.topic_model import fit_topics, model_selection

CHI_SQUARE_GROUPS = {
    "education": COL_A6_EDUCATION,
    "landholding": "landholding_bucket",
    "info_source": COL_B4_SOURCE,
    "scheme": "scheme_normalized",
    "gender": COL_A3_GENDER,
    "income": COL_A11_INCOME,
}

SATISFACTION_LABELS = {
    "Reduced cultivation expense burden": ["burden", "reduced", "expense", "cultivation"],
    "Crop protection / loss mitigation": ["protection", "loss", "loan"],
    "Input purchase support": ["buy", "requirement", "pesticide", "family"],
    "Timely direct bank transfer": ["timeliness", "directly", "bank", "transfer", "account"],
    "Seasonal input/insurance support": ["season", "insurance", "input", "useful", "buying"],
}

DISSATISFACTION_LABELS = {
    "Payment delays & bank verification": ["payment", "bank", "verification", "eligible", "account"],
    "Online registration & documents": ["online", "registration", "internet", "technical", "document"],
    "Insufficient support for crop needs": ["insufficient", "need", "requirement", "buy", "loan"],
    "Late credit of benefits": ["late", "credited", "receiving", "claim", "lately"],
    "Low awareness & slow office approval": ["awareness", "office", "approval", "lack", "long"],
}


def run_data_quality(raw):
    numeric = numeric_plausibility_checks(raw)
    logic = logical_consistency_checks(raw)
    numeric.to_csv(OUTPUT_DIR / "data_quality_numeric_issues.csv", index=False)
    logic.to_csv(OUTPUT_DIR / "data_quality_logic_issues.csv", index=False)
    print(f"Data quality: {len(numeric)} numeric issues, {len(logic)} consistency issues {logic['field'].value_counts().to_dict()}")


def run_statistical_tests(results):
    results = results.copy()
    results["landholding_bucket"] = results[COL_A7_LANDHOLDING].apply(landholding_bucket)

    chi_rows = []
    for label, col in CHI_SQUARE_GROUPS.items():
        row = chi_square_test(results, col)
        if row:
            chi_rows.append({**row, "label": label})
    pd.DataFrame(chi_rows).to_csv(OUTPUT_DIR / "statistical_tests_chisquare.csv", index=False)

    correlations = aspect_correlations(results, list(COL_C3_ASPECTS))
    correlations["p_fmt"] = correlations["p_value"].apply(fmt_p)
    correlations.to_csv(OUTPUT_DIR / "statistical_tests_correlations.csv", index=False)

    kappa = kappa_test(results, "vader_on_feedback_label", "transformer_label")
    bowker = mcnemar_bowker(results, "overall_vader_label", "C2_satisfaction_polarity")
    results["landholding_acres"] = pd.to_numeric(results[COL_A7_LANDHOLDING], errors="coerce")
    land = spearman_test(results, "landholding_acres", "overall_vader_score")
    agreement = pd.DataFrame([
        {"test": kappa["test"], "n": kappa["n"], "statistic_name": "kappa", "statistic": kappa["kappa"], "dof": None, "p_value": None},
        {"test": bowker["test"], "n": bowker["n"], "statistic_name": "chi2", "statistic": bowker["statistic"], "dof": bowker["dof"], "p_value": bowker["p_value"]},
        {"test": land["test"], "n": land["n"], "statistic_name": "spearman_rho", "statistic": land["spearman_rho"], "dof": None, "p_value": land["p_value"]},
    ])
    agreement.to_csv(OUTPUT_DIR / "statistical_tests_agreement.csv", index=False)
    print(f"Statistical tests: {len(chi_rows)} chi-square, {len(correlations)} correlations, kappa={kappa['kappa']}")


def name_topics(topics, label_anchors):
    overlaps = sorted(
        ((len(set(t["top_words"]) & set(anchors)), t["topic_id"], label)
         for t in topics for label, anchors in label_anchors.items()),
        reverse=True,
    )
    assigned, used_labels = {}, set()
    for overlap, topic_id, label in overlaps:
        if overlap and topic_id not in assigned and label not in used_labels:
            assigned[topic_id] = label
            used_labels.add(label)
    return {t["topic_id"]: assigned.get(t["topic_id"], "Unlabelled: " + ", ".join(t["top_words"][:4])) for t in topics}


def topics_frame(topics, names):
    return pd.DataFrame([
        {"topic_id": t["topic_id"], "label": names[t["topic_id"]], "top_words": ", ".join(t["top_words"]), "share": t["share"]}
        for t in topics
    ])


def substantive(series):
    return series.where(~series.apply(is_nil_response))


def run_topic_modeling(results):
    positive = substantive(results["C4_positive_clean"])
    problems = substantive(results["C5_problems_clean"])
    positive_norm = positive.apply(normalize_for_topics)
    problems_norm = problems.apply(normalize_for_topics)

    sat_topics, sat_dominant, sat_weights = fit_topics(positive_norm)
    dis_topics, dis_dominant, dis_weights = fit_topics(problems_norm)
    sat_names = name_topics(sat_topics, SATISFACTION_LABELS)
    dis_names = name_topics(dis_topics, DISSATISFACTION_LABELS)

    def terms_frame(corpus, topics, names):
        return pd.DataFrame([
            {"corpus": corpus, "topic_id": t["topic_id"], "label": names[t["topic_id"]],
             "term_rank": rank, "term": word, "weight": weight}
            for t in topics
            for rank, (word, weight) in enumerate(zip(t["top_words"], t["top_weights"]), start=1)
        ])

    def documents_frame(corpus, weights):
        long = weights.stack().rename("weight").reset_index()
        long.columns = ["row_index", "topic_id", "weight"]
        long.insert(0, "corpus", corpus)
        long.insert(1, "source_excel_row", long.pop("row_index") + 2)
        return long

    per_respondent = pd.DataFrame({
        "source_excel_row": results.index + 2,
        "scheme_normalized": results["scheme_normalized"],
        "satisfaction_topic": sat_dominant.map(sat_names),
        "dissatisfaction_topic": dis_dominant.map(dis_names),
        "C4_positive_clean": positive,
        "C5_problems_clean": problems,
    })

    def topic_by_scheme(column):
        return pd.crosstab(per_respondent["scheme_normalized"], per_respondent[column]).reset_index()

    prep = pd.DataFrame([
        {"field": "C4_positive", **preprocessing_report(positive, positive_norm)},
        {"field": "C5_problems", **preprocessing_report(problems, problems_norm)},
    ])

    texts = suggestion_texts(results)

    sheets = {
        "Satisfaction Topics": topics_frame(sat_topics, sat_names),
        "Dissatisfaction Topics": topics_frame(dis_topics, dis_names),
        "Per-Respondent Topics": per_respondent,
        "Topic Terms": pd.concat([terms_frame("C4", sat_topics, sat_names), terms_frame("C5", dis_topics, dis_names)], ignore_index=True),
        "Document Topics": pd.concat([documents_frame("C4", sat_weights), documents_frame("C5", dis_weights)], ignore_index=True),
        "Dissatisfaction x Scheme": topic_by_scheme("dissatisfaction_topic"),
        "Satisfaction x Scheme": topic_by_scheme("satisfaction_topic"),
        "Preprocessing Report": prep,
        "Topic Model Selection": pd.concat([
            model_selection(positive_norm).assign(corpus="C4"), model_selection(problems_norm).assign(corpus="C5"),
        ], ignore_index=True),
        "ABSA by Scheme": absa_negative_by_scheme(results),
        "Suggestion Themes Ranked": theme_table(texts),
        "Suggestion Keyword Frequency": keyword_frequency_table(texts),
    }
    save_workbook(OUTPUT_DIR / "topic_modeling_results.xlsx", sheets)
    print(f"Topic modeling: {len(sat_topics)} satisfaction + {len(dis_topics)} dissatisfaction topics, {len(texts)} suggestions mined")


def main():
    raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    results = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)

    run_data_quality(raw)
    run_statistical_tests(results)
    run_topic_modeling(results)


if __name__ == "__main__":
    main()
