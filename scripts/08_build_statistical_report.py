import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import OUTPUT_DIR, REPORTS_DIR
from src.report_docx import bullet, new_document, num, p_value, para, pct, signed, table

STATS_FILE = OUTPUT_DIR / "descriptive_statistics.xlsx"
NUMBERS_FILE = OUTPUT_DIR / "report_numbers.json"
VALIDATION_FILE = OUTPUT_DIR / "validation_results.xlsx"
OUT = REPORTS_DIR / "Full_Statistical_Analysis.docx"

LEXICON_LOG = [
    ("\"No\" (nil answers)", "Bare \"No\" carries negative valence in VADER, so \"Crop protection. No. No\" scored Negative",
     "Nil answers (No, Nothing, Nil, No problems, No suggestions) removed before scoring", "VADER-transformer agreement 30.7% -> 37.2% at the time of the fix"),
    ("burden, pressure, losses, unstable, uncertain", "Negative in VADER and not negated by \"reduce(d)\", so \"reduced financial burden\" scored Negative",
     "Neutralised to 0.0 (89-100% of uses were relief phrases)", "49 of 52 \"burden\" answers were mis-scored before the fix"),
    ("helped", "Missing from VADER although \"help\"/\"helps\" exist", "Added at +1.7", "110 of 237 answers using \"helped\" were mis-scored"),
    ("slow, complex, unnecessary", "Evaluative complaint words missing from VADER", "Added at -1.5 / -1.5 / -1.0", "34 complaint uses, 0 positive uses"),
    ("benifits (spelling)", "Misspelling of an already-neutralised process noun", "Neutralised to 0.0", "16 uses, all in complaint text"),
]


def pct_rows(frame):
    return [[r["Category"], num(r["Count"]), pct(r["Percent"])] for _, r in frame.iterrows()]


def section(doc, title, level=1):
    doc.add_heading(title, level=level)


def main():
    book = pd.ExcelFile(STATS_FILE)
    sheet = book.parse
    n = json.loads(NUMBERS_FILE.read_text(encoding="utf-8"))
    doc = new_document("Full Statistical Analysis")

    doc.add_heading("Full Statistical Analysis", 0)
    para(doc, f"Farmer Welfare Scheme Survey — Chittoor District, Andhra Pradesh — N = {num(n['n_respondents'])}", size=12)
    para(doc, "Percentages use the respondents who answered each question as the denominator; blank answers are excluded from every table.", italic=True, size=9)

    section(doc, "1. Overview")
    table(doc, ["Metric", "Value"], [
        ["Total respondents", num(n["n_respondents"])], ["District", "Chittoor, Andhra Pradesh"],
        ["Respondents naming a scheme for feedback", num(n["n_with_scheme"])],
        ["Respondents with scheme feedback text", num(n["n_with_feedback_text"])],
        ["Respondents with sentiment scored", num(n["n_scored"])],
    ])

    section(doc, "2. Demographics")
    a, l, e = n["age"], n["landholding"], n["experience"]
    para(doc, f"Age: mean {a['mean']:.1f} years, median {a['median']:.1f} (n = {num(a['n'])} valid answers).")
    para(doc, f"Landholding: mean {l['mean']:.2f} acres, median {l['median']:.2f} (n = {num(l['n'])}, right-skewed).")
    para(doc, f"Farming experience: mean {e['mean']:.1f} years, median {e['median']:.1f} (n = {num(e['n'])}).")
    for title, name in [("Age band", "Demo Age Band"), ("Gender", "Demo Gender"), ("Education level", "Demo Education"),
                        ("Landholding band", "Demo Landholding Band"), ("Type of land", "Demo Land Type"),
                        ("Primary occupation", "Demo Occupation"), ("Annual income from agriculture", "Demo Income Band"),
                        ("Top crops cultivated (multi-select)", "Demo Top Crops")]:
        section(doc, title, 2)
        table(doc, ["Category", "Count", "Percent"], pct_rows(sheet(name)))

    section(doc, "3. Awareness & Utilisation")
    section(doc, "Aware of any government welfare scheme (B1)", 2)
    table(doc, ["Category", "Count", "Percent"], pct_rows(sheet("Awareness B1")))
    section(doc, "How respondents first heard about schemes (B4)", 2)
    table(doc, ["Category", "Count", "Percent"], pct_rows(sheet("Awareness Info Source")))
    section(doc, "Awareness-to-enrolment gap", 2)
    g = n["awareness_gap"]
    para(doc, f"Among {num(g['n_usable'])} respondents who listed at least one scheme as heard of or enrolled in, the average was "
              f"{g['avg_heard']:.2f} schemes heard of versus {g['avg_enrolled']:.2f} enrolled in; {num(g['n_gap'])} respondents "
              f"({pct(g['pct_gap'])}) named more schemes heard of than enrolled in.")
    conversion = sheet("Awareness by Scheme")
    table(doc, list(conversion.columns), [[r["Scheme"], num(r["Heard of"]), num(r["Enrolled"]), num(r["Heard and enrolled"]), pct(r["Conversion %"])]
                                          for _, r in conversion.iterrows()])
    para(doc, "Conversion = respondents who heard of a scheme and are enrolled in it, as a share of all who heard of it. "
              f"{num(n['data_quality']['enrolled_not_heard'])} respondents listed a scheme as enrolled that they did not list as heard of; these are flagged in the data-quality audit (Section 15).", italic=True, size=9)

    section(doc, "4. Overall Sentiment")
    for title, name in [("VADER (lexicon), averaged across the five text fields", "Sentiment Overall VADER"),
                        ("RoBERTa transformer (zero-shot), combined feedback text", "Sentiment Transformer"),
                        ("VADER on the same combined feedback text", "Sentiment VADER Same Text")]:
        section(doc, title, 2)
        table(doc, ["Category", "Count", "Percent"], pct_rows(sheet(name)))
    agree = n["method_agreement"]
    section(doc, "VADER vs transformer on identical text", 2)
    confusion = sheet("VADER vs Transformer")
    table(doc, list(confusion.columns), confusion.astype(str).values.tolist())
    para(doc, f"Raw agreement {num(agree['n_agree'])} of {num(agree['n'])} ({pct(agree['n_agree'] / agree['n'] * 100)}); "
              f"Cohen's kappa = {n['kappa']:.3f} (slight agreement).")

    section(doc, "5. Aspect-Based Sentiment (ABSA)")
    absa = sheet("ABSA by Aspect")
    table(doc, list(absa.columns), [[r["Aspect"], num(r["Mentions"]), pct(r["% Negative"]), pct(r["% Neutral"]), pct(r["% Positive"])] for _, r in absa.iterrows()])

    section(doc, "6. Triangulation: Stated Rating vs Text Sentiment")
    div = sheet("Triangulation Divergence")
    table(doc, ["Comparison", "n comparable", "n divergent", "% divergent"],
          [[r["comparison"], num(r["n_comparable"]), num(r["n_divergent"]), pct(r["pct_divergent"])] for _, r in div.iterrows()])
    mb = n["mcnemar_bowker"]
    para(doc, f"McNemar-Bowker test, paired overall sentiment vs satisfaction polarity: chi-square = {mb['chi2']:.2f}, df = {mb['dof']}, "
              f"n = {num(mb['n'])}, p {p_value(mb['p'])}.")

    section(doc, "7. Scheme-wise Analysis")
    schemes = sheet("Scheme Summary")
    table(doc, ["Scheme", "N", "% of feedback", "% Positive", "% Neutral", "% Negative", "Avg satisfaction", "Avg impact rating"],
          [[r["Scheme"], num(r["N"]), pct(r["pct_of_feedback"]), pct(r["pct_positive"]), pct(r["pct_neutral"]), pct(r["pct_negative"]),
            f"{r['avg_satisfaction']:.2f}", "" if pd.isna(r["avg_impact"]) else f"{r['avg_impact']:.2f}"] for _, r in schemes.iterrows()], size=8)
    para(doc, "Satisfaction and impact are structured ratings on a −2 to +2 scale.", italic=True, size=9)

    section(doc, "8. ABSA by Scheme (% of aspect mentions scored Negative)")
    by_scheme = sheet("ABSA by Scheme")
    aspects = ["timeliness", "amount", "enrollment_ease", "official_behavior", "clarity", "impact"]
    table(doc, ["Scheme", "N", "Timeliness", "Amount", "Enrolment", "Officials", "Clarity", "Impact"],
          [[r["Scheme"], num(r["N"])] + [pct(r[f"{a}_pct_negative"]) for a in aspects] for _, r in by_scheme.iterrows()], size=8)

    section(doc, "9. Farmer-Authored Improvement Priorities")
    para(doc, f"Term frequency over {num(n['n_suggestions'])} substantive suggestions (C6; answers that were only \"No\"/\"Nothing\" excluded).")
    themes = sheet("Suggestion Themes")
    table(doc, list(themes.columns), themes.astype(str).values.tolist())

    section(doc, "10. Topic Modelling (LDA)")
    pre = n["preprocessing"]
    for title, name, field in [("Dissatisfaction topics (problems text, C5)", "Topics Dissatisfaction", "C5_problems"),
                               ("Satisfaction topics (positive-aspects text, C4)", "Topics Satisfaction", "C4_positive")]:
        section(doc, f"{title}, n = {num(pre[field]['n_documents'])}", 2)
        topics = sheet(name).sort_values("share", ascending=False)
        table(doc, ["Topic", "Top terms", "Share"], [[r["label"], r["top_words"], pct(r["share"] * 100)] for _, r in topics.iterrows()])

    section(doc, "11. Government Outlook (Section D)")
    for title, name in [("D1: Overall rating of government support", "Outlook D1 Support"),
                        ("D2: Compared to five years ago", "Outlook D2 Trend"),
                        ("D3: Confidence support improves in the next five years", "Outlook D3 Confidence")]:
        section(doc, title, 2)
        table(doc, ["Category", "Count", "Percent"], pct_rows(sheet(name)))

    section(doc, "12. Structured Ratings & Significance Tests")
    section(doc, "C2: Overall satisfaction", 2)
    table(doc, ["Category", "Count", "Percent"], pct_rows(sheet("Ratings C2 Satisfaction")))
    section(doc, "C3: Average aspect ratings (−2 to +2)", 2)
    ratings = sheet("Ratings C3 Mean")
    table(doc, ["Aspect", "N", "Average"], [[r["Aspect"], num(r["N"]), f"{r['Avg rating']:.2f}"] for _, r in ratings.iterrows()])
    section(doc, "Chi-square tests of independence (overall sentiment × group)", 2)
    names = {"education": "Education level", "income": "Income band", "scheme": "Scheme", "landholding": "Landholding band",
             "info_source": "Information source", "gender": "Gender"}
    chi = sorted(n["chi_square"].items(), key=lambda kv: -kv[1]["v"])
    table(doc, ["Grouping", "Chi-square", "df", "n", "p", "Cramér's V"],
          [[names[k], f"{v['chi2']:.2f}", v["dof"], num(v["n"]), p_value(v["p"]), f"{v['v']:.3f}"] for k, v in chi])
    section(doc, "Spearman correlation: C3 rating vs ABSA score", 2)
    labels = {"timeliness": "Timeliness", "amount": "Amount of support", "enrollment_ease": "Enrollment ease",
              "official_behavior": "Official behavior", "clarity": "Clarity of information", "impact": "Overall impact"}
    table(doc, ["Aspect", "n", "Spearman rho", "p"],
          [[labels[k], num(v["n"]), signed(v["rho"]), p_value(v["p"])] for k, v in n["spearman_aspects"].items()])
    s = n["landholding_spearman"]
    para(doc, f"Landholding (acres) vs overall sentiment score: rho = {s['rho']:.3f}, n = {num(s['n'])}, p {p_value(s['p'])}.")

    section(doc, "13. Sentiment by Demographic and Economic Group")
    for title, name in [("By annual income band", "Sentiment by Income"), ("By education level", "Sentiment by Education"),
                        ("By landholding band", "Sentiment by Landholding"), ("By gender", "Sentiment by Gender"),
                        ("By information source", "Sentiment by Info Source")]:
        section(doc, title, 2)
        frame = sheet(name)
        table(doc, ["Group", "% Negative", "% Neutral", "% Positive", "n"],
              [[r["Group"], pct(r["Negative"]), pct(r["Neutral"]), pct(r["Positive"]), num(r["n"])] for _, r in frame.iterrows()])

    section(doc, "14. Survey Quality (Section E)")
    section(doc, "Language of interview", 2)
    table(doc, ["Category", "Count", "Percent"], pct_rows(sheet("Survey Language")))
    section(doc, "Participant comfort level", 2)
    table(doc, ["Category", "Count", "Percent"], pct_rows(sheet("Survey Comfort")))
    d = n["duration"]
    para(doc, f"Interview duration: mean {d['mean']:.1f} minutes, range {d['min']:.0f}–{d['max']:.0f} (n = {num(d['n'])}).")

    section(doc, "15. Data Quality Audit")
    q = n["data_quality"]
    table(doc, ["Check", "Respondents", "% of sample", "Action"], [
        ["Implausible numeric values (age, landholding, duration)", num(q["numeric_issue_rows"]), pct(q["numeric_issue_rows"] / n["n_respondents"] * 100), "Listed in data_quality_numeric_issues.csv"],
        ["Farming experience ≥ age", num(q["age_experience_impossible"]), pct(q["age_experience_impossible"] / n["n_respondents"] * 100), "Both values set to missing"],
        ["Said 'not aware' (B1) but rated a scheme (C1)", num(q["unaware_but_rated"]), pct(q["unaware_but_rated"] / n["n_respondents"] * 100), "Flagged, original answers kept"],
        ["Enrolled in a scheme not listed as heard of (B3 vs B2)", num(q["enrolled_not_heard"]), pct(q["enrolled_not_heard"] / n["n_respondents"] * 100), "Flagged, original answers kept"],
        ["Missing or invalid age", num(q["age_imputed"]), pct(q["age_imputed"] / n["n_respondents"] * 100), "Mean-imputed in the analysis copy only"],
    ], size=8)

    section(doc, "16. Lexicon Correction Log")
    para(doc, "Words appearing at least 8 times across the positive (C4) and problems (C5) fields were checked; any word found in one field at least 80% of the time was compared with its VADER score.")
    table(doc, ["Word(s)", "Issue", "Fix", "Evidence"], [list(r) for r in LEXICON_LOG], size=8)

    section(doc, "17. Model Validation Against Human Labels")
    if VALIDATION_FILE.exists():
        summary = pd.read_excel(VALIDATION_FILE, sheet_name="Summary")
        for _, r in summary.iterrows():
            bullet(doc, f"{r['item']}: {r['value']}")
        methods = pd.read_excel(VALIDATION_FILE, sheet_name="Overall Methods")
        table(doc, ["Method", "n", "Accuracy (95% CI)", "Macro-F1 (95% CI)", "Kappa"],
              [[r["method"], num(r["n"]), f"{r['accuracy']:.3f} ({r['accuracy_ci95']})", f"{r['macro_f1']:.3f} ({r['macro_f1_ci95']})", f"{r['cohen_kappa']:.3f}"]
               for _, r in methods.iterrows()], size=8)
        polarity = pd.read_excel(VALIDATION_FILE, sheet_name="ABSA Polarity") if "ABSA Polarity" in pd.ExcelFile(VALIDATION_FILE).sheet_names else None
        if polarity is not None:
            table(doc, ["Aspect", "n", "Accuracy", "Macro-F1", "Kappa"],
                  [[r["aspect"], num(r["n"]), f"{r['accuracy']:.3f}", f"{r['macro_f1']:.3f}", f"{r['cohen_kappa']:.3f}"] for _, r in polarity.iterrows()], size=8)
    else:
        para(doc, "Human annotation of the stratified validation samples is in progress; results will be added when complete.", italic=True)

    doc.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
