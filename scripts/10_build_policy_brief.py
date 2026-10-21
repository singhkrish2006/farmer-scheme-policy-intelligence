import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy.stats import spearmanr

from src.config import COL_D1_GOVT_SUPPORT, LIKERT_5PT_MAP, MASTER_FILE, MASTER_SHEET, OUTPUT_DIR, REPORTS_DIR
from src.report_docx import num, p_value, pct
from src.triangulate import likert_to_numeric

STATS_FILE = OUTPUT_DIR / "descriptive_statistics.xlsx"
NUMBERS_FILE = OUTPUT_DIR / "report_numbers.json"
TOPIC_FILE = OUTPUT_DIR / "topic_modeling_results.xlsx"
VALIDATION_FILE = OUTPUT_DIR / "validation_results.xlsx"
OUT = REPORTS_DIR / "Policy_Intelligence_Brief.md"

MAJOR = ["Rythu Bharosa", "PM-KISAN", "Kisan Credit Card", "Input Subsidy"]
TIMING_TOPICS = ["Late credit of benefits", "Payment delays & bank verification"]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def share(frame, categories):
    frame = frame.set_index("Category")
    return frame.loc[[c for c in categories if c in frame.index], "Percent"].sum()


def main():
    n = json.loads(NUMBERS_FILE.read_text(encoding="utf-8"))
    s = pd.ExcelFile(STATS_FILE).parse
    topics = pd.ExcelFile(TOPIC_FILE)
    master = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)

    absa = s("ABSA by Scheme").set_index("Scheme")
    schemes = s("Scheme Summary").set_index("Scheme")
    conversion = s("Awareness by Scheme")
    div = n["divergence"]
    gap = n["awareness_gap"]
    dis = topics.parse("Dissatisfaction Topics").sort_values("share", ascending=False)
    timing_share = dis[dis["label"].isin(TIMING_TOPICS)]["share"].sum() * 100
    themes = s("Suggestion Themes")
    d1, d2, d3 = s("Outlook D1 Support"), s("Outlook D2 Trend"), s("Outlook D3 Confidence")
    income = s("Sentiment by Income").set_index("Group")
    low = [g for g in income.index if "Below" in str(g)]
    high = [g for g in income.index if "Above" in str(g)]
    chi_v = [v["v"] for v in n["chi_square"].values()]
    pk, ci = schemes.loc["PM-KISAN"], schemes.loc["Crop Insurance"]
    amount_rating = master.groupby("scheme_normalized")["C3_amount_numeric"].mean()
    crop_conv = conversion.set_index("Scheme").loc["Crop Insurance", "Conversion %"]
    widely = conversion[conversion["Heard of"] >= 50]
    best, worst = widely.loc[widely["Conversion %"].idxmax()], widely.loc[widely["Conversion %"].idxmin()]

    d1_score = master[COL_D1_GOVT_SUPPORT].apply(lambda v: likert_to_numeric(v, LIKERT_5PT_MAP))
    pair = pd.DataFrame({"d1": d1_score, "sent": master["overall_vader_score"]}).dropna()
    rho, rho_p = spearmanr(pair["d1"], pair["sent"])

    validated = VALIDATION_FILE.exists()
    if validated:
        methods = pd.read_excel(VALIDATION_FILE, sheet_name="Overall Methods").set_index("method")
        best_method = methods["macro_f1"].idxmax()
        validation_note = (f"Sentiment classification has been checked against {num(methods['n'].iloc[0])} human-labelled responses: VADER accuracy "
                           f"{methods.loc['VADER + agri lexicon', 'accuracy']:.2f}, transformer {methods.loc['RoBERTa transformer (zero-shot)', 'accuracy']:.2f}, "
                           f"best method by macro-F1 {best_method} ({methods.loc[best_method, 'macro_f1']:.2f}). Single annotator; see the research paper, Section 4.10.")
    else:
        validation_note = ("Sentiment classification is not yet validated against human labels; annotation of the stratified samples is in progress. "
                           "The suggestion-mining, awareness and triangulation findings do not depend on this validation.")

    enrol_rows = [[sc, num(absa.loc[sc, "N"])] + [
        f"**{pct(absa.loc[sc, f'{a}_pct_negative'])}**" if a == "enrollment_ease" else pct(absa.loc[sc, f"{a}_pct_negative"])
        for a in ["timeliness", "amount", "enrollment_ease", "official_behavior", "clarity"]] for sc in MAJOR]
    enrol_summary = ", ".join(f"{sc} ({pct(absa.loc[sc, 'enrollment_ease_pct_negative'])})" for sc in MAJOR)

    scheme_rows = []
    for sc, r in schemes.iterrows():
        bold = sc == "Crop Insurance"
        cell = (lambda t: f"**{t}**") if bold else (lambda t: t)
        scheme_rows.append([cell(sc), num(r["N"]), cell(pct(r["pct_positive"])), cell(pct(r["pct_negative"])), cell(f"{r['avg_satisfaction']:.2f}"),
                            f"**{r['avg_impact']:.2f}**" if sc == "PM-KISAN" else ("" if pd.isna(r["avg_impact"]) else f"{r['avg_impact']:.2f}")])

    div_names = {"divergence_overall": "Overall (C2 rating vs text sentiment)", "divergence_timeliness": "Timeliness",
                 "divergence_enrollment_ease": "Enrollment ease", "divergence_clarity": "Clarity of information",
                 "divergence_amount": "Amount of support", "divergence_impact": "Overall impact", "divergence_official_behavior": "Official behavior"}
    div_rows = [[div_names[k], num(div[k]["n"]), f"**{pct(div[k]['pct'])}**" if div[k]["pct"] >= 20 or k == "divergence_overall" else pct(div[k]["pct"])]
                for k in div_names]

    outlook_rows = []
    for i in range(max(len(d1), len(d2), len(d3))):
        row = []
        for frame in (d1, d2, d3):
            row += [frame["Category"].iloc[i], pct(frame["Percent"].iloc[i])] if i < len(frame) else ["", ""]
        outlook_rows.append(row)
    good = share(d1, ["Good", "Excellent"])
    improved = share(d2, ["Improved slightly", "Improved significantly"])
    worsened = share(d2, ["Worsened slightly", "Worsened significantly"])

    text = f"""# Policy Intelligence Brief — Farmer Welfare Scheme Feedback
Chittoor District, Andhra Pradesh · N = {num(n['n_respondents'])} farmers · Dataset verified by the Ethnic Community, Government of Andhra Pradesh

---

## Objectives

1. **Identify the problems farmers actually face**, in their own words rather than pre-set survey categories.
2. **Identify what farmers are asking for**, mined directly from their improvement suggestions.
3. **Test whether structured satisfaction scores can be taken at face value**, and find which schemes and service dimensions underperform.
4. **Turn each finding into a specific recommendation** for the Department of Agriculture and Farmers Welfare, tied to the evidence behind it.

---

## Executive summary

1. **Enrolment is the weakest part of every major scheme.** Enrollment ease is the most negatively described aspect for {enrol_summary} — a shared registration problem, not a scheme-specific one.
2. **Payment timing is the biggest single complaint.** Late credit and payment delays together make up {pct(timing_share)} of complaint text, while timely bank transfer is also a leading reason for satisfaction.
3. **Awareness does not turn into enrolment evenly.** {pct(best['Conversion %'])} of farmers aware of {best['Scheme']} are enrolled, but only {pct(crop_conv)} of those aware of Crop Insurance and {pct(worst['Conversion %'])} for {worst['Scheme']}.
4. **What farmers rate and what they write disagree {pct(div['divergence_overall']['pct'])} of the time**, rising to {pct(div['divergence_timeliness']['pct'])} on timeliness and {pct(div['divergence_enrollment_ease']['pct'])} on enrolment ease (p {p_value(n['mcnemar_bowker']['p'])}) — satisfaction scores understate problems on exactly these dimensions.
5. **Farmers' own top request is a larger benefit amount**, followed by simpler applications, more awareness programmes and faster disbursement.

---

## Results

### 1. Enrolment ease: a shared gap across schemes

{md_table(["Scheme", "N", "Timeliness", "Amount", "Enrollment ease", "Official behavior", "Clarity"], enrol_rows)}

*(% of aspect mentions scored Negative within each scheme)*

The same friction recurs across a state investment-support scheme (Rythu Bharosa), an income-support scheme (PM-KISAN), a credit instrument (KCC) and a subsidy programme (Input Subsidy). The most plausible explanation is a registration, documentation or verification process common to all of them.

### 2. What farmers are asking for

From {num(n['n_suggestions'])} substantive improvement suggestions (answers that were only "No"/"Nothing" excluded):

{md_table(list(themes.columns), themes.astype(str).values.tolist())}

Complaints and suggestions overlap on process friction (registration, documents, office approval) but differ in emphasis: complaints are dominated by payment timing, while suggestions put the size of the benefit first. Farmers describe delays as the problem they live with, and a bigger amount as the change they most want.

### 3. Scheme-specific findings

{md_table(["Scheme", "N", "% Positive", "% Negative", "Avg. satisfaction", "Avg. impact on farming"], scheme_rows)}

*(satisfaction and impact on a −2 to +2 scale)*

- **Crop Insurance** has by far the highest negative share ({pct(ci['pct_negative'])}) and the lowest satisfaction among adequately sampled schemes ({ci['avg_satisfaction']:.2f}); it is also the scheme farmers know but rarely join ({pct(crop_conv)} conversion). It is only {pct(ci['pct_of_feedback'])} of feedback, so pooled figures hide it.
- **PM-KISAN** has strong satisfaction ({pk['avg_satisfaction']:.2f}) but the lowest self-reported farming impact of any adequately sampled scheme ({pk['avg_impact']:.2f}) and the lowest Amount-of-support rating of any major scheme ({amount_rating['PM-KISAN']:.2f}). Farmers value receiving it reliably but do not feel the fixed instalment changes their farming.

### 4. Stated satisfaction vs what farmers write

{md_table(["Comparison", "n comparable", "% divergent"], div_rows)}

A McNemar–Bowker test confirms a systematic pattern (χ² = {n['mcnemar_bowker']['chi2']:.2f}, p {p_value(n['mcnemar_bowker']['p'])}). The divergence is concentrated in timeliness and enrolment — the same dimensions flagged above.

### 5. Awareness and enrolment

- **{pct(n['aware_pct'])}** are aware of at least one scheme ({num(n['aware_yes'])} of {num(n['aware_n'])}).
- On average **{gap['avg_heard']:.2f} schemes heard of vs {gap['avg_enrolled']:.2f} enrolled in**; **{pct(gap['pct_gap'])}** named more schemes heard of than enrolled in.

{md_table(["Scheme", "Heard of", "Heard and enrolled", "Conversion"], [[r["Scheme"], num(r["Heard of"]), num(r["Heard and enrolled"]), pct(r["Conversion %"])] for _, r in conversion.iterrows()])}

### 6. Government outlook

{md_table(["D1: Govt. support rating", "", "D2: Trend vs 5 years ago", "", "D3: Confidence in next 5 years", ""], outlook_rows)}

{pct(good)} rate government support Good or Excellent, and {pct(improved)} see improvement over five years against {pct(worsened)} who see decline. """ + (
        f"Higher ratings of government support go with more positive scheme feedback (Spearman ρ = {rho:.2f}, p {p_value(rho_p)}), so farmers' general view and their scheme experience point the same way, while the detail of their complaints is concentrated on specific mechanics."
        if rho > 0 and rho_p < 0.05 else "Farmers' general view of government support is more favourable than their scheme-specific feedback.") + f"""

### 7. Subgroup patterns

Sentiment differs significantly but modestly by education, income, landholding, gender, information source and scheme (Cramér's V {min(chi_v):.2f}–{max(chi_v):.2f}). """ + (
        f"Income shows the clearest gradient: {pct(income.loc[low[0], 'Negative'])} negative among farmers earning {low[0]} versus {pct(income.loc[high[0], 'Negative'])} among those earning {high[0]}. "
        if low and high else "") + f"""Marginal landholders (<2.5 acres) are the most negative landholding group. Full tables are in `Full_Statistical_Analysis.docx`.

---

## Recommendations

To the Department of Agriculture and Farmers Welfare, Government of Andhra Pradesh:

1. **Review the enrolment and registration process across schemes as one exercise.** Enrollment ease is the weakest aspect for every major scheme regardless of design; audit the shared registration, documentation and verification steps rather than commissioning separate scheme reviews.
2. **Make disbursement reliability an operational priority.** Timing is the largest complaint theme and also a leading source of satisfaction when it works.
3. **Monitor awareness-to-enrolment conversion for every scheme.** Schemes that farmers know but do not join — Crop Insurance ({pct(crop_conv)}), {worst['Scheme']} ({pct(worst['Conversion %'])}) — need enrolment support at the point of application, not more awareness campaigns.
4. **Commission a dedicated review of Crop Insurance**, which combines low conversion, the highest negative share and timing-dominated complaints.
5. **Review the adequacy of PM-KISAN's benefit amount**, separately from its delivery, since its problem is low felt impact rather than dissatisfaction.
6. **Add open-ended questions on enrolment experience and payment timing to monitoring surveys** — the dimensions where ratings and farmers' own words disagree most.
7. **Target additional support at lower-income and marginal farmers**, whose feedback is consistently the most negative.

---

## Limitations

- **Correlational, single district.** Findings come from {num(n['n_respondents'])} respondents in Chittoor district and should be read as hypothesis-generating.
- **Validation.** {validation_note}
- **Method disagreement.** VADER and the transformer agree only slightly on identical text (κ = {n['kappa']:.3f}), so every finding here is cross-checked against at least one other measure.
- **Small samples for minor schemes.** Crop Insurance (n = {num(ci['N'])}) and the smaller schemes should be confirmed with larger samples.
- **Topic shares are approximate.** Topic models are sensitive to pre-processing; the labels are interpretive.
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
