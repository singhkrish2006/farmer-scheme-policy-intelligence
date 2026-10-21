"""Rescore all written feedback with the validated models and recompute every sentiment-dependent finding.

Overall sentiment: ModernBERT fine-tuned on the 400 training labels (5 seeds, majority vote) — the best model in script 12.
Aspect sentiment: two candidate pipelines are scored against the human aspect labels and the better one is used:
  B  keyword aspect detection + DeBERTa ABSA polarity
  C  keyword aspect detection + ModernBERT polarity of the clauses that mention the aspect
Outputs: output/validated_sentiment.xlsx and output/validated_numbers.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import model_comparison as mc
from src.absa import _CLAUSE_SPLIT, extract_aspects
from src.config import COL_A3_GENDER, COL_A6_EDUCATION, COL_A7_LANDHOLDING, COL_A11_INCOME, COL_B4_SOURCE, MASTER_FILE, MASTER_SHEET, OUTPUT_DIR
from src.crosstabs import landholding_bucket
from src.evaluation import LABELS, read_worksheet, summary
from src.statistical_tests import chi_square_test, mcnemar_bowker
from src.triangulate import flag_divergence

ASPECTS = {"timeliness": "Timeliness", "amount": "Amount_of_Support", "enrollment_ease": "Enrollment_Ease",
           "official_behavior": "Official_Behavior", "clarity": "Clarity_of_Info", "impact": "Overall_Impact"}
ASPECT_NAMES = {"timeliness": "Timeliness", "amount": "Amount of support", "enrollment_ease": "Enrollment ease",
                "official_behavior": "Official behavior", "clarity": "Clarity of information", "impact": "Overall impact"}
CHI_GROUPS = {"education": COL_A6_EDUCATION, "landholding": "landholding_bucket", "info_source": COL_B4_SOURCE,
              "scheme": "scheme_normalized", "gender": COL_A3_GENDER, "income": COL_A11_INCOME}
SEEDS, BACKBONE, ABSA_MODEL = 5, "answerdotai/ModernBERT-base", "yangheng/deberta-v3-base-absa-v1.1"
POL = {"Positive": 1, "Neutral": 0, "Negative": -1}


def clauses(text):
    return [c.strip() for c in _CLAUSE_SPLIT.split(text) if c.strip()] or [text]


def aggregate(labels):
    total = sum(POL[l] for l in labels)
    return "Positive" if total > 0 else "Negative" if total < 0 else "Neutral"


def majority(runs):
    return [pd.Series(col).mode().iloc[0] for col in zip(*runs)]


def main():
    master = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)
    master["Source_Excel_Row"] = master.index + 2
    fb = master[master["scheme_feedback_text"].notna()].copy()
    texts = fb["scheme_feedback_text"].tolist()

    train, _ = read_worksheet(OUTPUT_DIR / "Training_Worksheet.xlsx", ["Human_Label"])
    train = train[train["Human_Label"].notna()]
    x_train, y_train = train["Feedback_Text"], train["Human_Label"].astype(str)

    # aspect mentions: (row index, aspect, clause list)
    mentions = []
    for idx, text in zip(fb.index, texts):
        per_aspect = {}
        for cl in clauses(text):
            for a in extract_aspects(cl):
                per_aspect.setdefault(a, []).append(cl)
        mentions += [(idx, a, cls) for a, cls in per_aspect.items()]
    all_clauses = sorted({cl for _, _, cls in mentions for cl in cls})
    print(f"{len(texts)} feedback texts, {len(mentions)} aspect mentions, {len(all_clauses)} distinct clauses")

    # ModernBERT: overall and clause predictions in one pass per seed
    cache = OUTPUT_DIR / "validated_predictions.json"
    inputs = texts + all_clauses
    runs = json.loads(cache.read_text(encoding="utf8")) if cache.exists() else []
    if runs and len(runs[0]) != len(inputs):
        runs = []
    for s in range(len(runs), SEEDS):
        runs.append([str(x) for x in mc.finetune_predict(x_train, y_train, inputs, s, BACKBONE)])
        cache.write_text(json.dumps(runs), encoding="utf8")
        print(f"  ModernBERT seed {s} done")
    voted = majority(runs)
    fb["validated_label"] = voted[:len(texts)]
    clause_label = dict(zip(all_clauses, voted[len(texts):]))

    # candidate aspect pipelines
    cand_c = {(i, a): aggregate([clause_label[c] for c in cls]) for i, a, cls in mentions}
    deb = mc.absa_predict([fb.loc[i, "scheme_feedback_text"] for i, _, _ in mentions], [mc.ABSA_PHRASES[ASPECTS[a]] for _, a, _ in mentions], ABSA_MODEL)
    cand_b = {(i, a): lab for (i, a, _), lab in zip(mentions, deb)}

    # validate both against the human aspect labels
    sheet, _ = read_worksheet(OUTPUT_DIR / "ABSA_Annotation_Worksheet.xlsx", list(ASPECTS.values()))
    row_to_idx = dict(zip(master["Source_Excel_Row"], master.index))
    human, pred_b, pred_c = [], [], []
    for r in sheet.itertuples():
        idx = row_to_idx[r.Source_Excel_Row]
        for a, col in ASPECTS.items():
            h = getattr(r, col)
            if pd.notna(h):
                human.append(h); pred_b.append(cand_b.get((idx, a), "Not detected")); pred_c.append(cand_c.get((idx, a), "Not detected"))
    val = pd.DataFrame([{"pipeline": "Keyword detection + DeBERTa ABSA", **summary(human, pred_b)},
                        {"pipeline": "Keyword detection + ModernBERT clauses", **summary(human, pred_c)}])
    detected = sum(p != "Not detected" for p in pred_b)
    print(val.to_string(index=False))
    best = val.sort_values(["macro_f1", "cohen_kappa"]).iloc[-1]["pipeline"]   # ties broken by kappa
    chosen, chosen_name = (cand_b if best.startswith("Keyword detection + DeBERTa") else cand_c), best

    for a in ASPECTS:
        fb[f"val_{a}"] = [chosen.get((i, a)) for i in fb.index]

    # sanity: overall model on the held-out test rows
    test, _ = read_worksheet(OUTPUT_DIR / "Annotation_Worksheet.xlsx", ["Human_Label"])
    t = test.merge(fb[["Source_Excel_Row", "validated_label"]], on="Source_Excel_Row")
    test_check = summary(t["Human_Label"].astype(str), t["validated_label"])

    # recompute findings
    dist = fb["validated_label"].value_counts(normalize=True).mul(100).round(1).reindex(LABELS).to_dict()
    aspect_rows = []
    for a, name in ASPECT_NAMES.items():
        v = fb[f"val_{a}"].dropna()
        aspect_rows.append({"Aspect": name, "Mentions": len(v), **{f"% {l}": round(100 * (v == l).mean(), 1) for l in ("Negative", "Neutral", "Positive")}})
    aspects = pd.DataFrame(aspect_rows)

    div = {}
    for key, text_col, rating_col in [("overall", "validated_label", "C2_satisfaction_polarity")] + [(a, f"val_{a}", f"C3_{a}_polarity") for a in ASPECTS]:
        flags = [flag_divergence(x, y) for x, y in zip(fb[text_col], fb[rating_col])]
        flags = [f for f in flags if f is not None]
        div[key] = {"n": len(flags), "n_div": int(sum(flags)), "pct": round(100 * sum(flags) / len(flags), 1) if flags else None}
    bowker = mcnemar_bowker(fb, "validated_label", "C2_satisfaction_polarity")

    schemes = []
    for s, g in fb[fb["scheme_normalized"].notna()].groupby("scheme_normalized"):
        lab = g["validated_label"].value_counts(normalize=True).mul(100)
        schemes.append({"Scheme": s, "N": len(g), "pct_positive": round(lab.get("Positive", 0), 1), "pct_neutral": round(lab.get("Neutral", 0), 1), "pct_negative": round(lab.get("Negative", 0), 1)})
    schemes = pd.DataFrame(schemes).sort_values("N", ascending=False)

    worst = []
    for s, g in fb.groupby("scheme_normalized"):
        best = None
        for a, name in ASPECT_NAMES.items():
            v = g[f"val_{a}"].dropna()
            if len(v) >= 10:
                share = round(100 * (v == "Negative").mean(), 1)
                if best is None or share > best[2]:
                    best = (s, name, share, len(v))
        if best: worst.append(best)
    worst = pd.DataFrame(worst, columns=["Scheme", "Weakest aspect", "% negative", "mentions"])

    income = pd.crosstab(fb[COL_A11_INCOME], fb["validated_label"], normalize="index").mul(100).round(1)
    income["n"] = fb.groupby(COL_A11_INCOME).size()
    income = income.reset_index().rename(columns={COL_A11_INCOME: "Group"})

    fb["landholding_bucket"] = fb[COL_A7_LANDHOLDING].apply(landholding_bucket)
    chi = {}
    for k, col in CHI_GROUPS.items():
        r = chi_square_test(fb, col, "validated_label")
        if r: chi[k] = {"chi2": r["chi2"], "dof": int(r["dof"]), "n": r["n"], "p": float(r["p_value"]), "v": r["cramers_v"]}

    numbers = {"n_scored": len(fb), "distribution": dist, "test_check": test_check, "aspect_pipeline": chosen_name,
               "aspect_validation": val.to_dict("records"), "aspect_detection_recall": round(detected / len(human), 3), "human_aspect_mentions": len(human),
               "divergence": div, "mcnemar_bowker": {"chi2": bowker["statistic"], "dof": bowker["dof"], "n": bowker["n"], "p": float(bowker["p_value"])}, "chi_square": chi}
    (OUTPUT_DIR / "validated_numbers.json").write_text(json.dumps(numbers, indent=1, default=float), encoding="utf8")
    with pd.ExcelWriter(OUTPUT_DIR / "validated_sentiment.xlsx") as w:
        for name, frame in [("Aspects", aspects), ("Schemes", schemes), ("Weakest Aspect", worst), ("Income", income), ("Aspect Validation", val),
                            ("Labels", fb[["Source_Excel_Row", "scheme_normalized", "validated_label"] + [f"val_{a}" for a in ASPECTS]])]:
            frame.to_excel(w, sheet_name=name, index=False)
    print(json.dumps({k: numbers[k] for k in ("distribution", "test_check", "aspect_pipeline", "divergence", "mcnemar_bowker")}, indent=1, default=float))
    print(aspects.to_string(index=False)); print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
