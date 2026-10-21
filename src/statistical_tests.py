import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import cohen_kappa_score

SENTIMENT_LABELS = ("Negative", "Neutral", "Positive")
MIN_GROUP_SIZE = 3


def chi_square_test(df, group_col, sentiment_col="overall_vader_label"):
    valid = df[[group_col, sentiment_col]].dropna()
    valid = valid[valid[group_col].astype(str).str.strip() != ""]
    table = pd.crosstab(valid[group_col], valid[sentiment_col])
    table = table[table.sum(axis=1) >= MIN_GROUP_SIZE]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None

    chi2, p, dof, _ = chi2_contingency(table)
    n = table.values.sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    return {
        "test": f"Chi-square: sentiment x {group_col}",
        "chi2": round(chi2, 2), "dof": dof, "n": int(n),
        "p_value": p, "cramers_v": round(np.sqrt(chi2 / (n * min_dim)), 3),
        "groups_tested": table.shape[0],
    }


def kappa_test(df, col_a, col_b):
    valid = df[[col_a, col_b]].dropna()
    return {
        "test": f"Cohen's kappa: {col_a} vs {col_b}",
        "kappa": round(cohen_kappa_score(valid[col_a], valid[col_b]), 3),
        "n": len(valid),
    }


def mcnemar_bowker(df, col_a, col_b, labels=SENTIMENT_LABELS):
    valid = df[[col_a, col_b]].dropna()
    counts = pd.crosstab(valid[col_a], valid[col_b]).reindex(index=labels, columns=labels, fill_value=0)
    m = counts.values.astype(float)

    statistic, dof = 0.0, 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if m[i, j] + m[j, i] > 0:
                statistic += (m[i, j] - m[j, i]) ** 2 / (m[i, j] + m[j, i])
                dof += 1

    p = 1 - stats.chi2.cdf(statistic, dof) if dof > 0 else float("nan")
    return {
        "test": f"McNemar-Bowker: {col_a} vs {col_b}",
        "statistic": round(statistic, 2), "dof": dof, "p_value": p, "n": int(m.sum()),
    }


def spearman_test(df, col_a, col_b):
    valid = df[[col_a, col_b]].dropna()
    rho, p = spearmanr(valid[col_a], valid[col_b])
    return {"test": f"Spearman: {col_a} vs {col_b}", "n": len(valid), "spearman_rho": round(rho, 3), "p_value": p}


def aspect_correlations(df, aspects):
    rows = []
    for aspect in aspects:
        valid = df[[f"C3_{aspect}_numeric", f"absa_{aspect}_score"]].dropna()
        if len(valid) < 5:
            continue
        rho, p = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
        rows.append({"aspect": aspect, "n": len(valid), "spearman_rho": round(rho, 3), "p_value": p})
    return pd.DataFrame(rows)


def fmt_p(p):
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
