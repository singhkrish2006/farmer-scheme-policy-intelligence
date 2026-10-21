import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

LABELS = ["Positive", "Neutral", "Negative"]
SHORTCUTS = {"1": "Positive", "2": "Neutral", "3": "Negative"}
WORKSHEET_SHEET = "To Annotate"


def normalise_labels(values):
    """Worksheet labels as Positive/Neutral/Negative; annotators may type 1/2/3 instead."""
    text = pd.Series(values).astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return text.replace(SHORTCUTS).str.capitalize()


def read_worksheet(path, label_cols):
    sheet = pd.read_excel(path, sheet_name=WORKSHEET_SHEET).rename(columns={"Label": "Human_Label"})
    for col in label_cols:
        sheet[col] = normalise_labels(sheet[col])
    invalid = {
        col: sorted(set(sheet[col].dropna()) - set(LABELS)) for col in label_cols
        if set(sheet[col].dropna()) - set(LABELS)
    }
    return sheet, invalid


def attach_master(sheet, master, columns):
    lookup = master.copy()
    lookup["Source_Excel_Row"] = lookup.index + 2
    merged = sheet.merge(lookup[["Source_Excel_Row", "scheme_feedback_text"] + columns], on="Source_Excel_Row", how="left")
    text_mismatch = int((merged["Feedback_Text"] != merged["scheme_feedback_text"]).sum())
    return merged, text_mismatch


def summary(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_precision": round(precision, 3),
        "macro_recall": round(recall, 3),
        "macro_f1": round(f1, 3),
        "cohen_kappa": round(cohen_kappa_score(y_true, y_pred, labels=LABELS), 3),
    }


def bootstrap_ci(y_true, y_pred, metric, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric(y_true[idx], y_pred[idx]))
    low, high = np.percentile(scores, [2.5, 97.5])
    return round(low, 3), round(high, 3)


def macro_f1(y_true, y_pred):
    return precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)[2]


def per_class(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return pd.DataFrame({
        "class": LABELS, "precision": precision.round(3), "recall": recall.round(3),
        "f1": f1.round(3), "support": support,
    })


def confusion(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    frame = pd.DataFrame(matrix, index=[f"human {l}" for l in LABELS], columns=[f"predicted {l}" for l in LABELS])
    return frame.reset_index().rename(columns={"index": ""})


def supervised_baselines(texts, labels, seed=42):
    labels = pd.Series(labels).reset_index(drop=True)
    texts = pd.Series(texts).reset_index(drop=True)
    n_splits = min(5, int(labels.value_counts().min()))
    if n_splits < 2:
        return {}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = {
        "Logistic regression (TF-IDF)": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Feed-forward network (TF-IDF)": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=2000, random_state=seed, early_stopping=True,
        ),
    }
    predictions = {}
    for name, model in models.items():
        pipeline = make_pipeline(TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=1), model)
        predictions[name] = cross_val_predict(pipeline, texts, labels, cv=cv)
    return predictions


def method_table(y_true, predictions):
    rows = []
    for method, y_pred in predictions.items():
        row = {"method": method, **summary(y_true, y_pred)}
        row["accuracy_ci95"] = "{} - {}".format(*bootstrap_ci(y_true, y_pred, accuracy_score))
        row["macro_f1_ci95"] = "{} - {}".format(*bootstrap_ci(y_true, y_pred, macro_f1))
        rows.append(row)
    return pd.DataFrame(rows)


def absa_metrics(sheet, aspect_cols, predicted_cols):
    mention_rows, polarity_rows, pooled_true, pooled_pred = [], [], [], []
    for aspect, human_col in aspect_cols.items():
        human = sheet[human_col]
        predicted = sheet[predicted_cols[aspect]]
        human_mentions = human.notna()
        predicted_mentions = predicted.notna()

        tp = int((human_mentions & predicted_mentions).sum())
        fp = int((~human_mentions & predicted_mentions).sum())
        fn = int((human_mentions & ~predicted_mentions).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if tp else float("nan")
        mention_rows.append({
            "aspect": aspect, "human_mentions": int(human_mentions.sum()),
            "pipeline_mentions": int(predicted_mentions.sum()),
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "accuracy": round(float((human_mentions == predicted_mentions).mean()), 3),
        })

        both = human_mentions & predicted_mentions
        if both.sum():
            y_true, y_pred = human[both].astype(str), predicted[both].astype(str)
            pooled_true += list(y_true)
            pooled_pred += list(y_pred)
            polarity_rows.append({"aspect": aspect, **summary(y_true, y_pred)})

    if pooled_true:
        polarity_rows.append({"aspect": "ALL (pooled)", **summary(pooled_true, pooled_pred)})
    return pd.DataFrame(mention_rows), pd.DataFrame(polarity_rows)


def inter_annotator_kappa(first, second, label_col):
    both = first[["Source_Excel_Row", label_col]].merge(
        second[["Source_Excel_Row", label_col]], on="Source_Excel_Row", suffixes=("_a", "_b"),
    ).dropna()
    if both.empty:
        return None
    return {
        "n": len(both),
        "percent_agreement": round(float((both[f"{label_col}_a"] == both[f"{label_col}_b"]).mean() * 100), 1),
        "cohen_kappa": round(cohen_kappa_score(both[f"{label_col}_a"], both[f"{label_col}_b"]), 3),
    }
