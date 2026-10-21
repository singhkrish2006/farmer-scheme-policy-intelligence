import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import MASTER_FILE, MASTER_SHEET, OUTPUT_DIR
from src.evaluation import (
    absa_metrics, attach_master, confusion, inter_annotator_kappa, method_table, per_class,
    read_worksheet, supervised_baselines,
)
from src.excel_style import save_workbook

OVERALL_FILE = OUTPUT_DIR / "Annotation_Worksheet.xlsx"
ABSA_FILE = OUTPUT_DIR / "ABSA_Annotation_Worksheet.xlsx"
SECOND_RATER_FILE = OUTPUT_DIR / "Annotation_Worksheet_rater2.xlsx"
RESULTS_FILE = OUTPUT_DIR / "validation_results.xlsx"

ASPECT_COLUMNS = {
    "timeliness": "Timeliness",
    "amount": "Amount_of_Support",
    "enrollment_ease": "Enrollment_Ease",
    "official_behavior": "Official_Behavior",
    "clarity": "Clarity_of_Info",
    "impact": "Overall_Impact",
}
ZERO_SHOT = {"VADER + agri lexicon": "vader_on_feedback_label", "RoBERTa transformer (zero-shot)": "transformer_label"}


def stop(message):
    sys.exit(f"Stopped: {message}")


def check(invalid, mismatched, name):
    if invalid:
        stop(f"{name} has labels other than Positive/Neutral/Negative: {invalid}")
    if mismatched:
        stop(f"{name}: {mismatched} rows no longer match the master results file. Rebuild the worksheet before annotating.")


def evaluate_overall(master, sheets):
    sheet, invalid = read_worksheet(OVERALL_FILE, ["Human_Label"])
    sheet, mismatched = attach_master(sheet, master, list(ZERO_SHOT.values()))
    check(invalid, mismatched, OVERALL_FILE.name)

    labelled = sheet[sheet["Human_Label"].notna()].reset_index(drop=True)
    if labelled.empty:
        print("Overall worksheet: no labels yet - skipped")
        return None
    y_true = labelled["Human_Label"].astype(str)

    predictions = {name: labelled[col].fillna("Neutral").astype(str) for name, col in ZERO_SHOT.items()}
    predictions.update(supervised_baselines(labelled["Feedback_Text"], y_true))

    methods = method_table(y_true, predictions)
    sheets["Overall Methods"] = methods
    sheets["Overall Per Class"] = pd.concat(
        [per_class(y_true, pred).assign(method=name) for name, pred in predictions.items()], ignore_index=True,
    )[["method", "class", "precision", "recall", "f1", "support"]]
    sheets["Overall Confusion"] = pd.concat(
        [confusion(y_true, pred).assign(method=name) for name, pred in predictions.items()], ignore_index=True,
    )
    sheets["Human Label Distribution"] = y_true.value_counts().rename_axis("label").reset_index(name="count")
    print(methods[["method", "n", "accuracy", "macro_f1", "cohen_kappa"]].to_string(index=False))
    return len(labelled), len(sheet)


def evaluate_absa(master, sheets):
    sheet, invalid = read_worksheet(ABSA_FILE, list(ASPECT_COLUMNS.values()))
    predicted = {aspect: f"absa_{aspect}_label" for aspect in ASPECT_COLUMNS}
    sheet, mismatched = attach_master(sheet, master, list(predicted.values()))
    check(invalid, mismatched, ABSA_FILE.name)

    read = sheet["Annotator_Initials"].notna() | sheet[list(ASPECT_COLUMNS.values())].notna().any(axis=1)
    annotated = sheet[read].reset_index(drop=True)
    if annotated.empty:
        print("ABSA worksheet: no annotations yet - skipped")
        return None

    mentions, polarity = absa_metrics(annotated, ASPECT_COLUMNS, predicted)
    sheets["ABSA Aspect Detection"] = mentions
    sheets["ABSA Polarity"] = polarity
    print(polarity.to_string(index=False))
    return len(annotated), len(sheet)


def main():
    master = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)
    sheets = {}
    overall = evaluate_overall(master, sheets)
    absa = evaluate_absa(master, sheets)
    if not sheets:
        stop("no annotations found in either worksheet")

    summary = [
        {"item": "Overall worksheet rows labelled", "value": f"{overall[0]} of {overall[1]}" if overall else "0"},
        {"item": "ABSA worksheet rows annotated", "value": f"{absa[0]} of {absa[1]}" if absa else "0"},
        {"item": "Supervised baselines", "value": "Stratified k-fold cross-validation on the human labels (k <= 5)"},
        {"item": "Confidence intervals", "value": "95% bootstrap, 1,000 resamples"},
    ]
    if SECOND_RATER_FILE.exists():
        first, _ = read_worksheet(OVERALL_FILE, ["Human_Label"])
        second, _ = read_worksheet(SECOND_RATER_FILE, ["Human_Label"])
        agreement = inter_annotator_kappa(first, second, "Human_Label")
        if agreement:
            summary.append({"item": "Inter-annotator agreement", "value": f"kappa {agreement['cohen_kappa']} on {agreement['n']} rows ({agreement['percent_agreement']}% raw)"})

    save_workbook(RESULTS_FILE, {"Summary": pd.DataFrame(summary), **sheets})
    print(f"Saved {RESULTS_FILE}")


if __name__ == "__main__":
    main()
