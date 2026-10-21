import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import MASTER_FILE, MASTER_SHEET, OUTPUT_DIR
from src.worksheets import (
    LABEL_GUIDE, OVERALL_FILE, OVERALL_LABELS, SECOND_RATER_FILE, SHEET, TRAINING_FILE, has_labels, worksheet_frame,
    write_worksheet,
)

SELECTION_FILE = OUTPUT_DIR / "training_batch_selection.csv"
BASE_COLUMNS = ["RespondentID", "Source_Excel_Row", "Participant_Name", "Scheme", "Feedback_Text"]

SECOND_RATER_GUIDE = [
    "PURPOSE: second, independent labelling of the same 243 texts as Annotation_Worksheet.xlsx,",
    "used to measure agreement between two human annotators (Cohen's kappa).",
    "",
    "Work on your own: do not look at Annotation_Worksheet.xlsx, any model output, or discuss individual rows",
    "with the first annotator until both of you have finished.",
    "",
    *LABEL_GUIDE,
    "",
    "When finished, save the file in place (same name, same folder) and run scripts/05_evaluate_models.py.",
]

TRAINING_GUIDE = [
    "PURPOSE: training data for the fine-tuned models (script 12). These rows never overlap the validation",
    "sample in Annotation_Worksheet.xlsx, which stays the untouched test set.",
    "",
    "Label each Feedback_Text yourself, without looking at any model output.",
    "",
    *LABEL_GUIDE,
    "",
    "When finished, save the file in place (same name, same folder) and run scripts/12_compare_models.py.",
]


def build_second_rater(force):
    if has_labels(SECOND_RATER_FILE, OVERALL_LABELS) and not force:
        sys.exit(f"{SECOND_RATER_FILE.name} already contains annotations; refusing to overwrite (use --force to rebuild).")
    frame = pd.read_excel(OVERALL_FILE, sheet_name=SHEET)[BASE_COLUMNS]
    for col in OVERALL_LABELS:
        frame[col] = None
    write_worksheet(SECOND_RATER_FILE, frame, {"Label"}, {"Feedback_Text", "Notes_optional"}, SECOND_RATER_GUIDE)
    print(f"Second-rater worksheet: {len(frame)} rows -> {SECOND_RATER_FILE.name}")


def build_training(size, force):
    from src.active_learning import select_training_batch

    if has_labels(TRAINING_FILE, OVERALL_LABELS) and not force:
        sys.exit(f"{TRAINING_FILE.name} already contains annotations; refusing to overwrite (use --force to rebuild).")
    master = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)
    pool = master[master["scheme_feedback_text"].notna()].copy()
    pool["scheme_normalized"] = pool["scheme_normalized"].fillna("Unspecified")

    test_rows = set(pd.read_excel(OVERALL_FILE, sheet_name=SHEET)["Source_Excel_Row"])
    pool = pool[~(pool.index + 2).isin(test_rows)]
    selected, method = select_training_batch(pool, master["vader_on_feedback_label"], size)

    frame = worksheet_frame(selected, "T", OVERALL_LABELS)
    assert not set(frame["Source_Excel_Row"]) & test_rows, "training rows overlap the test set"
    write_worksheet(TRAINING_FILE, frame, {"Label"}, {"Feedback_Text", "Notes_optional"}, TRAINING_GUIDE)
    pd.DataFrame({"Source_Excel_Row": selected.index + 2, "selection": method.values}).to_csv(SELECTION_FILE, index=False)
    print(f"Training worksheet: {len(frame)} rows from a pool of {len(pool)} "
          f"({(method == 'uncertainty').sum()} uncertainty, {(method == 'random').sum()} random) -> {TRAINING_FILE.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the second-rater and training worksheets.")
    parser.add_argument("--size", type=int, default=400, help="training rows to select (default 400)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_second_rater(args.force)
    build_training(args.size, args.force)
