import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import MASTER_FILE, MASTER_SHEET
from src.worksheets import (
    ABSA_FILE, ASPECT_COLUMNS, LABEL_GUIDE, OVERALL_FILE, OVERALL_LABELS, has_labels, stratified_sample,
    worksheet_frame, write_worksheet,
)

OVERALL_GUIDE = [
    "PURPOSE: 20% validation sample. Label each Feedback_Text yourself, without looking at any model output.",
    "",
    "TRACEABILITY: Source_Excel_Row is the row number in data/Research_farmers_dataset.xlsx ('Merged Data' sheet);",
    "Participant_Name is that row's respondent. Every row here can be checked against the original survey.",
    "",
    *LABEL_GUIDE,
    "",
    "When finished, save the file in place (same name, same folder) and run scripts/05_evaluate_models.py.",
]

ABSA_GUIDE = [
    "PURPOSE: validation sample for aspect-level sentiment. Independent of Annotation_Worksheet.xlsx.",
    "",
    "TRACEABILITY: Source_Excel_Row = row number in data/Research_farmers_dataset.xlsx ('Merged Data' sheet).",
    "",
    "For each of the six aspect columns:",
    "  - If the text says nothing about that aspect, LEAVE THE CELL BLANK.",
    "  - Otherwise enter Positive, Negative or Neutral.",
    "",
    "  Timeliness        = benefits arriving on time, delays, speed of payment",
    "  Amount_of_Support = whether the money / input amount was enough",
    "  Enrollment_Ease   = registration, application, documents, eligibility",
    "  Official_Behavior = how officials or staff treated the farmer",
    "  Clarity_of_Info   = whether the scheme rules and process were understood",
    "  Overall_Impact    = whether the scheme helped farming or income overall",
    "",
    "Most responses mention only one to three aspects, so most cells stay blank.",
    "Fill Annotator_Initials on EVERY row you have read, even when all six aspect cells are blank -",
    "that is how the evaluation tells 'read, no aspect mentioned' apart from 'not annotated yet'.",
    "",
    "When finished, save the file in place and run scripts/05_evaluate_models.py.",
]


def main(force):
    absa_labels = ASPECT_COLUMNS + ["Annotator_Initials"]
    for path, labels in [(OVERALL_FILE, OVERALL_LABELS), (ABSA_FILE, absa_labels)]:
        if has_labels(path, labels) and not force:
            sys.exit(f"{path.name} already contains annotations; refusing to overwrite (use --force to rebuild).")

    master = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)
    pool = master[master["scheme_feedback_text"].notna()].copy()
    pool["scheme_normalized"] = pool["scheme_normalized"].fillna("Unspecified")

    overall = worksheet_frame(stratified_sample(pool, round(len(pool) * 0.20), seed=42), "R", OVERALL_LABELS)
    absa = worksheet_frame(stratified_sample(pool, 130, seed=99), "A", absa_labels)

    write_worksheet(OVERALL_FILE, overall, {"Label"}, {"Feedback_Text", "Notes_optional"}, OVERALL_GUIDE)
    write_worksheet(ABSA_FILE, absa, set(ASPECT_COLUMNS), {"Feedback_Text"}, ABSA_GUIDE)
    print(f"Overall worksheet: {len(overall)} of {len(pool)} respondents with feedback")
    print(f"ABSA worksheet: {len(absa)} respondents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args().force)
