from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Research_farmers_dataset.xlsx"
SHEET_NAME = "Merged Data"
OUTPUT_DIR = ROOT / "output"
REPORTS_DIR = ROOT / "reports"

MASTER_FILE = OUTPUT_DIR / "sentiment_analysis_results.xlsx"
MASTER_SHEET = "Full Results"

TEXT_COLS = {
    "B5_first_experience": "B5. Describe your experience of first learning about these schemes (e.g., easy to understand, confusing, immediate action taken, etc.)",
    "C4_positive": "C4.Positive aspects of the scheme you experienced",
    "C5_problems": "C5.Problems or challenges faced while accessing or utilizing the scheme",
    "C6_suggestions": "C6.Suggestions for improvement for this specific scheme",
    "D4_challenge": "D4.Biggest challenge faced in farming today (e.g., water shortage, market prices, labor, debt)",
}

COL_A1_NAME = "A1. Name of Participant "
COL_A2_AGE = "A2. Age (in years)"
COL_A3_GENDER = "A3. Gender"
COL_A4_VILLAGE = "A4. Village / Mandal"
COL_A5_DISTRICT = "A5. District"
COL_A6_EDUCATION = "A6. Education Level"
COL_A7_LANDHOLDING = "A7. Landholding size (in acres)"
COL_A8_LAND_TYPE = "A8. Type of land"
COL_A9_CROPS = "A9. Primary crops cultivated (List major crops, separated by commas)"
COL_A10_OCCUPATION = "A10. Primary occupation"
COL_A11_INCOME = "A11. Annual income from agriculture (approx.)"
COL_A13_EXPERIENCE = "A13. Farming experience (years enter a numeric value, months enter a decimals allowed).  ex: 3.4"

COL_B1_AWARE = "B1. Are you aware of any government welfare schemes for farmers?"
COL_B2_HEARD = "B2. Which of the following schemes have you heard about? (Select all that apply)"
COL_B3_ENROLLED = "B3. Which schemes have you personally enrolled in or benefited from? (Select all that apply)"
COL_B4_SOURCE = "B4. How did you first come to know about these schemes?"

COL_C1_SCHEME = "C1. Scheme Name (Specify the scheme you are providing feedback for)"
COL_C2_SATISFACTION = "C2. Overall satisfaction with this specific scheme"

COL_C3_ASPECTS = {
    "timeliness": "C3.Rate the following aspects of the scheme: [Timeliness of benefits]",
    "amount": "C3.Rate the following aspects of the scheme: [Amount of financial support]",
    "enrollment_ease": "C3.Rate the following aspects of the scheme: [Ease of enrollment process]",
    "official_behavior": "C3.Rate the following aspects of the scheme: [Behavior of officials]",
    "clarity": "C3.Rate the following aspects of the scheme: [Clarity of information]",
    "impact": "C3.Rate the following aspects of the scheme: [Overall impact on farming]",
}

ASPECT_LABELS = {
    "timeliness": "Timeliness",
    "amount": "Amount of support",
    "enrollment_ease": "Enrollment ease",
    "official_behavior": "Official behavior",
    "clarity": "Clarity of information",
    "impact": "Overall impact",
}

COL_D1_GOVT_SUPPORT = "D1. Overall rating of government support for farmers in your area"
COL_D2_TREND = "D2. Compared to 5 years ago, support for farmers has:"
COL_D3_CONFIDENCE = "D3. Confidence in future improvement (How confident are you that government support will improve in the next 5 years?)"

COL_E1_LANGUAGE = "E1.Language of interview"
COL_E2_COMFORT = "E2.Participant comfort level during the interview"
COL_E4_DURATION = "E4.Interview duration (in minutes)"

LIKERT_5PT_MAP = {"very poor": -2, "poor": -1, "average": 0, "good": 1, "excellent": 2}
SATISFACTION_MAP = {"very dissatisfied": -2, "dissatisfied": -1, "neutral": 0, "satisfied": 1, "very satisfied": 2}
TREND_MAP = {
    "worsened significantly": -2, "worsened slightly": -1, "stayed the same": 0,
    "improved slightly": 1, "improved significantly": 2,
}
CONFIDENCE_MAP = {
    "not at all confident": -2, "not very confident": -1, "neutral": 0,
    "somewhat confident": 1, "very confident": 2,
}

CANONICAL_SCHEMES = [
    "PM-KISAN",
    "Rythu Bharosa",
    "Kisan Credit Card",
    "Crop Insurance",
    "Input Subsidy",
    "Minimum Support Price",
    "Soil Health Card",
]
