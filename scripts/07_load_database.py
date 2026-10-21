import re
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from src.absa import ASPECT_KEYWORDS
from src.config import (
    ASPECT_LABELS, CANONICAL_SCHEMES, COL_A1_NAME, COL_A3_GENDER, COL_A4_VILLAGE, COL_A5_DISTRICT, COL_A6_EDUCATION,
    COL_A7_LANDHOLDING, COL_A8_LAND_TYPE, COL_A9_CROPS, COL_A10_OCCUPATION, COL_A11_INCOME, COL_B1_AWARE,
    COL_B2_HEARD, COL_B3_ENROLLED, COL_B4_SOURCE, COL_C1_SCHEME, COL_C2_SATISFACTION, COL_C3_ASPECTS,
    COL_D1_GOVT_SUPPORT, COL_D2_TREND, COL_D3_CONFIDENCE, COL_E1_LANGUAGE, COL_E2_COMFORT, COL_E4_DURATION,
    CONFIDENCE_MAP, DATA_FILE, LIKERT_5PT_MAP, MASTER_FILE, MASTER_SHEET, OUTPUT_DIR, ROOT, SHEET_NAME, TREND_MAP,
)
from src.descriptive import scheme_table
from src.evaluation import normalise_labels
from src.preprocess import SCHEME_ABBREVIATIONS, clean_text, normalize_scheme_name
from src.schemes import SCHEME_CATALOGUE, schemes_in
from src.sentiment_lexicon import AGRI_DOMAIN_LEXICON, NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD
from src.sentiment_transformer import MAX_TOKENS, MODEL_NAME
from src.topic_model import LDA_PARAMS
from src.triangulate import likert_to_numeric

ENV_FILE = ROOT / "db" / ".env"
SCHEMA_FILE = ROOT / "db" / "schema.sql"
TOPIC_FILE = OUTPUT_DIR / "topic_modeling_results.xlsx"
VERIFICATION_FILE = OUTPUT_DIR / "database_verification.csv"
OVERALL_WORKSHEET = OUTPUT_DIR / "Annotation_Worksheet.xlsx"
ABSA_WORKSHEET = OUTPUT_DIR / "ABSA_Annotation_Worksheet.xlsx"

STUDY_DISTRICT, STUDY_STATE = "Chittoor", "Andhra Pradesh"
COL_E3 = "E3.Non-verbal observations (e.g., mood, level of engagement, reluctance to answer certain questions)"
STAGING_COLUMNS = [
    "a1_name", "a2_age", "a3_gender", "a4_village", "a5_district", "a6_education", "a7_landholding", "a8_land_type",
    "a9_crops", "a10_occupation", "a11_income", "a13_experience", "b1_aware", "b2_heard", "b3_enrolled", "b4_source",
    "b5_first_experience", "c1_scheme", "c2_satisfaction", "c3_timeliness", "c3_amount", "c3_enrollment_ease",
    "c3_official_behavior", "c3_clarity", "c3_impact", "c4_positive", "c5_problems", "c6_suggestions", "d1_support",
    "d2_trend", "d3_confidence", "d4_challenge", "e1_language", "e2_comfort", "e3_observations", "e4_duration",
]
TEXT_FIELD_CODES = {"B5_first_experience": "B5", "C4_positive": "C4", "C5_problems": "C5", "C6_suggestions": "C6", "D4_challenge": "D4"}
SOURCE_RENAMES = {"Volunteers": "Volunteer", "Through The Friend": "Friend"}
ASPECT_WORKSHEET_COLUMNS = {
    "timeliness": "Timeliness", "amount": "Amount_of_Support", "enrollment_ease": "Enrollment_Ease",
    "official_behavior": "Official_Behavior", "clarity": "Clarity_of_Info", "impact": "Overall_Impact",
}
LABELS = {"Positive", "Neutral", "Negative"}


def read_env():
    pairs = (line.split("=", 1) for line in ENV_FILE.read_text(encoding="utf-8").splitlines() if "=" in line)
    return {k.strip(): v.strip() for k, v in pairs}


def value(v):
    if v is None or (not isinstance(v, (list, dict, set)) and pd.isna(v)):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def rounded(v, digits):
    v = value(v)
    return None if v is None else round(float(v), digits)


def connect(env, dbname):
    return psycopg.connect(host=env["PGHOST"], port=env["PGPORT"], user=env["PGUSER"], password=env["PGPASSWORD"], dbname=dbname)


def ensure_database(env):
    with connect(env, "postgres") as conn:
        conn.autocommit = True
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (env["PGDATABASE"],)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(env["PGDATABASE"])))


def insert(cur, table, columns, rows, returning=None):
    query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.SQL(table), sql.SQL(", ").join(map(sql.Identifier, columns)), sql.SQL(", ").join(sql.Placeholder() * len(columns)),
    )
    rows = [tuple(value(v) for v in row) for row in rows]
    if returning:
        query += sql.SQL(" RETURNING {}").format(sql.Identifier(returning))
        return [cur.execute(query, row).fetchone()[0] for row in rows]
    cur.executemany(query, rows)
    return len(rows)


def title_case(text):
    text = clean_text(text)
    return " ".join(text.split()).title() if text else None


def info_source_name(raw):
    text = clean_text(raw)
    if not text:
        return None
    text = " ".join(text.split())
    if text.isupper() or text.islower():
        text = text.title()
    return SOURCE_RENAMES.get(text, text)


def alias_match(raw):
    text = clean_text(raw)
    first = re.split(r"[,/&]| and ", text)[0].strip() or text
    canonical = normalize_scheme_name(raw)
    if first.lower() in SCHEME_ABBREVIATIONS:
        return canonical, "abbreviation", 100.0
    _, score, _ = process.extractOne(first, CANONICAL_SCHEMES, scorer=fuzz.WRatio, processor=default_process)
    if canonical == "Other / Unmatched":
        return canonical, "unmatched", round(score, 1)
    method = "exact" if default_process(first) == default_process(canonical) else "fuzzy"
    return canonical, method, round(score, 1)


def load_staging(cur, raw):
    rows = [[i + 2] + [None if pd.isna(v) else str(v) for v in raw.iloc[i]] for i in range(len(raw))]
    return insert(cur, "staging.raw_survey", ["source_excel_row"] + STAGING_COLUMNS, rows)


def load_locations(cur, master, flags):
    districts = {STUDY_DISTRICT}
    districts.update(d for d in master[COL_A5_DISTRICT].apply(title_case).dropna())
    district_ids = dict(zip(sorted(districts), insert(cur, "survey.district", ["district_name", "state"],
                                                       [(d, STUDY_STATE) for d in sorted(districts)], returning="district_id")))
    village_district = {}
    for i, row in master.iterrows():
        village = title_case(row[COL_A4_VILLAGE])
        if not village:
            continue
        district = title_case(row[COL_A5_DISTRICT])
        if district is None:
            flags.append((i + 1, "district_inferred", "A5 District", f"District blank; set to study district {STUDY_DISTRICT} (village {village})"))
            district = STUDY_DISTRICT
        village_district.setdefault((village, district), None)
    keys = sorted(village_district)
    village_ids = dict(zip(keys, insert(cur, "survey.village", ["village_name", "district_id"],
                                        [(v, district_ids[d]) for v, d in keys], returning="village_id")))
    return district_ids, village_ids


def respondent_village(row, village_ids):
    village = title_case(row[COL_A4_VILLAGE])
    if not village:
        return None
    return village_ids[(village, title_case(row[COL_A5_DISTRICT]) or STUDY_DISTRICT)]


def load_respondents(cur, master, village_ids, flags):
    sources = sorted({s for s in master[COL_B4_SOURCE].apply(info_source_name).dropna()})
    source_ids = dict(zip(sources, insert(cur, "survey.info_source", ["source_name"], [(s,) for s in sources], returning="info_source_id")))

    rows, interviews = [], []
    for i, row in master.iterrows():
        rid = i + 1
        age = value(row["age_numeric_clean"])
        if age is not None and not 15 <= age <= 100:
            flags.append((rid, "age_set_null", "A2 Age", f"Recorded age {age} outside 15-100; stored as NULL"))
            age = None
        land = value(pd.to_numeric(row[COL_A7_LANDHOLDING], errors="coerce"))
        aware = clean_text(row[COL_B1_AWARE])
        education = clean_text(row[COL_A6_EDUCATION])
        rows.append((
            rid, i + 2, clean_text(row[COL_A1_NAME]), None if age is None else int(round(age)),
            clean_text(row[COL_A3_GENDER]), "Diploma" if education and education.lower() == "diploma" else education,
            None if land is None or land < 0 else round(land, 2), clean_text(row[COL_A8_LAND_TYPE]),
            clean_text(row[COL_A10_OCCUPATION]), clean_text(row[COL_A11_INCOME]),
            rounded(row["farming_experience_clean"], 1),
            None if aware is None else aware.lower().startswith("yes"),
            row["B5_first_experience_clean"],
            likert_to_numeric(row[COL_D1_GOVT_SUPPORT], LIKERT_5PT_MAP),
            likert_to_numeric(row[COL_D2_TREND], TREND_MAP),
            likert_to_numeric(row[COL_D3_CONFIDENCE], CONFIDENCE_MAP),
            row["D4_challenge_clean"],
            respondent_village(row, village_ids),
            source_ids.get(info_source_name(row[COL_B4_SOURCE])),
        ))
        duration = value(pd.to_numeric(row[COL_E4_DURATION], errors="coerce"))
        interview = (clean_text(row[COL_E1_LANGUAGE]), clean_text(row[COL_E2_COMFORT]), clean_text(row[COL_E3]),
                     None if duration is None or duration <= 0 else int(round(duration)))
        if any(v is not None for v in interview):
            interviews.append((rid,) + interview)

    insert(cur, "survey.respondent", [
        "respondent_id", "source_excel_row", "name", "age", "gender", "education", "landholding_acres", "land_type",
        "occupation", "income_band", "farming_experience_yrs", "is_aware", "first_experience_text", "govt_support_score",
        "support_trend_score", "future_confidence_score", "biggest_challenge_text", "village_id", "info_source_id",
    ], rows)
    insert(cur, "survey.interview", ["respondent_id", "language", "comfort_level", "observations", "duration_min"], interviews)


def load_crops(cur, master):
    per_respondent = {}
    for i, entry in master[COL_A9_CROPS].items():
        if pd.isna(entry):
            continue
        crops = {" ".join(c.split()).title() for c in str(entry).split(",") if c.strip()}
        if crops:
            per_respondent[i + 1] = crops
    names = sorted(set().union(*per_respondent.values()))
    crop_ids = dict(zip(names, insert(cur, "survey.crop", ["crop_name"], [(n,) for n in names], returning="crop_id")))
    insert(cur, "survey.respondent_crop", ["respondent_id", "crop_id"],
           [(rid, crop_ids[c]) for rid, crops in per_respondent.items() for c in sorted(crops)])


def load_schemes(cur, raw, master):
    scheme_ids = dict(zip(
        [s[0] for s in SCHEME_CATALOGUE],
        insert(cur, "survey.scheme", ["scheme_name", "govt_level", "category"], [s[:3] for s in SCHEME_CATALOGUE], returning="scheme_id"),
    ))

    aliases = []
    for alias in sorted(raw[COL_C1_SCHEME].dropna().astype(str).unique()):
        if normalize_scheme_name(alias) is None:
            continue
        canonical, method, score = alias_match(alias)
        aliases.append((alias, scheme_ids[canonical], method, score))
    insert(cur, "survey.scheme_alias", ["alias_text", "scheme_id", "match_method", "match_score"], aliases)

    awareness = []
    for i, row in master.iterrows():
        heard, enrolled = schemes_in(row[COL_B2_HEARD]), schemes_in(row[COL_B3_ENROLLED])
        awareness += [(i + 1, scheme_ids[s], s in heard, s in enrolled) for s in sorted(heard | enrolled)]
    insert(cur, "survey.scheme_awareness", ["respondent_id", "scheme_id", "has_heard", "is_enrolled"], awareness)
    return scheme_ids, len(aliases)


def load_feedback(cur, master, scheme_ids):
    c3_numeric = [f"C3_{a}_numeric" for a in COL_C3_ASPECTS]
    has_feedback = (master["scheme_normalized"].notna() | master["scheme_feedback_text"].notna()
                    | master[COL_C2_SATISFACTION].notna() | master[c3_numeric].notna().any(axis=1))
    feedback_ids = {}
    for i, row in master[has_feedback].iterrows():
        scheme = row["scheme_normalized"] if pd.notna(row["scheme_normalized"]) else "Unspecified"
        feedback_ids[i + 1] = insert(cur, "survey.scheme_feedback", [
            "respondent_id", "scheme_id", "satisfaction_score", "positive_text", "problems_text", "suggestions_text",
        ], [(i + 1, scheme_ids[scheme], None if pd.isna(row["C2_satisfaction_numeric"]) else int(row["C2_satisfaction_numeric"]), row["C4_positive_clean"],
             row["C5_problems_clean"], row["C6_suggestions_clean"])], returning="feedback_id")[0]

    aspect_ids = {aspect: n for n, aspect in enumerate(COL_C3_ASPECTS, start=1)}
    insert(cur, "survey.aspect", ["aspect_id", "aspect_name"], [(n, ASPECT_LABELS[a]) for a, n in aspect_ids.items()])
    ratings = [
        (feedback_ids[i + 1], aspect_ids[a], int(row[f"C3_{a}_numeric"]))
        for i, row in master[has_feedback].iterrows() for a in COL_C3_ASPECTS if pd.notna(row[f"C3_{a}_numeric"])
    ]
    insert(cur, "survey.aspect_rating", ["feedback_id", "aspect_id", "rating_score"], ratings)
    return feedback_ids, aspect_ids


def load_models(cur):
    models = [
        ("VADER + agri lexicon", "Lexicon", f"NLTK {version('nltk')} VADER",
         {"positive_threshold": POSITIVE_THRESHOLD, "negative_threshold": NEGATIVE_THRESHOLD, "domain_terms": len(AGRI_DOMAIN_LEXICON)}),
        ("RoBERTa sentiment (zero-shot)", "Transformer", f"{MODEL_NAME} / transformers {version('transformers')}",
         {"max_tokens": MAX_TOKENS, "fine_tuned_on_this_data": False}),
        ("ABSA clause keywords + VADER", "ABSA", f"NLTK {version('nltk')} VADER",
         {"aspects": list(ASPECT_KEYWORDS), "keywords_per_aspect": {a: len(k) for a, k in ASPECT_KEYWORDS.items()}}),
        ("LDA topic model", "Topic model", f"scikit-learn {version('scikit-learn')}", LDA_PARAMS),
    ]
    ids = insert(cur, "analytics.nlp_model", ["model_name", "method_type", "version", "parameters"],
                 [(n, t, v, Jsonb(p)) for n, t, v, p in models], returning="model_id")
    return dict(zip(["vader", "transformer", "absa", "lda"], ids))


def load_sentiment(cur, master, model_ids, feedback_ids, aspect_ids):
    rows = []
    for i, row in master.iterrows():
        rid = i + 1
        for key, code in TEXT_FIELD_CODES.items():
            if pd.notna(row[f"{key}_vader_score"]):
                rows.append((rid, model_ids["vader"], code, rounded(row[f"{key}_vader_score"], 4), row[f"{key}_vader_label"]))
        if pd.notna(row["vader_on_feedback_score"]):
            rows.append((rid, model_ids["vader"], "COMBINED", rounded(row["vader_on_feedback_score"], 4), row["vader_on_feedback_label"]))
        if pd.notna(row.get("transformer_score")):
            rows.append((rid, model_ids["transformer"], "COMBINED", rounded(row["transformer_score"], 4), row["transformer_label"]))
    insert(cur, "analytics.text_sentiment", ["respondent_id", "model_id", "source_field", "score", "label"], rows)

    aspects = [
        (feedback_ids[i + 1], aspect_ids[a], model_ids["absa"], rounded(row[f"absa_{a}_score"], 4), row[f"absa_{a}_label"])
        for i, row in master.iterrows() for a in COL_C3_ASPECTS if pd.notna(row[f"absa_{a}_score"])
    ]
    insert(cur, "analytics.aspect_sentiment", ["feedback_id", "aspect_id", "model_id", "score", "label"], aspects)
    return len(rows), len(aspects)


def load_topics(cur, model_ids, feedback_ids):
    book = pd.ExcelFile(TOPIC_FILE)
    topic_ids = {}
    for corpus, sheet in [("C4", "Satisfaction Topics"), ("C5", "Dissatisfaction Topics")]:
        topics = book.parse(sheet)
        ids = insert(cur, "analytics.topic", ["model_id", "corpus", "topic_number", "topic_label", "corpus_share"],
                     [(model_ids["lda"], corpus, t.topic_id, t.label, t.share) for t in topics.itertuples()], returning="topic_id")
        topic_ids.update({(corpus, t): tid for t, tid in zip(topics["topic_id"], ids)})

    terms = book.parse("Topic Terms")
    insert(cur, "analytics.topic_term", ["topic_id", "term_rank", "term", "weight"],
           [(topic_ids[(t.corpus, t.topic_id)], t.term_rank, t.term, t.weight) for t in terms.itertuples()])

    docs = book.parse("Document Topics")
    dominant = docs.loc[docs.groupby(["source_excel_row", "corpus"])["weight"].idxmax()].index
    docs["is_dominant"] = docs.index.isin(dominant)
    rows = [(feedback_ids[d.source_excel_row - 1], topic_ids[(d.corpus, d.topic_id)], d.weight, d.is_dominant)
            for d in docs.itertuples()]
    insert(cur, "analytics.feedback_topic", ["feedback_id", "topic_id", "weight", "is_dominant"], rows)
    return len(rows)


def load_quality_flags(cur, flags):
    numeric = pd.read_csv(OUTPUT_DIR / "data_quality_numeric_issues.csv")
    logic = pd.read_csv(OUTPUT_DIR / "data_quality_logic_issues.csv")
    types = {"B1 vs C1": "unaware_but_rated", "B2 vs B3": "enrolled_not_heard"}
    rows = [(r.source_excel_row - 1, "implausible_value", r.field, f"{r.issue} ({r.value})") for r in numeric.itertuples()]
    rows += [(r.source_excel_row - 1, types[r.field], r.field, f"{r.issue}: {r.value}") for r in logic.itertuples()]
    rows += flags
    insert(cur, "analytics.quality_flag", ["respondent_id", "flag_type", "field_name", "description"], rows)
    return len(rows)


def worksheet_labels(path, columns):
    if not path.exists():
        return pd.DataFrame()
    sheet = pd.read_excel(path, sheet_name="To Annotate").rename(columns={"Label": "Human_Label"})
    for col in columns:
        sheet[col] = normalise_labels(sheet[col])
    return sheet


def load_annotations(cur, feedback_ids, aspect_ids):
    overall = worksheet_labels(OVERALL_WORKSHEET, ["Human_Label"])
    absa = worksheet_labels(ABSA_WORKSHEET, list(ASPECT_WORKSHEET_COLUMNS.values()))
    entries = []
    if not overall.empty:
        for r in overall[overall["Human_Label"].isin(LABELS)].itertuples():
            entries.append((r.Source_Excel_Row, None, r.Human_Label, r.Annotator_Initials))
    if not absa.empty:
        for r in absa.itertuples():
            for aspect, col in ASPECT_WORKSHEET_COLUMNS.items():
                label = getattr(r, col)
                if label in LABELS:
                    entries.append((r.Source_Excel_Row, aspect_ids[aspect], label, r.Annotator_Initials))
    if not entries:
        return 0

    initials = sorted({str(e[3]).strip().upper() if pd.notna(e[3]) else "R1" for e in entries})
    annotator_ids = dict(zip(initials, insert(cur, "analytics.annotator", ["initials", "role"],
                                              [(i, "Human annotator") for i in initials], returning="annotator_id")))
    rows = [(feedback_ids[row - 1], aspect, annotator_ids[str(ini).strip().upper() if pd.notna(ini) else "R1"], label)
            for row, aspect, label, ini in entries]
    return insert(cur, "analytics.annotation", ["feedback_id", "aspect_id", "annotator_id", "human_label"], rows)


def verify(conn, master):
    checks = []

    def check(name, expected, actual):
        checks.append({"check": name, "expected": expected, "actual": actual, "status": "PASS" if expected == actual else "FAIL"})

    check("respondents", len(master), conn.execute("SELECT count(*) FROM survey.respondent").fetchone()[0])
    check("staging rows", len(master), conn.execute("SELECT count(*) FROM staging.raw_survey").fetchone()[0])

    expected_labels = master["overall_vader_label"].value_counts().to_dict()
    actual_labels = dict(conn.execute("SELECT overall_label, count(*) FROM reporting.v_respondent_sentiment GROUP BY 1").fetchall())
    for label in ("Positive", "Neutral", "Negative"):
        check(f"overall sentiment {label}", int(expected_labels.get(label, 0)), int(actual_labels.get(label, 0)))

    divergence = pd.read_csv(OUTPUT_DIR / "divergence_summary.csv")
    names = {"divergence_overall": "Overall satisfaction (C2)", **{f"divergence_{a}": ASPECT_LABELS[a] for a in COL_C3_ASPECTS}}
    db = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT comparison, n_comparable, n_divergent FROM reporting.v_aspect_divergence")}
    for r in divergence.itertuples():
        check(f"divergence {names[r.comparison]}", (int(r.n_comparable), int(r.n_divergent)), tuple(int(x) for x in db.get(names[r.comparison], (-1, -1))))

    expected_schemes = scheme_table(master).set_index("Scheme")
    db_schemes = {r[0]: (r[1], float(r[2]) if r[2] is not None else None)
                  for r in conn.execute("SELECT scheme_name, n_feedback, pct_negative FROM reporting.v_scheme_summary")}
    for scheme, r in expected_schemes.iterrows():
        n_db = conn.execute(
            "SELECT count(*) FROM survey.scheme_feedback f JOIN survey.scheme s USING (scheme_id) WHERE s.scheme_name = %s", (scheme,)
        ).fetchone()[0]
        check(f"scheme {scheme} feedback rows", int(r["N"]), int(n_db))
        check(f"scheme {scheme} % negative", float(r["pct_negative"]), db_schemes.get(scheme, (None, None))[1])

    heard = master[COL_B2_HEARD].apply(schemes_in)
    enrolled = master[COL_B3_ENROLLED].apply(schemes_in)
    db_gap = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT scheme_name, heard, enrolled FROM reporting.v_awareness_gap")}
    for scheme in sorted(set().union(*heard)):
        expected = (int(sum(scheme in h for h in heard)), int(sum(scheme in e for e in enrolled)))
        check(f"awareness {scheme}", expected, tuple(int(x) for x in db_gap.get(scheme, (-1, -1))))

    report = pd.DataFrame(checks)
    report.to_csv(VERIFICATION_FILE, index=False)
    return report


def main():
    env = read_env()
    raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    master = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)

    ensure_database(env)
    with connect(env, env["PGDATABASE"]) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
            flags = []
            load_staging(cur, raw)
            _, village_ids = load_locations(cur, master, flags)
            load_respondents(cur, master, village_ids, flags)
            load_crops(cur, master)
            scheme_ids, n_alias = load_schemes(cur, raw, master)
            feedback_ids, aspect_ids = load_feedback(cur, master, scheme_ids)
            model_ids = load_models(cur)
            n_text, n_aspect = load_sentiment(cur, master, model_ids, feedback_ids, aspect_ids)
            n_doc_topics = load_topics(cur, model_ids, feedback_ids)
            n_flags = load_quality_flags(cur, flags)
            n_annotations = load_annotations(cur, feedback_ids, aspect_ids)
        conn.commit()

        counts = conn.execute("""
            SELECT table_schema || '.' || table_name FROM information_schema.tables
            WHERE table_schema IN ('staging', 'survey', 'analytics') AND table_type = 'BASE TABLE' ORDER BY 1
        """).fetchall()
        for (table,) in counts:
            n = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.SQL(table))).fetchone()[0]
            print(f"  {table:34s} {n:6d}")
        print(f"Loaded {n_alias} scheme spellings, {n_text} text scores, {n_aspect} aspect scores, "
              f"{n_doc_topics} document-topic weights, {n_flags} quality flags, {n_annotations} human labels")

        report = verify(conn, master)
    failed = report[report["status"] == "FAIL"]
    print(f"Verification: {len(report) - len(failed)}/{len(report)} checks passed -> {VERIFICATION_FILE.name}")
    if not failed.empty:
        print(failed.to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
