-- Farmer welfare scheme feedback database (PostgreSQL 16)
-- Layers: staging (raw sheet as received) -> survey (normalised, 3NF) -> analytics (NLP output, validation) -> reporting (views)

DROP SCHEMA IF EXISTS reporting CASCADE;
DROP SCHEMA IF EXISTS analytics CASCADE;
DROP SCHEMA IF EXISTS survey CASCADE;
DROP SCHEMA IF EXISTS staging CASCADE;

CREATE SCHEMA staging;
CREATE SCHEMA survey;
CREATE SCHEMA analytics;
CREATE SCHEMA reporting;

---------------------------------------------------------------- staging
CREATE TABLE staging.raw_survey (
    source_excel_row     INT PRIMARY KEY,
    a1_name              TEXT,
    a2_age               TEXT,
    a3_gender            TEXT,
    a4_village           TEXT,
    a5_district          TEXT,
    a6_education         TEXT,
    a7_landholding       TEXT,
    a8_land_type         TEXT,
    a9_crops             TEXT,
    a10_occupation       TEXT,
    a11_income           TEXT,
    a13_experience       TEXT,
    b1_aware             TEXT,
    b2_heard             TEXT,
    b3_enrolled          TEXT,
    b4_source            TEXT,
    b5_first_experience  TEXT,
    c1_scheme            TEXT,
    c2_satisfaction      TEXT,
    c3_timeliness        TEXT,
    c3_amount            TEXT,
    c3_enrollment_ease   TEXT,
    c3_official_behavior TEXT,
    c3_clarity           TEXT,
    c3_impact            TEXT,
    c4_positive          TEXT,
    c5_problems          TEXT,
    c6_suggestions       TEXT,
    d1_support           TEXT,
    d2_trend             TEXT,
    d3_confidence        TEXT,
    d4_challenge         TEXT,
    e1_language          TEXT,
    e2_comfort           TEXT,
    e3_observations      TEXT,
    e4_duration          TEXT
);

---------------------------------------------------------------- survey
CREATE TABLE survey.district (
    district_id   SERIAL PRIMARY KEY,
    district_name VARCHAR(60) NOT NULL UNIQUE,
    state         VARCHAR(60) NOT NULL
);

CREATE TABLE survey.village (
    village_id   SERIAL PRIMARY KEY,
    village_name VARCHAR(120) NOT NULL,
    district_id  INT NOT NULL REFERENCES survey.district ON DELETE RESTRICT,
    UNIQUE (village_name, district_id)
);

CREATE TABLE survey.info_source (
    info_source_id SERIAL PRIMARY KEY,
    source_name    VARCHAR(80) NOT NULL UNIQUE
);

CREATE TABLE survey.respondent (
    respondent_id           INT PRIMARY KEY,
    source_excel_row        INT NOT NULL UNIQUE,
    name                    VARCHAR(120),
    age                     SMALLINT CHECK (age BETWEEN 15 AND 100),
    gender                  VARCHAR(20),
    education               VARCHAR(60),
    landholding_acres       NUMERIC(7,2) CHECK (landholding_acres >= 0),
    land_type               VARCHAR(20),
    occupation              VARCHAR(80),
    income_band             VARCHAR(40),
    farming_experience_yrs  NUMERIC(4,1) CHECK (farming_experience_yrs >= 0),
    is_aware                BOOLEAN,
    first_experience_text   TEXT,
    govt_support_score      SMALLINT CHECK (govt_support_score BETWEEN -2 AND 2),
    support_trend_score     SMALLINT CHECK (support_trend_score BETWEEN -2 AND 2),
    future_confidence_score SMALLINT CHECK (future_confidence_score BETWEEN -2 AND 2),
    biggest_challenge_text  TEXT,
    village_id              INT REFERENCES survey.village ON DELETE RESTRICT,
    info_source_id          INT REFERENCES survey.info_source ON DELETE RESTRICT,
    CHECK (farming_experience_yrs IS NULL OR age IS NULL OR farming_experience_yrs <= age)
);

CREATE TABLE survey.interview (
    respondent_id INT PRIMARY KEY REFERENCES survey.respondent ON DELETE CASCADE,
    language      VARCHAR(20),
    comfort_level VARCHAR(30),
    observations  TEXT,
    duration_min  SMALLINT CHECK (duration_min > 0)
);

CREATE TABLE survey.crop (
    crop_id   SERIAL PRIMARY KEY,
    crop_name VARCHAR(80) NOT NULL UNIQUE
);

CREATE TABLE survey.respondent_crop (
    respondent_id INT REFERENCES survey.respondent ON DELETE CASCADE,
    crop_id       INT REFERENCES survey.crop ON DELETE RESTRICT,
    PRIMARY KEY (respondent_id, crop_id)
);

CREATE TABLE survey.scheme (
    scheme_id   SERIAL PRIMARY KEY,
    scheme_name VARCHAR(60) NOT NULL UNIQUE,
    govt_level  VARCHAR(10) CHECK (govt_level IN ('Central', 'State')),
    category    VARCHAR(40)
);

CREATE TABLE survey.scheme_alias (
    alias_text   VARCHAR(200) PRIMARY KEY,
    scheme_id    INT NOT NULL REFERENCES survey.scheme ON DELETE CASCADE,
    match_method VARCHAR(12) NOT NULL CHECK (match_method IN ('exact', 'abbreviation', 'fuzzy', 'unmatched')),
    match_score  NUMERIC(5,1) CHECK (match_score BETWEEN 0 AND 100)
);

CREATE TABLE survey.scheme_awareness (
    respondent_id INT REFERENCES survey.respondent ON DELETE CASCADE,
    scheme_id     INT REFERENCES survey.scheme ON DELETE RESTRICT,
    has_heard     BOOLEAN NOT NULL,
    is_enrolled   BOOLEAN NOT NULL,
    PRIMARY KEY (respondent_id, scheme_id),
    CHECK (has_heard OR is_enrolled)
);

CREATE TABLE survey.scheme_feedback (
    feedback_id        SERIAL PRIMARY KEY,
    respondent_id      INT NOT NULL REFERENCES survey.respondent ON DELETE CASCADE,
    scheme_id          INT NOT NULL REFERENCES survey.scheme ON DELETE RESTRICT,
    satisfaction_score SMALLINT CHECK (satisfaction_score BETWEEN -2 AND 2),
    positive_text      TEXT,
    problems_text      TEXT,
    suggestions_text   TEXT,
    UNIQUE (respondent_id, scheme_id)
);

CREATE TABLE survey.aspect (
    aspect_id   SMALLINT PRIMARY KEY,
    aspect_name VARCHAR(40) NOT NULL UNIQUE
);

CREATE TABLE survey.aspect_rating (
    feedback_id  INT REFERENCES survey.scheme_feedback ON DELETE CASCADE,
    aspect_id    SMALLINT REFERENCES survey.aspect ON DELETE RESTRICT,
    rating_score SMALLINT NOT NULL CHECK (rating_score BETWEEN -2 AND 2),
    PRIMARY KEY (feedback_id, aspect_id)
);

CREATE INDEX idx_respondent_village ON survey.respondent (village_id);
CREATE INDEX idx_respondent_source ON survey.respondent (info_source_id);
CREATE INDEX idx_awareness_scheme ON survey.scheme_awareness (scheme_id);
CREATE INDEX idx_feedback_scheme ON survey.scheme_feedback (scheme_id);
CREATE INDEX idx_feedback_fulltext ON survey.scheme_feedback USING GIN (
    to_tsvector('english', coalesce(positive_text, '') || ' ' || coalesce(problems_text, '') || ' ' || coalesce(suggestions_text, ''))
);

---------------------------------------------------------------- analytics
CREATE TABLE analytics.nlp_model (
    model_id    SERIAL PRIMARY KEY,
    model_name  VARCHAR(80) NOT NULL UNIQUE,
    method_type VARCHAR(20) NOT NULL CHECK (method_type IN ('Lexicon', 'Transformer', 'ABSA', 'Topic model')),
    version     VARCHAR(120),
    parameters  JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE analytics.text_sentiment (
    sentiment_id  SERIAL PRIMARY KEY,
    respondent_id INT NOT NULL REFERENCES survey.respondent ON DELETE CASCADE,
    model_id      INT NOT NULL REFERENCES analytics.nlp_model ON DELETE CASCADE,
    source_field  VARCHAR(10) NOT NULL CHECK (source_field IN ('B5', 'C4', 'C5', 'C6', 'D4', 'COMBINED')),
    score         NUMERIC(6,4) NOT NULL CHECK (score BETWEEN -1 AND 1),
    label         VARCHAR(8) NOT NULL CHECK (label IN ('Positive', 'Neutral', 'Negative')),
    scored_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (respondent_id, model_id, source_field)
);

CREATE TABLE analytics.aspect_sentiment (
    feedback_id INT REFERENCES survey.scheme_feedback ON DELETE CASCADE,
    aspect_id   SMALLINT REFERENCES survey.aspect ON DELETE RESTRICT,
    model_id    INT REFERENCES analytics.nlp_model ON DELETE CASCADE,
    score       NUMERIC(6,4) NOT NULL CHECK (score BETWEEN -1 AND 1),
    label       VARCHAR(8) NOT NULL CHECK (label IN ('Positive', 'Neutral', 'Negative')),
    PRIMARY KEY (feedback_id, aspect_id, model_id)
);

CREATE TABLE analytics.topic (
    topic_id     SERIAL PRIMARY KEY,
    model_id     INT NOT NULL REFERENCES analytics.nlp_model ON DELETE CASCADE,
    corpus       CHAR(2) NOT NULL CHECK (corpus IN ('C4', 'C5')),
    topic_number SMALLINT NOT NULL,
    topic_label  VARCHAR(80) NOT NULL,
    corpus_share NUMERIC(5,4) CHECK (corpus_share BETWEEN 0 AND 1),
    UNIQUE (model_id, corpus, topic_number)
);

CREATE TABLE analytics.topic_term (
    topic_id  INT REFERENCES analytics.topic ON DELETE CASCADE,
    term_rank SMALLINT CHECK (term_rank > 0),
    term      VARCHAR(40) NOT NULL,
    weight    NUMERIC(7,5) NOT NULL,
    PRIMARY KEY (topic_id, term_rank)
);

CREATE TABLE analytics.feedback_topic (
    feedback_id INT REFERENCES survey.scheme_feedback ON DELETE CASCADE,
    topic_id    INT REFERENCES analytics.topic ON DELETE CASCADE,
    weight      NUMERIC(5,4) NOT NULL CHECK (weight BETWEEN 0 AND 1),
    is_dominant BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (feedback_id, topic_id)
);

CREATE TABLE analytics.quality_flag (
    flag_id       SERIAL PRIMARY KEY,
    respondent_id INT NOT NULL REFERENCES survey.respondent ON DELETE CASCADE,
    flag_type     VARCHAR(30) NOT NULL,
    field_name    VARCHAR(40) NOT NULL,
    description   TEXT NOT NULL,
    flagged_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE analytics.annotator (
    annotator_id SERIAL PRIMARY KEY,
    initials     VARCHAR(10) NOT NULL UNIQUE,
    role         VARCHAR(40)
);

CREATE TABLE analytics.annotation (
    annotation_id SERIAL PRIMARY KEY,
    feedback_id   INT NOT NULL REFERENCES survey.scheme_feedback ON DELETE CASCADE,
    aspect_id     SMALLINT REFERENCES survey.aspect ON DELETE RESTRICT,
    annotator_id  INT NOT NULL REFERENCES analytics.annotator ON DELETE RESTRICT,
    human_label   VARCHAR(8) NOT NULL CHECK (human_label IN ('Positive', 'Neutral', 'Negative')),
    annotated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (feedback_id, aspect_id, annotator_id)
);

CREATE INDEX idx_text_sentiment_model ON analytics.text_sentiment (model_id, source_field);
CREATE INDEX idx_quality_flag_respondent ON analytics.quality_flag (respondent_id);

---------------------------------------------------------------- reporting
CREATE VIEW reporting.v_respondent_sentiment AS
SELECT ts.respondent_id,
       round(avg(ts.score), 4) AS overall_score,
       CASE WHEN avg(ts.score) > 0.05 THEN 'Positive'
            WHEN avg(ts.score) < -0.05 THEN 'Negative'
            ELSE 'Neutral' END AS overall_label
FROM analytics.text_sentiment ts
JOIN analytics.nlp_model m USING (model_id)
WHERE m.method_type = 'Lexicon' AND ts.source_field <> 'COMBINED'
GROUP BY ts.respondent_id;

CREATE VIEW reporting.v_awareness_gap AS
SELECT s.scheme_name,
       count(*) FILTER (WHERE a.has_heard)   AS heard,
       count(*) FILTER (WHERE a.is_enrolled) AS enrolled,
       round(100.0 * count(*) FILTER (WHERE a.is_enrolled AND a.has_heard)
             / NULLIF(count(*) FILTER (WHERE a.has_heard), 0), 1) AS conversion_pct
FROM survey.scheme_awareness a
JOIN survey.scheme s USING (scheme_id)
GROUP BY s.scheme_name;

CREATE VIEW reporting.v_scheme_summary AS
SELECT s.scheme_name,
       count(*) AS n_feedback,
       round(100.0 * count(*) FILTER (WHERE rs.overall_label = 'Positive') / NULLIF(count(rs.overall_label), 0), 1) AS pct_positive,
       round(100.0 * count(*) FILTER (WHERE rs.overall_label = 'Neutral')  / NULLIF(count(rs.overall_label), 0), 1) AS pct_neutral,
       round(100.0 * count(*) FILTER (WHERE rs.overall_label = 'Negative') / NULLIF(count(rs.overall_label), 0), 1) AS pct_negative,
       round(avg(f.satisfaction_score), 2) AS avg_satisfaction,
       round(avg(ir.rating_score), 2) AS avg_impact_rating
FROM survey.scheme_feedback f
JOIN survey.scheme s USING (scheme_id)
LEFT JOIN reporting.v_respondent_sentiment rs USING (respondent_id)
LEFT JOIN survey.aspect_rating ir
       ON ir.feedback_id = f.feedback_id
      AND ir.aspect_id = (SELECT aspect_id FROM survey.aspect WHERE aspect_name = 'Overall impact')
GROUP BY s.scheme_name;

CREATE VIEW reporting.v_aspect_divergence AS
WITH aspect_pairs AS (
    SELECT a.aspect_name AS comparison,
           sign(r.rating_score) AS rating_sign,
           s.label AS text_label
    FROM survey.aspect_rating r
    JOIN analytics.aspect_sentiment s USING (feedback_id, aspect_id)
    JOIN analytics.nlp_model m ON m.model_id = s.model_id AND m.method_type = 'ABSA'
    JOIN survey.aspect a ON a.aspect_id = r.aspect_id
), overall_pairs AS (
    SELECT 'Overall satisfaction (C2)'::VARCHAR AS comparison,
           sign(f.satisfaction_score) AS rating_sign,
           rs.overall_label AS text_label
    FROM survey.scheme_feedback f
    JOIN reporting.v_respondent_sentiment rs USING (respondent_id)
    WHERE f.satisfaction_score IS NOT NULL
), pairs AS (
    SELECT * FROM aspect_pairs UNION ALL SELECT * FROM overall_pairs
)
SELECT comparison,
       count(*) AS n_comparable,
       count(*) FILTER (WHERE (rating_sign > 0 AND text_label = 'Negative') OR (rating_sign < 0 AND text_label = 'Positive')) AS n_divergent,
       round(100.0 * count(*) FILTER (WHERE (rating_sign > 0 AND text_label = 'Negative') OR (rating_sign < 0 AND text_label = 'Positive')) / count(*), 1) AS pct_divergent
FROM pairs
GROUP BY comparison;

CREATE VIEW reporting.v_validation_metrics AS
SELECT m.model_name,
       count(*) AS n_labelled,
       count(*) FILTER (WHERE ts.label = an.human_label) AS n_agree_with_human,
       round(100.0 * count(*) FILTER (WHERE ts.label = an.human_label) / count(*), 1) AS accuracy_pct
FROM analytics.annotation an
JOIN survey.scheme_feedback f USING (feedback_id)
JOIN analytics.text_sentiment ts ON ts.respondent_id = f.respondent_id AND ts.source_field = 'COMBINED'
JOIN analytics.nlp_model m ON m.model_id = ts.model_id
WHERE an.aspect_id IS NULL
GROUP BY m.model_name;

CREATE VIEW reporting.v_respondent_overview AS
SELECT r.respondent_id,
       r.source_excel_row,
       d.district_name,
       v.village_name,
       r.age,
       r.gender,
       r.education,
       r.landholding_acres,
       CASE WHEN r.landholding_acres < 2.5 THEN 'Marginal (<2.5 ac)'
            WHEN r.landholding_acres < 5 THEN 'Small (2.5-5 ac)'
            WHEN r.landholding_acres < 10 THEN 'Semi-medium (5-10 ac)'
            WHEN r.landholding_acres >= 10 THEN 'Large (10+ ac)' END AS landholding_band,
       r.income_band,
       r.is_aware,
       i.source_name AS info_source,
       s.scheme_name AS feedback_scheme,
       f.satisfaction_score,
       rs.overall_score AS text_sentiment_score,
       rs.overall_label AS text_sentiment_label,
       r.govt_support_score,
       r.support_trend_score,
       r.future_confidence_score
FROM survey.respondent r
LEFT JOIN survey.village v USING (village_id)
LEFT JOIN survey.district d USING (district_id)
LEFT JOIN survey.info_source i USING (info_source_id)
LEFT JOIN survey.scheme_feedback f USING (respondent_id)
LEFT JOIN survey.scheme s ON s.scheme_id = f.scheme_id
LEFT JOIN reporting.v_respondent_sentiment rs USING (respondent_id);
