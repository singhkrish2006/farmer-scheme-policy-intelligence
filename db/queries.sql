-- Analysis queries on the farmer_policy database (PostgreSQL 16)

-- 1. Awareness-to-enrolment conversion per scheme
SELECT * FROM reporting.v_awareness_gap ORDER BY heard DESC;

-- 2. Sentiment and satisfaction per scheme
SELECT * FROM reporting.v_scheme_summary ORDER BY n_feedback DESC;

-- 3. Where ratings and written feedback disagree
SELECT * FROM reporting.v_aspect_divergence ORDER BY pct_divergent DESC;

-- 4. Most negatively described aspect for each scheme (window function)
WITH aspect_negative AS (
    SELECT s.scheme_name, a.aspect_name,
           round(100.0 * count(*) FILTER (WHERE x.label = 'Negative') / count(*), 1) AS pct_negative,
           count(*) AS mentions
    FROM analytics.aspect_sentiment x
    JOIN survey.scheme_feedback f USING (feedback_id)
    JOIN survey.scheme s USING (scheme_id)
    JOIN survey.aspect a USING (aspect_id)
    GROUP BY s.scheme_name, a.aspect_name
    HAVING count(*) >= 10
)
SELECT scheme_name, aspect_name, pct_negative, mentions
FROM (
    SELECT *, rank() OVER (PARTITION BY scheme_name ORDER BY pct_negative DESC) AS rnk
    FROM aspect_negative
) ranked
WHERE rnk = 1
ORDER BY pct_negative DESC;

-- 5. Sentiment by income band
SELECT r.income_band,
       count(*) AS respondents,
       round(100.0 * count(*) FILTER (WHERE rs.overall_label = 'Negative') / count(*), 1) AS pct_negative,
       round(100.0 * count(*) FILTER (WHERE rs.overall_label = 'Positive') / count(*), 1) AS pct_positive
FROM survey.respondent r
JOIN reporting.v_respondent_sentiment rs USING (respondent_id)
WHERE r.income_band IS NOT NULL
GROUP BY r.income_band
ORDER BY pct_negative DESC;

-- 6. Full-text search: complaints mentioning delays in documents or verification
SELECT f.feedback_id, s.scheme_name, f.problems_text
FROM survey.scheme_feedback f
JOIN survey.scheme s USING (scheme_id)
WHERE to_tsvector('english', coalesce(f.positive_text, '') || ' ' || coalesce(f.problems_text, '') || ' ' || coalesce(f.suggestions_text, ''))
      @@ to_tsquery('english', 'delay & (document | verification)')
LIMIT 20;

-- 7. Top terms of each complaint topic
SELECT t.topic_label, string_agg(tt.term, ', ' ORDER BY tt.term_rank) AS top_terms, t.corpus_share
FROM analytics.topic t
JOIN analytics.topic_term tt USING (topic_id)
WHERE t.corpus = 'C5'
GROUP BY t.topic_label, t.corpus_share
ORDER BY t.corpus_share DESC;

-- 8. Scheme-name spellings mapped to each scheme
SELECT s.scheme_name, count(*) AS spellings, string_agg(a.alias_text, ' | ' ORDER BY a.alias_text) AS examples
FROM survey.scheme_alias a
JOIN survey.scheme s USING (scheme_id)
GROUP BY s.scheme_name
ORDER BY spellings DESC;

-- 9. Data-quality flags by type
SELECT flag_type, count(*) AS flags, count(DISTINCT respondent_id) AS respondents
FROM analytics.quality_flag
GROUP BY flag_type
ORDER BY flags DESC;

-- 10. Model accuracy against human labels (after annotation is loaded)
SELECT * FROM reporting.v_validation_metrics;

-- 11. Trace any respondent back to the raw survey row
SELECT r.respondent_id, r.source_excel_row, r.name, raw.c1_scheme, raw.c4_positive, raw.c5_problems
FROM survey.respondent r
JOIN staging.raw_survey raw USING (source_excel_row)
WHERE r.respondent_id = 42;

-- 12. Integrity check: CHECK constraint rejects an out-of-range rating
-- INSERT INTO survey.aspect_rating (feedback_id, aspect_id, rating_score) VALUES (1, 1, 5);
