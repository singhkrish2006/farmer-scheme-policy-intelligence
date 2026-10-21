import re

import pandas as pd

from src.sentiment_lexicon import label_from_score

ASPECT_KEYWORDS = {
    "timeliness": [
        "delay", "delayed", "delays", "late", "time", "wait", "waiting",
        "quick", "quickly", "fast", "immediate", "immediately", "slow", "on time", "timely",
    ],
    "amount": [
        "amount", "money", "subsidy", "payment", "fund", "funds", "insufficient",
        "inadequate", "less amount", "enough", "financial", "cash", "compensation",
    ],
    "enrollment_ease": [
        "apply", "application", "process", "document", "documentation", "paperwork",
        "form", "register", "registration", "enroll", "enrollment", "online", "procedure",
        "eligibility", "eligible",
    ],
    "official_behavior": [
        "official", "officials", "staff", "officer", "behavior", "behaviour", "rude",
        "helpful", "cooperative", "corrupt", "corruption", "bribe", "harassment",
        "attitude", "response",
    ],
    "clarity": [
        "information", "clear", "clarity", "confusing", "confused", "understand",
        "explain", "explained", "aware", "awareness", "unclear", "communication",
    ],
    "impact": [
        "benefit", "benefits", "benefited", "helped", "help", "improved", "improvement",
        "useful", "impact", "income", "yield", "crop", "farming", "livelihood",
    ],
}

_ASPECT_PATTERNS = {
    aspect: re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b")
    for aspect, keywords in ASPECT_KEYWORDS.items()
}

_CLAUSE_SPLIT = re.compile(r"[.,;]| but | and | however | although ")


def extract_aspects(text):
    if pd.isna(text) or not text:
        return set()
    lowered = text.lower()
    return {aspect for aspect, pattern in _ASPECT_PATTERNS.items() if pattern.search(lowered)}


def score_aspects(text, analyzer, score_fn):
    if pd.isna(text) or not text:
        return {}

    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(text) if c.strip()] or [text]

    scores_by_aspect = {}
    for clause in clauses:
        aspects = extract_aspects(clause)
        if not aspects:
            continue
        score, _ = score_fn(analyzer, clause)
        if score is None:
            continue
        for aspect in aspects:
            scores_by_aspect.setdefault(aspect, []).append(score)

    return {aspect: sum(s) / len(s) for aspect, s in scores_by_aspect.items()}


aspect_label = label_from_score
