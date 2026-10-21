import re
from collections import Counter

import pandas as pd

from src.preprocess import is_nil_response

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its", "as", "at", "by", "from",
    "your", "you", "i", "we", "they", "he", "she", "them", "his", "her", "our", "their",
    "scheme", "schemes", "farmer", "farmers", "should", "would", "could", "can", "will", "not", "no",
    "please", "also", "more", "very", "much", "many", "need", "needs", "make", "made", "get", "gets",
    "provide", "provided",
}

THEMES = [
    ("Increase the benefit amount", ["amount", "increase"]),
    ("Simplify / speed up the application process", ["process", "application", "reduce", "procedures"]),
    ("Run more awareness programs", ["awareness", "programmes", "programs", "conduct"]),
    ("Ensure faster, more reliable disbursement", ["timely", "faster", "ensure"]),
]


def suggestion_texts(df, col="C6_suggestions_clean"):
    texts = df[col].dropna()
    return texts[~texts.apply(is_nil_response)]


def keyword_counts(texts):
    counter = Counter()
    for text in texts:
        for word in re.findall(r"[a-zA-Z']+", str(text).lower()):
            if len(word) > 2 and word not in STOPWORDS:
                counter[word] += 1
    return counter


def keyword_frequency_table(texts, top_n=30):
    counter = keyword_counts(texts)
    n = len(texts)
    table = pd.DataFrame(counter.most_common(top_n), columns=["Keyword", "Frequency"])
    table[f"Pct of suggestion responses (n={n})"] = (table["Frequency"] / n * 100).round(1)
    return table


def theme_table(texts):
    counter = keyword_counts(texts)
    rows = []
    for rank, (theme, keywords) in enumerate(THEMES, start=1):
        terms = ", ".join(f'"{k}" ({counter[k]})' for k in keywords if counter[k] > 0)
        rows.append({"Rank": rank, "Theme": theme, "Representative terms": terms})
    return pd.DataFrame(rows)
