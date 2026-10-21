import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src.preprocess import is_nil_response

NEGATIVE_TERMS = {
    "delay": -2.0, "delays": -2.0, "delayed": -2.0,
    "corruption": -3.0, "corrupt": -3.0, "bribe": -3.0, "bribes": -3.0,
    "harassment": -2.5, "rude": -2.0,
    "confusing": -1.5, "confused": -1.5, "difficult": -1.5, "difficulty": -1.5,
    "paperwork": -1.0, "documentation": -0.5,
    "insufficient": -1.5, "inadequate": -1.5,
    "slow": -1.5, "complex": -1.5, "unnecessary": -1.0,
}

POSITIVE_TERMS = {
    "timely": 2.0, "helpful": 2.0, "cooperative": 2.0, "beneficial": 2.0,
    "easy": 1.5, "smooth": 1.5, "useful": 1.5, "transparent": 1.5,
    "quick": 1.5, "immediate": 1.5,
    "helped": 1.7,
}

# process nouns that VADER scores as positive ("approval", "benefit") but carry no sentiment here
NEUTRALISED_PROCESS_NOUNS = {
    "approval": 0.0, "approved": 0.0, "benefit": 0.0, "benefits": 0.0, "benifits": 0.0,
}

# words VADER reads as negative inside relief phrases ("reduced financial burden")
NEUTRALISED_RELIEF_NOUNS = {
    "burden": 0.0, "burdens": 0.0, "pressure": 0.0, "unstable": 0.0,
    "uncertain": 0.0, "losses": 0.0,
}

AGRI_DOMAIN_LEXICON = {
    **NEGATIVE_TERMS, **POSITIVE_TERMS,
    **NEUTRALISED_PROCESS_NOUNS, **NEUTRALISED_RELIEF_NOUNS,
}

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def ensure_vader_lexicon():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def build_analyzer():
    ensure_vader_lexicon()
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(AGRI_DOMAIN_LEXICON)
    return analyzer


def label_from_score(score):
    if pd.isna(score):
        return None
    if score > POSITIVE_THRESHOLD:
        return "Positive"
    if score < NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def score_text(analyzer, text):
    if pd.isna(text):
        return None, None
    if is_nil_response(text):
        return 0.0, "Neutral"
    compound = analyzer.polarity_scores(text)["compound"]
    return compound, label_from_score(compound)


def score_column(analyzer, series):
    results = series.apply(lambda text: score_text(analyzer, text))
    return results.apply(lambda r: r[0]), results.apply(lambda r: r[1])
