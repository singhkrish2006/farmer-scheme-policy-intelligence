import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_WORD = re.compile(r"[a-zA-Z]{2,}")


def normalize_for_topics(text):
    if pd.isna(text):
        return None
    tokens = [t for t in word_tokenize(str(text).lower()) if _WORD.fullmatch(t) and t not in _STOPWORDS]
    lemmas = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(lemmas) if lemmas else None


def preprocessing_report(raw_series, normalized_series):
    raw_docs = raw_series.dropna().astype(str)
    normalized_docs = normalized_series.dropna()

    raw_vocab = {token for doc in raw_docs for token in word_tokenize(doc.lower())}
    normalized_vocab = {token for doc in normalized_docs for token in doc.split()}
    tokens_before = raw_docs.apply(lambda t: len(word_tokenize(t)))
    tokens_after = normalized_docs.apply(lambda t: len(t.split()))

    return {
        "n_documents": int(len(raw_docs)),
        "n_documents_nonempty_after_normalization": int(len(normalized_docs)),
        "avg_tokens_before": round(float(tokens_before.mean()), 2) if len(tokens_before) else 0,
        "avg_tokens_after": round(float(tokens_after.mean()), 2) if len(tokens_after) else 0,
        "vocab_size_before": len(raw_vocab),
        "vocab_size_after_stopword_lemmatization": len(normalized_vocab),
        "vocab_reduction_pct": round((1 - len(normalized_vocab) / len(raw_vocab)) * 100, 1) if raw_vocab else 0,
    }
