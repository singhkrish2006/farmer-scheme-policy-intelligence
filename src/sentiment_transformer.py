import pandas as pd

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TOKENS = 128

_LABEL_MAP = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME,
            truncation=True, max_length=MAX_TOKENS,
        )
    return _pipeline


def _signed(label, confidence):
    if label == "Positive":
        return confidence
    if label == "Negative":
        return -confidence
    return 0.0


def score_texts(texts, batch_size=32):
    classifier = get_pipeline()
    valid_idx = [i for i, t in enumerate(texts) if pd.notna(t) and t]

    scores = [None] * len(texts)
    labels = [None] * len(texts)
    if valid_idx:
        predictions = classifier([texts[i] for i in valid_idx], batch_size=batch_size)
        for i, pred in zip(valid_idx, predictions):
            label = _LABEL_MAP.get(pred["label"].lower(), pred["label"])
            labels[i] = label
            scores[i] = _signed(label, pred["score"])
    return scores, labels
