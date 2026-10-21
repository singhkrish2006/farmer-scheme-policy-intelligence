import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.sentiment_transformer import MAX_TOKENS, MODEL_NAME
from src.worksheets import stratified_sample


def transformer_probabilities(texts, batch_size=32):
    """Full positive/neutral/negative probabilities from the zero-shot RoBERTa model."""
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME,
                          truncation=True, max_length=MAX_TOKENS, top_k=None)
    rows = []
    for scores in classifier(list(texts), batch_size=batch_size):
        rows.append({s["label"].capitalize(): s["score"] for s in scores})
    return pd.DataFrame(rows, index=texts.index)[["Positive", "Neutral", "Negative"]]


def uncertainty(probabilities, vader_labels):
    """Normalised entropy of the transformer, plus 0.25 when VADER disagrees with it."""
    p = probabilities.clip(lower=1e-12).to_numpy()
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])
    disagree = (probabilities.idxmax(axis=1) != vader_labels.reindex(probabilities.index)).to_numpy()
    return pd.Series(entropy + 0.25 * disagree, index=probabilities.index)


def diverse_uncertain(pool, scores, size, n_clusters=20, seed=42):
    """Most uncertain texts, taken round-robin across TF-IDF clusters so they are not near-duplicates."""
    vectors = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english").fit_transform(pool["scheme_feedback_text"])
    clusters = pd.Series(KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(vectors), index=pool.index)
    queues = [scores[clusters == c].sort_values(ascending=False).index.tolist() for c in range(n_clusters)]
    queues.sort(key=lambda q: -scores[q[0]] if q else 0)

    chosen = []
    while len(chosen) < size and any(queues):
        for queue in queues:
            if queue and len(chosen) < size:
                chosen.append(queue.pop(0))
    return chosen


def select_training_batch(pool, vader_labels, size, seed=42):
    """Half uncertainty-based (diverse), half stratified random from the rest; returns rows and their selection method."""
    scores = uncertainty(transformer_probabilities(pool["scheme_feedback_text"]), vader_labels)
    uncertain = diverse_uncertain(pool, scores, size // 2, seed=seed)
    rest = pool.drop(index=uncertain)
    random_part = stratified_sample(rest, size - len(uncertain), seed=seed).head(size - len(uncertain))
    selected = pd.concat([pool.loc[uncertain], random_part]).sample(frac=1, random_state=seed)
    method = pd.Series("random", index=selected.index)
    method[uncertain] = "uncertainty"
    return selected, method
