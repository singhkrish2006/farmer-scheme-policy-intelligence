import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

CUSTOM_STOPWORDS = {
    "scheme", "schemes", "farmer", "farmers", "farming", "government", "govt",
    "amount", "benefit", "benefits", "process", "provided", "given", "received",
    "getting", "get", "got", "also", "well", "good", "no", "none", "nothing", "na",
}

LDA_PARAMS = {"n_topics": 5, "max_features": 400, "min_df": 5, "max_df": 0.85, "max_iter": 50, "random_state": 42}


def document_term_matrix(texts, max_features=400):
    docs = texts.dropna()
    docs = docs[docs.str.strip().str.len() > 2]
    vectorizer = CountVectorizer(
        max_df=0.85, min_df=5, stop_words="english", max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    dtm = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()
    keep = [i for i, word in enumerate(features) if word not in CUSTOM_STOPWORDS]
    return docs, dtm[:, keep].tocsc(), [features[i] for i in keep]


def fit_lda(dtm, n_topics, random_state):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, learning_method="batch", max_iter=50)
    return lda, lda.fit_transform(dtm)


def fit_topics(texts, n_topics=5, n_top_words=10, max_features=400, random_state=42):
    docs, dtm, vocab = document_term_matrix(texts, max_features)
    if len(docs) < n_topics * 5:
        raise ValueError(f"Not enough documents ({len(docs)}) for {n_topics} topics")
    lda, doc_topic = fit_lda(dtm, n_topics, random_state)

    topics = []
    for topic_id, weights in enumerate(lda.components_):
        order = weights.argsort()[::-1][:n_top_words]
        normalised = weights / weights.sum()
        topics.append({
            "topic_id": topic_id,
            "top_words": [vocab[i] for i in order],
            "top_weights": [round(float(normalised[i]), 5) for i in order],
            "share": round(float(doc_topic[:, topic_id].mean()), 4),
        })

    weights = pd.DataFrame(doc_topic, index=docs.index).round(4)
    per_document = pd.Series(index=texts.index, dtype="Int64")
    per_document.loc[docs.index] = doc_topic.argmax(axis=1)
    return topics, per_document, weights


def umass_coherence(components, dtm, n_top_words=10):
    presence = (dtm > 0).astype(np.int32)
    doc_freq = np.asarray(presence.sum(axis=0)).ravel()
    co_occurrence = (presence.T @ presence).toarray()
    scores = []
    for weights in components:
        top = weights.argsort()[::-1][:n_top_words]
        pairs = [np.log((co_occurrence[top[m], top[l]] + 1) / doc_freq[top[l]]) for m in range(1, len(top)) for l in range(m)]
        scores.append(np.mean(pairs))
    return float(np.mean(scores))


def _top_sets(components, n_top_words=10):
    return [set(w.argsort()[::-1][:n_top_words]) for w in components]


def model_selection(texts, k_values=range(3, 11), seeds=(0, 1, 2, 3, 4), reference_seed=42):
    _, dtm, _ = document_term_matrix(texts)
    rows = []
    for k in k_values:
        reference, _ = fit_lda(dtm, k, reference_seed)
        ref_sets = _top_sets(reference.components_)
        overlaps = []
        for seed in seeds:
            other, _ = fit_lda(dtm, k, seed)
            other_sets = _top_sets(other.components_)
            overlaps += [max(len(r & o) / len(r | o) for o in other_sets) for r in ref_sets]
        rows.append({
            "n_topics": k,
            "umass_coherence": round(umass_coherence(reference.components_, dtm), 3),
            "perplexity": round(float(reference.perplexity(dtm)), 1),
            "seed_stability_jaccard": round(float(np.mean(overlaps)), 3),
        })
    return pd.DataFrame(rows)
