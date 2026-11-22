"""Evaluation helpers."""
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def score(y_true, y_pred) -> Dict[str, object]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def top_terms(pipeline, n: int = 10) -> Dict[str, List[str]]:
    """Most positive and most negative TF-IDF features by logistic regression weight."""
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return {"negative": names[order[:n]].tolist(), "positive": names[order[-n:][::-1]].tolist()}


def nearest_words(w2v_vectorizer, words: List[str], topn: int = 5) -> Dict[str, List[str]]:
    wv = w2v_vectorizer.model_.wv
    return {w: [t for t, _ in wv.most_similar(w, topn=topn)] for w in words if w in wv}
