"""Model pipelines: TF-IDF + logistic regression, Word2Vec + logistic regression, VADER baseline."""
from typing import Iterable, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .features import Word2VecVectorizer, build_tfidf


def tfidf_logreg(C: float = 4.0) -> Pipeline:
    return Pipeline([
        ("tfidf", build_tfidf()),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


def word2vec_logreg(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("w2v", Word2VecVectorizer()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


class VaderBaseline:
    """Lexicon and rule-based baseline: positive when the VADER compound score is above zero."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return np.array([1 if self.analyzer.polarity_scores(t)["compound"] > 0 else 0 for t in texts])
