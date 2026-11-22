"""Feature extraction: TF-IDF vectors and averaged Word2Vec embeddings."""
from typing import Iterable, List

import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import preprocess


def stemmed_tokens(text: str) -> List[str]:
    return preprocess(text, use_stemming=True)


def plain_tokens(text: str) -> List[str]:
    return preprocess(text, use_stemming=False)


def build_tfidf(ngram_range=(1, 2), min_df: int = 2) -> TfidfVectorizer:
    """TF-IDF over NLTK-preprocessed tokens, with unigrams and bigrams."""
    return TfidfVectorizer(
        tokenizer=stemmed_tokens,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that learns word embeddings and averages them per document.

    Word2Vec is trained inside fit(), on the training documents only, so cross
    validation never leaks information from held-out folds.
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2,
                 epochs: int = 20, sg: int = 1, seed: int = 42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.seed = seed

    def fit(self, X: Iterable[str], y=None):
        sentences = [plain_tokens(doc) for doc in X]
        # workers=1 keeps training deterministic for a fixed seed
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            seed=self.seed,
            workers=1,
        )
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        wv = self.model_.wv
        out = np.zeros((len(list(X)) if not hasattr(X, "__len__") else len(X), self.vector_size))
        for i, doc in enumerate(X):
            vectors = [wv[t] for t in plain_tokens(doc) if t in wv]
            if vectors:
                out[i] = np.mean(vectors, axis=0)
        return out
