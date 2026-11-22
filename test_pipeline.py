import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from review_nlp.features import Word2VecVectorizer, build_tfidf  # noqa: E402
from review_nlp.models import VaderBaseline, tfidf_logreg  # noqa: E402
from review_nlp.preprocess import preprocess, remove_stop_words, stem, tokenize  # noqa: E402

DOCS = ["great movie , loved the acting", "terrible plot and awful acting",
        "wonderful and moving story", "boring , slow and bad",
        "excellent film , great cast", "worst film i have seen , awful"] * 3
LABELS = [1, 0, 1, 0, 1, 0] * 3


def test_tokenize_lowercases_and_keeps_contractions():
    assert tokenize("Don't LIKE it!") == ["don't", "like", "it"]


def test_stop_words_removed_but_negations_kept():
    tokens = remove_stop_words(["the", "movie", "was", "not", "good"])
    assert "the" not in tokens and "not" in tokens and "good" in tokens


def test_stemming_reduces_inflections():
    assert stem(["loved", "loving"]) == ["love", "love"]


def test_preprocess_end_to_end():
    assert preprocess("The acting was amazing") == ["act", "amaz"]


def test_tfidf_shape_and_vocabulary():
    X = build_tfidf(min_df=1).fit_transform(DOCS)
    assert X.shape[0] == len(DOCS) and X.shape[1] > 0


def test_word2vec_vectorizer_output_shape():
    vec = Word2VecVectorizer(vector_size=16, min_count=1, epochs=5).fit(DOCS)
    out = vec.transform(DOCS)
    assert out.shape == (len(DOCS), 16) and np.isfinite(out).all()


def test_tfidf_model_learns_toy_task():
    model = tfidf_logreg()
    model.named_steps["tfidf"].set_params(min_df=1)
    model.fit(DOCS, LABELS)
    assert model.predict(["great acting , loved it"])[0] == 1
    assert model.predict(["awful and boring"])[0] == 0


def test_vader_baseline_polarity():
    pred = VaderBaseline().predict(["I love this, it is wonderful", "This is horrible and awful"])
    assert pred.tolist() == [1, 0]
