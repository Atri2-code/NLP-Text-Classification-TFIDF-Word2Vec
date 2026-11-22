"""Text preprocessing built on NLTK tokenizer and stemmer.

Both tools used here need no downloaded corpora, so the project runs offline.
"""
from typing import List

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations carry sentiment, so they are kept even though they are stop words.
NEGATIONS = {"no", "not", "nor", "never", "cannot", "nothing", "none", "neither"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

_TOKENIZER = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?")
_STEMMER = SnowballStemmer("english")


def tokenize(text: str) -> List[str]:
    """Lowercase and split text into word tokens (keeps contractions like don't)."""
    return _TOKENIZER.tokenize(text.lower())


def remove_stop_words(tokens: List[str]) -> List[str]:
    """Drop common stop words, keeping negations and single-letter noise out."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def stem(tokens: List[str]) -> List[str]:
    """Reduce each token to its stem with the Snowball (Porter2) stemmer."""
    return [_STEMMER.stem(t) for t in tokens]


def preprocess(text: str, use_stemming: bool = True) -> List[str]:
    """Full pipeline: tokenize, remove stop words, optionally stem."""
    tokens = remove_stop_words(tokenize(text))
    return stem(tokens) if use_stemming else tokens
