# NLP Text Classification: NLTK, TF-IDF and Word2Vec Embeddings

A compact NLP project that classifies review text as positive or negative and compares three approaches
side by side on the same split: a lexicon baseline (VADER), TF-IDF vectors with logistic regression, and
Word2Vec word embeddings with logistic regression.

## What it does
1. **Preprocessing (NLTK):** regex tokenizer, stop word removal (negations like "not" are kept because they flip sentiment) and the Snowball stemmer.
2. **TF-IDF vectors (scikit-learn):** unigrams and bigrams, sublinear term frequency, logistic regression classifier.
3. **Word embeddings (gensim):** a skip-gram Word2Vec model trained on the training documents only, averaged into one vector per document, then logistic regression.
4. **Baseline:** VADER lexicon and rule-based sentiment, to measure how much a trained model adds.
5. **Evaluation:** stratified 80/20 test split plus 5-fold cross validation. Vectorizers are refit inside each fold so no information leaks from held-out data.

## Results (2,000 labelled movie reviews, seed 42)
| Model | Test accuracy | Test macro F1 | 5-fold CV accuracy |
|---|---|---|---|
| VADER lexicon baseline | 0.645 | 0.633 | n/a |
| TF-IDF + logistic regression | 0.845 | 0.845 | 0.858 +/- 0.015 |
| Word2Vec average + logistic regression | 0.807 | 0.807 | 0.768 +/- 0.028 |

Training a model beat the lexicon baseline by about 20 accuracy points. TF-IDF beat averaged Word2Vec here, which is
expected: 1,600 training documents is too small to learn strong embeddings from scratch. The nearest neighbours of
words like "good" are noisy for the same reason, while "boring" and "plot" give sensible neighbours
(see `results/report.md`). Pretrained embeddings or a transformer would be the next step.

## Run it
```bash
pip install -r requirements.txt
# Put a folder corpus at data/raw/movie_reviews/{pos,neg}/*.txt  (the NLTK movie_reviews corpus works)
python run_experiment.py
# or use your own CSV with a text column and a 0/1 label column
python run_experiment.py --csv raw_reviews.csv --text-col text --label-col label
pytest -q
```
Full run takes about 6 minutes on a laptop (the cross validation retrains Word2Vec five times).

## Layout
```
src/review_nlp/   data.py  preprocess.py  features.py  models.py  evaluate.py
tests/            unit tests for preprocessing, features, models
results/          metrics.json and report.md written by run_experiment.py
run_experiment.py command line entry point
```

## Limitations
- Movie reviews are a public benchmark, not Trustpilot data. Swap in your own labelled reviews with `--csv`.
- Word2Vec is trained from scratch on a small corpus. Averaging word vectors also ignores word order.
- Binary labels only.

## Data
The movie review corpus (Pang and Lee) is distributed with NLTK data and is not committed to this repository.
