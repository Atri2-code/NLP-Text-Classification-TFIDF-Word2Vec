"""Train and compare text classifiers, then write results/metrics.json and results/report.md.

Usage:
    python run_experiment.py                          # bundled folder corpus in data/raw/movie_reviews
    python run_experiment.py --csv reviews.csv --text-col text --label-col label
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))

from review_nlp.data import load_csv, load_folder_corpus  # noqa: E402
from review_nlp.evaluate import nearest_words, score, top_terms  # noqa: E402
from review_nlp.models import VaderBaseline, tfidf_logreg, word2vec_logreg  # noqa: E402

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/movie_reviews")
    parser.add_argument("--csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    texts, labels = (load_csv(args.csv, args.text_col, args.label_col) if args.csv
                     else load_folder_corpus(args.data_dir))
    y = np.array(labels)
    print(f"Loaded {len(texts)} documents ({int(y.sum())} positive, {int((1 - y).sum())} negative)")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=SEED)

    results = {"documents": len(texts), "train": len(X_train), "test": len(X_test), "seed": SEED, "models": {}}

    # 1. Lexicon and rule-based baseline
    vader = VaderBaseline()
    results["models"]["vader_baseline"] = score(y_test, vader.predict(X_test))

    # 2. TF-IDF + logistic regression
    tfidf = tfidf_logreg().fit(X_train, y_train)
    results["models"]["tfidf_logreg"] = score(y_test, tfidf.predict(X_test))
    results["top_terms"] = top_terms(tfidf)

    # 3. Word2Vec averaged embeddings + logistic regression
    w2v = word2vec_logreg().fit(X_train, y_train)
    results["models"]["word2vec_logreg"] = score(y_test, w2v.predict(X_test))
    results["nearest_words"] = nearest_words(w2v.named_steps["w2v"], ["good", "bad", "boring", "great", "plot"])
    results["vocab_size_word2vec"] = len(w2v.named_steps["w2v"].model_.wv)

    # 5-fold cross validation on the full dataset (vectorizers refit inside each fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, factory in (("tfidf_logreg", tfidf_logreg), ("word2vec_logreg", word2vec_logreg)):
        scores = cross_val_score(factory(), texts, y, cv=cv, scoring="accuracy")
        results["models"][name]["cv_accuracy_mean"] = round(float(scores.mean()), 4)
        results["models"][name]["cv_accuracy_std"] = round(float(scores.std()), 4)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    lines = ["# Results", "",
             f"Documents: {results['documents']} (train {results['train']}, test {results['test']}, seed {SEED})", "",
             "| Model | Test accuracy | Test macro F1 | 5-fold CV accuracy |", "|---|---|---|---|"]
    for name, m in results["models"].items():
        cvtxt = f"{m['cv_accuracy_mean']:.3f} +/- {m['cv_accuracy_std']:.3f}" if "cv_accuracy_mean" in m else "n/a"
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {cvtxt} |")
    lines += ["", f"Top positive terms: {', '.join(results['top_terms']['positive'])}",
              f"Top negative terms: {', '.join(results['top_terms']['negative'])}", "",
              "Nearest words in the learned embedding space:"]
    for w, near in results["nearest_words"].items():
        lines.append(f"- {w}: {', '.join(near)}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
