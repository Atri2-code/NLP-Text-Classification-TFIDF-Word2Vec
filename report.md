# Results

Documents: 2000 (train 1600, test 400, seed 42)

| Model | Test accuracy | Test macro F1 | 5-fold CV accuracy |
|---|---|---|---|
| vader_baseline | 0.645 | 0.633 | n/a |
| tfidf_logreg | 0.845 | 0.845 | 0.858 +/- 0.015 |
| word2vec_logreg | 0.807 | 0.807 | 0.768 +/- 0.028 |

Top positive terms: perfect, great, excel, perform, enjoy, hilari, terrif, overal, life, entertain
Top negative terms: bad, wast, bore, worst, suppos, noth, poor, stupid, plot, ridicul

Nearest words in the learned embedding space:
- good: forte, lawerence, mandy, kaye's, mailed
- bad: awesomely, guys, ahh, bandages, spate
- boring: stupefying, disasterous, unengaging, excruciatingly, unexciting
- great: deal, phoniness, reshooting, differ, consistent
- plot: holes, twists, beards, heel, storyline
