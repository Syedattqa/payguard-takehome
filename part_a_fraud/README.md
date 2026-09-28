# Part A: Fraud Detection

- `fraud_detection.ipynb`: the deliverable. Open in Google Colab and Run all (about 2 min on CPU).
- `fraud_detection.py`: the same notebook as a plain-text script ([jupytext](https://jupytext.readthedocs.io)
  percent format). It's easier to diff and review. Regenerate the notebook with
  `jupytext --to notebook fraud_detection.py`.
- `explore.py`, `temporal_analysis.py`: initial exploration and the 70/15/15 time-split check.

Notebook sections: data checks → evaluation design (leakage risks and how each is handled) → baseline,
logistic regression, XGBoost → PR-AUC comparison with bootstrap and paired-bootstrap CIs → cost-based
threshold with sensitivity analysis → real-time inference note → limitations.
