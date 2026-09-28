# PayGuard Feasibility Build: Fraud Detection + Complaint Triage

Stixor take-home (AI Engineer / Data Scientist). Two working prototypes and a production memo.

| Part | What | Where |
|---|---|---|
| A | Fraud detection on the ULB credit-card dataset (Colab notebook) | [`part_a_fraud/fraud_detection.ipynb`](part_a_fraud/fraud_detection.ipynb) |
| B | LLM complaint triage pipeline with evaluation | [`part_b_triage/`](part_b_triage/README.md) |
| C | Production design memo (complaint triage) | [`part_c_memo/memo.md`](part_c_memo/memo.md) |
| – | AI tool disclosure | [`AI_USAGE.md`](AI_USAGE.md) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Part A (Colab):** open `part_a_fraud/fraud_detection.ipynb` in Google Colab (File → Upload notebook)
and run all cells. The loader looks for `creditcard.csv` locally, then downloads it with `kagglehub`
(Kaggle login), then offers a file upload. It also runs locally with `jupyter lab`, with the CSV at `data/creditcard.csv`.
Runtime is about 2 minutes on CPU.

**Part B:** see [`part_b_triage/README.md`](part_b_triage/README.md). Needs a free Gemini API key
(or Groq/OpenAI, or a local Ollama model). `pytest -q part_b_triage/tests` runs offline.

## Headline results

**Part A: fraud (time-ordered hold-out: last 6 of 48 hours, 52 frauds)**

| Model | PR-AUC [95% CI] | ROC-AUC | Recall @ 80% precision |
|---|---|---|---|
| Prevalence baseline | 0.001 | 0.50 | 0.00 |
| Logistic regression (balanced) | 0.69 [0.55, 0.84] | 0.977 | 0.69 |
| XGBoost | 0.74 [0.61, 0.85] | 0.978 | 0.71 |

At a cost-chosen threshold (~0.12), XGBoost catches 38/52 test frauds at 68% precision (about 187
alerts/day at this volume), cutting fraud-plus-review cost by about 60% vs. no model. XGBoost's lead
over logistic regression is **not** statistically clear: the paired bootstrap interval includes 0.
The honest conclusion is "ML is viable; confirm the model choice on more weeks of data."

**Part B: triage (400 complaints, `gpt-6-luna`; 80 labelled by a blind `gpt-6-sol` judge)**

| requires_human recall | Unsafe misses | Category accuracy | Urgency exact | Adversarial | Cost at 50k/month |
|---|---|---|---|---|---|
| 45/45 = 100% (CI 92–100%) | 0 | 91% (CI 83–96%) | 84% | 11/11 | ~$13 |

0 fallbacks, 1 retry and 0 invented categories in 400 calls. Trade-off: 75% of complaints are routed to
a human (precision 71%), a deliberately conservative starting point. Reference labels come from a
stronger model, not humans, which is the main evaluation limitation (see the Part B README).

## Approach, in one paragraph each

**A.** The main risk in this dataset is fooling yourself. So: time-ordered split (no future leakage),
duplicates removed before splitting, scaling inside a pipeline fit on train only, threshold and early
stopping chosen on validation, and test touched once. PR-AUC is the headline metric because accuracy
and ROC-AUC are meaningless at 0.17% prevalence. Every test metric has a bootstrap CI because 52
frauds is a small number. The threshold comes from an explicit cost model with a sensitivity check.
That check showed one ~$1,000 validation fraud was setting the threshold, which is itself a finding
about data volume.

**B.** The LLM is treated as an unreliable component: schema-validated output, the validator's error
fed back for repair, bounded retries, a conservative fallback, and rules that can only make the
system *more* cautious. Evaluation is ordered by cost of error: first "did every case that needs a
human reach one?", then routing accuracy, urgency, and summary faithfulness.

## Deviations from the brief (and why)

- **CFPB narratives are no longer published** (discontinued 14 Aug 2026; the live API returns
  complaints without text). Part B uses a public pre-change snapshot of the CFPB database on
  Hugging Face; the exact slice is defined in `fetch_complaints.py`, and running it writes the file hash and every complaint id to `part_b_triage/data/sample_manifest.json`.
  I'd have raised this with you by email in a real engagement before building on it.
- **Memo covers triage**, not fraud: residency makes the LLM system the harder production problem,
  and it can start in human-assist mode.

## What I cut, and why

- Hyperparameter search, ensembling, SMOTE, deep learning (brief asks for reasoning, not leaderboard gains).
- Probability calibration and SHAP explanations for fraud (the next step, listed in the notebook).
- Fine-tuning, embeddings classifiers, LLM-as-judge, UI (out of scope or not worth the extra trust surface).
- Human-labelled ground truth for Part B: an LLM judge was used instead; a human-validated subset is the next step.

## Repository layout

```
data/creditcard.csv            (not committed, 150 MB; notebook downloads it)
part_a_fraud/                  fraud_detection.ipynb (+ .py source), explore.py, temporal_analysis.py
part_b_triage/                 taxonomy, schema, prompts, guardrails, llm_client, triage, run_batch,
                               fetch_complaints, label_tool, evaluate, cost, tests/, data/, outputs/
part_c_memo/memo.md
```
