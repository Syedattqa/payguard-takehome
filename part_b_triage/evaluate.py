"""Score predictions against hand labels and write a markdown report.

  python evaluate.py --pred outputs/predictions_gemini-3.5-flash-lite.jsonl
  python evaluate.py --pred outputs/adversarial.jsonl --adversarial data/adversarial_cases.csv

What "correct" means (the metrics are ordered by how much each error costs PayGuard):
  1. requires_human RECALL: of the complaints a human said need review, how many did the pipeline
     send to a human? A miss here means a fraud or legal case could be auto-handled. Target: ~100%.
  2. category accuracy / macro-F1: is the complaint routed to the right team? A miss costs a
     re-route (hours of delay), not a safety failure.
  3. urgency: exact agreement, plus "severe misses" (gold high predicted low), the costly ones.
  4. summary faithfulness: a cheap automatic check that numbers in the summary appear in the
     narrative (catches invented amounts/dates); a human spot-check covers the rest.
  5. pipeline health: fallback rate, retry rate, invalid-output rate, latency, tokens.
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from taxonomy import CATEGORY_IDS

HERE = Path(__file__).parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion; honest error bars on small labelled sets."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} (95% CI {lo:.0%}-{hi:.0%})" if n else "n/a"


def load_preds(path: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    return pd.DataFrame(rows).astype({"complaint_id": str})


def ungrounded_numbers(summary: str, narrative: str) -> list[str]:
    """Numbers (amounts, dates, counts) in the summary that never appear in the narrative."""
    norm = lambda s: re.sub(r"[,$]", "", s)
    source = norm(narrative)
    nums = re.findall(r"\$?\d[\d,]*(?:\.\d+)?", summary)
    return [n for n in nums if norm(n).rstrip(".") not in source]


# Weak labels for the FULL sample: CFPB's own "issue" field, chosen by the consumer. Noisy, but it
# costs nothing and scales to every complaint, so it is a useful drift/sanity signal at 50k/month.
WEAK_MAP = [
    (r"fraud|scam|unauthori|identity theft", "fraud_scam"),
    (r"purchase shown on your statement|problem with a purchase|billing dispute|charged for", "transaction_dispute"),
    (r"money was not available|wrong amount|transfer|payment process|making payments|deposit", "payments_transfers"),
    (r"closing|frozen|access|getting a card|card declined", "account_access"),
    (r"fee|interest|rewards|credit limit|advertising|promotion|apr", "fees_interest_terms"),
    (r"credit report|collect|debt|company's investigation", "credit_reporting_collections"),
]


def weak_label(issue: str) -> str | None:
    for pattern, cat in WEAK_MAP:
        if re.search(pattern, str(issue), re.I):
            return cat
    return None


def evaluate(pred_path: str, labels_path: str, sample_path: str) -> str:
    preds = load_preds(pred_path)
    sample = pd.read_csv(sample_path, dtype={"complaint_id": str})
    out: list[str] = [f"# Triage evaluation: `{Path(pred_path).name}`\n",
                      f"Model: {preds['model'].iloc[0]} | predictions: {len(preds)}\n"]

    # ---------------- pipeline health (all predictions, no labels needed) ----------------
    n = len(preds)
    out += ["## 1. Pipeline health (all predictions)\n",
            f"- Fallback records (LLM never produced valid output): {rate((preds.source == 'fallback').sum(), n)}",
            f"- Needed a retry (first reply invalid): {rate((preds.attempts > 1).sum(), n)}",
            f"- Any validation error seen: {rate((preds.validation_errors.str.len() > 0).sum(), n)}",
            f"- Invalid category attempts (hallucinated labels caught by the validator): "
            f"{sum('category' in e for errs in preds.validation_errors for e in errs)}",
            f"- requires_human overridden by safety rules: {rate((preds.overrides.str.len() > 0).sum(), n)}",
            f"- Routed to a human overall: {rate(preds.requires_human.sum(), n)}",
            f"- Latency per complaint: p50 {preds.latency_s.median():.2f}s, p95 {preds.latency_s.quantile(.95):.2f}s",
            f"- Tokens per complaint: {preds.prompt_tokens.mean():.0f} in / {preds.completion_tokens.mean():.0f} out (mean)\n"]

    # ---------------- summary grounding (all predictions) ----------------
    merged_all = preds.merge(sample[["complaint_id", "narrative", "issue"]], on="complaint_id", how="left")
    llm_rows = merged_all[merged_all.source == "llm"]
    ungrounded = [(r.complaint_id, u) for r in llm_rows.itertuples()
                  if (u := ungrounded_numbers(r.summary, str(r.narrative)))]
    out += ["## 2. Summary grounding (automatic check)\n",
            f"Summaries containing a number not found in the complaint: {rate(len(ungrounded), len(llm_rows))}",
            *[f"- {cid}: {nums}" for cid, nums in ungrounded[:15]], ""]

    # ---------------- weak-label agreement (all predictions) ----------------
    merged_all["weak"] = merged_all["issue"].map(weak_label)
    wl = merged_all.dropna(subset=["weak"])
    out += ["## 3. Agreement with CFPB issue field (weak labels, all predictions)\n",
            f"Coverage: {len(wl)}/{n} complaints have a mappable issue. "
            f"Agreement: {rate((wl.weak == wl.category).sum(), len(wl))}. "
            "Consumer-chosen issues are noisy, so this is a trend signal, not ground truth.\n"]

    # ---------------- against hand labels ----------------
    if not Path(labels_path).exists():
        out.append("_No reference labels found (run label_tool.py or judge_label.py); skipped sections 4-5._")
        return "\n".join(out)
    gold = pd.read_csv(labels_path, dtype=str).fillna("")
    gold["requires_human"] = gold["requires_human"].str.lower().isin(["true", "1", "y", "yes"])
    m = gold.merge(preds, on="complaint_id", suffixes=("_gold", ""))
    k = len(m)
    out.append(f"## 4. Against reference labels `{Path(labels_path).name}` (n = {k})\n")

    # Routing to a human: the safety metric
    need = m[m.requires_human_gold]
    missed = need[~need.requires_human]
    rule_only = need.rule_flags.str.len() > 0
    llm_only = need.llm_requires_human.fillna(False).astype(bool)  # fallback = LLM gave no answer = miss
    out += ["### requires_human (safety)\n",
            f"- **Recall** (gold=needs human, pipeline sent to human): {rate(need.requires_human.sum(), len(need))}",
            f"- Precision: {rate((m.requires_human & m.requires_human_gold).sum(), m.requires_human.sum())}",
            "- Ablation, recall of each safety layer on its own:",
            f"  - LLM decision alone: {rate(llm_only.sum(), len(need))}",
            f"  - Keyword rules alone: {rate(rule_only.sum(), len(need))}",
            f"  - Combined (shipped): {rate(need.requires_human.sum(), len(need))}",
            f"- Unsafe misses (need human, auto-routed): {len(missed)} {list(missed.complaint_id)}"]
    sens = m[m.sensitive_flags_gold.str.len() > 0]
    out += [f"- Sensitive cases (gold has any sensitive flag) sent to human: {rate(sens.requires_human.sum(), len(sens))}\n"]

    # Category
    acc = (m.category_gold == m.category).sum()
    out += ["### Category (routing)\n",
            f"- Accuracy: {rate(acc, k)}",
            f"- Macro-F1: {f1_score(m.category_gold, m.category, labels=list(CATEGORY_IDS), average='macro', zero_division=0):.2f}",
            "- Confusion matrix (rows = gold, columns = predicted):\n"]
    cm = pd.DataFrame(confusion_matrix(m.category_gold, m.category, labels=list(CATEGORY_IDS)),
                      index=CATEGORY_IDS, columns=[c[:10] for c in CATEGORY_IDS])
    out += ["```", cm.to_string(), "```", ""]

    # Urgency
    order = {"low": 0, "medium": 1, "high": 2}
    g, p = m.urgency_gold.map(order), m.urgency.map(order)
    out += ["### Urgency\n",
            f"- Exact agreement: {rate((g == p).sum(), k)}",
            f"- Within one level: {rate(((g - p).abs() <= 1).sum(), k)}",
            f"- Severe under-calls (gold high, predicted low): {int(((g == 2) & (p == 0)).sum())}",
            f"- Linear-weighted Cohen's kappa: {cohen_kappa_score(g, p, weights='linear'):.2f}\n"]

    # Errors worth reading
    wrong = m[(m.category_gold != m.category) | (m.requires_human_gold & ~m.requires_human)]
    out += ["## 5. Disagreements to read (error analysis)\n",
            "| id | gold | predicted | urgency gold/pred | model's confidence note |", "|---|---|---|---|---|"]
    out += [f"| {r.complaint_id} | {r.category_gold} | {r.category} | {r.urgency_gold}/{r.urgency} | "
            f"{r.confidence_note[:110].replace('|', '/')} |" for r in wrong.itertuples()]
    return "\n".join(out)


def evaluate_adversarial(pred_path: str, cases_path: str) -> str:
    preds = load_preds(pred_path)
    cases = pd.read_csv(cases_path, dtype=str).fillna("")
    m = cases.merge(preds, on="complaint_id", how="left")
    lines = ["# Adversarial / red-team cases\n",
             "| id | attack | requires_human (expected -> got) | category (expected -> got) | pass |",
             "|---|---|---|---|---|"]
    passed = 0
    for r in m.itertuples():
        ok_h = r.expect_requires_human == "" or (str(r.requires_human).lower() == r.expect_requires_human)
        ok_c = r.expect_category == "" or r.category == r.expect_category
        ok = ok_h and ok_c
        passed += ok
        lines.append(f"| {r.complaint_id} | {r.attack_type} | {r.expect_requires_human or 'any'} -> "
                     f"{r.requires_human} | {r.expect_category or 'any'} -> {r.category} | {'yes' if ok else 'NO'} |")
    lines.append(f"\n**{passed}/{len(m)} passed.** Category mismatches on ambiguous cases are acceptable "
                 "if requires_human is correct; requires_human failures are not.")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels", default=str(HERE / "data/labels.csv"))
    ap.add_argument("--sample", default=str(HERE / "data/complaints_sample.csv"))
    ap.add_argument("--adversarial", default=None, help="path to adversarial_cases.csv")
    args = ap.parse_args()

    report = (evaluate_adversarial(args.pred, args.adversarial) if args.adversarial
              else evaluate(args.pred, args.labels, args.sample))
    out = Path(args.pred).with_name(Path(args.pred).stem + "_report.md")
    out.write_text(report)
    print(report)
    print(f"\nReport written to {out}")
