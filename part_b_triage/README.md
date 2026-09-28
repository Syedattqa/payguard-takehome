# Part B: LLM Complaint Triage

A batch pipeline that turns a raw complaint narrative into a validated triage record:

```json
{"category": "payments_transfers", "urgency": "medium",
 "summary": "Customer's $500 transfer to a relative shows completed but never arrived; wants it traced.",
 "requires_human": true,
 "confidence_note": "Could be fraud if the recipient account was a scammer's; flagged for review."}
```

Each record also carries an audit trail: routing team, sensitive flags, which safety rule overrode
the model and why, attempts, validation errors, tokens and latency.

## Run it

```bash
pip install -r ../requirements.txt
python fetch_complaints.py                          # builds data/complaints_sample.csv (400 rows)
export OPENAI_API_KEY=...                           # or GEMINI_API_KEY with --provider gemini (free tier)
python run_batch.py --provider openai               # -> outputs/predictions_<model>.jsonl (resumable)
python judge_label.py                               # blind LLM-judge labels (gpt-6-sol) -> data/labels_judge.csv
python evaluate.py --pred outputs/predictions_gpt-6-luna.jsonl --labels data/labels_judge.csv
python judge_label.py --compare                     # judge vs. any hand labels (label_tool.py)
python run_batch.py --provider gemini --input data/adversarial_cases.csv --out outputs/adversarial.jsonl
python evaluate.py --pred outputs/adversarial.jsonl --adversarial data/adversarial_cases.csv
python cost.py --pred outputs/predictions_gemini-3.5-flash-lite.jsonl
pytest -q tests                                     # offline tests, no API key needed
```

Other providers: `--provider groq | openai | ollama` (Ollama runs locally; nothing leaves the machine).
Any OpenAI-compatible endpoint works; see `llm_client.py`.

## The data slice (reproducible)

**Important:** on 14 August 2026 the CFPB stopped publishing complaint narratives. The live
database and API still return complaints, but without the narrative text (checked 26 Sept 2026).
I therefore use a public snapshot taken before the change:

| | |
|---|---|
| Source | Hugging Face `davidheineman/consumer-finance-complaints-large` (7.18M CFPB complaints with narratives), shard `train/0000.parquet` |
| Filter | non-empty narrative; product ∈ {Credit card, Credit card or prepaid card, Checking or savings account, Money transfer / virtual currency / money service}; received 2024-07-01 to 2024-12-31 |
| Sample | 400 complaints, `sample(random_state=42)` after sorting by complaint_id |
| Proof | `data/sample_manifest.json` records the shard's SHA-256, filter counts and every complaint id |

Why these products: they match PayGuard's business (cards, accounts, money movement). Credit
reporting, mortgages and student loans dominate the raw CFPB data but are not payments problems.

## Design

```
narrative ─► sanitise ─► keyword rules (sensitive? injection?) ─────────────────────┐
                 │                                                                  │
                 └─► LLM (temperature 0, JSON mode) ─► parse ─► Pydantic validate   │
                          ▲                                   │ invalid              │
                          └──── error fed back, retry ≤3 ◄────┘                      │
                                        │ still invalid / provider down              │
                                        ▼                                            ▼
                               conservative fallback ──────────► safety overrides ─► TriageRecord
                               (requires_human = true)           (can only escalate)
```

**Taxonomy (7 routing destinations)**, defined once in `taxonomy.py`:
`fraud_scam`, `transaction_dispute`, `payments_transfers`, `account_access`, `fees_interest_terms`,
`credit_reporting_collections`, `customer_service_other`. The hardest boundary is fraud vs.
dispute: "I didn't authorise it" is fraud, while "I authorised it but it's wrong" is a dispute.

**Treating the LLM as an unreliable component**

| Failure | Handling |
|---|---|
| Timeout, rate limit, 5xx | `llm_client.py`: exponential backoff with jitter, 4 retries, then `TransportError` → fallback |
| Provider refuses one request (content filter, unsupported parameter) | Drops `temperature`/`response_format` if that was the cause; otherwise → fallback for that complaint only. Wrong API key or model name stops the batch loudly (config error) |
| Not JSON / wrapped in prose or fences | `extract_json` tolerates fences and chatter; otherwise it's a validation error |
| Wrong keys, types, or an **invented category** | Pydantic `Literal` enums + `extra="forbid"`. The validator's error message is sent back to the model ("repair prompt") and it retries, up to 3 attempts |
| Still invalid, or provider down | Fallback record: `requires_human=true`, low confidence, category guessed by rules. The batch never crashes |
| Model rejects JSON mode | Client drops `response_format` once and relies on the prompt + validator |

**Safety layer (`guardrails.py`)**: deterministic, and it can only escalate:
- Keyword rules for the five sensitive flags (fraud allegation, legal threat, vulnerable customer,
  discrimination, financial hardship). They're deliberately over-inclusive: a false match costs a
  review, a miss could auto-close a fraud case.
- `requires_human` is forced to true on any sensitive flag (LLM or rule), high urgency, low confidence,
  or fallback. `llm_requires_human` is stored separately so the evaluation can measure each layer.
- **Prompt injection:** the complaint is wrapped in `<complaint>` tags; the system prompt says its
  content is data; fake closing tags are stripped; outputs are restricted to enums (so an attacker
  can't make the model emit arbitrary routing); and instruction-like text forces human review.
  The regex is crude and will miss paraphrased attacks. It's a tripwire, not a defence, and the
  real defence is that the worst an injected complaint can do is pick a wrong enum value, which
  a human then sees.
- Two real bugs were found and fixed, each with a regression test:
  (1) the red-team harness showed that sanitising before injection detection hid the `</complaint>`
  tag-escape attack (`adv-03`); detection now runs on the raw text.
  (2) A code review showed that keyword rules ran on the *truncated* text, so a legal threat in the
  middle of a very long complaint was invisible to them; rules now see the full text and only the
  LLM gets the truncated version.

## Evaluation: how we know the triage is any good

**What "correct" means**, in order of how much an error costs PayGuard:

1. **requires_human recall.** Of complaints a human judges need review, what share did the pipeline
   send to a human? A miss means a fraud or legal case could be auto-handled. This is the release gate
   (target: no misses on the labelled set, with the confidence interval reported honestly).
2. **Category accuracy / macro-F1.** A misroute costs a re-route and delay, not a safety failure.
3. **Urgency.** Exact agreement, within-one-level, and *severe under-calls* (gold high, predicted low).
4. **Summary faithfulness.** An automatic check flags any number in the summary that isn't in the
   narrative (invented amounts or dates), plus a human read of flagged cases.
5. **Pipeline health.** Fallback rate, retry rate, invalid-category attempts, overrides, latency, tokens.

**Labelling protocol: LLM-as-judge.** The 80-complaint evaluation set (fixed seed) is labelled by
`judge_label.py` using a *different, stronger* model (`gpt-6-sol`) from the one under test
(`gpt-6-luna`). The judge labels **blind**: it sees only the complaint, never the triage output,
so it can disagree rather than rubber-stamp. It uses the same taxonomy definitions, plus an explicit
fraud-vs-dispute rule, and records a one-line rationale per label (`data/labels_judge.csv`).

*Limitation, stated plainly:* judge labels measure **agreement between two models, not
correctness**. Two models from the same provider can share blind spots (e.g. both misread
"fraudulent" in a merchant dispute). The judge is validated with `python judge_label.py --compare`,
which reports judge-vs-human agreement on any complaints hand-labelled with `label_tool.py`.
In production the judge would only pre-screen; a weekly human audit remains the ground truth.

**Ablation.** The report shows `requires_human` recall for the LLM alone, rules alone, and combined,
so the value of each safety layer is measured, not asserted.

**Adversarial set** (`data/adversarial_cases.csv`, 11 hand-written cases): injection by instruction,
fake JSON, tag escape and role-play; subtle legal threat; elderly scam victim; Spanish text; empty
input; redacted-text hallucination bait; loose use of "fraudulent" in a merchant dispute.

**What I'd measure at 50k/month**, where hand-labelling everything is impossible:
- *Human override rate* by category: how often agents re-route or change urgency. It's the cheapest
  ground truth, generated by normal work.
- *Weekly audit sample:* 100 random + 50 auto-routed (`requires_human=false`) records, labelled by QA.
  The auto-routed stream is where silent safety failures hide.
- *Drift:* category mix, fallback rate, and agreement with CFPB-style issue fields (weak labels,
  already computed in section 3 of the report) week over week.
- *Downstream outcomes:* reopen rate and time-to-resolution by predicted category.
- *Canary set:* the labelled + adversarial sets are re-run on every prompt or model change, since the
  provider can change the model underneath you.

## Results

Run on 400 complaints with `gpt-6-luna`; reference labels for 80 of them from a blind `gpt-6-sol`
judge (see Labelling protocol: these measure agreement with a stronger model, not verified truth).
Full reports: `outputs/predictions_gpt-6-luna_report.md`, `outputs/adversarial_report.md`.

| Metric | Result (95% CI) |
|---|---|
| **requires_human recall** (n = 45 needing review) | **45/45 = 100%** (92–100%) |
| Unsafe misses (needed human, auto-routed) | **0** |
| Sensitive cases sent to human | 37/37 = 100% (91–100%) |
| requires_human precision | 45/63 = 71% (59–81%) |
| Category accuracy / macro-F1 (n = 80) | 73/80 = 91% (83–96%) / 0.89 |
| Urgency exact / within one level / severe under-calls | 84% (74–90%) / 100% / 0 |
| Summaries with a number not in the complaint | 2/400 (0–2%) |
| Fallback / retry rate (n = 400) | 0% / 1 retry; 0 invented categories |
| Adversarial cases passed | **11/11** |
| Tokens per complaint (in / out), latency p50 / p95 | 978 / 313; 4.3 s / 6.5 s |

**What the numbers say**
- **Safety holds on this sample:** no complaint that needed a human was auto-routed, and all 11 red-team
  cases (4 injection styles, subtle legal threat, elderly scam victim, Spanish, empty input) behaved correctly.
- **The ablation is the interesting finding:** here the LLM alone already reached 100% recall; the keyword
  rules alone reached only 62%. The rules added no recall on ordinary complaints. **On the adversarial set they
  did the real work:** in 3 of 11 cases the LLM set `requires_human=false` and the rules overrode it: the
  tag-escape injection (adv-03), the role-play injection (adv-11), and the dispute that used the word
  "fraudulent" (adv-10). In other words, the LLM was partly talked out of escalating by injected text, and the
  independent layer caught it. That is the case for defence in depth.
- **The cost of safety is automation rate:** 75% of complaints go to a human, and 29% of those didn't strictly
  need to (precision 71%). The pipeline is conservative by design. The next tuning step is to relax the
  "low confidence / high urgency ⇒ human" rules per category, while watching that recall stays at 100%.
- **Errors are boundary cases, not nonsense:** 4 of the 7 misroutes are fraud vs. dispute / credit-reporting
  (e.g. identity-theft accounts on a credit report), and in every case the model's `confidence_note` named the
  correct alternative, which shows the confidence notes are informative.
- **Grounding:** 2 summaries contain an amount not found verbatim in the complaint. Manual read: both are
  **totals the model computed**, not invented numbers. 10523563: $7,700 + $4,200 + $10,000 + $10,000 = $31,900
  (correct). 11110164: "$2,000" from debits of $400/$400/$800/$1,200, which is ambiguous ($2,000 or $2,800
  depending on which are duplicates). Lesson: the prompt should forbid derived totals ("quote amounts as
  stated") so every number in a summary is traceable to the source.

## Cost and latency

Measured on the 400-complaint run: **978 input + 313 output tokens per complaint** (retries included),
p50 latency 4.3 s. From `cost.py`:

| Model (Sept 2026 list prices, $ per 1M in / out) | $/complaint | $/month at 50k |
|---|---|---|
| **gpt-6-luna** ($0.10 / $0.50), used for this run | $0.00025 | **$13** |
| gemini-3.1-flash-lite ($0.25 / $1.50) | $0.0007 | $36 |
| gemini-3.8-flash ($0.75 / $3.75 promo; $1.50 / $7.50 from 2027) | $0.0019–0.0038 | $95–191 |
| gpt-6-sol ($2.00 / $10.00), the judge model | $0.0051 | $254 |

The actual cost of this whole prototype (400 triage calls + 80 judge calls + 11 adversarial) was well under $1.

**Implication:** LLM cost is negligible next to one support agent's salary. So model choice should
be driven by triage quality and **data residency** (see the memo), not price. 50k/month is ~70
complaints/hour. Latency (1–3 s per call) doesn't matter for an asynchronous queue, and a single
worker is enough. Hidden costs to watch: retries, "thinking" tokens on reasoning models (billed as
output), and human review time, which will dominate the total.

## What I cut, and why

- **No fine-tuning or embeddings classifier.** Out of scope; a labelled set of a few thousand
  override-corrected records would make a cheap classifier a strong baseline later.
- **No LLM-as-judge for summaries.** It adds a second model to trust; a number-grounding check plus
  human spot-checks is more defensible at this size.
- **No PII redaction step.** CFPB already redacts (XXXX). PayGuard's real complaints will not be, and
  the memo covers this: it's mandatory before any external API call.
- **LLM-judge labels instead of a full hand-labelled set.** Faster; the trade-off is that metrics
  show model-vs-model agreement. A human-validated subset is the fix, and the next step.
