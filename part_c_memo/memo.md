# Memo: Taking complaint triage to production

**To:** CTO, PayGuard  **From:** Attqa Batool, consulting data scientist  **Re:** Production plan for LLM complaint triage

## Recommendation

Put complaint triage into production first; fraud scoring second. Triage is lower-risk (a wrong
suggestion costs a re-route, not a lost payment), it can start as a human-assist tool, and it frees
agent time immediately. The prototype shows the core loop works. Model cost is negligible (under
$150/month at 50,000 complaints on a hosted model). The binding constraints are **data residency**
and **proving the pipeline never auto-handles a sensitive case**. This plan is built around those two.

## 1. Architecture

```
 Ticketing system (email, app, web forms)
        │ new complaint event
        ▼
 ┌──────────────── PayGuard in-country environment ─────────────────────────────┐
 │  Ingest queue ─► PII scrubber ─► Triage worker ───────► Triage store (audit) │
 │                  (names, card/   │  rules + LLM +          │                 │
 │                   account nos.,  │  schema validator       ▼                 │
 │                   phone, IDs)    ▼                    Ticketing API:         │
 │                           LLM inference server        set queue, priority,   │
 │                           (open-weights model on      summary, "needs human" │
 │                            in-country GPUs)                                   │
 │  Monitoring: metrics + weekly audit sample + canary set ◄── agent overrides   │
 └──────────────────────────────────────────────────────────────────────────────┘
```

- **Asynchronous by design.** 50k/month is about 70 per hour. One worker and one GPU are enough. Latency
  of a few seconds is irrelevant, so no real-time serving complexity.
- The pipeline **writes suggestions into the existing ticketing system**. It doesn't replace it.
  Agents keep working where they already do, and every correction becomes a label.
- The triage store keeps the input hash, prompt version, model version, raw output, overrides and
  final decision for every complaint. That audit trail is what you show the regulator.

## 2. Data: privacy, residency, retention

**Residency drives the model choice.** Customer data must stay in-country, and complaint narratives
are full of personal data. Three options:

| Option | Verdict |
|---|---|
| Hosted API abroad (as in the prototype) | **No** for production. Data leaves the country, even if the provider promises not to retain it. |
| Redact PII in-country, then call a hosted API | **Not as the primary design.** Free-text redaction is imperfect, and a missed account number is a reportable breach. Acceptable only if the regulator explicitly signs off. |
| **Open-weights model self-hosted in-country** (cloud region in-country or PayGuard's data centre) | **Recommended.** No data leaves. About $1.5–3k/month for one GPU server, still small relative to agent cost. |

Before committing, re-run the prototype's evaluation with 2–3 open-weights models of different sizes
against the hosted baseline. Adopt the smallest model that matches the baseline on requires_human recall.

**Other controls:**
- PII scrubbing *before* the LLM even in-country (least privilege), with the mapping kept in the
  ticketing system, not the triage store.
- Retention: raw LLM inputs/outputs kept 90 days for debugging, then only the structured record and
  audit fields. Align the complaint record itself with the regulator's retention rule.
- No training on customer complaints without a documented legal basis. Access to raw narratives
  is role-based and logged.

## 3. Monitoring: what breaks silently and how we catch it

| Silent failure | Detection |
|---|---|
| Model or prompt change shifts behaviour | Frozen **canary set** (labelled + adversarial cases) run on every deploy; release blocked if requires_human recall drops |
| Complaint mix drifts (new product, new scam wave) | Daily category/urgency distribution vs. 4-week baseline; alert on large shifts |
| Pipeline degrades to fallbacks | Alert if fallback or retry rate exceeds 2% in any hour |
| Sensitive case auto-routed | **Weekly audit** of 50 auto-routed records by QA; any miss triggers a review and a new rule/test |
| Summaries start inventing facts | Automated number-grounding check on every record; sample human review |
| Agents rubber-stamp suggestions (automation bias) | Track override rate per agent; near-zero overrides is a warning, not a success |
| Language shift | Share of non-English / code-mixed text; the prototype was only evaluated on English |

The single most useful metric is the **agent override rate by category**. It's free, continuous,
and it's the ground truth the business actually cares about.

## 4. Phased rollout with a human in the loop

| Phase | What happens | Exit gate |
|---|---|---|
| 0. Offline (3 weeks) | Label 1,000 historical PayGuard complaints; build the taxonomy with support leads; choose the in-country model | requires_human recall ≥ 98% with lower CI ≥ 95%; category accuracy ≥ 85% |
| 1. Shadow (4 weeks) | Runs on live traffic; agents don't see output; compare with agents' actual routing | Same gates on live data; fallback < 1% |
| 2. Assist (6–8 weeks) | Agents see suggested queue, priority and summary; every ticket still confirmed by a human | Override rate stable; handling time falls; no sensitive misses in audits |
| 3. Partial automation | Auto-route (never auto-close) only `requires_human=false` + high confidence in low-risk categories; everything else stays with humans | Monthly audit clean; kill switch tested |

Nothing is ever auto-*closed* at any phase: automation decides where a complaint goes, not how it ends.
A kill switch reverts to manual routing in one step.

## 5. What I need from PayGuard

1. **Data:** 5–10k historical complaints with the team that actually resolved each one, in all
   languages customers use.
2. **People:** two senior support agents for about 3 days of labelling and taxonomy review; a named
   compliance owner who signs off the sensitive-case definitions and escalation SLAs.
3. **Regulatory clarity:** written confirmation of what counts as "customer data leaving the country"
   (does PII-redacted text count?) and required retention periods.
4. **Infrastructure:** in-country GPU capacity (one inference server plus a standby), ticketing-system
   API access, and your existing monitoring stack.
5. **A decision owner** who can accept a measured error rate and approve each phase gate.

**Main risks:** (1) local-language and code-mixed complaints perform worse than the English prototype,
which is mitigated by labelling in those languages in phase 0; (2) automation bias among agents,
mitigated by override tracking and audits; (3) scope creep into auto-replies to customers, which
should be a separate project with its own safety case.
