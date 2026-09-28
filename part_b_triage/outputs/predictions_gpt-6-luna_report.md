# Triage evaluation: `predictions_gpt-6-luna.jsonl`

Model: gpt-6-luna | predictions: 400

## 1. Pipeline health (all predictions)

- Fallback records (LLM never produced valid output): 0/400 = 0% (95% CI 0%-1%)
- Needed a retry (first reply invalid): 1/400 = 0% (95% CI 0%-1%)
- Any validation error seen: 1/400 = 0% (95% CI 0%-1%)
- Invalid category attempts (hallucinated labels caught by the validator): 0
- requires_human overridden by safety rules: 7/400 = 2% (95% CI 1%-4%)
- Routed to a human overall: 300/400 = 75% (95% CI 71%-79%)
- Latency per complaint: p50 4.26s, p95 6.50s
- Tokens per complaint: 978 in / 313 out (mean)

## 2. Summary grounding (automatic check)

Summaries containing a number not found in the complaint: 2/400 = 0% (95% CI 0%-2%)
- 10523563: ['$31,900']
- 11110164: ['$2,000']

## 3. Agreement with CFPB issue field (weak labels, all predictions)

Coverage: 166/400 complaints have a mappable issue. Agreement: 106/166 = 64% (95% CI 56%-71%). Consumer-chosen issues are noisy, so this is a trend signal, not ground truth.

## 4. Against hand labels (n = 80)

### requires_human (safety)

- **Recall** (gold=needs human, pipeline sent to human): 45/45 = 100% (95% CI 92%-100%)
- Precision: 45/63 = 71% (95% CI 59%-81%)
- Ablation, recall of each safety layer on its own:
  - LLM decision alone: 45/45 = 100% (95% CI 92%-100%)
  - Keyword rules alone: 28/45 = 62% (95% CI 48%-75%)
  - Combined (shipped): 45/45 = 100% (95% CI 92%-100%)
- Unsafe misses (need human, auto-routed): 0 []
- Sensitive cases (gold has any sensitive flag) sent to human: 37/37 = 100% (95% CI 91%-100%)

### Category (routing)

- Accuracy: 73/80 = 91% (95% CI 83%-96%)
- Macro-F1: 0.89
- Confusion matrix (rows = gold, columns = predicted):

```
                              fraud_scam  transactio  payments_t  account_ac  fees_inter  credit_rep  customer_s
fraud_scam                            20           0           0           0           0           0           0
transaction_dispute                    1          10           0           0           0           0           0
payments_transfers                     0           0          12           1           0           1           0
account_access                         0           0           0          16           0           0           0
fees_interest_terms                    1           0           0           0           5           0           0
credit_reporting_collections           2           0           0           0           0           8           0
customer_service_other                 0           0           0           0           0           1           2
```

### Urgency

- Exact agreement: 67/80 = 84% (95% CI 74%-90%)
- Within one level: 80/80 = 100% (95% CI 95%-100%)
- Severe under-calls (gold high, predicted low): 0
- Linear-weighted Cohen's kappa: 0.67

## 5. Disagreements to read (error analysis)

| id | gold | predicted | urgency gold/pred | model's confidence note |
|---|---|---|---|---|
| 10589967 | payments_transfers | credit_reporting_collections | medium/medium | This could also be payments_transfers or account_access because the complaint centers partly on failed autopay |
| 9963476 | transaction_dispute | fraud_scam | medium/medium | The customer does not explain whether they authorized the charges, so this could also be a transaction_dispute |
| 9891694 | fees_interest_terms | fraud_scam | medium/medium | This could also fit fees_interest_terms because the complaint centers on recurring fees the customer says were |
| 11059443 | customer_service_other | credit_reporting_collections | low/medium | This could also fit customer_service_other because the main issue is an account the customer says they did not |
| 11164547 | credit_reporting_collections | fraud_scam | high/medium | This could also be categorized as credit_reporting_collections because the requested remedy is to block and de |
| 10436040 | credit_reporting_collections | fraud_scam | medium/high | This could also fit credit_reporting_collections because the customer is disputing an account on their credit  |
| 10420316 | payments_transfers | account_access | medium/high | This could also be classified as payments_transfers because the registration problem may affect the customer’s |