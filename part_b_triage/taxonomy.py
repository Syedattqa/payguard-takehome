"""Routing taxonomy: the single source of truth for categories, urgency and sensitive flags.

Everything else (prompt, schema validation, evaluation, labelling tool) imports from here,
so adding or renaming a category is a one-file change.
"""

# category id -> (team the ticket is routed to, definition shown to the LLM and to human labellers)
CATEGORIES: dict[str, tuple[str, str]] = {
    "fraud_scam": (
        "Fraud Investigations",
        "Consumer says they did NOT authorise a transaction, their card/account/identity was stolen or "
        "taken over, or they were tricked by a scammer into sending money.",
    ),
    "transaction_dispute": (
        "Disputes & Chargebacks",
        "Consumer made or authorised the transaction but disputes it: merchant didn't deliver, wrong "
        "amount, double charge, refund not received, billing error, chargeback handling.",
    ),
    "payments_transfers": (
        "Payments Operations",
        "Money movement problems: transfer, wire, remittance, Zelle/P2P, deposit or bill payment that "
        "failed, is delayed, went to the wrong place, or a payment that was not credited/posted.",
    ),
    "account_access": (
        "Account Services",
        "Consumer cannot use their account or money: account frozen, restricted, closed by the company, "
        "funds on hold, locked out of login, card blocked or declined.",
    ),
    "fees_interest_terms": (
        "Product & Pricing",
        "Unexpected or disputed fees, interest, APR, overdraft charges, rewards/promotions/bonuses not "
        "honoured, credit limit changes, or terms changed.",
    ),
    "credit_reporting_collections": (
        "Credit Bureau & Collections",
        "How this account is reported to credit bureaus (late marks, wrong balance), or debt collection "
        "activity, collection calls, charge-offs.",
    ),
    "customer_service_other": (
        "General Support",
        "Poor service, unresponsive staff, communication problems, account opening, or anything that "
        "does not fit the other categories.",
    ),
}
CATEGORY_IDS = tuple(CATEGORIES)

URGENCY_LEVELS = ("low", "medium", "high")
URGENCY_GUIDE = {
    "high": "Ongoing financial harm or loss of access to money the consumer needs now; active fraud; "
            "vulnerable consumer; threat of legal action; or large sums at immediate risk.",
    "medium": "Money is at stake or a dispute is unresolved, but nothing is getting worse by the hour.",
    "low": "Service-quality complaints, information requests, small fees, or issues already resolved.",
}

# Cases that must never be auto-closed. Detected by the LLM AND by deterministic rules (guardrails.py).
SENSITIVE_FLAGS = (
    "fraud_allegation",     # alleges fraud, theft, scam or identity theft
    "legal_threat",         # lawyer, lawsuit, court, attorney general, legal action
    "vulnerable_customer",  # elderly, disability, serious illness, bereavement, domestic abuse
    "discrimination",       # alleges unfair treatment because of a protected characteristic
    "financial_hardship",   # cannot pay rent/food, eviction, bankruptcy, severe hardship
)
