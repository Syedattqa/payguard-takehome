"""Deterministic safety layer that runs alongside the LLM.

Why: the LLM can miss a sensitive case, or be talked out of flagging it by text in the complaint
(prompt injection). These regexes are deliberately high-recall / low-precision: a false match only
costs a human review, while a miss could auto-close a fraud or legal case. They can only ever
ESCALATE (set requires_human=True); they never downgrade the LLM's decision.
"""
import re

_W = r"(?:\w+\s+){0,3}"  # allow a few words in between, e.g. "did not ever authorize"

SENSITIVE_PATTERNS: dict[str, re.Pattern] = {
    "fraud_allegation": re.compile(
        r"\b(unauthori[sz]ed|fraud\w*|scam\w*|stolen|stole|identity theft|hack(ed|er)?|phish\w*|"
        r"account takeover|did\s*n[o']?t " + _W + r"(make|authori[sz]e|recogni[sz]e)|never " + _W +
        r"authori[sz]ed|not authori[sz]ed by me)\b", re.I),
    "legal_threat": re.compile(
        r"\b(lawyer|attorney|lawsuit|sue|suing|legal action|court|small claims|litigation|"
        r"attorney general|class action)\b", re.I),
    "vulnerable_customer": re.compile(
        r"\b(elderly|senior citizen|disab(led|ility)|dementia|alzheimer|deceased|passed away|"
        r"widow(ed)?|terminal|chemo\w*|hospitali[sz]ed|domestic (abuse|violence))\b", re.I),
    "discrimination": re.compile(
        r"\b(discriminat\w*|racis[mt]\w*|because (of )?my (race|religion|age|gender|disability|"
        r"national origin)|redlin\w*)\b", re.I),
    "financial_hardship": re.compile(
        r"\b(homeless|evict\w*|bankrupt\w*|can\s*n[o']?t (afford|pay) " + _W +
        r"(rent|food|bills|medication)|no money (for|to buy) (food|rent))\b", re.I),
}

INJECTION_PATTERN = re.compile(
    r"(ignore|disregard|forget|override)\s+(?:(?:all|any|the|your|my|of|previous|prior|above|earlier|"
    r"these|those)\s+){1,4}(instructions|rules|prompts?|guidelines)|system prompt|you are now|as an ai|"
    r"(set|mark|classify)\s+(this|the)?\s*(complaint|ticket)?\s*(as|to)\s+(low|high|medium|"
    r"not urgent)|requires_human|\"category\"\s*:|</?complaint>", re.I)

MAX_CHARS = 6000  # ~1.5k tokens; very long narratives keep head + tail


def detect_sensitive(text: str) -> list[str]:
    return [flag for flag, pattern in SENSITIVE_PATTERNS.items() if pattern.search(text)]


def detect_injection(text: str) -> bool:
    return bool(INJECTION_PATTERN.search(text))


def normalise(text: str) -> str:
    """Stop the complaint from closing our delimiter tag and collapse whitespace (no truncation)."""
    text = re.sub(r"</?\s*complaint\s*>", "[tag removed]", text or "", flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def sanitise(text: str) -> str:
    """normalise() plus a length cap, for what is sent to the LLM."""
    text = normalise(text)
    if len(text) > MAX_CHARS:
        head, tail = text[: MAX_CHARS * 2 // 3], text[-MAX_CHARS // 3:]
        text = f"{head} [... truncated ...] {tail}"
    return text
