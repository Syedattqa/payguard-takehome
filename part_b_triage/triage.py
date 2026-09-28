"""Core pipeline: one complaint in, one validated TriageRecord out. Never raises on bad model output.

Flow:
  sanitise -> rule-based flags -> LLM call -> parse JSON -> validate (pydantic)
     invalid? -> send the validation error back to the model and retry (up to MAX_ATTEMPTS)
     still invalid, or provider down? -> conservative FALLBACK record (always requires_human)
  -> apply safety overrides (rules can only escalate) -> TriageRecord
"""
import json
import re

from pydantic import ValidationError

from guardrails import detect_injection, detect_sensitive, normalise, sanitise
from llm_client import TransportError
from prompts import build_messages, repair_message
from schema import LLMTriage, TriageRecord
from taxonomy import CATEGORIES

MAX_ATTEMPTS = 3


def extract_json(text: str) -> dict:
    """Parse the model's reply. Tolerates code fences or chatter around one JSON object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)  # first '{' to last '}'
        if not match:
            raise ValueError(f"no JSON object in reply: {text[:120]!r}")
        return json.loads(match.group(0))


def _fallback(rule_flags: list[str], reason: str) -> LLMTriage:
    """Used when the LLM cannot produce a valid answer. Conservative by design: route to a human."""
    return LLMTriage(
        category="fraud_scam" if "fraud_allegation" in rule_flags else "customer_service_other",
        urgency="high" if rule_flags else "medium",
        summary="Automatic triage failed; a human must read the original complaint.",
        sensitive_flags=rule_flags,
        requires_human=True,
        confidence="low",
        confidence_note=f"Fallback record, not an LLM judgement: {reason}"[:400],
    )


def triage_one(complaint_id: str, narrative: str, client) -> TriageRecord:
    # Injection check runs on the RAW text: sanitise() removes fake </complaint> tags, which would
    # otherwise hide the strongest injection signal. (Found by adversarial case adv-03.)
    injection = detect_injection(narrative or "")
    # Keyword rules see the FULL text; only the LLM gets the truncated version. (Found in code review:
    # a legal threat in the middle of a long complaint was being cut off before the rules ran.)
    rule_flags = detect_sensitive(normalise(narrative))
    text = sanitise(narrative)

    messages = build_messages(text)
    errors: list[str] = []
    result: LLMTriage | None = None
    prompt_tokens = completion_tokens = 0
    latency = 0.0
    attempts = 0

    if len(text) < 20:
        errors.append("narrative too short to triage")
    else:
        for attempts in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = client.complete(messages)
            except TransportError as e:   # provider down/refused after retries: fall back, don't crash
                errors.append(f"transport: {e}")
                break
            prompt_tokens += resp.prompt_tokens
            completion_tokens += resp.completion_tokens
            latency += resp.latency_s
            try:
                result = LLMTriage.model_validate(extract_json(resp.text))
                break
            except (ValueError, ValidationError) as e:
                err = str(e).replace("\n", " ")
                errors.append(err[:300])
                messages = messages + [{"role": "assistant", "content": resp.text[:2000]},
                                       repair_message(err)]

    source = "llm" if result else "fallback"
    llm_requires_human = result.requires_human if result else None
    if result is None:
        result = _fallback(rule_flags, errors[-1] if errors else "unknown error")

    # ---- Safety overrides: can only escalate, never downgrade ----
    overrides = []
    requires_human = result.requires_human
    flags = sorted(set(result.sensitive_flags) | set(rule_flags))
    if rule_flags and not requires_human:
        overrides.append(f"rule flags {rule_flags} forced human review")
    if result.sensitive_flags and not requires_human:
        overrides.append("LLM set a sensitive flag but requires_human=false; forced true")
    if result.urgency == "high" and not requires_human:
        overrides.append("high urgency forced human review")
    if result.confidence == "low" and not requires_human:
        overrides.append("low confidence forced human review")
    if overrides:
        requires_human = True

    note = result.confidence_note
    if injection:
        requires_human = True
        overrides.append("possible prompt injection in complaint text")
        note = f"Complaint text contains instruction-like content (possible prompt injection). {note}"[:400]

    return TriageRecord(
        complaint_id=str(complaint_id),
        category=result.category,
        routing_team=CATEGORIES[result.category][0],
        urgency=result.urgency,
        summary=result.summary,
        requires_human=requires_human,
        confidence_note=note,
        sensitive_flags=flags,
        source=source,
        overrides=overrides,
        llm_requires_human=llm_requires_human,
        rule_flags=rule_flags,
        attempts=attempts,
        validation_errors=errors,
        model=getattr(client, "model", "unknown"),
        latency_s=round(latency, 3),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
