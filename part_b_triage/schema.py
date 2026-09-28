"""Pydantic schemas.

LLMTriage   - what the model must return. Validated strictly: a category outside the taxonomy,
              a missing field or a wrong type is a validation error that triggers a retry.
TriageRecord - the final record the pipeline emits (the five fields from the brief plus audit metadata).
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from taxonomy import CATEGORY_IDS, SENSITIVE_FLAGS, URGENCY_LEVELS

Category = Literal[CATEGORY_IDS]          # type: ignore[valid-type]
Urgency = Literal[URGENCY_LEVELS]         # type: ignore[valid-type]
SensitiveFlag = Literal[SENSITIVE_FLAGS]  # type: ignore[valid-type]


class LLMTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unexpected keys = malformed output

    category: Category
    urgency: Urgency
    summary: str = Field(min_length=10, max_length=400)
    sensitive_flags: list[SensitiveFlag] = Field(default_factory=list)
    requires_human: bool
    confidence: Literal["high", "medium", "low"]
    confidence_note: str = Field(min_length=3, max_length=400)

    @field_validator("category", "urgency", "confidence", mode="before")
    @classmethod
    def _normalise(cls, v):
        # Tolerate harmless formatting drift ("High", " fraud_scam ") but NOT invented values.
        return v.strip().lower() if isinstance(v, str) else v


class TriageRecord(BaseModel):
    complaint_id: str
    category: Category
    routing_team: str
    urgency: Urgency
    summary: str
    requires_human: bool
    confidence_note: str
    sensitive_flags: list[str]
    # audit trail
    source: Literal["llm", "fallback"]
    overrides: list[str]          # why requires_human / urgency were forced, if they were
    llm_requires_human: bool | None
    rule_flags: list[str]
    attempts: int
    validation_errors: list[str]
    model: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
