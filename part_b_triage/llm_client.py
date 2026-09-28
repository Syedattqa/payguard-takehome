"""Provider-agnostic LLM client.

Most providers (Gemini, Groq, OpenAI, Ollama) expose an OpenAI-compatible Chat Completions API,
so one client with a different base_url covers all of them. This class owns TRANSPORT failures
(timeouts, rate limits, 5xx) with exponential backoff. CONTENT failures (bad JSON, invalid
category) are handled one level up in triage.py, because fixing those needs a different prompt.
"""
import os
import random
import time
from dataclasses import dataclass

PROVIDERS = {
    # name: (base_url, env var holding the key, default model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.5-flash-lite"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-6-luna"),  # cheapest current OpenAI model (Sept 2026)
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b"),  # local: no key, no data leaves the machine
}


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class TransportError(RuntimeError):
    """The provider could not give an answer for this request (down, rate-limited, or refused it).
    triage.py turns this into a fallback record, so one bad request never stops the batch."""


class LLMClient:
    def __init__(self, provider: str = "gemini", model: str | None = None, timeout_s: float = 30,
                 max_retries: int = 4, json_mode: bool = True):
        from openai import OpenAI  # imported here so tests can run without the package configured

        base_url, key_env, default_model = PROVIDERS[provider]
        api_key = os.environ.get(key_env, "") if key_env else "ollama"
        if key_env and not api_key:
            raise SystemExit(f"Set {key_env} to use provider '{provider}'.")
        self.model = model or os.environ.get("LLM_MODEL") or default_model
        self.max_retries = max_retries
        self.json_mode = json_mode
        # SDK-level retries off: we control retries ourselves so they are visible and logged.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)

    def complete(self, messages: list[dict]) -> LLMResponse:
        import openai

        kwargs = {"model": self.model, "messages": messages, "temperature": 0}
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                r = self._client.chat.completions.create(**kwargs)
                usage = r.usage
                return LLMResponse(
                    text=r.choices[0].message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_s=time.perf_counter() - start,
                )
            except openai.BadRequestError as e:
                msg = str(e).lower()
                # Some models reject response_format or temperature=0; drop the option once and retry.
                if self.json_mode and "response_format" in msg:
                    self.json_mode = False
                    kwargs.pop("response_format", None)
                    continue
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                # Anything else (e.g. content filter on this complaint): fail this request only.
                raise TransportError(f"BadRequest: {msg[:200]}") from e
            except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError):
                raise  # wrong key or model name: a config error, so stop the batch loudly
            except (openai.APITimeoutError, openai.APIConnectionError,
                    openai.RateLimitError, openai.InternalServerError) as e:
                if attempt == self.max_retries:
                    raise TransportError(f"{type(e).__name__} after {attempt + 1} attempts") from e
                # Exponential backoff with jitter: 1s, 2s, 4s, 8s (+ up to 1s) to respect rate limits.
                time.sleep(2 ** attempt + random.random())
        raise TransportError("unreachable")
