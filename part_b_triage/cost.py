"""Per-complaint cost and the 50,000 complaints/month projection, from MEASURED token counts.

  python cost.py --pred outputs/predictions_gemini-3.5-flash-lite.jsonl
  python cost.py                       # no predictions yet: uses an assumed token profile

Prices are USD per 1M tokens, paid tier, taken from the providers' pricing pages in Sept 2026.
Check them before quoting: they change often, and Gemini 3.x Flash promo prices end 31 Dec 2026.
Sources: ai.google.dev/gemini-api/docs/pricing, developers.openai.com/api/docs/pricing
"""
import argparse
import json
from pathlib import Path

PRICES = {  # model: (input $/1M, output $/1M)
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),        # promo until 2026-12-31, then 1.50 / 7.50
    "gemini-3.8-flash (2027 price)": (1.50, 7.50),
    "gpt-6-luna": (0.10, 0.50),              # OpenAI, used for the reported run
    "gpt-6-sol": (2.00, 10.00),
}
MONTHLY_VOLUME = 50_000
ASSUMED_TOKENS = (1_100, 150)  # system prompt ~780 + typical narrative ~300 in; JSON out ~150


def measured_tokens(pred_path: str) -> tuple[float, float, float, float]:
    rows = [json.loads(l) for l in Path(pred_path).open() if l.strip()]
    n = len(rows)
    tin = sum(r["prompt_tokens"] for r in rows) / n     # includes retries: they cost money too
    tout = sum(r["completion_tokens"] for r in rows) / n
    lat = sorted(r["latency_s"] for r in rows)
    return tin, tout, lat[n // 2], lat[int(n * 0.95)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None)
    args = ap.parse_args()

    if args.pred:
        tin, tout, p50, p95 = measured_tokens(args.pred)
        print(f"Measured: {tin:.0f} input + {tout:.0f} output tokens per complaint; "
              f"latency p50 {p50:.1f}s, p95 {p95:.1f}s")
    else:
        tin, tout = ASSUMED_TOKENS
        print(f"Assumed: {tin} input + {tout} output tokens per complaint (run with --pred to measure)")

    print(f"\n{'model':<32}{'$/complaint':>14}{'$/month @50k':>15}")
    for model, (pin, pout) in PRICES.items():
        per = (tin * pin + tout * pout) / 1e6
        print(f"{model:<32}{per:>14.5f}{per * MONTHLY_VOLUME:>15.2f}")
    print("\nThroughput: 50k/month is ~1,700/day or ~70/hour, well within a single worker. "
          "Latency is irrelevant for an async queue; cost and quality decide the model.")


if __name__ == "__main__":
    main()
