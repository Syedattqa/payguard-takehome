"""Triage a batch of complaints and write one JSON line per complaint.

  python run_batch.py --provider gemini --limit 50
  python run_batch.py --provider gemini --input data/adversarial_cases.csv --out outputs/adversarial.jsonl

Resumable: complaint ids already present in the output file are skipped, so a crash or a
rate-limit stop halfway through loses nothing. Rerun the same command to continue.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from llm_client import LLMClient
from triage import triage_one

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--input", default=str(HERE / "data/complaints_sample.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2, help="keep low on free tiers (rate limits)")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between requests per worker")
    args = ap.parse_args()

    client = LLMClient(args.provider, args.model)
    out = Path(args.out or HERE / f"outputs/predictions_{client.model.replace('/', '_')}.jsonl")
    out.parent.mkdir(exist_ok=True)

    df = pd.read_csv(args.input, dtype={"complaint_id": str})
    if args.limit:
        df = df.head(args.limit)
    done = set()
    if out.exists():
        done = {json.loads(line)["complaint_id"] for line in out.open() if line.strip()}
    todo = df[~df["complaint_id"].isin(done)]
    print(f"{len(df)} complaints, {len(done)} already done, {len(todo)} to run -> {out}")

    def work(row):
        rec = triage_one(row.complaint_id, row.narrative, client)
        if args.pause:
            time.sleep(args.pause)
        return rec

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a") as f:
        for i, rec in enumerate(pool.map(work, todo.itertuples(index=False)), 1):
            f.write(rec.model_dump_json() + "\n")
            f.flush()
            flag = "FALLBACK" if rec.source == "fallback" else ("retry" if rec.attempts > 1 else "")
            print(f"[{i}/{len(todo)}] {rec.complaint_id} {rec.category:<28} {rec.urgency:<6} "
                  f"human={rec.requires_human!s:<5} {flag}")
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
