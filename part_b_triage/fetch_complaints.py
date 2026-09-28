"""Build the reproducible complaint slice used in Part B.

Why not the CFPB API? On 14 Aug 2026 the CFPB stopped publishing consumer narratives; the live API
and exports now return complaints without the narrative text. This script uses a public snapshot of
the CFPB database taken before that change (7.18M complaints, including narratives), hosted on
Hugging Face: davidheineman/consumer-finance-complaints-large. It downloads ONE parquet shard
(~124 MB), filters it and draws a seeded random sample.

Slice definition (the README repeats this):
  source      : shard train/0000.parquet of the dataset above (parquet conversion branch)
  filter      : non-empty narrative; product in PRODUCTS; date_received in [DATE_MIN, DATE_MAX)
  sample      : 400 rows, pandas .sample(random_state=42) after sorting by complaint_id
  output      : data/complaints_sample.csv  + data/sample_manifest.json (ids, file sha256, counts)

Usage:  python fetch_complaints.py            (or --n 300, --seed 7)
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

DATASET = "davidheineman/consumer-finance-complaints-large"
SHARD_URL = (f"https://huggingface.co/datasets/{DATASET}/resolve/refs%2Fconvert%2Fparquet/"
             "default/train/0000.parquet")
PRODUCTS = [  # the products closest to PayGuard's business (cards, accounts, money movement)
    "Credit card",
    "Credit card or prepaid card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
]
DATE_MIN, DATE_MAX = "2024-07-01", "2025-01-01"
DATA = Path(__file__).parent / "data"


def download(url: str, dest: Path) -> str:
    if not dest.exists():
        print(f"Downloading {url} ...")
        partial = dest.with_suffix(".part")  # rename only when complete, so a crash can't leave a truncated file
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        partial.rename(dest)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def main(n: int, seed: int) -> None:
    DATA.mkdir(exist_ok=True)
    shard = DATA / "cfpb_shard_0000.parquet"
    sha = download(SHARD_URL, shard)

    cols = ["complaint_id", "date_received", "product", "sub_product", "issue", "sub_issue",
            "company", "state", "tags", "complaint_what_happened"]
    df = pd.read_parquet(shard, columns=cols)
    total = len(df)
    df["narrative"] = df.pop("complaint_what_happened").fillna("").str.strip()
    received = pd.to_datetime(df["date_received"], errors="coerce")
    df = df[(df["narrative"].str.len() > 0)
            & df["product"].isin(PRODUCTS)
            & (received >= DATE_MIN) & (received < DATE_MAX)]
    pool = len(df)
    sample = df.sort_values("complaint_id").sample(n=min(n, pool), random_state=seed)
    sample = sample.sort_values("complaint_id").reset_index(drop=True)

    sample.to_csv(DATA / "complaints_sample.csv", index=False)
    manifest = {
        "dataset": DATASET, "shard_url": SHARD_URL, "shard_sha256": sha,
        "filters": {"products": PRODUCTS, "date_min": DATE_MIN, "date_max_exclusive": DATE_MAX,
                    "narrative": "non-empty"},
        "seed": seed, "rows_in_shard": total, "rows_matching_filter": pool, "rows_sampled": len(sample),
        "product_counts": sample["product"].value_counts().to_dict(),
        "complaint_ids": sample["complaint_id"].astype(str).tolist(),
    }
    (DATA / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{total:,} rows in shard -> {pool:,} match filter -> {len(sample)} sampled")
    print(sample["product"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n, args.seed)
