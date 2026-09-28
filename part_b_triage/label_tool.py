"""Hand-label a random subset of complaints, BLIND to the model's predictions.

  python label_tool.py            # label (resumes where you stopped; Ctrl+C any time)
  python label_tool.py --n 80     # size of the labelled set (fixed seed, so always the same 80)

Writes data/labels.csv. You can also open that CSV in Excel/Sheets and edit it directly.
Labelling rules (what "correct" means) are in README.md > "Evaluation" and in taxonomy.py.
"""
import argparse
import csv
import textwrap
from pathlib import Path

import pandas as pd

from taxonomy import CATEGORIES, CATEGORY_IDS, SENSITIVE_FLAGS, URGENCY_GUIDE

HERE = Path(__file__).parent
LABELS = HERE / "data/labels.csv"
FIELDS = ["complaint_id", "category", "urgency", "requires_human", "sensitive_flags", "notes"]


def ask(prompt: str, valid: set[str]) -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in valid:
            return answer
        print(f"  choose one of: {sorted(valid)}")


def main(n: int, seed: int) -> None:
    df = pd.read_csv(HERE / "data/complaints_sample.csv", dtype={"complaint_id": str})
    subset = df.sample(n=min(n, len(df)), random_state=seed)
    done = set(pd.read_csv(LABELS, dtype=str)["complaint_id"]) if LABELS.exists() else set()
    new_file = not LABELS.exists()

    cat_menu = "\n".join(f"  {i}. {cid:<30} {CATEGORIES[cid][1][:70]}..." for i, cid in enumerate(CATEGORY_IDS, 1))
    flag_menu = "  " + "  ".join(f"{i}={f}" for i, f in enumerate(SENSITIVE_FLAGS, 1))

    with LABELS.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        todo = subset[~subset["complaint_id"].isin(done)]
        for k, row in enumerate(todo.itertuples(index=False), len(done) + 1):
            print("\n" + "=" * 100)
            print(f"[{k}/{len(subset)}] complaint {row.complaint_id} | CFPB product: {row.product} | issue: {row.issue}")
            print(textwrap.fill(row.narrative, 100))
            print("-" * 100 + "\n" + cat_menu)
            cat = CATEGORY_IDS[int(ask("category #: ", {str(i) for i in range(1, len(CATEGORY_IDS) + 1)})) - 1]
            print("  " + " | ".join(f"{lvl[0]}={text[:45]}..." for lvl, text in URGENCY_GUIDE.items()))
            urg = {"l": "low", "m": "medium", "h": "high"}[ask("urgency (l/m/h): ", {"l", "m", "h"})]
            print(flag_menu)
            raw = input("sensitive flags (e.g. 1,2 or blank for none): ").strip()
            flags = [SENSITIVE_FLAGS[int(x) - 1] for x in raw.split(",") if x.strip().isdigit()
                     and 1 <= int(x) <= len(SENSITIVE_FLAGS)]
            human = ask("requires human review? (y/n): ", {"y", "n"}) == "y"
            notes = input("notes (optional): ").strip()
            writer.writerow({"complaint_id": row.complaint_id, "category": cat, "urgency": urg,
                             "requires_human": human, "sensitive_flags": ";".join(flags), "notes": notes})
            fh.flush()
    print(f"\nLabels saved to {LABELS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)  # different seed from the sampling step on purpose
    args = ap.parse_args()
    main(args.n, args.seed)
