import pandas as pd


def main():
    df = pd.read_csv("../data/creditcard.csv")

    df = df.sort_values("Time").reset_index(drop=True)

    print("Time range:")
    print(f"Start: {df['Time'].min():,.0f}")
    print(f"End:   {df['Time'].max():,.0f}")

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    for name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(f"\n{name}")
        print(f"Rows: {len(split):,}")
        print(f"Fraud: {split['Class'].sum()}")
        print(f"Fraud rate: {split['Class'].mean() * 100:.4f}%")
        print(
            f"Time: {split['Time'].min():,.0f}"
            f" → {split['Time'].max():,.0f}"
        )


if __name__ == "__main__":
    main()