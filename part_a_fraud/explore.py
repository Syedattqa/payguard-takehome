import pandas as pd


def main():
    df = pd.read_csv("../data/creditcard.csv")

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass distribution (%):")
    print(df["Class"].value_counts(normalize=True) * 100)

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()