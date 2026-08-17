from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance_raw.csv"
)


def main():

    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nPerformance distribution:")
    print(df["performance"].value_counts())

    print("\nStatistical summary:")
    print(df.describe())


if __name__ == "__main__":
    main()