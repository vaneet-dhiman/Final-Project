from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance_dirty.csv"
)



def main():

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 60)
    print("DIRTY DATASET PROFILE")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nNumerical summary:")
    print(df.describe())

    print("\nUnique values in categorical columns:")

    categorical_columns = [
        "gender",
        "internet_access",
        "parental_education",
        "family_income",
        "extracurricular",
        "study_environment",
        "performance",
    ]

    for column in categorical_columns:

        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))


if __name__ == "__main__":
    main()