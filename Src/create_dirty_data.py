from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance_raw.csv"
)

DIRTY_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance_dirty.csv"
)


def create_dirty_dataset():

    df = pd.read_csv(RAW_DATA_PATH)

    print("Original dataset:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    missing_columns = [
        "study_hours",
        "attendance_percentage",
        "previous_score",
        "sleep_hours",
        "internet_access",
    ]

    for column in missing_columns:

        random_indices = np.random.choice(
            df.index,
            size=20,
            replace=False
        )

        df.loc[random_indices, column] = np.nan

    duplicate_rows = df.sample(
        n=15,
        random_state=RANDOM_SEED
    )

    df = pd.concat(
        [df, duplicate_rows],
        ignore_index=True
    )


    gender_indices = df.sample(
        n=10,
        random_state=10
    ).index

    df.loc[gender_indices, "gender"] = " male "

    internet_indices = df.sample(
        n=10,
        random_state=20
    ).index

    df.loc[internet_indices, "internet_access"] = "yes"

    invalid_age_indices = df.sample(
        n=5,
        random_state=30
    ).index

    df.loc[invalid_age_indices, "age"] = 45

    study_indices = df.sample(
        n=5,
        random_state=40
    ).index

    df.loc[study_indices, "study_hours"] = 25


    attendance_indices = df.sample(
        n=5,
        random_state=50
    ).index

    df.loc[
        attendance_indices,
        "attendance_percentage"
    ] = 150


    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)


    df.to_csv(
        DIRTY_DATA_PATH,
        index=False
    )

    print("\nDirty dataset created successfully!")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print(
        f"\nSaved to:\n{DIRTY_DATA_PATH}"
    )



if __name__ == "__main__":
    create_dirty_dataset()