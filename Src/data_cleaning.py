from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "Data"
    / "raw"
    / "student_performance_dirty.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "student_performance_clean.csv"
)


def load_data():

    df = pd.read_csv(INPUT_PATH)

    print("Dirty dataset loaded.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print("\nDuplicate removal:")
    print(f"Removed: {before - after}")

    return df


def clean_categorical_values(df):
    df["gender"] = (
        df["gender"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    df["internet_access"] = (
        df["internet_access"]
        .astype("string")
        .str.strip()
        .str.title()
    )
    df["extracurricular"] = (
        df["extracurricular"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    categorical_columns = [
        "parental_education",
        "family_income",
        "study_environment",
        "performance",
    ]

    for column in categorical_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.title()
        )

    print("\nCategorical values cleaned.")

    return df


def clean_age(df):

    invalid_age = (
        (df["age"] < 15)
        | (df["age"] > 22)
    )

    count = invalid_age.sum()

    print("\nInvalid ages found:", count)

    df.loc[invalid_age, "age"] = pd.NA

    median_age = df["age"].median()

    df["age"] = df["age"].fillna(
        median_age
    )

    print("Invalid ages replaced with median.")

    return df

def clean_study_hours(df):

    invalid_hours = (
        (df["study_hours"] < 0)
        | (df["study_hours"] > 8)
    )

    count = invalid_hours.sum()

    print("\nInvalid study-hour records:", count)

    df.loc[invalid_hours, "study_hours"] = pd.NA

    median_hours = df["study_hours"].median()

    df["study_hours"] = df["study_hours"].fillna(
        median_hours
    )

    return df

def clean_attendance(df):

    invalid_attendance = (
        (df["attendance_percentage"] < 0)
        | (df["attendance_percentage"] > 100)
    )

    count = invalid_attendance.sum()

    print("\nInvalid attendance records:", count)

    df.loc[
        invalid_attendance,
        "attendance_percentage"
    ] = pd.NA

    median_attendance = (
        df["attendance_percentage"].median()
    )

    df["attendance_percentage"] = (
        df["attendance_percentage"]
        .fillna(median_attendance)
    )

    return df


def handle_missing_numeric_values(df):

    numeric_columns = [
        "study_hours",
        "attendance_percentage",
        "previous_score",
        "sleep_hours",
    ]

    print("\nMissing numerical values before filling:")

    print(
        df[numeric_columns]
        .isnull()
        .sum()
    )

    for column in numeric_columns:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )

    print("\nMissing numerical values filled using median.")

    return df


def handle_missing_categorical_values(df):

    categorical_columns = [
        "internet_access",
    ]

    print("\nMissing categorical values before filling:")

    print(
        df[categorical_columns]
        .isnull()
        .sum()
    )

    for column in categorical_columns:

        mode_value = df[column].mode()[0]

        df[column] = df[column].fillna(
            mode_value
        )

    print("\nMissing categorical values filled using mode.")

    return df


def validate_data(df):

    print("DATA VALIDATION")

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nAge range:")
    print(
        df["age"].min(),
        "to",
        df["age"].max()
    )

    print("\nStudy-hour range:")
    print(
        df["study_hours"].min(),
        "to",
        df["study_hours"].max()
    )

    print("\nAttendance range:")
    print(
        df["attendance_percentage"].min(),
        "to",
        df["attendance_percentage"].max()
    )


def main():

    df = load_data()

    df = remove_duplicates(df)

    df = clean_categorical_values(df)

    df = clean_age(df)

    df = clean_study_hours(df)

    df = clean_attendance(df)

    df = handle_missing_numeric_values(df)

    df = handle_missing_categorical_values(df)

    validate_data(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nClean dataset saved successfully!")

    print(
        f"Location:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()