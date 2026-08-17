from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "student_performance_clean.csv"
)


def load_data():

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def descriptive_statistics(df):

    numerical_columns = [
        "age",
        "study_hours",
        "attendance_percentage",
        "previous_score",
        "assignment_score",
        "midterm_score",
        "absences",
        "sleep_hours",
        "final_score"
    ]

    print("DESCRIPTIVE STATISTICS")

    statistics = df[numerical_columns].agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )

    print(statistics.round(2))


def correlation_analysis(df):

    numerical_columns = [
        "age",
        "study_hours",
        "attendance_percentage",
        "previous_score",
        "assignment_score",
        "midterm_score",
        "absences",
        "sleep_hours",
        "final_score"
    ]

    correlation = (
        df[numerical_columns]
        .corr()["final_score"]
        .sort_values(ascending=False)
    )

    print("CORRELATION WITH FINAL SCORE")

    print(correlation.round(3))


def group_statistics(df):

    print("PASS VS FAIL STATISTICS")

    variables = [
        "study_hours",
        "attendance_percentage",
        "previous_score",
        "assignment_score",
        "midterm_score",
        "absences",
        "sleep_hours",
        "final_score"
    ]

    result = (
        df.groupby("performance")[variables]
        .agg(["mean", "median", "std"])
        .round(2)
    )

    print(result)


def independent_t_test(df, column):

    pass_group = df.loc[
        df["performance"] == "Pass",
        column
    ].dropna()

    fail_group = df.loc[
        df["performance"] == "Fail",
        column
    ].dropna()

    statistic, p_value = stats.ttest_ind(
        pass_group,
        fail_group,
        equal_var=False
    )

    print(f"T-TEST: {column}")

    print(f"Pass mean: {pass_group.mean():.2f}")
    print(f"Fail mean: {fail_group.mean():.2f}")
    print(f"T-statistic: {statistic:.3f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference.")
    else:
        print("Result: No statistically significant difference.")


def main():

    df = load_data()

    descriptive_statistics(df)

    correlation_analysis(df)

    group_statistics(df)

    independent_t_test(
        df,
        "study_hours"
    )

    independent_t_test(
        df,
        "attendance_percentage"
    )

    independent_t_test(
        df,
        "previous_score"
    )

    print("STATISTICAL ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()