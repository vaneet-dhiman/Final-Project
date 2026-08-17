from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "student_performance_clean.csv"
)

FIGURES_PATH = (
    PROJECT_ROOT
    / "Outputs"
    / "figures"
)

FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def load_data():

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def basic_overview(df):

    print("BASIC DATASET OVERVIEW")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


def statistical_summary(df):

    print("STATISTICAL SUMMARY")

    print(
        df.describe().round(2)
    )


def performance_distribution(df):

    counts = df["performance"].value_counts()

    print("PERFORMANCE DISTRIBUTION")
    print(counts)

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="performance"
    )

    plt.title("Student Performance Distribution")
    plt.xlabel("Performance")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "performance_distribution.png"
    )

    plt.close()


def final_score_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="final_score",
        bins=20,
        kde=True
    )

    plt.title("Distribution of Final Scores")
    plt.xlabel("Final Score")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "final_score_distribution.png"
    )

    plt.close()


def study_hours_vs_score(df):

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="study_hours",
        y="final_score",
        hue="performance"
    )

    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Final Score")

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "study_hours_vs_score.png"
    )

    plt.close()


def attendance_vs_score(df):

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="attendance_percentage",
        y="final_score",
        hue="performance"
    )

    plt.title("Attendance vs Final Score")
    plt.xlabel("Attendance Percentage")
    plt.ylabel("Final Score")

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "attendance_vs_score.png"
    )

    plt.close()


def performance_by_gender(df):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="gender",
        hue="performance"
    )

    plt.title("Student Performance by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "performance_by_gender.png"
    )

    plt.close()


def performance_by_parental_education(df):

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="parental_education",
        y="final_score"
    )

    plt.title("Final Score by Parental Education")
    plt.xlabel("Parental Education")
    plt.ylabel("Final Score")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "score_by_parental_education.png"
    )

    plt.close()


def absences_vs_score(df):

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="absences",
        y="final_score",
        hue="performance"
    )

    plt.title("Absences vs Final Score")
    plt.xlabel("Number of Absences")
    plt.ylabel("Final Score")

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "absences_vs_score.png"
    )

    plt.close()


def sleep_vs_score(df):

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="sleep_hours",
        y="final_score",
        hue="performance"
    )

    plt.title("Sleep Hours vs Final Score")
    plt.xlabel("Sleep Hours")
    plt.ylabel("Final Score")

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "sleep_vs_score.png"
    )

    plt.close()


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

    correlation_matrix = (
        df[numerical_columns]
        .corr()
    )
    print("CORRELATION WITH FINAL SCORE")
    print(
        correlation_matrix["final_score"]
        .sort_values(ascending=False)
        .round(3)
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH / "correlation_matrix.png"
    )

    plt.close()


def group_statistics(df):

    print("GROUP STATISTICS")
    print("\nAverage score by performance:")

    print(
        df.groupby("performance")["final_score"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )

    print("\nAverage attendance by performance:")

    print(
        df.groupby("performance")["attendance_percentage"]
        .mean()
        .round(2)
    )

    print("\nAverage study hours by performance:")

    print(
        df.groupby("performance")["study_hours"]
        .mean()
        .round(2)
    )


def main():

    df = load_data()

    basic_overview(df)

    statistical_summary(df)

    performance_distribution(df)

    final_score_distribution(df)

    study_hours_vs_score(df)

    attendance_vs_score(df)

    performance_by_gender(df)

    performance_by_parental_education(df)

    absences_vs_score(df)

    sleep_vs_score(df)

    correlation_analysis(df)

    group_statistics(df)

    print("EDA COMPLETED SUCCESSFULLY")
    print(
        f"\nCharts saved to:\n{FIGURES_PATH}"
    )


if __name__ == "__main__":
    main()