from pathlib import Path
import sqlite3

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "student_performance_clean.csv"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "Database"
    / "student_performance.db"
)


def create_database():

    print("Loading cleaned dataset...")

    df = pd.read_csv(CSV_PATH)

    print(f"Rows loaded: {len(df)}")

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)


    df.to_sql(
        "students",
        connection,
        if_exists="replace",
        index=False
    )

    connection.close()

    print("\nDatabase created successfully!")
    print(f"Database: {DATABASE_PATH}")
    print("Table: students")


if __name__ == "__main__":
    create_database()