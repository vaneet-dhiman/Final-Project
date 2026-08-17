from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    PROJECT_ROOT
    / "Database"
    / "student_performance.db"
)

SQL_PATH = (
    PROJECT_ROOT
    / "Sql"
    / "analysis.sql"
)

def main():

    print("Connecting to database...")

    connection = sqlite3.connect(DATABASE_PATH)

    print("Database connected.")

    # Read SQL file
    with open(SQL_PATH, "r", encoding="utf-8") as file:
        sql_script = file.read()

    print("\nSQL file loaded successfully.")

    # Execute the SQL script
    connection.executescript(sql_script)

    print("SQL script executed successfully.")

    connection.close()

    print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()