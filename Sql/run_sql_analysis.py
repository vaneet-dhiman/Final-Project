from pathlib import Path
import sqlite3

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    PROJECT_ROOT
    / "Database"
    / "student_performance.db"
)


def run_query(connection, query, title):

    print(title)


    result = pd.read_sql_query(
        query,
        connection
    )

    print(result.to_string(index=False))


def main():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    # Query 1
    run_query(
        connection,
        """
        SELECT COUNT(*) AS total_students
        FROM students;
        """,
        "TOTAL STUDENTS"
    )

    # Query 2
    run_query(
        connection,
        """
        SELECT
            performance,
            COUNT(*) AS student_count
        FROM students
        GROUP BY performance;
        """,
        "PASS / FAIL DISTRIBUTION"
    )

    # Query 3
    run_query(
        connection,
        """
        SELECT
            performance,
            ROUND(AVG(final_score), 2)
                AS average_score
        FROM students
        GROUP BY performance;
        """,
        "AVERAGE SCORE BY PERFORMANCE"
    )

    # Query 4
    run_query(
        connection,
        """
        SELECT
            performance,
            ROUND(AVG(attendance_percentage), 2)
                AS average_attendance
        FROM students
        GROUP BY performance;
        """,
        "AVERAGE ATTENDANCE BY PERFORMANCE"
    )

    # Query 5
    run_query(
        connection,
        """
        SELECT
            performance,
            ROUND(AVG(study_hours), 2)
                AS average_study_hours
        FROM students
        GROUP BY performance;
        """,
        "AVERAGE STUDY HOURS BY PERFORMANCE"
    )

    # Query 6
    run_query(
        connection,
        """
        SELECT
            parental_education,
            COUNT(*) AS student_count,
            ROUND(AVG(final_score), 2)
                AS average_score
        FROM students
        GROUP BY parental_education
        ORDER BY average_score DESC;
        """,
        "PERFORMANCE BY PARENTAL EDUCATION"
    )

    connection.close()


if __name__ == "__main__":
    main()