from pathlib import Path
import random

import numpy as np
import pandas as pd
from faker import Faker

NUMBER_OF_STUDENTS = 2000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fake = Faker()
Faker.seed(RANDOM_SEED)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance_raw.csv"
)


def generate_performance_score(
    study_hours,
    attendance,
    previous_score,
    assignment_score,
    midterm_score,
    absences,
    sleep_hours,
):
    """
    Generate a synthetic final score using several
    student-related variables plus random noise.
    """

    score = (
        0.15 * (study_hours / 8 * 100)
        + 0.20 * attendance
        + 0.15 * previous_score
        + 0.15 * assignment_score
        + 0.25 * midterm_score
        + 0.05 * (sleep_hours / 10 * 100)
        - 0.10 * absences
    )

    noise = np.random.normal(0, 5)

    score += noise

    # Keep score between 0 and 100
    score = np.clip(score, 0, 100)

    return round(score, 2)


def generate_students(number_of_students):
    """Generate synthetic student records."""

    students = []

    for i in range(1, number_of_students + 1):

        age = random.randint(15, 22)

        gender = random.choice(
            ["Male", "Female"]
        )

        study_hours = round(
            np.random.uniform(0.5, 8.0),
            1
        )

        attendance = round(
            np.random.uniform(50, 100),
            1
        )

        previous_score = round(
            np.random.uniform(30, 95),
            1
        )

        assignment_score = round(
            np.random.uniform(30, 100),
            1
        )

        midterm_score = round(
            np.random.uniform(30, 100),
            1
        )

        absences = random.randint(0, 30)

        sleep_hours = round(
            np.random.uniform(4, 10),
            1
        )

        internet_access = random.choice(
            ["Yes", "No"]
        )

        parental_education = random.choice(
            [
                "High School",
                "Diploma",
                "Graduate",
                "Postgraduate",
            ]
        )

        family_income = random.choice(
            [
                "Low",
                "Middle",
                "High",
            ]
        )

        extracurricular = random.choice(
            ["Yes", "No"]
        )

        study_environment = random.choice(
            [
                "Poor",
                "Average",
                "Good",
                "Excellent",
            ]
        )

        final_score = generate_performance_score(
            study_hours,
            attendance,
            previous_score,
            assignment_score,
            midterm_score,
            absences,
            sleep_hours,
        )

        performance = (
            "Pass"
            if final_score >= 50
            else "Fail"
        )

        students.append(
            {
                "student_id": f"STU{i:05d}",
                "name": fake.name(),
                "age": age,
                "gender": gender,
                "study_hours": study_hours,
                "attendance_percentage": attendance,
                "previous_score": previous_score,
                "assignment_score": assignment_score,
                "midterm_score": midterm_score,
                "absences": absences,
                "sleep_hours": sleep_hours,
                "internet_access": internet_access,
                "parental_education": parental_education,
                "family_income": family_income,
                "extracurricular": extracurricular,
                "study_environment": study_environment,
                "final_score": final_score,
                "performance": performance,
            }
        )

    return pd.DataFrame(students)

def main():

    print("Generating synthetic student dataset...")

    df = generate_students(
        NUMBER_OF_STUDENTS
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nDataset generated successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFirst 5 records:")
    print(df.head())


if __name__ == "__main__":
    main()