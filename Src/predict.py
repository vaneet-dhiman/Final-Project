from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "student_performance_model.pkl"
)


def get_student_data():

    age = int(input("Age: "))

    gender = input("Gender (Male/Female): ")

    study_hours = float(
        input("Study hours per day: ")
    )

    attendance_percentage = float(
        input("Attendance percentage: ")
    )

    previous_score = float(
        input("Previous score: ")
    )

    assignment_score = float(
        input("Assignment score: ")
    )

    midterm_score = float(
        input("Midterm score: ")
    )

    absences = int(
        input("Number of absences: ")
    )

    sleep_hours = float(
        input("Sleep hours per day: ")
    )

    internet_access = input(
        "Internet access (Yes/No): "
    )

    parental_education = input(
        "Parental education: "
    )

    family_income = input(
        "Family income: "
    )

    extracurricular = input(
        "Extracurricular activities (Yes/No): "
    )

    study_environment = input(
        "Study environment: "
    )

    data = {
        "age": age,
        "gender": gender,
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "previous_score": previous_score,
        "assignment_score": assignment_score,
        "midterm_score": midterm_score,
        "absences": absences,
        "sleep_hours": sleep_hours,
        "internet_access": internet_access,
        "parental_education": parental_education,
        "family_income": family_income,
        "extracurricular": extracurricular,
        "study_environment": study_environment
    }

    return pd.DataFrame([data])


def main():

    print("STUDENT PERFORMANCE PREDICTION")

    if not MODEL_PATH.exists():

        print("Model file not found.")

        return

    model = joblib.load(MODEL_PATH)

    print("\nEnter student information:\n")

    student = get_student_data()

    prediction = model.predict(student)[0]

    probabilities = model.predict_proba(student)[0]

    classes = model.classes_

    probability_dict = dict(
        zip(classes, probabilities)
    )

    fail_probability = probability_dict.get(
        "Fail",
        0
    )

    pass_probability = probability_dict.get(
        "Pass",
        0
    )

    print()
    print("PREDICTION RESULT")

    print(
        f"Predicted Performance: {prediction}"
    )

    print(
        f"Pass Probability: {pass_probability:.2%}"
    )

    print(
        f"Fail Probability: {fail_probability:.2%}"
    )

    if fail_probability >= 0.50:

        print("Risk Level: HIGH")

    elif fail_probability >= 0.30:

        print("Risk Level: MEDIUM")

    else:

        print("Risk Level: LOW")


if __name__ == "__main__":
    main()