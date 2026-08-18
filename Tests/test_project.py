from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "student_performance_clean.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "student_performance_model.pkl"
)


def test_clean_dataset_exists():

    assert DATA_PATH.exists()


def test_clean_dataset_can_be_loaded():

    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0


def test_required_columns_exist():

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "age",
        "gender",
        "study_hours",
        "attendance_percentage",
        "previous_score",
        "assignment_score",
        "midterm_score",
        "absences",
        "sleep_hours",
        "performance"
    ]

    for column in required_columns:

        assert column in df.columns


def test_target_values_are_valid():

    df = pd.read_csv(DATA_PATH)

    valid_values = {
        "Pass",
        "Fail"
    }

    assert set(
        df["performance"].unique()
    ).issubset(valid_values)


def test_model_exists():

    assert MODEL_PATH.exists()


def test_model_can_be_loaded():

    model = joblib.load(
        MODEL_PATH
    )

    assert model is not None


def test_model_can_predict():

    model = joblib.load(
        MODEL_PATH
    )

    student = pd.DataFrame(
        [{
            "age": 18,
            "gender": "Female",
            "study_hours": 5,
            "attendance_percentage": 85,
            "previous_score": 75,
            "assignment_score": 80,
            "midterm_score": 78,
            "absences": 5,
            "sleep_hours": 7,
            "internet_access": "Yes",
            "parental_education": "Graduate",
            "family_income": "Medium",
            "extracurricular": "Yes",
            "study_environment": "Good"
        }]
    )

    prediction = model.predict(
        student
    )

    assert prediction[0] in {
        "Pass",
        "Fail"
    }