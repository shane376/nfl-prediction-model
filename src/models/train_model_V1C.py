import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/nfl_v1_training.csv"

FEATURES = [
    "win_pct_diff",
    "streak_diff",
    "point_diff_diff",
]

TARGET = "target"

TRAIN_SEASONS = [2023, 2024]
TEST_SEASON = 2025


# ============================================================
# DATA
# ============================================================

def load_data():
    return pd.read_csv(DATA_PATH)


def split_data(data):

    train = data[data["season"].isin(TRAIN_SEASONS)].copy()
    test = data[data["season"] == TEST_SEASON].copy()

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    return train, test, X_train, y_train, X_test, y_test


# ============================================================
# MODEL
# ============================================================

def build_model(C=1.0):

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42
        ))
    ])


# ============================================================
# STANDARD EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\n===================================")
    print("MODEL RESULTS")
    print("===================================")

    print(f"Accuracy: {accuracy:.2%}")

    print("\nConfusion Matrix:")
    print(matrix)

    return accuracy, predictions


def show_predictions(test, predictions):

    results = test[
        [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "target",
        ]
    ].copy()

    results["prediction"] = predictions
    results["correct"] = (
        results["prediction"] == results["target"]
    )

    print("\n===================================")
    print("SAMPLE PREDICTIONS")
    print("===================================")

    print(
        results[
            [
                "week",
                "home_team",
                "away_team",
                "prediction",
                "target",
                "correct",
            ]
        ].head(20).to_string(index=False)
    )


# ============================================================
# PARAMETER EXPERIMENT
# ============================================================

def test_parameter(
    X_train,
    y_train,
    X_test,
    y_test,
    C
):

    model = build_model(C=C)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return accuracy


def compare_parameter(
    values,
    X_train,
    y_train,
    X_test,
    y_test
):

    results = []

    for value in values:

        accuracy = test_parameter(
            X_train,
            y_train,
            X_test,
            y_test,
            value
        )

        results.append({
            "C": value,
            "accuracy": accuracy
        })

    return pd.DataFrame(results)

#threshold
def test_threshold(
    model,
    X_test,
    y_test,
    threshold
):
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return accuracy


def compare_thresholds(
    model,
    thresholds,
    X_test,
    y_test
):

    results = []

    for threshold in thresholds:

        accuracy = test_threshold(
            model,
            X_test,
            y_test,
            threshold
        )

        results.append({
            "threshold": threshold,
            "accuracy": accuracy
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    data = load_data()

    train, test, X_train, y_train, X_test, y_test = (
        split_data(data)
    )

    print("===================================")
    print("V1C MODEL PLAYGROUND")
    print("===================================")

    print(f"Training seasons: {TRAIN_SEASONS}")
    print(f"Testing season:   {TEST_SEASON}")

    print("\nFeatures:")
    for feature in FEATURES:
        print(f"- {feature}")

    print(f"\nTraining games: {len(train)}")
    print(f"Testing games:  {len(test)}")


    # ========================================================
    # C PARAMETER EXPERIMENT
    # ========================================================

    C_VALUES = [
        0.001,
        0.01,
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        100.0,
    ]

    results = compare_parameter(
        C_VALUES,
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("\n===================================")
    print("C PARAMETER COMPARISON")
    print("===================================")

    print(
        results.to_string(
            index=False,
            formatters={
                "accuracy": "{:.2%}".format
            }
        )
    )


    # ========================================================
    # BEST PARAMETER
    # ========================================================

    best_row = results.loc[
        results["accuracy"].idxmax()
    ]

    best_C = best_row["C"]
    best_accuracy = best_row["accuracy"]

    print("\n===================================")
    print("BEST C")
    print("===================================")

    print(f"C:        {best_C}")
    print(f"Accuracy: {best_accuracy:.2%}")


    # ========================================================
    # FINAL MODEL USING BEST PARAMETER
    # ========================================================

    best_model = build_model(C=best_C)

    best_model.fit(
        X_train,
        y_train
    )

    THRESHOLDS = [
        0.40,
        0.42,
        0.44,
        0.46,
        0.48,
        0.50,
        0.52,
        0.54,
        0.56,
        0.58,
        0.60,
    ]

    threshold_results = compare_thresholds(
        best_model,
        THRESHOLDS,
        X_test,
        y_test
    )

    print("\n===================================")
    print("THRESHOLD COMPARISON")
    print("===================================")

    print(
        threshold_results.to_string(
            index=False,
            formatters={
                "accuracy": "{:.2%}".format
            }
        )
    )

    accuracy, predictions = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    show_predictions(
        test,
        predictions
    )


if __name__ == "__main__":
    main()