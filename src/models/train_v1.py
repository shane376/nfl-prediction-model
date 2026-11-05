"""
Train and compare V1 NFL prediction models.

Models:
    A - Minimal baseline
    B - Expanded feature set
    C - Matchup-based features
    D - Matchup features + scoring

Training:
    2023 + 2024

Testing:
    2025

Target:
    1 = Home team wins
    0 = Away team wins
"""

from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# FILE PATH
# ---------------------------------------------------------

DATA_FILE = Path("data/processed/nfl_v1_training.csv")

TARGET = "target"


# ---------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------

MODEL_FEATURES = {
    "A": [
        "win_pct_diff",
        "point_diff_diff",
    ],

    "B": [
        "home_win_pct",
        "home_streak",
        "home_avg_points",
        "home_avg_points_allowed",
        "home_point_diff",
        "away_win_pct",
        "away_streak",
        "away_avg_points",
        "away_avg_points_allowed",
        "away_point_diff",
        "win_pct_diff",
        "streak_diff",
        "point_diff_diff",
    ],

    "C": [
        "win_pct_diff",
        "streak_diff",
        "point_diff_diff",
    ],

    "D": [
        "win_pct_diff",
        "streak_diff",
        "point_diff_diff",
        "home_avg_points",
        "away_avg_points",
    ],
}


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    print("Loading V1 training data...")

    data = pd.read_csv(DATA_FILE)

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return data


# ---------------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------------

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
        ),
    ])


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

def train_model(data, features):
    train = data[data["season"].isin([2023, 2024])].copy()
    test = data[data["season"] == 2025].copy()

    X_train = train[features]
    y_train = train[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, test, y_test, predictions


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Away Win",
                "Home Win",
            ],
        )
    )

    return accuracy


# ---------------------------------------------------------
# WEEKLY PERFORMANCE
# ---------------------------------------------------------

def evaluate_by_week(data, features):
    train = data[data["season"].isin([2023, 2024])].copy()
    test = data[data["season"] == 2025].copy()

    model = build_model()

    model.fit(
        train[features],
        train[TARGET],
    )

    weekly_results = []

    for week in sorted(test["week"].unique()):
        week_data = test[test["week"] == week]

        predictions = model.predict(
            week_data[features]
        )

        accuracy = accuracy_score(
            week_data[TARGET],
            predictions,
        )

        weekly_results.append({
            "week": int(week),
            "games": len(week_data),
            "accuracy": accuracy,
        })

    return pd.DataFrame(weekly_results)


# ---------------------------------------------------------
# SAMPLE PREDICTIONS
# ---------------------------------------------------------

def show_predictions(
    model,
    test,
    predictions,
    features,
    model_name,
):
    probabilities = model.predict_proba(
        test[features]
    )

    results = test[
        [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            TARGET,
        ]
    ].copy()

    results["prediction"] = predictions

    results["home_win_probability"] = probabilities[:, 1]

    results["predicted_winner"] = results.apply(
        lambda row:
        row["home_team"]
        if row["prediction"] == 1
        else row["away_team"],
        axis=1,
    )

    results["actual_winner"] = results.apply(
        lambda row:
        row["home_team"]
        if row[TARGET] == 1
        else row["away_team"],
        axis=1,
    )

    print("\n===================================")
    print(f"MODEL {model_name} SAMPLE PREDICTIONS")
    print("===================================")

    display_columns = [
        "season",
        "week",
        "home_team",
        "away_team",
        "predicted_winner",
        "actual_winner",
        "home_win_probability",
    ]

    print(
        results[display_columns]
        .head(20)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    data = load_data()

    train_games = data[
        data["season"].isin([2023, 2024])
    ]

    test_games = data[
        data["season"] == 2025
    ]

    print("\n===================================")
    print("DATA SPLIT")
    print("===================================")

    print(f"Training games: {len(train_games)}")
    print(f"Testing games:  {len(test_games)}")

    print("\nTraining seasons:")
    print(sorted(train_games["season"].unique()))

    print("\nTesting seasons:")
    print(sorted(test_games["season"].unique()))

    # -----------------------------------------------------
    # TRAIN ALL MODELS
    # -----------------------------------------------------

    results = {}

    for model_name, features in MODEL_FEATURES.items():

        print("\n===================================")
        print(f"MODEL {model_name}")
        print("===================================")

        print("\nFeatures:")
        print(features)

        model, test, y_test, predictions = train_model(
            data,
            features,
        )

        accuracy = evaluate_model(
            y_test,
            predictions,
        )

        results[model_name] = {
            "model": model,
            "test": test,
            "y_test": y_test,
            "predictions": predictions,
            "accuracy": accuracy,
            "features": features,
        }

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    print("\n===================================")
    print("MODEL COMPARISON")
    print("===================================")

    comparison = []

    for model_name, result in results.items():
        comparison.append({
            "model": model_name,
            "features": len(result["features"]),
            "accuracy": result["accuracy"],
        })

    comparison = pd.DataFrame(comparison)

    comparison["accuracy"] = (
        comparison["accuracy"] * 100
    )

    comparison["difference_from_best"] = (
        comparison["accuracy"]
        - comparison["accuracy"].max()
    )

    print(
        comparison.to_string(
            index=False,
            formatters={
                "accuracy": "{:.2f}%".format,
                "difference_from_best": "{:+.2f}%".format,
            },
        )
    )

    # -----------------------------------------------------
    # WEEKLY COMPARISON
    # -----------------------------------------------------

    print("\n===================================")
    print("WEEKLY MODEL COMPARISON")
    print("===================================")

    weekly_results = {}

    for model_name, features in MODEL_FEATURES.items():

        weekly_results[model_name] = evaluate_by_week(
            data,
            features,
        )

    weekly_comparison = pd.DataFrame({
        "week": weekly_results["A"]["week"],
        "games": weekly_results["A"]["games"],
    })

    for model_name, weekly in weekly_results.items():
        weekly_comparison[
            f"accuracy_{model_name}"
        ] = weekly["accuracy"]

    print(
        weekly_comparison.to_string(
            index=False,
            formatters={
                column: "{:.2%}".format
                for column in weekly_comparison.columns
                if column.startswith("accuracy_")
            },
        )
    )

    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    best_model_name = max(
        results,
        key=lambda name: results[name]["accuracy"],
    )

    best_model = results[best_model_name]

    print("\n===================================")
    print("BEST MODEL")
    print("===================================")

    print(f"Model: {best_model_name}")
    print(
        f"Accuracy: "
        f"{best_model['accuracy'] * 100:.2f}%"
    )

    print(
        f"Features: "
        f"{len(best_model['features'])}"
    )

    # -----------------------------------------------------
    # SAMPLE PREDICTIONS
    # -----------------------------------------------------

    show_predictions(
        best_model["model"],
        best_model["test"],
        best_model["predictions"],
        best_model["features"],
        best_model_name,
    )

    print("\nModel comparison complete!")


if __name__ == "__main__":
    main()