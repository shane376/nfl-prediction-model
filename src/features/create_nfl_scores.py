"""
Extract NFL game scores for the 2023-2025 regular seasons.

Output:
    data/raw/2023-2025 nfl scores.csv

Columns:
    season
    week
    game_id
    home_team
    away_team
    home_score
    away_score
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/"
    "master/data/games.csv"
)

YEARS = [2023, 2024, 2025]

OUTPUT_FILE = Path(
    "data/raw/2023-2025 nfl scores.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_games():

    print("Downloading NFL game data...")

    games = pd.read_csv(DATA_URL)

    games = games[
        games["season"].isin(YEARS)
        & (games["game_type"] == "REG")
        & games["week"].between(1, 18)
    ].copy()

    return games


# ---------------------------------------------------------
# CREATE SCORE DATASET
# ---------------------------------------------------------

def create_score_dataset(games):

    columns = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    scores = games[columns].copy()

    scores = scores.sort_values(
        ["season", "week", "game_id"]
    )

    return scores.reset_index(drop=True)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    games = load_games()

    scores = create_score_dataset(games)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    scores.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nScore dataset created!")

    print(f"File: {OUTPUT_FILE.resolve()}")

    print(f"Games: {len(scores)}")

    print(f"Columns: {len(scores.columns)}")

    print("\nColumns:")
    print(scores.columns.tolist())

    print("\nFirst 5 games:")
    print(scores.head().to_string(index=False))


if __name__ == "__main__":
    main()