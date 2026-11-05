"""
Extract basic NFL game information for the 2023-2025 seasons.

Output:
    data/raw/2023-2025 nfl games.csv

Columns:
    season
    week
    game_id
    home_team
    away_team
"""

from pathlib import Path
import pandas as pd


DATA_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

YEARS = [2023, 2024, 2025]

OUTPUT_FILE = Path("data/raw/2023-2025 nfl games.csv")


def load_games():
    print("Downloading NFL game data...")

    games = pd.read_csv(DATA_URL)

    games = games[
        games["season"].isin(YEARS)
        & (games["game_type"] == "REG")
        & games["week"].between(1, 18)
    ].copy()

    return games


def create_dataset(games):
    columns = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
    ]

    return games[columns].sort_values(
        ["season", "week", "game_id"]
    ).reset_index(drop=True)


def main():
    games = load_games()

    dataset = create_dataset(games)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCreated: {OUTPUT_FILE.resolve()}")
    print(f"Games: {len(dataset)}")
    print(f"Columns: {list(dataset.columns)}")

    print("\nFirst 10 games:")
    print(dataset.head(10).to_string(index=False))


if __name__ == "__main__":
    main()