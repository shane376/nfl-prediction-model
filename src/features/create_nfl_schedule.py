"""
Build a 2023-2025 NFL weekly W/L matrix.

Output:
    2023-2025 nfl schedule.csv

Rows = NFL teams
Columns = 2023_W1 ... 2025_W18
Values = W, L, T, or BYE

The script downloads nflverse's games.csv, filters regular-season
games for 2023-2025, and builds the matrix automatically.
"""

from pathlib import Path
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
YEARS = [2023, 2024, 2025]
MAX_WEEKS = 18
OUTPUT_FILE = Path("2023-2025 nfl schedule.csv")

# 32 current NFL team abbreviations.
TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]


def load_games():
    """Download and filter regular-season games."""
    print("Downloading NFL game data...")
    games = pd.read_csv(DATA_URL)

    games = games[
        games["season"].isin(YEARS)
        & (games["game_type"] == "REG")
        & games["week"].between(1, MAX_WEEKS)
    ].copy()

    required = {"season", "week", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    return games


def game_result(home_score, away_score):
    """Return the result from a team's perspective."""
    if home_score > away_score:
        return "W"
    if home_score < away_score:
        return "L"
    return "T"


def build_matrix(games):
    """Create one row per team and one column per NFL week."""
    columns = [
        f"{year}_W{week}"
        for year in YEARS
        for week in range(1, MAX_WEEKS + 1)
    ]

    matrix = pd.DataFrame("BYE", index=TEAMS, columns=columns)
    matrix.index.name = "Team"

    for _, game in games.iterrows():
        season = int(game["season"])
        week = int(game["week"])
        home = game["home_team"]
        away = game["away_team"]
        home_score = game["home_score"]
        away_score = game["away_score"]

        # Ignore games with missing scores.
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        column = f"{season}_W{week}"

        matrix.loc[home, column] = game_result(home_score, away_score)
        matrix.loc[away, column] = game_result(away_score, home_score)

    return matrix.reset_index()


def main():
    games = load_games()
    matrix = build_matrix(games)

    matrix.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCreated: {OUTPUT_FILE.resolve()}")
    print(f"Shape: {matrix.shape[0]} teams x {matrix.shape[1] - 1} week columns")
    print(f"Regular-season games loaded: {len(games)}")
    print("\nFirst 5 teams:")
    print(matrix.head().to_string(index=False))


if __name__ == "__main__":
    main()
