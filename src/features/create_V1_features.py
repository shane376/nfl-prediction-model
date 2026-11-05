"""
Build V1 NFL prediction features from:

1. 2023-2025 nfl schedule.csv
   - Team-by-week W/L/T/BYE matrix

2. 2023-2025 nfl games.csv
   - Game descriptors:
     season, week, game_id, home_team, away_team

Output:
    data/processed/nfl_v1_features.csv

V1 features:
    season
    week
    game_id
    home_team
    away_team

    home_wins
    home_losses
    home_win_pct
    home_streak

    away_wins
    away_losses
    away_win_pct
    away_streak

    win_pct_diff
    streak_diff
    home_field

    target
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

SCHEDULE_FILE = Path("data/raw/2023-2025 nfl schedule.csv")
GAMES_FILE = Path("data/raw/2023-2025 nfl games.csv")

OUTPUT_FILE = Path("data/processed/nfl_v1_features.csv")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    print("Loading NFL data...")

    schedule = pd.read_csv(SCHEDULE_FILE)
    games = pd.read_csv(GAMES_FILE)

    return schedule, games


# ---------------------------------------------------------
# TEAM RECORD
# ---------------------------------------------------------

def get_team_record(schedule, team, season, week):
    """
    Calculate a team's record BEFORE the current game.

    Only weeks before the current week are used.
    This prevents data leakage.
    """

    columns = [
        f"{season}_W{i}"
        for i in range(1, week)
    ]

    # Week 1 has no previous games
    if not columns:
        return 0, 0

    results = schedule.loc[
        schedule["Team"] == team,
        columns
    ].iloc[0]

    wins = (results == "W").sum()
    losses = (results == "L").sum()

    return int(wins), int(losses)


# ---------------------------------------------------------
# WIN PERCENTAGE
# ---------------------------------------------------------

def calculate_win_pct(wins, losses):
    games_played = wins + losses

    if games_played == 0:
        return 0.5

    return wins / games_played


# ---------------------------------------------------------
# CURRENT STREAK
# ---------------------------------------------------------

def get_team_streak(schedule, team, season, week):
    """
    Calculate the team's current streak BEFORE the game.

    Positive number = consecutive wins
    Negative number = consecutive losses
    0 = no previous games
    """

    columns = [
        f"{season}_W{i}"
        for i in range(1, week)
    ]

    if not columns:
        return 0

    results = schedule.loc[
        schedule["Team"] == team,
        columns
    ].iloc[0]

    # Ignore BYE weeks
    results = results[results != "BYE"]

    if len(results) == 0:
        return 0

    streak = 0

    # Work backward from most recent game
    for result in reversed(results.tolist()):

        if result == "W":

            if streak < 0:
                break

            streak += 1

        elif result == "L":

            if streak > 0:
                break

            streak -= 1

        elif result == "T":

            break

    return streak


# ---------------------------------------------------------
# DETERMINE GAME WINNER
# ---------------------------------------------------------

def get_target(schedule, home_team, away_team, season, week):
    """
    Determine the actual winner from the W/L matrix.

    1 = home team won
    0 = away team won
    """

    column = f"{season}_W{week}"

    home_result = schedule.loc[
        schedule["Team"] == home_team,
        column
    ].iloc[0]

    away_result = schedule.loc[
        schedule["Team"] == away_team,
        column
    ].iloc[0]

    if home_result == "W":
        return 1

    if away_result == "W":
        return 0

    # Currently handles ties separately.
    # We will decide how to treat ties when needed.
    return None


# ---------------------------------------------------------
# BUILD FEATURES
# ---------------------------------------------------------

def build_features(schedule, games):

    features = []

    games = games.sort_values(
        ["season", "week", "game_id"]
    )

    for _, game in games.iterrows():

        season = int(game["season"])
        week = int(game["week"])

        home_team = game["home_team"]
        away_team = game["away_team"]

        # -----------------------------
        # Home team
        # -----------------------------

        home_wins, home_losses = get_team_record(
            schedule,
            home_team,
            season,
            week
        )

        home_win_pct = calculate_win_pct(
            home_wins,
            home_losses
        )

        home_streak = get_team_streak(
            schedule,
            home_team,
            season,
            week
        )

        # -----------------------------
        # Away team
        # -----------------------------

        away_wins, away_losses = get_team_record(
            schedule,
            away_team,
            season,
            week
        )

        away_win_pct = calculate_win_pct(
            away_wins,
            away_losses
        )

        away_streak = get_team_streak(
            schedule,
            away_team,
            season,
            week
        )

        # -----------------------------
        # Matchup features
        # -----------------------------

        win_pct_diff = home_win_pct - away_win_pct

        streak_diff = home_streak - away_streak

        # Every game in this dataset has a designated home team
        home_field = 1

        # -----------------------------
        # Target
        # -----------------------------

        target = get_target(
            schedule,
            home_team,
            away_team,
            season,
            week
        )

        # Skip ties for V1
        if target is None:
            continue

        # -----------------------------
        # Store row
        # -----------------------------

        features.append({
            "season": season,
            "week": week,
            "game_id": game["game_id"],
            "home_team": home_team,
            "away_team": away_team,

            "home_wins": home_wins,
            "home_losses": home_losses,
            "home_win_pct": home_win_pct,
            "home_streak": home_streak,

            "away_wins": away_wins,
            "away_losses": away_losses,
            "away_win_pct": away_win_pct,
            "away_streak": away_streak,

            "win_pct_diff": win_pct_diff,
            "streak_diff": streak_diff,

            "home_field": home_field,

            "target": target
        })

    return pd.DataFrame(features)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    schedule, games = load_data()

    print("Building V1 features...")

    features = build_features(
        schedule,
        games
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nV1 feature dataset created!")
    print(f"File: {OUTPUT_FILE.resolve()}")
    print(f"Rows: {len(features)}")
    print(f"Columns: {len(features.columns)}")

    print("\nColumns:")
    print(features.columns.tolist())

    print("\nFirst 5 rows:")
    print(features.head().to_string(index=False))


if __name__ == "__main__":
    main()