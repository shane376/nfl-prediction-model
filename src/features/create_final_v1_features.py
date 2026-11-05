"""
Build the final V1 NFL training dataset.

INPUTS:
    data/raw/2023-2025 nfl schedule.csv
    data/raw/2023-2025 nfl games.csv
    data/raw/2023-2025 nfl scores.csv

OUTPUT:
    data/processed/nfl_v1_training.csv

IMPORTANT:
All team statistics are calculated using games BEFORE
the current game. This prevents data leakage.
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

SCHEDULE_FILE = Path(
    "data/raw/2023-2025 nfl schedule.csv"
)

GAMES_FILE = Path(
    "data/raw/2023-2025 nfl games.csv"
)

SCORES_FILE = Path(
    "data/raw/2023-2025 nfl scores.csv"
)

OUTPUT_FILE = Path(
    "data/processed/nfl_v1_training.csv"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    print("Loading NFL datasets...")

    schedule = pd.read_csv(SCHEDULE_FILE)
    games = pd.read_csv(GAMES_FILE)
    scores = pd.read_csv(SCORES_FILE)

    return schedule, games, scores


# ---------------------------------------------------------
# TEAM W/L HISTORY
# ---------------------------------------------------------

def get_record(schedule, team, season, week):

    columns = [
        f"{season}_W{i}"
        for i in range(1, week)
    ]

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

def win_pct(wins, losses):

    games_played = wins + losses

    if games_played == 0:
        return 0.5

    return wins / games_played


# ---------------------------------------------------------
# STREAK
# ---------------------------------------------------------

def get_streak(schedule, team, season, week):

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

    results = results[
        results.isin(["W", "L", "T"])
    ]

    if len(results) == 0:
        return 0

    streak = 0

    for result in reversed(results.tolist()):

        if result == "W":

            if streak < 0:
                break

            streak += 1

        elif result == "L":

            if streak > 0:
                break

            streak -= 1

        else:
            break

    return streak


# ---------------------------------------------------------
# SCORING STATISTICS
# ---------------------------------------------------------

def get_scoring_stats(scores, team, season, week):

    """
    Calculate scoring statistics using only games
    BEFORE the current week.
    """

    previous_games = scores[
        (scores["season"] == season)
        & (scores["week"] < week)
        & (
            (scores["home_team"] == team)
            | (scores["away_team"] == team)
        )
    ]

    if previous_games.empty:

        return {
            "avg_points": 0.0,
            "avg_points_allowed": 0.0,
            "point_diff": 0.0
        }

    points_scored = []
    points_allowed = []

    for _, game in previous_games.iterrows():

        if game["home_team"] == team:

            points_scored.append(
                game["home_score"]
            )

            points_allowed.append(
                game["away_score"]
            )

        else:

            points_scored.append(
                game["away_score"]
            )

            points_allowed.append(
                game["home_score"]
            )

    avg_points = sum(points_scored) / len(points_scored)

    avg_points_allowed = (
        sum(points_allowed)
        / len(points_allowed)
    )

    point_diff = (
        avg_points
        - avg_points_allowed
    )

    return {
        "avg_points": avg_points,
        "avg_points_allowed": avg_points_allowed,
        "point_diff": point_diff
    }


# ---------------------------------------------------------
# TARGET
# ---------------------------------------------------------

def get_target(scores, game_id):

    game = scores[
        scores["game_id"] == game_id
    ].iloc[0]

    if game["home_score"] > game["away_score"]:
        return 1

    if game["away_score"] > game["home_score"]:
        return 0

    # Ignore ties for V1
    return None


# ---------------------------------------------------------
# BUILD DATASET
# ---------------------------------------------------------

def build_features(schedule, games, scores):

    features = []

    games = games.sort_values(
        ["season", "week", "game_id"]
    )

    for _, game in games.iterrows():

        season = int(game["season"])
        week = int(game["week"])

        home_team = game["home_team"]
        away_team = game["away_team"]

        game_id = game["game_id"]

        # -------------------------------------------------
        # HOME TEAM
        # -------------------------------------------------

        home_wins, home_losses = get_record(
            schedule,
            home_team,
            season,
            week
        )

        home_win_pct = win_pct(
            home_wins,
            home_losses
        )

        home_streak = get_streak(
            schedule,
            home_team,
            season,
            week
        )

        home_scoring = get_scoring_stats(
            scores,
            home_team,
            season,
            week
        )

        # -------------------------------------------------
        # AWAY TEAM
        # -------------------------------------------------

        away_wins, away_losses = get_record(
            schedule,
            away_team,
            season,
            week
        )

        away_win_pct = win_pct(
            away_wins,
            away_losses
        )

        away_streak = get_streak(
            schedule,
            away_team,
            season,
            week
        )

        away_scoring = get_scoring_stats(
            scores,
            away_team,
            season,
            week
        )

        # -------------------------------------------------
        # MATCHUP FEATURES
        # -------------------------------------------------

        win_pct_diff = (
            home_win_pct
            - away_win_pct
        )

        streak_diff = (
            home_streak
            - away_streak
        )

        point_diff_diff = (
            home_scoring["point_diff"]
            - away_scoring["point_diff"]
        )

        home_field = 1

        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        target = get_target(
            scores,
            game_id
        )

        # Skip ties
        if target is None:
            continue

        # -------------------------------------------------
        # SAVE ROW
        # -------------------------------------------------

        features.append({

            "season": season,
            "week": week,
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,

            "home_wins": home_wins,
            "home_losses": home_losses,
            "home_win_pct": home_win_pct,
            "home_streak": home_streak,

            "home_avg_points":
                home_scoring["avg_points"],

            "home_avg_points_allowed":
                home_scoring["avg_points_allowed"],

            "home_point_diff":
                home_scoring["point_diff"],

            "away_wins": away_wins,
            "away_losses": away_losses,
            "away_win_pct": away_win_pct,
            "away_streak": away_streak,

            "away_avg_points":
                away_scoring["avg_points"],

            "away_avg_points_allowed":
                away_scoring["avg_points_allowed"],

            "away_point_diff":
                away_scoring["point_diff"],

            "win_pct_diff": win_pct_diff,
            "streak_diff": streak_diff,
            "point_diff_diff": point_diff_diff,

            "home_field": home_field,

            "target": target
        })

    return pd.DataFrame(features)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    schedule, games, scores = load_data()

    print("Building final V1 training dataset...")

    dataset = build_features(
        schedule,
        games,
        scores
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Keep numeric values manageable
    numeric_columns = dataset.select_dtypes(
        include="number"
    ).columns

    dataset[numeric_columns] = dataset[
        numeric_columns
    ].round(4)

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===================================")
    print("V1 TRAINING DATASET CREATED")
    print("===================================")

    print(f"File: {OUTPUT_FILE.resolve()}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nFirst 5 rows:")
    print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()