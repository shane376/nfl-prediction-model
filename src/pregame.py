"""Identical season-to-date feature definitions for training and inference."""
import pandas as pd


def features(games, game, cutoff):
    cutoff = min(pd.Timestamp(cutoff), game.kickoff)
    history = games.loc[(games.season == game.season) & (games.available_at < cutoff) & games.status.eq("completed")]
    result = {}
    for side in ("home", "away"):
        team = game[f"{side}_team"]
        prior = history.loc[history.home_team.eq(team) | history.away_team.eq(team)].sort_values("kickoff")
        scored, allowed, outcomes = [], [], []
        for row in prior.itertuples():
            a, b = (row.home_score, row.away_score) if row.home_team == team else (row.away_score, row.home_score)
            scored.append(a)
            allowed.append(b)
            outcomes.append(1 if a > b else -1 if a < b else 0)
        streak = 0
        for outcome in reversed(outcomes):
            if outcome == 0 or (streak and (streak > 0) != (outcome > 0)):
                break
            streak += outcome
        count = len(outcomes)
        points = sum(scored) / count if count else 0.0
        conceded = sum(allowed) / count if count else 0.0
        result.update({f"{side}_win_pct": (outcomes.count(1) + .5 * outcomes.count(0)) / count if count else .5,
                       f"{side}_streak": streak, f"{side}_avg_points": points,
                       f"{side}_avg_points_allowed": conceded, f"{side}_point_diff": points - conceded})
    for name, stat in (("win_pct_diff", "win_pct"), ("streak_diff", "streak"), ("point_diff_diff", "point_diff")):
        result[name] = result[f"home_{stat}"] - result[f"away_{stat}"]
    return result


def training_rows(games, cutoff):
    rows = []
    for _, game in games.loc[games.status.eq("completed") & (games.available_at < cutoff)].iterrows():
        if game.home_score == game.away_score:
            continue
        rows.append({**game.to_dict(), **features(games, game, game.kickoff), "target": int(game.home_score > game.away_score)})
    return pd.DataFrame(rows)
