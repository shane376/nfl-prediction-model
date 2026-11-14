"""Deterministic schedule features at kickoff minus 24 hours, research only."""
import math
import pandas as pd
from src.pregame import features as v1_features
from src.research.schedule_data import audit

BASE = ["win_pct_diff", "point_diff_diff"]
CONFIGS = {
    "v1_a": BASE,
    "v1_a_prior_elo": BASE + ["prior_elo_diff"],
    "v1_a_prior_elo_rest": BASE + ["prior_elo_diff", "rest_diff", "home_opener", "away_opener", "home_rest_unavailable", "away_rest_unavailable"],
}


def elo_probability(diff, home_adjustment=50):
    return 1 / (1 + 10 ** (-(diff + home_adjustment) / 400))


def opening_ratings(games, venue_enabled=False):
    """Update earlier REG seasons only, regress once per offseason, then freeze."""
    games = games.loc[games.game_type.eq("REG")]
    ratings, counts, openings = {}, {}, {}
    years = sorted(games.season.unique())
    previous = None
    for season in years:
        season_games = games.loc[games.season.eq(season)]
        first_cutoff = season_games.kickoff.min() - pd.Timedelta(hours=24)
        if previous is not None:
            # Eligible labels are bounded by the first forecast in this season.
            prior = games.loc[games.season.ge(previous) & games.season.lt(season)
                              & games.status.eq("completed") & games.kickoff.notna() & (games.available_at < first_cutoff)
                              & games.home_score.notna() & games.away_score.notna()]
            for g in prior.sort_values(["kickoff", "game_id"]).itertuples():
                home, away = ratings.get(g.home_team, 1500.), ratings.get(g.away_team, 1500.)
                advantage = 0 if venue_enabled and g.location == "Neutral" else 50
                outcome = .5 if g.home_score == g.away_score else float(g.home_score > g.away_score)
                change = 20 * (outcome - elo_probability(home - away, advantage))
                ratings[g.home_team], ratings[g.away_team] = home + change, away - change
                for team in (g.home_team, g.away_team):
                    counts[team] = counts.get(team, 0) + 1
            for team in ratings:
                ratings[team] = 1500 + (ratings[team] - 1500) * .75 ** (int(season) - int(previous))
        openings[int(season)] = (ratings.copy(), counts.copy())
        previous = season
    return openings


def rest_for(games, game, side, cutoff):
    team = game[f"{side}_team"]
    team_games = games.loc[games.season.eq(game.season) & games.game_id.ne(game.game_id)
                           & (games.home_team.eq(team) | games.away_team.eq(team))]
    preceding = team_games.loc[(team_games.kickoff < game.kickoff) | (team_games.kickoff.isna() & (team_games.week < game.week))]
    eligible = preceding.loc[preceding.status.eq("completed") & preceding.kickoff.notna() & (preceding.available_at < cutoff)
                             & preceding.home_score.notna() & preceding.away_score.notna()]
    opener = int(preceding.empty)
    if eligible.empty:
        return 7, opener, 1
    last = eligible.kickoff.max()
    days = (game.kickoff.tz_convert("America/New_York").date() - last.tz_convert("America/New_York").date()).days
    if days <= 0:
        raise ValueError("Nonpositive played-game interval")
    return min(days, 14), opener, 0


def build_features(games, venue_enabled=False):
    """Keep missing labels; eligibility/exclusion belongs to the evaluator."""
    openings = opening_ratings(games, venue_enabled)
    complete = audit(games).set_index("season").complete.to_dict()
    rows = []
    for _, game in games.iterrows():
        if pd.isna(game.kickoff):
            continue  # Exclusion is recorded separately; never synthesize midnight.
        cutoff = game.kickoff - pd.Timedelta(hours=24)
        ratings, counts = openings[int(game.season)]
        row = dict(game_id=game.game_id, season=int(game.season), week=int(game.week),
                   home_team=game.home_team, away_team=game.away_team, kickoff=game.kickoff,
                   forecast_cutoff=cutoff, label_available_at=game.available_at,
                   target=(int(game.home_score > game.away_score) if game.status == "completed" and game.home_score != game.away_score else None),
                   **{k: v for k, v in v1_features(games, game, cutoff).items() if k in BASE})
        for side in ("home", "away"):
            team = game[f"{side}_team"]
            row[f"{side}_prior_elo"] = ratings.get(team, 1500.)
            row[f"{side}_elo_history_games"] = counts.get(team, 0)
            row[f"{side}_elo_initialized"] = int(counts.get(team, 0) == 0)
            row[f"{side}_elo_history_incomplete"] = int(not all(complete.get(int(game.season) - n, False) for n in (1, 2)))
            rest, opener, unavailable = rest_for(games, game, side, cutoff)
            row.update({f"{side}_rest": rest, f"{side}_opener": opener, f"{side}_rest_unavailable": unavailable})
        row["prior_elo_diff"] = row["home_prior_elo"] - row["away_prior_elo"]
        row["rest_diff"] = row["home_rest"] - row["away_rest"]
        row["neutral"] = int(game.location == "Neutral")
        row["venue_unavailable"] = int(game.location not in ("Home", "Neutral"))
        row["venue_enabled"] = venue_enabled
        row["fixed_elo_probability"] = elo_probability(row["prior_elo_diff"], 0 if venue_enabled and row["neutral"] else 50)
        assert math.isfinite(row["prior_elo_diff"])
        rows.append(row)
    return pd.DataFrame(rows)
