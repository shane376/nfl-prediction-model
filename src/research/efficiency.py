"""Bounded exploratory PBP aggregation. Never promotion evidence without vintage proof."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.research.schedule_data import ALIASES

COLUMNS = ["game_id", "play_id", "season_type", "posteam", "defteam", "play_type", "qb_kneel", "qb_spike", "epa", "success", "qb_dropback", "passer_player_id"]


def aggregate_batches(batches):
    """Return team-game sums/counts; never materialize more than one PBP batch."""
    accum, audit = [], dict(rows=0, eligible=0, missing_epa=0, missing_success=0, missing_team=0, dropbacks=0, missing_passer_id=0)
    for frame in batches:
        if set(COLUMNS) - set(frame):
            raise ValueError("Required PBP columns unavailable")
        audit["rows"] += len(frame)
        # nflfastR play_type pass includes sacks, run includes QB scrambles.
        plays = frame.loc[frame.season_type.eq("REG") & frame.play_type.isin(["pass", "run"])
                          & frame.qb_kneel.eq(0) & frame.qb_spike.eq(0)].copy()
        audit["eligible"] += len(plays)
        plays["epa"] = pd.to_numeric(plays.epa, errors="raise").replace([np.inf, -np.inf], np.nan)
        plays["success"] = pd.to_numeric(plays.success, errors="raise").where(plays.success.isin([0, 1]))
        audit["missing_epa"] += int(plays.epa.isna().sum())
        audit["missing_success"] += int(plays.success.isna().sum())
        dropbacks = plays.loc[plays.qb_dropback.eq(1)]
        audit["dropbacks"] += len(dropbacks)
        audit["missing_passer_id"] += int(dropbacks.passer_player_id.isna().sum())
        missing_team = plays.posteam.isna() | plays.defteam.isna() | plays.game_id.isna()
        audit["missing_team"] += int(missing_team.sum())
        plays = plays.loc[~missing_team]
        for role, team_col in (("offense", "posteam"), ("defense_allowed", "defteam")):
            values = plays.assign(team=plays[team_col].replace(ALIASES))
            grouped = values.groupby(["game_id", "team"], sort=True).agg(
                plays=("play_id", "size"), epa_sum=("epa", "sum"), epa_n=("epa", "count"),
                success_sum=("success", "sum"), success_n=("success", "count")).reset_index()
            grouped["role"] = role
            accum.append(grouped)
    if not accum:
        return pd.DataFrame(columns=["game_id", "team", "role", "plays", "epa_sum", "epa_n", "success_sum", "success_n"]), audit
    table = pd.concat(accum).groupby(["game_id", "team", "role"], sort=True).sum().reset_index()
    table["epa_valid"] = table.epa_n / table.plays >= .95
    table["success_valid"] = table.success_n / table.plays >= .95
    # Keep counts for audit; incomplete game totals cannot enter rate estimates.
    return table, audit


def aggregate_parquet(path):
    file = pq.ParquetFile(path)
    return aggregate_batches(batch.to_pandas() for batch in file.iter_batches(batch_size=65536, columns=COLUMNS))


def training_priors(aggregates, games, first_forecast, evaluation_season):
    """League priors from strictly earlier training seasons/eligible labels only."""
    first_forecast = pd.Timestamp(first_forecast)
    if first_forecast.tzinfo is None:
        raise ValueError("Timezone-aware cutoff required")
    joined = aggregates.merge(games[["game_id", "season", "available_at", "status"]], on="game_id", validate="many_to_one")
    allowed = joined.loc[(joined.available_at < first_forecast) & joined.status.eq("completed")
                         & joined.season.lt(evaluation_season)]
    priors = {}
    for role in ("offense", "defense_allowed"):
        for metric in ("epa", "success"):
            frame = allowed.loc[allowed.role.eq(role) & allowed[f"{metric}_valid"]]
            n = int(frame[f"{metric}_n"].sum())
            if not n:
                raise ValueError("No eligible training plays for league prior")
            priors[f"{role}_{metric}"] = float(frame[f"{metric}_sum"].sum() / n)
    return priors


def pregame_efficiency(aggregates, games, game, cutoff, priors):
    """Season-to-date, target-excluded, 200-play shrinkage with explicit fallbacks."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or cutoff > game.kickoff:
        raise ValueError("Known pregame cutoff required")
    history = games.loc[games.season.eq(game.season) & games.status.eq("completed")
                        & (games.available_at < cutoff) & (games.kickoff < cutoff) & games.game_id.ne(game.game_id)]
    values = aggregates.loc[aggregates.game_id.isin(history.game_id)]
    result = {}
    for side in ("home", "away"):
        for role in ("offense", "defense_allowed"):
            for metric in ("epa", "success"):
                key = f"{role}_{metric}"
                prior = priors[key]
                if not np.isfinite(prior):
                    raise ValueError("A finite training-only prior is required")
                all_team = values.loc[values.team.eq(game[f"{side}_team"]) & values.role.eq(role)]
                team = all_team.loc[all_team[f"{metric}_valid"]]
                n = int(team[f"{metric}_n"].sum())
                result[f"{side}_{key}"] = float((team[f"{metric}_sum"].sum() + 200 * prior) / (n + 200))
                result[f"{side}_{key}_plays"] = n
                team_games = history.loc[history.home_team.eq(game[f"{side}_team"]) | history.away_team.eq(game[f"{side}_team"])]
                result[f"{side}_{key}_unavailable"] = int(n == 0 or len(team) < len(team_games))
    for role in ("offense", "defense_allowed"):
        for metric in ("epa", "success"):
            result[f"{role}_{metric}_diff"] = result[f"home_{role}_{metric}"] - result[f"away_{role}_{metric}"]
    return result
