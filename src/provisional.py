"""Ephemeral production-model inference. No training or artifact writes."""
import math
import pandas as pd
from src.data import utc_time
from src.pregame import features
from src.teams import NAMES


def forecast(games, game, source, bundle, generated_at):
    """Use observed season-to-date inputs; never project future team statistics.

    Caller supplies the actual computation time, retained unchanged on cache hits.
    For TBD times, only a published date strictly after today is eligible.
    """
    unavailable = lambda reason: {"kind": "unavailable", "reason": reason}
    try:
        generated = utc_time(generated_at)
        freshness = utc_time(source.get("fetched_at"))
        if source.get("error") or source.get("stale") or not pd.Timedelta(0) <= generated - freshness <= pd.Timedelta(hours=24):
            return unavailable("Source snapshot is stale or invalid; refresh through the pipeline.")
        if game.status not in ("scheduled", "kickoff_unavailable") or pd.notna(game.home_score) or pd.notna(game.away_score):
            return unavailable("Only unplayed scheduled matchups are eligible.")
        if game.home_team not in NAMES or game.away_team not in NAMES or game.home_team == game.away_team:
            return unavailable("Known, distinct NFL teams are required.")
        if pd.notna(game.kickoff):
            if utc_time(game.kickoff) <= generated + pd.Timedelta(seconds=30):
                return unavailable("The prediction window has closed.")
        else:
            date = pd.Timestamp(game.get("gameday"))
            if pd.isna(date) or date.date() <= generated.tz_convert("America/New_York").date():
                return unavailable("Kickoff TBD: a published future game date is required.")
        lock = game.get("prediction_lock_at")
        if pd.notna(lock) and utc_time(lock) <= generated:
            return unavailable("The earliest known kickoff has passed.")
        if bundle is None:
            return unavailable("Reviewed production model artifact is unavailable.")
        meta = bundle["metadata"]
        if utc_time(meta["training_cutoff"]) > generated:
            return unavailable("Model training cutoff is in the future.")
        # A calculation boundary, never a claimed kickoff. No future results enter.
        boundary = game.copy()
        boundary["kickoff"] = generated
        values = features(games, boundary, min(generated, freshness))
        p = float(bundle["model"].predict_proba(pd.DataFrame([values])[meta["features"]])[0, 1])
        if not math.isfinite(p) or not 0 <= p <= 1:
            return unavailable("Model returned an invalid probability.")
        return {"kind": "provisional", "game_id": game.game_id,
                "generated_at": generated.isoformat(), "data_cutoff": min(generated, freshness).isoformat(),
                "source_freshness": freshness.isoformat(), "model_version": meta["version"],
                "home_probability": p, "away_probability": 1-p,
                "predicted_winner": game.home_team if p >= .5 else game.away_team,
                "result": "pending", "target": None,
                "policy": "Current observed statistics only; no projection of intervening games. Excluded from official metrics."}
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        return unavailable(f"Safe provisional inference unavailable: {exc}")
