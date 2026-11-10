"""Pure dashboard formatting and selection helpers."""
import pandas as pd
from math import isfinite
from src.teams import NAMES, display

LABELS = {"live": "Recorded pregame pick", "retrospective_backtest": "Retrospective backtest",
          "provisional": "Provisional forecast", "unavailable": "Prediction unavailable"}
STATUSES = {"completed": "Final", "scheduled": "Upcoming", "kickoff_unavailable": "Kickoff TBD",
            "underway_or_unavailable": "Result pending", "result_unavailable": "Incomplete score",
            "postponed": "Postponed", "canceled": "Canceled", "suspended": "Suspended"}


def kickoff_label(value):
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return "Kickoff TBD" if pd.isna(stamp) else stamp.tz_convert("America/New_York").strftime("%a, %b %d · %I:%M %p %Z")


def default_week(games, current):
    future = games.loc[games.kickoff.ge(current)].sort_values("kickoff")
    return int(future.iloc[0].week) if len(future) else (int(games.week.max()) if len(games) else 1)


def prediction_label(record):
    return LABELS["unavailable"] if not record or record.get("integrity_error") else LABELS.get(record.get("kind"), LABELS["unavailable"])


def data_status(source, games, current):
    stamp = pd.to_datetime(source.get("fetched_at"), utc=True, errors="coerce")
    stale = bool(source.get("stale") or source.get("error") or pd.isna(stamp) or
                 not pd.Timedelta(0) <= current - stamp <= pd.Timedelta(hours=24))
    pending = int((games.status.isin(["underway_or_unavailable", "result_unavailable"]) |
                   (games.status.eq("completed") & games.available_at.ge(current))).sum())
    return {"stale": stale, "refresh": kickoff_label(stamp) if pd.notna(stamp) else "Unknown",
            "pending": pending}


def matchup_rows(games, recorded, retrospective, provisional, current, stale=False):
    """Join display data without mixing cohorts or replacing an invalid archive.

    All maps contain reconciled results or ephemeral estimates, never raw picks.
    Cache eligibility is rechecked for both weekly cards and Team Explorer.
    """
    rows = []
    for _, game in games.iterrows():
        record = next((m[game.game_id] for m in (recorded, retrospective, provisional) if game.game_id in m), None)
        reason = "No eligible forecast. Completed games are never forecast now."
        if record:
            reason = record.get("integrity_error") or record.get("reason") or reason
            if record.get("kind") == "provisional":
                date = pd.to_datetime(game.get("gameday"), errors="coerce")
                closed = (pd.notna(game.kickoff) and game.kickoff <= current + pd.Timedelta(seconds=30))
                tbd_closed = pd.isna(game.kickoff) and (pd.isna(date) or date.date() <= current.tz_convert("America/New_York").date())
                lock = game.get("prediction_lock_at")
                if stale or closed or tbd_closed or (pd.notna(lock) and lock <= current) or game.status not in ("scheduled", "kickoff_unavailable") or pd.notna(game.home_score) or pd.notna(game.away_score):
                    reason = "Source became stale or the pregame window closed."
                    record = {**record, "integrity_error": reason}
        probability = record.get("home_probability") if record else None
        valid = bool(record and not record.get("integrity_error") and
                     record.get("kind") in ("live", "retrospective_backtest", "provisional") and
                     isinstance(probability, (int, float)) and isfinite(probability) and 0 <= probability <= 1)
        kind = record["kind"] if valid else "unavailable"
        rows.append({**game.to_dict(), "forecast": record, "forecast_type": kind,
                     "provenance": LABELS[kind], "reason": reason,
                     "winner": record.get("predicted_winner") if valid else None,
                     "confidence": max(probability, 1-probability) if valid else None,
                     "status_label": STATUSES.get(game.status, game.status)})
    return rows


def filter_matchups(rows, *, week=None, team=None, query="", statuses=(), forecast_types=(),
                    winner=None, confidence=None, sort="Kickoff"):
    """AND filters; confidence is an optional inclusive (low, high) probability range."""
    tokens = [t for t in query.casefold().replace("@", " ").split() if t != "at"]
    selected = []
    for row in rows:
        text = f"{display(row['away_team'])} {display(row['home_team'])}".casefold()
        if week is not None and row["week"] != week:
            continue
        if team and team not in (row["home_team"], row["away_team"]):
            continue
        if any(t not in text for t in tokens):
            continue
        if statuses and row["status"] not in statuses:
            continue
        if forecast_types and row["forecast_type"] not in forecast_types:
            continue
        if winner and row["winner"] != winner:
            continue
        if confidence is not None and (row["confidence"] is None or not confidence[0] <= row["confidence"] <= confidence[1]):
            continue
        selected.append(row)
    def key(row):
        tie = (int(row["week"]), str(row["game_id"]))
        if sort in ("Confidence: high to low", "Confidence: low to high"):
            value = row["confidence"]
            return (value is None, (-value if sort == "Confidence: high to low" else value) if value is not None else 0, *tie)
        if sort == "Matchup A-Z":
            return (NAMES.get(row["away_team"], row["away_team"]), NAMES.get(row["home_team"], row["home_team"]), *tie)
        return (pd.isna(row["kickoff"]), row["kickoff"] if pd.notna(row["kickoff"]) else pd.Timestamp.max.tz_localize("UTC"), *tie)
    return sorted(selected, key=key)


def team_schedule(rows, team):
    """Keep every source matchup, including unavailable forecasts and TBD times."""
    schedule = []
    for row in sorted(rows, key=lambda r: (int(r["week"]), str(r["game_id"]))):
        if team not in (row["home_team"], row["away_team"]):
            continue
        home = row["home_team"] == team
        scored, allowed = (row["home_score"], row["away_score"]) if home else (row["away_score"], row["home_score"])
        final = row["status"] == "completed" and pd.notna(scored) and pd.notna(allowed)
        result = ("W" if scored > allowed else "L" if scored < allowed else "T") if final else None
        schedule.append({**row, "opponent": row["away_team"] if home else row["home_team"],
                         "venue": "Home" if home else "Away", "actual_result": result,
                         "score": f"{scored:g}-{allowed:g}" if final else None})
    return schedule


def team_record(schedule):
    """Actual W-L-T only, independent of forecast coverage and result embargo."""
    return {key: sum(row["actual_result"] == result for row in schedule)
            for key, result in (("wins", "W"), ("losses", "L"), ("ties", "T"))}


def remaining_wins_estimate(schedule, team):
    """No extrapolation: withhold the estimate unless every remaining game is covered."""
    remaining = [r for r in schedule if r["actual_result"] is None and r["status"] != "canceled"]
    covered = [r for r in remaining if r["forecast_type"] in ("live", "provisional") and
               r["status"] in ("scheduled", "kickoff_unavailable")]
    expected = sum(r["forecast"]["home_probability"] if r["home_team"] == team else 1-r["forecast"]["home_probability"] for r in covered)
    return {"remaining": len(remaining), "covered": len(covered),
            "estimated_wins": expected if remaining and len(covered) == len(remaining) else None}
