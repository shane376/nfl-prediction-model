"""Pure dashboard formatting and selection helpers."""
import pandas as pd

LABELS = {"live": "Recorded pregame pick", "retrospective_backtest": "Retrospective backtest",
          "provisional": "Provisional forecast", "unavailable": "Prediction unavailable"}


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
