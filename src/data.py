"""Public nflverse schedule snapshots; no synthetic production data."""
import json
import os
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get("NFL_STORE", ROOT / "artifacts"))
URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def now():
    return pd.Timestamp.now(tz="UTC")


def utc_time(value):
    """Reject unknown or timezone-naive timestamps rather than guessing UTC."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("A known, timezone-aware timestamp is required")
    return stamp.tz_convert("UTC")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def normalize(raw):
    games = raw.loc[raw.season.between(2023, 2026) & raw.game_type.eq("REG")].copy()
    for side in ("home", "away"):
        games[f"{side}_team"] = games[f"{side}_team"].replace({"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC"})
    if "kickoff" not in games:
        # A date without a published time must stay TBD, never become midnight.
        clock = games.gametime.astype("string")
        timestamp = (games.gameday.astype("string") + " " + clock).where(
            clock.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", na=False))
        local = pd.to_datetime(timestamp, errors="coerce")
        games["kickoff"] = local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    else:
        def known_kickoff(value):
            try:
                return utc_time(value)
            except (ValueError, TypeError):
                return pd.NaT
        games["kickoff"] = pd.to_datetime(games.kickoff.map(known_kickoff), utc=True)
    # Preserve the earliest published kickoff so a later reschedule cannot reopen
    # a prediction window that has already closed in this store.
    if "prediction_lock_at" in games:
        games["prediction_lock_at"] = pd.to_datetime(games.prediction_lock_at, utc=True, errors="coerce")
        games["prediction_lock_at"] = games[["prediction_lock_at", "kickoff"]].min(axis=1)
    else:
        games["prediction_lock_at"] = games.kickoff
    for column in ("home_score", "away_score"):
        games[column] = pd.to_numeric(games[column], errors="raise")
        invalid = games[column].notna() & ((games[column] < 0) | (games[column] % 1 != 0))
        if invalid.any():
            raise ValueError(f"Invalid {column}; scores must be finite nonnegative integers")
    unavailable = games.get("status", pd.Series("", index=games.index)).astype("string").str.lower().isin(
        ["postponed", "cancelled", "canceled", "suspended"])
    scored = games.home_score.notna() & games.away_score.notna()
    # nflverse supplies final scores, but no final whistle timestamp. A conservative
    # 12-hour embargo is conservative, not proof of an actual final whistle.
    games["available_at"] = games.kickoff + pd.Timedelta(hours=12)
    games["status"] = "scheduled"
    games.loc[~scored & games.kickoff.le(now()), "status"] = "underway_or_unavailable"
    games.loc[games.home_score.notna() ^ games.away_score.notna(), "status"] = "result_unavailable"
    games.loc[scored, "status"] = "completed"
    games.loc[unavailable, "status"] = "postponed"
    games.loc[games.kickoff.isna(), "status"] = "kickoff_unavailable"
    if games.game_id.isna().any() or not games.game_id.astype(str).str.fullmatch(r"[A-Za-z0-9_-]+").all():
        raise ValueError("Missing or unsafe source game ID")
    if games.game_id.duplicated().any():
        raise ValueError("Duplicate source game IDs")
    return games.sort_values("kickoff").reset_index(drop=True)


def load(offline=False, refresh=False):
    path = STORE / "games.csv"
    meta_path = STORE / "source.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if refresh:
        try:
            with urlopen(URL, timeout=45) as response:
                payload = response.read()
            import io
            candidate = normalize(pd.read_csv(io.BytesIO(payload)))
            if candidate.empty:
                raise ValueError("Empty schedule received")
            if path.exists():
                previous = normalize(pd.read_csv(path)).set_index("game_id")
                if set(previous.index) - set(candidate.game_id):
                    raise ValueError("Source dropped cached game IDs; review the upstream change before replacing the snapshot")
                prior_lock = candidate.game_id.map(previous.prediction_lock_at)
                candidate["prediction_lock_at"] = pd.concat([candidate.prediction_lock_at, prior_lock], axis=1).min(axis=1)
            STORE.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            candidate.to_csv(temp, index=False)
            temp.replace(path)
            meta = {"source": URL, "fetched_at": now().isoformat(), "error": None}
            write_json(meta_path, meta)
        except Exception as exc:
            meta["error"] = str(exc)
            write_json(meta_path, meta)
            if not path.exists():
                raise RuntimeError(f"Source unavailable and no cached snapshot: {exc}") from exc
    if not path.exists():
        raise FileNotFoundError("No schedule snapshot. Run python -m src.pipeline refresh")
    games = normalize(pd.read_csv(path))
    try:
        age = now() - utc_time(meta.get("fetched_at"))
        invalid_age = age < pd.Timedelta(0) or age > pd.Timedelta(hours=24)
    except (ValueError, TypeError):
        invalid_age = True
    meta["stale"] = bool(meta.get("error")) or invalid_age
    meta["offline"] = offline
    return games, meta
