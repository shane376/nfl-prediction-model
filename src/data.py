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
        timestamp = games.gameday.astype("string") + " " + games.gametime.astype("string")
        local = pd.to_datetime(timestamp, errors="coerce")
        games["kickoff"] = local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    else:
        games["kickoff"] = pd.to_datetime(games.kickoff, utc=True)
    scored = games.home_score.notna() & games.away_score.notna()
    # nflverse supplies final scores, but no final whistle timestamp. A conservative
    # 12-hour embargo prevents ongoing games from entering pregame features.
    games["available_at"] = games.kickoff + pd.Timedelta(hours=12)
    games["status"] = "scheduled"
    games.loc[~scored & games.kickoff.le(now()), "status"] = "underway_or_unavailable"
    games.loc[scored, "status"] = "completed"
    if games.game_id.duplicated().any():
        raise ValueError("Duplicate source game IDs")
    return games.sort_values("kickoff").reset_index(drop=True)


def load(offline=False, refresh=False):
    path = STORE / "games.csv"
    meta_path = STORE / "source.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if refresh:
        try:
            with urlopen(URL, timeout=45) as response:
                payload = response.read()
            import io
            candidate = normalize(pd.read_csv(io.BytesIO(payload)))
            if candidate.empty:
                raise ValueError("Empty schedule received")
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
    meta["stale"] = bool(meta.get("error")) or not meta.get("fetched_at") or now() - pd.Timestamp(meta["fetched_at"]) > pd.Timedelta(hours=24)
    meta["offline"] = offline
    return games, meta
