"""Pinned historical schedule ingestion, separate from production artifacts."""
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from src.teams import NAMES

ALIASES = {"JAC": "JAX", "LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "WSH": "WAS"}
RESEARCH = Path(__file__).resolve().parents[2] / "research/v2_0_2"
TRANSFORM = "schedule-context-v2.0.2-1"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def retrieve(directory=RESEARCH):
    """Explicit network action; offline experiments only read this pinned input."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    api = "https://api.github.com/repos/nflverse/nfldata/commits?path=data/games.csv&per_page=1"
    with urlopen(Request(api, headers={"User-Agent": "GridironLab-research"}), timeout=60) as response:
        commit = json.load(response)[0]
    url = f'https://raw.githubusercontent.com/nflverse/nfldata/{commit["sha"]}/data/games.csv'
    with urlopen(url, timeout=60) as response:
        raw = response.read()
    metadata = dict(url=url, discovery_url=api, commit=commit["sha"],
                    commit_timestamp=commit["commit"]["committer"]["date"],
                    retrieved_at=datetime.now(timezone.utc).isoformat(), raw_sha256=digest(raw),
                    bytes=len(raw), schema=pd.read_csv(io.BytesIO(raw), nrows=0).columns.tolist(),
                    availability="latest-data reconstruction; no historical publication/ingestion proof",
                    transform_version=TRANSFORM)
    # Never silently replace an input used in a published experiment.
    target = directory / "games.csv.gz"
    if target.exists() and gzip.decompress(target.read_bytes()) != raw:
        raise FileExistsError("Research input is pinned; use a new research directory for a new source revision")
    if target.exists() and (directory / "source.json").exists():
        return json.loads((directory / "source.json").read_text(encoding="utf-8"))
    target.write_bytes(gzip.compress(raw, mtime=0))
    (directory / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def normalize(raw, lag_hours=48):
    """Latest-data reconstruction only; lag is an assumption, never publication proof."""
    required = {"season", "week", "game_id", "game_type", "gameday", "gametime", "home_team", "away_team", "home_score", "away_score"}
    if required - set(raw):
        raise ValueError(f"Missing schedule fields: {sorted(required - set(raw))}")
    if lag_hours not in (24, 48, 72):
        raise ValueError("Use the prespecified 24/48/72-hour availability assumptions")
    games = raw.loc[raw.season.between(2015, 2024) & raw.game_type.eq("REG")].copy()
    for side in ("home", "away"):
        games[f"{side}_team"] = games[f"{side}_team"].replace(ALIASES)
        if not games[f"{side}_team"].isin(NAMES).all():
            raise ValueError("Unknown historical team alias")
        scores = pd.to_numeric(games[f"{side}_score"], errors="raise")
        if (scores.notna() & ((scores < 0) | scores.mod(1).ne(0))).any():
            raise ValueError("Invalid score")
        games[f"{side}_score"] = scores
    if games.game_id.isna().any() or games.game_id.duplicated().any():
        raise ValueError("Missing or duplicate game IDs")
    if games.home_team.eq(games.away_team).any():
        raise ValueError("Self matchup")
    if not games.week.between(1, 18).all() or not games.week.mod(1).eq(0).all():
        raise ValueError("Invalid regular-season week")
    clock = games.gametime.astype("string")
    valid = clock.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", na=False)
    local = pd.to_datetime((games.gameday.astype("string") + " " + clock).where(valid), errors="coerce")
    games["kickoff"] = local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    known = games.loc[games.kickoff.notna()]
    appearances = pd.concat([known[["kickoff", f"{side}_team"]].rename(columns={f"{side}_team": "team"}) for side in ("home", "away")])
    if appearances.duplicated(["kickoff", "team"]).any():
        raise ValueError("Team scheduled twice at the same kickoff")
    # No final-whistle timestamps exist here. A fixed four-hour end proxy plus
    # the prespecified postgame lag is a reconstruction assumption only.
    games["available_at"] = games.kickoff + pd.Timedelta(hours=4 + lag_hours)
    scored = games.home_score.notna() & games.away_score.notna()
    status = games.get("status", pd.Series("", index=games.index)).astype(str).str.lower()
    games["status"] = "unavailable"
    games.loc[scored & games.kickoff.notna() & ~status.isin(["postponed", "canceled", "cancelled", "suspended"]), "status"] = "completed"
    games["location"] = games.get("location", pd.Series("", index=games.index)).fillna("Unknown")
    return games.sort_values(["kickoff", "game_id"], kind="stable").reset_index(drop=True)


def audit(games):
    rows = []
    for season in range(2015, 2025):
        frame = games.loc[games.season.eq(season)]
        # BUF-CIN 2022 was canceled without a result; never invent its outcome.
        expected = 271 if season == 2022 else 272 if season >= 2021 else 256
        appearances = pd.concat([frame.home_team, frame.away_team]).value_counts()
        expected_counts = {team: (16 if season < 2021 or (season == 2022 and team in ("BUF", "CIN")) else 17) for team in NAMES}
        counts_ok = appearances.to_dict() == expected_counts
        def consistent_id(row):
            parts = str(row.game_id).split("_")
            return (len(parts) == 4 and parts[:2] == [str(season), f"{int(row.week):02d}"]
                    and ALIASES.get(parts[2], parts[2]) == row.away_team
                    and ALIASES.get(parts[3], parts[3]) == row.home_team)
        id_errors = int((~frame.apply(consistent_id, axis=1)).sum()) if len(frame) else 0
        rows.append(dict(season=season, games=len(frame), expected=expected,
                         completed=int(frame.status.eq("completed").sum()),
                         missing_kickoff=int(frame.kickoff.isna().sum()),
                         missing_scores=int((frame.home_score.isna() | frame.away_score.isna()).sum()),
                         venue_unknown=int((~frame.location.isin(["Home", "Neutral"])).sum()),
                         neutral=int(frame.location.eq("Neutral").sum()),
                         id_metadata_errors=id_errors,
                         teams=len(appearances), team_counts_valid=counts_ok,
                         complete=bool(len(frame) == expected and counts_ok and id_errors == 0 and frame.status.eq("completed").all())))
    return pd.DataFrame(rows)


def read(directory=RESEARCH, lag_hours=48):
    directory = Path(directory)
    metadata = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    raw = gzip.decompress((directory / "games.csv.gz").read_bytes())
    if digest(raw) != metadata["raw_sha256"]:
        raise ValueError("Research source checksum mismatch")
    return normalize(pd.read_csv(io.BytesIO(raw)), lag_hours), metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--directory", type=Path, default=RESEARCH)
    args = parser.parse_args()
    if args.fetch:
        print(json.dumps(retrieve(args.directory), indent=2))
    games, _ = read(args.directory)
    table = audit(games)
    table.to_csv(args.directory / "coverage.csv", index=False, lineterminator="\n")
    print(table.to_string(index=False))
