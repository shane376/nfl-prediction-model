"""Append-only live prediction records and reproducible reconciliation."""
import json
import math
import os
from contextlib import contextmanager
from uuid import uuid4
import joblib
import pandas as pd
from src.data import STORE, now, write_json, utc_time
from src.pregame import features
from src.modeling import metrics


def records():
    result = []
    for path in sorted((STORE / "predictions").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or not record.get("game_id"):
                raise ValueError("Expected a record with game_id")
            result.append(record)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot read prediction {path.name}: {exc}") from exc
    return result


@contextmanager
def prediction_writer():
    """Fail closed on concurrent writers; a crashed writer requires manual recovery."""
    STORE.mkdir(parents=True, exist_ok=True)
    lock = STORE / ".prediction-writer.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def save_record(path, record, deadline=None):
    """Publish a complete JSON file atomically, without replacing an existing name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        if deadline is not None and now() + pd.Timedelta(seconds=30) >= deadline:
            return False
        os.link(temp, path)  # Atomic create-only publication on NTFS and Linux.
        return True
    finally:
        temp.unlink(missing_ok=True)


def integrity_error(record, game, current=None):
    """Validate a saved live pick against both its recorded and latest kickoff."""
    try:
        if record["kind"] != "live":
            return "Not an authentic live prediction"
        if not isinstance(record["model_version"], str) or not record["model_version"]:
            return "Missing model version"
        generated = utc_time(record["generated_at"])
        kickoff = utc_time(record["kickoff"])
        data_cutoff = utc_time(record["data_cutoff"])
        training_cutoff = utc_time(record["training_cutoff"])
        freshness = utc_time(record["source_freshness"])
        if not (training_cutoff <= data_cutoff <= generated < kickoff):
            return "Invalid pregame timestamp chain"
        if freshness > data_cutoff or generated > (current if current is not None else now()):
            return "Source or prediction timestamp is in the future"
        if any(type(record[key]) not in (int, float) for key in ("home_probability", "away_probability")):
            return "Probabilities must be JSON numbers"
        p, away = float(record["home_probability"]), float(record["away_probability"])
        if not (math.isfinite(p) and math.isfinite(away) and 0 <= p <= 1 and 0 <= away <= 1 and math.isclose(p + away, 1, abs_tol=1e-9)):
            return "Invalid probabilities"
        if record["predicted_winner"] != record["home_team" if p >= .5 else "away_team"]:
            return "Winner does not match probabilities"
        if game is None:
            return "Game missing from latest schedule"
        actual_kickoff = utc_time(game.kickoff)
        if generated >= actual_kickoff:
            return "Prediction is late against latest source kickoff"
        lock_at = game.get("prediction_lock_at", actual_kickoff)
        if pd.notna(lock_at) and generated >= lock_at:
            return "Prediction is late against earliest known kickoff"
        if game.status in ("postponed", "canceled", "suspended", "kickoff_unavailable"):
            return "Kickoff postponed or uncertain"
        if any(record[key] != game[key] for key in ("game_id", "season", "home_team", "away_team")):
            return "Matchup changed in latest schedule"
    except (KeyError, ValueError, TypeError, OverflowError):
        return "Missing or invalid prediction metadata/kickoff"
    return None


def audit(games):
    lookup = games.set_index("game_id", drop=False)
    return [{"game_id": row["game_id"], "revision": row.get("revision"),
             "error": integrity_error(row, lookup.loc[row["game_id"]] if row["game_id"] in lookup.index else None)}
            for row in records()]


def predict(games, source, season=2026, week=None, rerun=False):
    with prediction_writer():
        return _predict(games, source, season, week, rerun)


def _predict(games, source, season, week, rerun):
    bundle = joblib.load(STORE / "model.joblib")
    meta = bundle["metadata"]
    started = now()
    cutoff = utc_time(meta["training_cutoff"])
    if cutoff > started:
        raise ValueError("Model training cutoff is in the future")
    if source.get("offline") or source.get("fixture"):
        raise ValueError("Live predictions cannot be recorded from offline fixture mode")
    freshness = utc_time(source.get("fetched_at"))
    if source.get("error") or source.get("stale") or not pd.Timedelta(0) <= started - freshness <= pd.Timedelta(hours=24):
        raise ValueError("Live predictions require a successful source snapshot no older than 24 hours")
    selected = games.loc[games.season.eq(season)]
    if week is not None:
        selected = selected.loc[selected.week.eq(week)]
    else:
        selected = selected.loc[selected.kickoff <= started + pd.Timedelta(days=7)]
    existing = records()
    added = 0
    for _, game in selected.iterrows():
        data_cutoff = now()
        # Never backdate or predict underway/completed/TBD games.
        if pd.isna(game.kickoff) or game.kickoff <= data_cutoff or game.status != "scheduled":
            continue
        if pd.notna(game.get("prediction_lock_at")) and game.prediction_lock_at <= data_cutoff:
            continue
        old = [r for r in existing if r["game_id"] == game.game_id]
        if any(pd.isna(pd.to_datetime(r.get("kickoff"), utc=True, errors="coerce")) or utc_time(r["kickoff"]) <= data_cutoff for r in old):
            continue
        if old and not rerun:
            continue
        if cutoff >= game.kickoff:
            continue
        x = pd.DataFrame([features(games, game, data_cutoff)])[meta["features"]]
        p = float(bundle["model"].predict_proba(x)[0, 1])
        revision = max((int(r.get("revision", 0)) for r in old), default=0) + 1
        # Check the real clock AFTER computation; never backdate to run start.
        generated = now()
        if generated + pd.Timedelta(seconds=30) >= game.kickoff:
            continue
        record = {"season": int(game.season), "week": int(game.week), "game_id": game.game_id,
                  "home_team": game.home_team, "away_team": game.away_team, "kickoff": game.kickoff.isoformat(),
                  "predicted_winner": game.home_team if p >= .5 else game.away_team,
                  "home_probability": p, "away_probability": 1 - p, "model_version": meta["version"],
                  "generated_at": generated.isoformat(), "data_cutoff": data_cutoff.isoformat(),
                  "training_cutoff": cutoff.isoformat(), "source_freshness": source.get("fetched_at"),
                  "source_stale": source.get("stale", False), "prediction_status": "recorded", "kind": "live", "revision": revision}
        error = integrity_error(record, game, generated)
        if error:
            raise ValueError(error)
        path = STORE / "predictions" / f"{game.game_id}-r{revision:04d}.json"
        if not save_record(path, record, game.kickoff):
            continue
        existing.append(record)
        added += 1
    return added


def reconciliation(games):
    """Pure read-only reconciliation, shared by CLI and dashboard."""
    # Score the first authentic pregame record only; later revisions remain auditable.
    first = {}
    for record in sorted(records(), key=lambda r: (r["game_id"], int(r.get("revision", 0)))):
        if record.get("kind") == "live":
            first.setdefault(record["game_id"], record)
    results = []
    current = now()
    lookup = games.set_index("game_id", drop=False)
    for gid, record in first.items():
        row = {**record, "actual": None, "result": "pending", "target": None}
        game = lookup.loc[gid] if gid in lookup.index else None
        error = integrity_error(record, game, current)
        row["integrity_error"] = error
        if error:
            row["result"] = "unavailable"
        elif game is not None:
            if game.status == "completed" and game.available_at < current and pd.notna(game.home_score) and pd.notna(game.away_score):
                if game.home_score == game.away_score:
                    row.update(actual="TIE", result="tie")
                else:
                    y = int(game.home_score > game.away_score)
                    actual = game.home_team if y else game.away_team
                    row.update(actual=actual, target=y, result="correct" if record["predicted_winner"] == actual else "incorrect")
        results.append(row)
    scored = [r for r in results if r["target"] is not None]
    summary = metrics([r["target"] for r in scored], [r["home_probability"] for r in scored])
    summary.update(correct=sum(r["result"] == "correct" for r in results), incorrect=sum(r["result"] == "incorrect" for r in results))
    return results, summary


def reconcile(games):
    results, summary = reconciliation(games)
    write_json(STORE / "live-results.json", results)
    write_json(STORE / "live-metrics.json", summary)
    return summary
