"""Append-only live prediction records and reproducible reconciliation."""
import json
import joblib
import pandas as pd
from src.data import STORE, now, write_json
from src.pregame import features
from src.modeling import metrics


def records():
    return [json.loads(path.read_text()) for path in sorted((STORE / "predictions").glob("*.json"))]


def predict(games, source, season=2026, week=None, rerun=False):
    bundle = joblib.load(STORE / "model.joblib")
    meta = bundle["metadata"]
    generated = now()
    cutoff = pd.Timestamp(meta["training_cutoff"])
    if cutoff > generated:
        raise ValueError("Model training cutoff is in the future")
    if source.get("offline") or source.get("fixture"):
        raise ValueError("Live predictions cannot be recorded from offline fixture mode")
    selected = games.loc[games.season.eq(season)]
    if week is not None:
        selected = selected.loc[selected.week.eq(week)]
    else:
        selected = selected.loc[selected.kickoff <= generated + pd.Timedelta(days=7)]
    existing = records()
    added = 0
    for _, game in selected.iterrows():
        # Never backdate or predict underway/completed/TBD games.
        if pd.isna(game.kickoff) or game.kickoff <= generated or game.status != "scheduled":
            continue
        old = [r for r in existing if r["game_id"] == game.game_id]
        if any(pd.Timestamp(r["kickoff"]) <= generated for r in old):
            continue
        if old and not rerun:
            continue
        if cutoff >= game.kickoff:
            continue
        x = pd.DataFrame([features(games, game, generated)])[meta["features"]]
        p = float(bundle["model"].predict_proba(x)[0, 1])
        revision = len(old) + 1
        record = {"season": int(game.season), "week": int(game.week), "game_id": game.game_id,
                  "home_team": game.home_team, "away_team": game.away_team, "kickoff": game.kickoff.isoformat(),
                  "predicted_winner": game.home_team if p >= .5 else game.away_team,
                  "home_probability": p, "away_probability": 1 - p, "model_version": meta["version"],
                  "generated_at": generated.isoformat(), "data_cutoff": generated.isoformat(),
                  "training_cutoff": cutoff.isoformat(), "source_freshness": source.get("fetched_at"),
                  "source_stale": source.get("stale", False), "prediction_status": "recorded", "kind": "live", "revision": revision}
        write_json(STORE / "predictions" / f"{game.game_id}-r{revision:04d}.json", record)
        existing.append(record)
        added += 1
    return added


def reconcile(games):
    # Score the first authentic pregame record only; later revisions remain auditable.
    first = {}
    for record in sorted(records(), key=lambda r: r["generated_at"]):
        if record["kind"] == "live" and pd.Timestamp(record["generated_at"]) < pd.Timestamp(record["kickoff"]):
            first.setdefault(record["game_id"], record)
    results = []
    current = now()
    lookup = games.set_index("game_id")
    for gid, record in first.items():
        row = {**record, "actual": None, "result": "pending", "target": None}
        if gid in lookup.index:
            game = lookup.loc[gid]
            # A changed earlier kickoff invalidates a late pick.
            if pd.notna(game.kickoff) and pd.Timestamp(record["generated_at"]) >= game.kickoff:
                row["result"] = "unavailable"
            elif game.status == "completed" and game.available_at < current:
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
    write_json(STORE / "live-results.json", results)
    write_json(STORE / "live-metrics.json", summary)
    return summary
