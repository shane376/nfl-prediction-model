"""Reconstructed 2026 backtests, never archived pregame forecasts."""
import hashlib
import json
import os
import joblib
import pandas as pd
from src.data import STORE, now, utc_time
from src.modeling import MODEL_FEATURES, build_model
from src.pregame import features, training_rows
from src.predictions import records, save_record, prediction_writer

TRAINING_CUTOFF = pd.Timestamp("2026-02-01T00:00:00Z")
KIND = "retrospective_backtest"


def fit_frozen(games):
    """A is fixed before this sprint's 2026 evaluation; no 2026 model selection."""
    prior = games.loc[games.season.isin([2023, 2024, 2025])]
    rows = training_rows(prior, TRAINING_CUTOFF)
    if rows.empty or set(rows.season) != {2023, 2024, 2025}:
        raise ValueError("Frozen model requires eligible 2023–2025 training seasons")
    columns = MODEL_FEATURES["A"]
    digest = hashlib.sha256(rows[["game_id", "target"] + columns].to_csv(index=False).encode()).hexdigest()
    model = build_model().fit(rows[columns], rows.target)
    metadata = {"version": f"v1A-pre2026-{digest[:12]}", "variant": "A", "features": columns,
                "training_cutoff": TRAINING_CUTOFF.isoformat(), "training_seasons": [2023, 2024, 2025],
                "training_games": len(rows), "training_data_sha256": digest,
                "last_training_result_available_at": rows.available_at.max().isoformat(),
                "generated_at": now().isoformat(),
                "policy": "Fixed V1 A, C=1, threshold=.5; fit only 2023–2025 regular seasons available before 2026-02-01. Reconstructed now and frozen for 2026; no 2026 outcomes used for fitting or selection."}
    return {"model": model, "metadata": metadata}


def frozen_model(games):
    path = STORE / "retrospective" / "model.joblib"
    if path.exists():
        bundle = joblib.load(path)
    else:
        if any((STORE / "retrospective/predictions").glob("*.json")):
            raise ValueError("Frozen model missing for existing retrospective records; recover the reviewed artifact")
        bundle = fit_frozen(games)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".joblib.tmp")
        try:
            joblib.dump(bundle, temp)
            os.link(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    meta = bundle["metadata"]
    if meta["training_seasons"] != [2023, 2024, 2025] or utc_time(meta["training_cutoff"]) != TRAINING_CUTOFF or meta["features"] != MODEL_FEATURES["A"]:
        raise ValueError("Frozen model provenance does not match the pre-2026 policy")
    if utc_time(meta["last_training_result_available_at"]) >= TRAINING_CUTOFF:
        raise ValueError("Frozen model contains results after its training cutoff")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        save_record(metadata_path, meta)
    return bundle


def retrospective_records():
    result = []
    for path in sorted((STORE / "retrospective" / "predictions").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or row.get("kind") != KIND or not {"game_id", "season", "week"}.issubset(row):
            raise ValueError(f"Unexpected record type in {path.name}")
        result.append(row)
    return result


def backtest(games, source):
    """Append only completed games that have no original archived record."""
    with prediction_writer():
        bundle = frozen_model(games)
        meta = bundle["metadata"]
        archived = {row["game_id"] for row in records()}
        existing = {row["game_id"] for row in retrospective_records()}
        current = now()
        completed = games.loc[games.season.eq(2026) & games.status.eq("completed")
                              & games.kickoff.notna() & (games.available_at < current)]
        added = 0
        for _, game in completed.iterrows():
            if game.game_id in archived or game.game_id in existing:
                continue
            if game.kickoff <= TRAINING_CUTOFF:
                raise ValueError("Backtest kickoff must follow frozen model training cutoff")
            values = features(games, game, game.kickoff)
            p = float(bundle["model"].predict_proba(pd.DataFrame([values])[meta["features"]])[0, 1])
            row = {"kind": KIND, "season": int(game.season), "week": int(game.week), "game_id": game.game_id,
                   "home_team": game.home_team, "away_team": game.away_team, "kickoff": game.kickoff.isoformat(),
                   "generated_at": now().isoformat(), "feature_cutoff": game.kickoff.isoformat(),
                   "training_cutoff": meta["training_cutoff"], "training_seasons": meta["training_seasons"],
                   "model_version": meta["version"], "model_generated_at": meta["generated_at"],
                   "home_probability": p, "away_probability": 1 - p,
                   "predicted_winner": game.home_team if p >= .5 else game.away_team,
                   "source_freshness": source.get("fetched_at"), "source_stale": bool(source.get("stale")),
                   "features": values,
                   "availability_assumption": "Latest final scores reconstructed with a 12-hour post-kickoff embargo; not an archived point-in-time source replay."}
            save_record(STORE / "retrospective" / "predictions" / f"{game.game_id}.json", row)
            added += 1
        return added


def retrospective_results(games):
    """Read-only scoring; changed inputs are visible, never silently regenerated."""
    lookup = games.set_index("game_id", drop=False)
    archived = {row["game_id"] for row in records()}
    metadata_path = STORE / "retrospective/model.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        raise ValueError("Invalid frozen model metadata")
    results = []
    for row in retrospective_records():
        result = {**row, "result": "unavailable", "actual": None, "target": None, "integrity_error": None}
        try:
            game = lookup.loc[row["game_id"]]
            cutoff = utc_time(row["feature_cutoff"])
            if row["game_id"] in archived:
                raise ValueError("An archived pick takes precedence; retrospective excluded")
            if row["training_seasons"] != [2023, 2024, 2025] or utc_time(row["training_cutoff"]) != TRAINING_CUTOFF:
                raise ValueError("Invalid frozen training provenance")
            if row.get("model_version") != metadata.get("version"):
                raise ValueError("Frozen model metadata missing or version mismatch")
            if row.get("model_generated_at") != metadata.get("generated_at") or utc_time(row["model_generated_at"]) > utc_time(row["generated_at"]):
                raise ValueError("Invalid frozen model generation provenance")
            if not (TRAINING_CUTOFF < cutoff == game.kickoff == utc_time(row["kickoff"]) < utc_time(row["generated_at"]) <= now()):
                raise ValueError("Invalid retrospective timestamps or changed kickoff")
            if any(row[key] != game[key] for key in ("home_team", "away_team", "season", "week")):
                raise ValueError("Retrospective matchup changed")
            from math import isfinite, isclose
            p, away = row["home_probability"], row["away_probability"]
            if not (type(p) in (int, float) and type(away) in (int, float) and isfinite(p) and 0 <= p <= 1 and isfinite(away) and 0 <= away <= 1 and isclose(p + away, 1, abs_tol=1e-9)):
                raise ValueError("Invalid retrospective probabilities")
            winner = game.home_team if p >= .5 else game.away_team
            if row["predicted_winner"] != winner:
                raise ValueError("Retrospective winner/probability mismatch")
            if features(games, game, cutoff) != row["features"]:
                raise ValueError("Historical feature inputs changed since generation; review source corrections")
            if game.status != "completed" or pd.isna(game.home_score) or pd.isna(game.away_score) or game.available_at >= now():
                raise ValueError("Result not yet eligible")
            if utc_time(row["generated_at"]) < game.available_at:
                raise ValueError("Backtest generation precedes result eligibility")
            if game.home_score == game.away_score:
                result.update(result="tie", actual="TIE")
            else:
                y = int(game.home_score > game.away_score)
                actual = game.home_team if y else game.away_team
                result.update(target=y, actual=actual, result="correct" if actual == row["predicted_winner"] else "incorrect")
        except (KeyError, TypeError, ValueError) as exc:
            result["integrity_error"] = str(exc)
        results.append(result)
    return results
