"""Chronological evaluation and separately fitted production artifact."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from src.models.train_v1 import MODEL_FEATURES, build_model
from src.pregame import training_rows
from src.data import STORE, write_json, utc_time, now


def metrics(y, probabilities):
    if len(y) == 0:
        return {"games": 0, "accuracy": None, "log_loss": None, "brier": None}
    p = np.asarray(probabilities)
    observed, predicted = calibration_curve(y, p, n_bins=5, strategy="uniform")
    return {"games": len(y), "accuracy": float(accuracy_score(y, p >= .5)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
            "calibration": {"predicted": predicted.tolist(), "observed": observed.tolist()}}


def train(games, cutoff, source):
    cutoff = utc_time(cutoff)
    if cutoff > now():
        raise ValueError("Training cutoff is in the future")
    rows = training_rows(games, cutoff)
    if rows.empty or not {2023, 2024, 2025}.issubset(set(rows.season)):
        raise ValueError("Training requires completed 2023, 2024 and 2025 data")
    validation = {}
    for name, columns in MODEL_FEATURES.items():
        fit, val = rows.loc[rows.season.eq(2023)], rows.loc[rows.season.eq(2024)]
        model = build_model().fit(fit[columns], fit.target)
        validation[name] = metrics(val.target, model.predict_proba(val[columns])[:, 1])
    chosen = min(validation, key=lambda key: validation[key]["log_loss"])
    fit, test = rows.loc[rows.season.isin([2023, 2024])], rows.loc[rows.season.eq(2025)]
    evaluation = {}
    backtest = []
    for name, columns in MODEL_FEATURES.items():
        model = build_model().fit(fit[columns], fit.target)
        p = model.predict_proba(test[columns])[:, 1]
        evaluation[name] = metrics(test.target, p)
        if name == chosen:
            backtest = [{"game_id": row.game_id, "week": int(row.week), "target": int(row.target), "home_probability": float(prob), "kind": "backtest"} for row, prob in zip(test.itertuples(), p)]
    evaluation["home_baseline"] = {"games": len(test), "accuracy": float(test.target.mean())}
    evaluation["home_rate_probability_baseline"] = metrics(test.target, np.full(len(test), fit.target.mean()))
    columns = MODEL_FEATURES[chosen]
    model = build_model().fit(rows[columns], rows.target)
    fingerprint = hashlib.sha256(rows[["game_id", "target"] + columns].to_csv(index=False).encode()).hexdigest()[:10]
    version = f"v1{chosen}-{fingerprint}"
    metadata = {"version": version, "variant": chosen, "features": columns, "training_cutoff": cutoff.isoformat(),
                "training_games": len(rows), "training_seasons": sorted(rows.season.unique().tolist()),
                "validation_2024": validation, "holdout_2025": evaluation, "source": source,
                "policy": "Select A-D by 2024 log loss after fitting 2023; fixed C=1 and threshold=.5. Holdout 2025. Production refit on all eligible completed games. Calibration evaluated, not corrected."}
    STORE.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, STORE / "model.joblib.tmp")
    (STORE / "model.joblib.tmp").replace(STORE / "model.joblib")
    write_json(STORE / "model.json", metadata)
    write_json(STORE / "backtest.json", backtest)
    return metadata
