"""Offline V2.0.2 schedule-feature reconstruction; writes research outputs only."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.research.schedule_data import RESEARCH, TRANSFORM, audit, read
from src.research.team_context import CONFIGS, build_features

ROOT = Path(__file__).resolve().parents[1]


def manifest_identifier(path):
    """Repo-relative keys internally; canonical file URIs for external inputs."""
    resolved = Path(path).resolve(strict=True)
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_uri()


def hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != ".prediction-writer.lock"}


def score(target, probability):
    p = np.asarray(probability, dtype=np.float64)
    return dict(games=len(target), accuracy=float(accuracy_score(target, p >= .5)),
                brier=float(brier_score_loss(target, p)), log_loss=float(log_loss(target, p, labels=[0, 1])))


def split_fold(rows, season, first_training_season=2017):
    evaluation = rows.loc[rows.season.eq(season) & rows.target.notna()].copy()
    if evaluation.empty:
        raise ValueError("No eligible evaluation games")
    # Include even excluded opener games when establishing the fold's first t.
    first_cutoff = rows.loc[rows.season.eq(season), "forecast_cutoff"].min()
    train = rows.loc[rows.season.between(first_training_season, season - 1) & rows.target.notna()
                     & (rows.label_available_at < first_cutoff)].copy()
    if train.empty or not train.season.lt(season).all():
        raise ValueError("Insufficient strictly earlier training labels")
    return train, evaluation, first_cutoff


def evaluate_fold(rows, season):
    train, evaluation, first_cutoff = split_fold(rows, season)
    predictions = evaluation[["game_id", "season", "week", "target", "forecast_cutoff"]].copy()
    fitted = {}
    for name, cols in CONFIGS.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42))
        model.fit(train[cols], train.target.astype(int))
        predictions[name] = model.predict_proba(evaluation[cols])[:, 1]
        scaler, logistic = model.steps[0][1], model.steps[1][1]
        fitted[name] = dict(features=cols, scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                            coefficients=logistic.coef_[0].tolist(), intercept=float(logistic.intercept_[0]))
    predictions["coin"] = .5
    predictions["training_home_rate"] = float(train.target.mean())
    predictions["fixed_elo"] = evaluation.fixed_elo_probability.to_numpy()
    metrics = [dict(season=season, model=name, **score(predictions.target, predictions[name]))
               for name in [*CONFIGS, "coin", "training_home_rate", "fixed_elo"]]
    provenance = dict(season=season, first_forecast=first_cutoff.isoformat(), training_games=len(train),
                      training_seasons=sorted(int(s) for s in train.season.unique()),
                      max_training_label_time=train.label_available_at.max().isoformat(),
                      training_game_ids=train.game_id.tolist(), evaluation_game_ids=evaluation.game_id.tolist(),
                      fitted_parameters=fitted)
    return predictions, metrics, provenance


def paired_deltas(predictions, draws=5000):
    """Descriptive development uncertainty; shared paired week/multiweek draws."""
    output = []
    for lag, frame in predictions.groupby("lag_hours"):
        y = frame.target.to_numpy()
        losses = {}
        for name in [*CONFIGS, "coin", "training_home_rate", "fixed_elo"]:
            p = frame[name].to_numpy()
            safe = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
            losses[name] = np.column_stack(((p-y)**2, -(y*np.log(safe)+(1-y)*np.log1p(-safe))))
        for width in (1, 2):
            labels = list(zip(frame.season, (frame.week - 1) // width))
            keys = sorted(set(labels))
            index = {k: i for i, k in enumerate(keys)}
            codes = np.array([index[k] for k in labels])
            counts = np.bincount(codes, minlength=len(keys))
            picks = np.random.default_rng(20202).integers(0, len(keys), size=(draws, len(keys)))
            denominator = counts[picks].sum(axis=1)
            for candidate in list(CONFIGS)[1:]:
                for comparator in ("v1_a", "coin", "training_home_rate", "fixed_elo"):
                    delta = losses[candidate] - losses[comparator]
                    blocks = np.column_stack([np.bincount(codes, weights=delta[:, j], minlength=len(keys)) for j in (0, 1)])
                    sampled = blocks[picks].sum(axis=1) / denominator[:, None]
                    row = dict(lag_hours=int(lag), candidate=candidate, comparator=comparator,
                               block_weeks=width, blocks=len(keys), draws=draws, games=len(frame))
                    for j, metric in enumerate(("brier", "log_loss")):
                        low, high = np.quantile(sampled[:, j], [.025, .975])
                        row.update({f"{metric}_delta": float(delta[:, j].mean()), f"{metric}_low": float(low), f"{metric}_high": float(high)})
                    output.append(row)
    return pd.DataFrame(output)


def run(directory=RESEARCH, output=None):
    directory = Path(directory).resolve()
    output = Path(output) if output else directory / "outputs"
    # Refuse any output path in production storage, including through symlinks.
    artifacts = (ROOT / "artifacts").resolve()
    if output.resolve() == artifacts or artifacts in output.resolve().parents:
        raise ValueError("Research output cannot be inside production artifacts")
    before = hashes(ROOT / "artifacts")
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol["configs"] != CONFIGS or protocol["venue_enabled"] or protocol["evaluation_seasons"] != [2021, 2022, 2023, 2024]:
        raise ValueError("Protocol/code mismatch; predeclare a new experiment instead")
    expected = {"warmup_seasons": [2015, 2016], "first_training_season": 2017,
                "forecast_hours_before_kickoff": 24, "assumed_availability_lag_hours": [24, 48, 72],
                "primary_lag_hours": 48, "assumed_game_duration_hours": 4,
                "model": dict(scaler="StandardScaler", penalty="L2", C=1, threshold=.5, random_state=42, max_iter=1000),
                "elo": dict(initial=1500, K=20, home_advantage=50, offseason_regression=.25, margin_multiplier=False, frozen_within_target_season=True),
                "rest": dict(calendar_timezone="America/New_York", clip_days=14, default_days=7, same_season_only=True)}
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Protocol parameters differ from implemented fixed design")
    output.mkdir(parents=True, exist_ok=True)
    metrics, predictions, folds, exclusions, fallback = [], [], [], [], []
    for lag in (24, 48, 72):
        games, source = read(directory, lag)
        coverage = audit(games)
        if not coverage.complete.all():
            raise ValueError("Incomplete audited history: stop affected comparisons")
        coverage.to_csv(output / "coverage.csv", index=False, lineterminator="\n")
        rows = build_features(games, venue_enabled=False)
        rows.to_csv(output / f"features_lag{lag}.csv", index=False, lineterminator="\n")
        for season in (2021, 2022, 2023, 2024):
            forecast, scores, provenance = evaluate_fold(rows, season)
            forecast["lag_hours"] = lag
            predictions.append(forecast)
            metrics.extend(dict(lag_hours=lag, **m) for m in scores)
            folds.append(dict(lag_hours=lag, **provenance))
            cohort = games.loc[games.season.eq(season)]
            selected = rows.loc[rows.season.eq(season)]
            fallback.append(dict(lag_hours=lag, season=season, games=len(selected),
                                 rest_unavailable_home=int(selected.home_rest_unavailable.sum()),
                                 rest_unavailable_away=int(selected.away_rest_unavailable.sum()),
                                 elo_initialized_sides=int(selected.home_elo_initialized.sum() + selected.away_elo_initialized.sum()),
                                 elo_incomplete_sides=int(selected.home_elo_history_incomplete.sum() + selected.away_elo_history_incomplete.sum()),
                                 venue_unknown=int(selected.venue_unavailable.sum()), neutral=int(selected.neutral.sum()), venue_enabled=False))
            used = set(forecast.game_id)
            for g in cohort.loc[~cohort.game_id.isin(used)].itertuples():
                exclusions.append(dict(lag_hours=lag, season=season, game_id=g.game_id,
                                       reason="missing kickoff" if pd.isna(g.kickoff) else "unavailable label" if g.status != "completed" else "tie"))
    pd.DataFrame(fallback).to_csv(output / "fallbacks.csv", index=False, lineterminator="\n")
    pd.DataFrame(exclusions, columns=["lag_hours", "season", "game_id", "reason"]).to_csv(output / "exclusions.csv", index=False, lineterminator="\n")
    metrics = pd.DataFrame(metrics)
    base = metrics.loc[metrics.model.eq("v1_a")].set_index(["lag_hours", "season"])
    for metric in ("accuracy", "brier", "log_loss"):
        metrics[f"{metric}_delta_v1"] = [r[metric] - base.loc[(r.lag_hours, r.season), metric] for _, r in metrics.iterrows()]
    metrics.to_csv(output / "season_metrics.csv", index=False, lineterminator="\n")
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False, lineterminator="\n")
    paired_deltas(predictions).to_csv(output / "paired_deltas.csv", index=False, lineterminator="\n")
    pooled = []
    for lag, frame in predictions.groupby("lag_hours"):
        for name in [*CONFIGS, "coin", "training_home_rate", "fixed_elo"]:
            pooled.append(dict(lag_hours=int(lag), model=name, **score(frame.target, frame[name])))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(output / "pooled_metrics.csv", index=False, lineterminator="\n")
    (output / "folds.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    assert hashes(ROOT / "artifacts") == before, "Production artifacts changed during research"
    files = [*sorted((ROOT / "src/research").glob("*.py")), Path(__file__), directory / "protocol.json", directory / "source.json"]
    manifest = dict(version="v2.0.2", transform=TRANSFORM, source=source, protocol=protocol,
                    python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                    evidence="latest-data reconstruction; historical publication and venue provenance unverified",
                    log_loss_epsilon=float(np.finfo(np.float64).eps),
                    production_artifact_sha256=before,
                    code_sha256={manifest_identifier(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    output_sha256={k: v for k, v in hashes(output).items() if k != "manifest.json"},
                    promotion_allowed=False, prospective_confirmation="not performed; requires frozen future season protocol")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pooled.to_string(index=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=RESEARCH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.directory, args.output)
