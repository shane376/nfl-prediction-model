"""Offline research only. Run: python -m scripts.model_diagnostics.

Writes exclusively to docs/research_v1_3; never invokes production writers.
Selection: 2023 fit, early 2024 calibration, late 2024 validation.
Only the selected candidate and fixed V1 reference see the 2025 holdout.
"""
import hashlib
import json
import platform
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.data import normalize, STORE, utc_time
from src.modeling import metrics
from src.models.train_v1 import MODEL_FEATURES, build_model
from src.pregame import training_rows
from src.predictions import reconciliation
from src.retrospective import retrospective_results, TRAINING_CUTOFF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/research_v1_3"


def replay_frozen(games, bundle):
    """Verify the ordered historical inputs before refitting in memory.

    The committed frozen artifact hashed pandas CSV on Windows: UTF-8, CRLF,
    no index, and the original row/column order. pandas' default is os.linesep;
    reproducing that *format* explicitly is essential on a Unix CI runner.
    Do not sort, round, re-normalize, or accept a different data fingerprint.
    """
    meta = bundle["metadata"]
    columns = MODEL_FEATURES["A"]
    if (meta["training_seasons"] != [2023, 2024, 2025]
            or pd.Timestamp(meta["training_cutoff"]) != TRAINING_CUTOFF
            or meta["variant"] != "A" or meta["features"] != columns):
        raise ValueError("Frozen model provenance does not match the pre-2026 policy")
    prior = games.loc[games.season.isin(meta["training_seasons"])]
    rows = training_rows(prior, TRAINING_CUTOFF)
    table = rows[["game_id", "target"] + columns]
    digest = hashlib.sha256(table.to_csv(index=False, lineterminator="\r\n").encode("utf-8")).hexdigest()
    if digest != meta["training_data_sha256"]:
        raise ValueError(
            "Frozen training_data_sha256 mismatch using historical UTF-8/CRLF CSV: "
            f"expected {meta['training_data_sha256']}, reproduced {digest}. "
            "Review the source snapshot, row order, normalization and historical features; "
            "current-snapshot reconstruction is not verified frozen-model replay."
        )
    if (len(rows) != meta["training_games"]
            or rows.available_at.max() != pd.Timestamp(meta["last_training_result_available_at"])
            or rows.available_at.max() >= TRAINING_CUTOFF):
        raise ValueError("Frozen training row count or result-availability provenance changed")
    model = build_model().fit(rows[columns], rows.target)
    reproduced = model.predict_proba(rows[columns])[:, 1]
    archived = bundle["model"].predict_proba(rows[columns])[:, 1]
    if not np.allclose(reproduced, archived, atol=1e-12, rtol=0):
        raise ValueError("Frozen model probabilities differ despite matching training inputs")
    return model, dict(
        status="verified", training_games=len(rows), training_data_sha256=digest,
        serialization="pandas CSV; UTF-8; CRLF; index=False; original row and column order",
        training_probability_max_error=float(np.max(np.abs(reproduced-archived))),
    )


def score(y, p):
    result = metrics(np.asarray(y, dtype=int), np.asarray(p))
    result.pop("calibration", None)
    n = len(y)
    if n:
        a, z = result["accuracy"], 1.96
        center = (a + z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(a*(1-a)/n+z*z/(4*n*n))/(1+z*z/n)
        result.update(accuracy_low=center-half, accuracy_high=center+half,
                      home_rate=float(np.mean(y)), mean_probability=float(np.mean(p)))
    return result


def engineered(games, window=None, shrink=0):
    """Latest-snapshot replay; every result must clear kickoff AND embargo.

    Elo resets toward 1500 by 25% each season; K=20, home advantage=50.
    Ratings update only on eligible results, including tied games.
    """
    rows, ratings, updated = [], {}, set()
    last_season = None
    ordered = games.sort_values(["kickoff", "game_id"])
    for _, g in ordered.iterrows():
        if pd.isna(g.kickoff):
            continue
        eligible = ordered.loc[(ordered.available_at < g.kickoff) &
                               (ordered.kickoff < g.kickoff) & ordered.status.eq("completed") &
                               ordered.home_score.notna() & ordered.away_score.notna()]
        pending = eligible.loc[~eligible.game_id.isin(updated)]
        for h in pending.sort_values(["available_at", "game_id"]).itertuples():
            if last_season != h.season:
                ratings = {t: 1500 + .75*(r-1500) for t, r in ratings.items()}
                last_season = h.season
            rh, ra = ratings.get(h.home_team, 1500), ratings.get(h.away_team, 1500)
            expected = 1/(1+10**(-(rh-ra+50)/400))
            outcome = 1 if h.home_score > h.away_score else 0 if h.home_score < h.away_score else .5
            delta = 20*(outcome-expected)
            ratings[h.home_team], ratings[h.away_team] = rh+delta, ra-delta
            updated.add(h.game_id)
        if last_season != g.season:
            ratings = {t: 1500 + .75*(r-1500) for t, r in ratings.items()}
            last_season = g.season
        f = {}
        history = eligible.loc[eligible.season.eq(g.season)]
        for side in ("home", "away"):
            team = g[f"{side}_team"]
            h = history.loc[history.home_team.eq(team) | history.away_team.eq(team)].sort_values("kickoff")
            if window:
                h = h.tail(window)
            margin = np.where(h.home_team.eq(team), h.home_score-h.away_score, h.away_score-h.home_score)
            n = len(h)
            f[f"{side}_count"] = n
            f[f"{side}_win"] = (np.sum(margin > 0)+.5*np.sum(margin == 0)+.5*shrink)/(n+shrink) if n+shrink else .5
            f[f"{side}_margin"] = float(np.sum(margin)/(n+shrink)) if n+shrink else 0.
        f.update(win_pct_diff=f["home_win"]-f["away_win"],
                 point_diff_diff=f["home_margin"]-f["away_margin"],
                 elo_diff=ratings.get(g.home_team, 1500)-ratings.get(g.away_team, 1500))
        rows.append({**g.to_dict(), **f})
    return pd.DataFrame(rows).set_index("game_id")


def paired(y, candidate, reference, weeks):
    """Paired week-block bootstrap; descriptive, not selection inference."""
    y, candidate, reference, weeks = map(np.asarray, (y, candidate, reference, weeks))
    loss = lambda p: -(y*np.log(np.clip(p, 1e-15, 1-1e-15))+(1-y)*np.log(np.clip(1-p, 1e-15, 1)))
    delta = loss(candidate)-loss(reference)
    blocks = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    rng = np.random.default_rng(130)
    draws = [delta[np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])].mean() for _ in range(2000)]
    return dict(log_loss_delta=float(delta.mean()), low=float(np.quantile(draws, .025)), high=float(np.quantile(draws, .975)))


def run(as_of=None):
    """Replay at the source time, or an explicit historical report cutoff."""
    if STORE.resolve() != (ROOT/"artifacts").resolve():
        raise ValueError("Unset NFL_STORE: diagnostics require the committed snapshot")
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT/"artifacts").rglob("*"))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    source = json.loads((ROOT/"artifacts/source.json").read_text())
    asof = utc_time(source["fetched_at"] if as_of is None else as_of)
    games = normalize(pd.read_csv(ROOT/"artifacts/games.csv"))
    rows = training_rows(games, asof).set_index("game_id")
    fit, hold = rows.loc[rows.season.isin([2023, 2024])], rows.loc[rows.season.eq(2025)]
    reference = build_model().fit(fit[MODEL_FEATURES["A"]], fit.target)
    metadata = json.loads((ROOT/"artifacts/model.json").read_text())
    reproduced = []
    for variant, cols in MODEL_FEATURES.items():
        for label, train, test in (("validation_2024", rows.loc[rows.season.eq(2023)], rows.loc[rows.season.eq(2024)]),
                                   ("holdout_2025", fit, hold)):
            m = build_model().fit(train[cols], train.target)
            result = score(test.target, m.predict_proba(test[cols])[:, 1])
            for metric in ("accuracy", "brier", "log_loss"):
                assert abs(result[metric]-metadata[label][variant][metric]) < 1e-12
            reproduced.append(dict(population=label, variant=variant, **result))
    pd.DataFrame(reproduced).to_csv(OUT/"variant_reproduction.csv", index=False)
    hold = hold.copy()
    hold["home_probability"] = reference.predict_proba(hold[MODEL_FEATURES["A"]])[:, 1]
    saved = pd.DataFrame(json.loads((ROOT/"artifacts/backtest.json").read_text())).set_index("game_id")
    assert set(saved.index) == set(hold.index)
    assert np.allclose(saved.loc[hold.index].home_probability, hold.home_probability, atol=1e-12, rtol=0)
    with patch("src.predictions.now", return_value=asof), patch("src.retrospective.now", return_value=asof):
        live, _ = reconciliation(games)
        retro = retrospective_results(games)
    populations = {"holdout_2025": hold.reset_index()}
    inventory = []
    for name, records in (("retrospective_2026", retro), ("recorded_2026", live)):
        df = pd.DataFrame(records)
        inventory.append(dict(population=name, records=len(df), scored=int(df.target.notna().sum()),
                              integrity_errors=int(df.integrity_error.notna().sum()), results=df.result.value_counts().to_dict()))
        populations[name] = df.loc[df.target.notna()].copy()
    checked = joblib.load(ROOT/"artifacts/retrospective/model.joblib")
    assert checked["metadata"] == json.loads((ROOT/"artifacts/retrospective/model.json").read_text())
    frozen, frozen_verification = replay_frozen(games, checked)
    r = populations["retrospective_2026"]
    xr = rows.loc[r.game_id, MODEL_FEATURES["A"]]
    for model in (frozen, checked["model"]):
        assert np.allclose(model.predict_proba(xr)[:, 1], r.home_probability, atol=1e-12, rtol=0)
    summaries, groups, reliability = [], [], []
    pre2026 = rows.loc[rows.season.lt(2026)]
    production_cutoff = pd.Timestamp(json.loads((ROOT/"artifacts/model.json").read_text())["training_cutoff"])
    for name, df in populations.items():
        if df.empty:
            continue
        prior = fit if name == "holdout_2025" else pre2026 if name == "retrospective_2026" else rows.loc[rows.available_at < production_cutoff]
        for label, p in (("V1", df.home_probability), ("coin", np.full(len(df), .5)),
                         ("prior_home_rate", np.full(len(df), prior.target.mean())), ("always_home", np.ones(len(df)))):
            summaries.append(dict(population=name, model=label, **score(df.target, p)))
        if "home_team" not in df:
            df = df.merge(games[["game_id", "home_team", "away_team"]], on="game_id", validate="one_to_one")
        df["confidence"] = np.maximum(df.home_probability, 1-df.home_probability)
        df["confidence_band"] = pd.cut(df.confidence, [.499999, .55, .6, .7, .8, 1], right=False).astype(str)
        df["pick_side"] = np.where(df.home_probability >= .5, "home", "away")
        df["actual_side"] = np.where(df.target.eq(1), "home", "away")
        df["matchup"] = np.where(df.confidence < .55, "close_<55%", np.where(df.confidence >= .7, "high_>=70%", "middle"))
        for dimension in ("week", "confidence_band", "pick_side", "actual_side", "matchup"):
            for key, sub in df.groupby(dimension):
                groups.append(dict(population=name, dimension=dimension, group=str(key), **score(sub.target, sub.home_probability)))
        for team in sorted(set(df.home_team) | set(df.away_team)):
            sub = df.loc[df.home_team.eq(team) | df.away_team.eq(team)]
            groups.append(dict(population=name, dimension="team", group=team, **score(sub.target, sub.home_probability)))
        df["probability_bin"] = pd.cut(df.home_probability, np.linspace(0, 1, 6), include_lowest=True)
        for key, sub in df.groupby("probability_bin", observed=True):
            interval = score(sub.target, sub.home_probability)
            # Wilson limits for observed home frequency, rather than accuracy.
            freq = score(sub.target, np.ones(len(sub)))
            reliability.append(dict(population=name, bin=str(key), n=len(sub), predicted=float(sub.home_probability.mean()),
                                    observed=float(sub.target.mean()), low=freq["accuracy_low"], high=freq["accuracy_high"]))
        df.drop(columns="probability_bin").to_csv(OUT/f"{name}_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT/"population_metrics.csv", index=False)
    pd.DataFrame(groups).to_csv(OUT/"subgroups.csv", index=False)
    pd.DataFrame(reliability).to_csv(OUT/"reliability.csv", index=False)

    # The grids and selection window are fixed here, before any holdout scoring.
    datasets = {"std": rows, "roll4": engineered(games, 4), "roll8": engineered(games, 8),
                "shrink4": engineered(games, shrink=4), "elo": engineered(games)}
    configs = [(f"A_C{c}", "std", c, "none", "2023") for c in (.01, .1, 1, 10, 100)]
    configs += [(k, k, 1, "none", "2023") for k in ("roll4", "roll8", "shrink4", "elo")]
    configs += [(k, "std", 1, k, "2023") for k in ("sigmoid", "isotonic")]
    configs += [("recent_half2023", "std", 1, "none", "recent")]
    def predict(config, final=False, full_validation=False):
        name, dataset, c, calibration, window = config
        data = datasets[dataset].loc[rows.index].copy()
        data["target"] = rows.target
        cols = ["elo_diff"] if dataset == "elo" else MODEL_FEATURES["A"]
        train = data.loc[data.season.eq(2023)]
        if window == "recent":
            train = train.loc[train.week.ge(10)]
        cal = data.loc[data.season.eq(2024) & data.week.le(9)]
        val = data.loc[data.season.eq(2024) & data.week.ge(10)]
        if full_validation:
            assert calibration == "none"
            val = data.loc[data.season.eq(2024)]
        if final:
            train = data.loc[data.season.isin([2023, 2024])]
            if window == "recent":
                train = data.loc[data.season.eq(2024)]
            if calibration != "none":
                train = data.loc[data.season.eq(2023)]
                cal = data.loc[data.season.eq(2024)]
            val = data.loc[data.season.eq(2025)]
        assert train.available_at.max() < val.kickoff.min()
        model = build_model().set_params(logistic_regression__C=c).fit(train[cols], train.target)
        p = model.predict_proba(val[cols])[:, 1]
        if calibration != "none":
            assert train.available_at.max() < cal.kickoff.min() and cal.available_at.max() < val.kickoff.min()
            cp = model.predict_proba(cal[cols])[:, 1]
            if calibration == "sigmoid":
                logit = lambda a: np.log(a/(1-a)).reshape(-1, 1)
                calibrator = LogisticRegression(C=1, random_state=42).fit(logit(cp), cal.target)
                p = calibrator.predict_proba(logit(p))[:, 1]
            else:
                p = IsotonicRegression(out_of_bounds="clip").fit(cp, cal.target).predict(p)
        return val, p
    comparisons = []
    val_ref, p_ref = predict(configs[2])
    for config in configs:
        val, p = predict(config)
        assert val.index.equals(val_ref.index)
        comparisons.append(dict(candidate=config[0], **score(val.target, p), **paired(val.target, p, p_ref, val.week)))
    winner = min(comparisons, key=lambda x: (x["log_loss"], x["candidate"]))["candidate"]
    chosen = next(c for c in configs if c[0] == winner)
    final, final_p = predict(chosen, final=True)
    assert final.index.equals(hold.index)
    final_table = [dict(candidate="V1_A", **score(hold.target, hold.home_probability)),
                   dict(candidate=winner, **score(final.target, final_p), **paired(final.target, final_p, hold.home_probability, final.week))]
    pd.DataFrame(comparisons).to_csv(OUT/"candidate_validation.csv", index=False)
    sensitivity = []
    for config in configs:
        if config[3] != "none":
            continue  # Early 2024 is calibration training, so cannot score it.
        val, p = predict(config, full_validation=True)
        sensitivity.append(dict(candidate=config[0], window="full_2024", **score(val.target, p)))
        early = val.week.le(9).to_numpy()
        sensitivity.append(dict(candidate=config[0], window="early_2024", **score(val.target.iloc[np.flatnonzero(early)], p[early])))
    pd.DataFrame(sensitivity).to_csv(OUT/"validation_window_sensitivity.csv", index=False)
    pd.DataFrame(final_table).to_csv(OUT/"selected_holdout.csv", index=False)
    pd.DataFrame(dict(game_id=final.index, target=final.target, selected_probability=final_p,
                      reference_probability=hold.home_probability)).to_csv(OUT/"selected_holdout_predictions.csv", index=False)
    coefficients, distributions = [], []
    for season in (2023, 2024, 2025, 2026):
        tr = rows.loc[rows.season.le(season)]
        for variant, cols in (MODEL_FEATURES.items() if season < 2026 else []):
            m = build_model().fit(tr[cols], tr.target)
            for col, value, scale in zip(cols, m[-1].coef_[0], m[0].scale_):
                coefficients.append(dict(through_season=season, variant=variant, feature=col, standardized=value, raw=value/scale, intercept=m[-1].intercept_[0]))
        for col in MODEL_FEATURES["B"]:
            series = rows.loc[rows.season.eq(season), col]
            distributions.append(dict(season=season, feature=col, mean=series.mean(), std=series.std(), min=series.min(), max=series.max(), missing=int(series.isna().sum())))
    pd.DataFrame(coefficients).to_csv(OUT/"coefficients.csv", index=False)
    pd.DataFrame(distributions).to_csv(OUT/"feature_distributions.csv", index=False)
    rows[MODEL_FEATURES["B"]].corr().to_csv(OUT/"feature_correlations.csv")
    pd.DataFrame([dict(season=int(s), rows=len(d), ties=int(d.home_score.eq(d.away_score).sum()),
                       missing_scores=int((d.home_score.isna() | d.away_score.isna()).sum()),
                       missing_kickoff=int(d.kickoff.isna().sum())) for s,d in games.groupby("season")]).to_csv(OUT/"data_audit.csv", index=False)
    manifest = dict(as_of=asof.isoformat(), python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__,
                    selected_candidate=winner, inventory=inventory, artifact_sha256=hashes,
                    selection="2023 fit; 2024 weeks 1-9 calibration only; 2024 weeks 10-18 matched validation; lowest log loss; 2025 selected-only holdout",
                    holdout_reproduction_max_error=float(np.max(np.abs(saved.loc[hold.index].home_probability-hold.home_probability))))
    manifest["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["frozen_verification"] = frozen_verification
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p,digest in hashes.items())
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(pd.DataFrame(final_table).to_string(index=False))


if __name__ == "__main__":
    run()
