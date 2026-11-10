import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from src import data, modeling, predictions, retrospective, season
from src.data import normalize


def archive(game, store):
    row = {"kind": "live", "game_id": game.game_id, "season": int(game.season), "week": int(game.week),
           "home_team": game.home_team, "away_team": game.away_team, "kickoff": game.kickoff.isoformat(),
           "generated_at": "2026-08-02T12:00:00Z", "data_cutoff": "2026-08-02T11:00:00Z",
           "training_cutoff": "2026-02-01T00:00:00Z", "source_freshness": "2026-08-02T10:00:00Z",
           "home_probability": .7, "away_probability": .3, "predicted_winner": game.home_team,
           "revision": 1, "model_version": "archived-original"}
    path = store / "predictions" / f"{game.game_id}-r0001.json"
    predictions.save_record(path, row)
    return path


def test_frozen_fit_ignores_all_2026_outcomes(games):
    first = retrospective.fit_frozen(games)
    changed = games.copy()
    changed.loc[changed.season.eq(2026), ["home_score", "away_score"]] = [100, 0]
    second = retrospective.fit_frozen(changed)
    assert first["metadata"]["training_seasons"] == [2023, 2024, 2025]
    assert first["metadata"]["training_cutoff"] == "2026-02-01T00:00:00+00:00"
    assert first["metadata"]["version"] == second["metadata"]["version"]
    np.testing.assert_array_equal(first["model"][-1].coef_, second["model"][-1].coef_)


def test_prior_season_january_results_are_included(games):
    late = games.loc[games.season.eq(2025)].iloc[-1].copy()
    late["game_id"] = "2025_january"
    late["kickoff"] = pd.Timestamp("2026-01-04T18:00:00Z")
    late["prediction_lock_at"] = late.kickoff
    late["week"] = 18
    modified = normalize(pd.concat([games, pd.DataFrame([late])], ignore_index=True))
    assert retrospective.fit_frozen(modified)["metadata"]["training_games"] == 25


def test_backtest_is_immutable_and_separate(games, store):
    production = store / "model.joblib"
    production.write_bytes(b"untouched production model")
    path = archive(games.loc[games.season.eq(2026)].iloc[0], store)
    before = path.read_bytes()
    assert retrospective.backtest(games, {"stale": True}) == 7
    rows = retrospective.retrospective_records()
    assert all(r["kind"] == "retrospective_backtest" for r in rows)
    assert all(pd.Timestamp(r["generated_at"]) > pd.Timestamp(r["feature_cutoff"]) for r in rows)
    assert all(r["source_stale"] for r in rows)
    assert path.read_bytes() == before
    assert production.read_bytes() == b"untouched production model"
    files = {p: p.read_bytes() for p in (store / "retrospective").rglob("*") if p.is_file()}
    assert retrospective.backtest(games, {}) == 0
    assert all(p.read_bytes() == contents for p, contents in files.items())
    report = season.season_report(games, {})
    assert report["recorded"]["summary"]["games"] == 1
    assert report["retrospective_backtest"]["summary"]["games"] == 7


@pytest.mark.parametrize("perturbation", ["target", "future", "both"])
def test_backtest_pipeline_detects_temporal_leakage(games, store, monkeypatch, perturbation):
    retrospective.backtest(games, {})
    game = games.loc[games.season.eq(2026) & games.week.eq(4)].iloc[0]
    first = next(r for r in retrospective.retrospective_records() if r["game_id"] == game.game_id)
    changed = games.copy()
    target = changed.game_id.eq(game.game_id)
    future = changed.season.eq(2026) & (changed.kickoff > game.kickoff)
    mask = target if perturbation == "target" else future if perturbation == "future" else target | future
    changed.loc[mask, ["home_score", "away_score"]] = [1000, 999]
    other = store / "independent"
    for module in (data, modeling, predictions, retrospective, season):
        monkeypatch.setattr(module, "STORE", other)
    retrospective.backtest(changed, {})
    second = next(r for r in retrospective.retrospective_records() if r["game_id"] == game.game_id)
    assert first["model_version"] == second["model_version"]
    assert first["features"] == second["features"]
    assert first["home_probability"] == second["home_probability"]


def test_frozen_model_with_2026_provenance_rejected(games, store):
    retrospective.backtest(games, {})
    path = store / "retrospective/model.joblib"
    bundle = joblib.load(path)
    bundle["metadata"]["training_seasons"].append(2026)
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="provenance"):
        retrospective.backtest(games, {})


def test_missing_frozen_model_does_not_refit_existing_history(games, store):
    retrospective.backtest(games, {})
    (store / "retrospective/model.joblib").unlink()
    with pytest.raises(ValueError, match="recover"):
        retrospective.backtest(games, {})
    assert len(retrospective.retrospective_records()) == 8


def test_imported_archive_takes_precedence_without_deleting_retrospective(games, store):
    retrospective.backtest(games, {})
    path = archive(games.loc[games.season.eq(2026)].iloc[0], store)
    original = path.read_bytes()
    report = season.season_report(games, {})
    assert report["recorded"]["summary"]["games"] == 1
    assert report["retrospective_backtest"]["summary"]["games"] == 7
    assert report["retrospective_backtest"]["summary"]["unavailable"] == 1
    assert len(retrospective.retrospective_records()) == 8
    assert path.read_bytes() == original


def test_ties_canceled_postponed_future_and_unknown(games, store):
    ids = games.index[games.season.eq(2026)].tolist()
    games.loc[ids[0], ["home_score", "away_score"]] = [10, 10]
    games.loc[ids[1], "status"] = "canceled"
    games.loc[ids[2], "status"] = "postponed"
    games.loc[ids[3], ["home_score", "away_score"]] = float("nan")
    games.loc[ids[3], "kickoff"] = pd.Timestamp("2026-11-10T20:00:00Z")
    games.loc[ids[4], "kickoff"] = pd.NaT
    games.loc[ids[5], "kickoff"] = pd.Timestamp("2026-11-05T18:00:00Z")  # 12h embargo not met
    normalized = normalize(games)
    assert retrospective.backtest(normalized, {}) == 3
    results = retrospective.retrospective_results(normalized)
    summary = season.score_records(results)
    assert summary["ties"] == 1 and summary["games"] == 2
    report = season.season_report(normalized, {})
    assert report["coverage_totals"]["canceled"] == 1
    assert report["coverage_totals"]["postponed"] == 1
    assert report["coverage_totals"]["awaiting_result_embargo"] == 1


def test_coverage_does_not_confuse_missing_records_with_schedule(games, store):
    before = season.season_report(games, {})
    assert len(before["coverage_by_week"]) == 18
    assert before["coverage_totals"]["scheduled_games"] == 8
    assert before["coverage_totals"]["missing_results_after_kickoff"] == 0
    assert before["coverage_totals"]["missing_authentic_predictions"] == 8
    assert before["coverage_totals"]["missing_retrospective_backtests"] == 8
    retrospective.backtest(games, {})
    after = season.season_report(games, {})
    assert after["coverage_totals"]["missing_authentic_predictions"] == 8
    assert after["coverage_totals"]["missing_retrospective_backtests"] == 0
    assert after["coverage_totals"]["retrospective_predictions"] == 8


def test_baseline_and_cumulative_metrics_use_same_non_tied_cohort():
    rows = [{"week": 1, "target": 1, "home_probability": .8, "result": "correct"},
            {"week": 2, "target": 0, "home_probability": .7, "result": "incorrect"},
            {"week": 2, "target": None, "home_probability": .2, "result": "tie"}]
    weekly = season.weekly_performance(rows)
    summary = season.score_records(rows)
    assert summary["accuracy"] == summary["home_baseline_accuracy"] == .5
    assert summary["brier"] == pytest.approx((.04 + .49) / 2)
    assert weekly[0]["cumulative_accuracy"] == 1
    assert weekly[1]["cumulative_accuracy"] == .5
    assert weekly[17]["cumulative_games"] == 2


def test_source_corrections_are_flagged_without_rewriting_backtests(games, store):
    retrospective.backtest(games, {})
    original = {p: p.read_bytes() for p in (store / "retrospective/predictions").glob("*.json")}
    games.loc[games.season.eq(2026) & games.week.eq(1), "home_score"] = 100
    results = retrospective.retrospective_results(games)
    assert any(r.get("integrity_error") and "inputs changed" in r["integrity_error"] for r in results)
    assert all(path.read_bytes() == value for path, value in original.items())


@pytest.mark.parametrize("case", ["duplicate_id", "duplicate_matchup", "unknown_team", "same_team", "bad_week"])
def test_schedule_validation(games, case):
    if case.startswith("duplicate"):
        duplicate = games.iloc[0].copy()
        if case == "duplicate_matchup":
            duplicate["game_id"] = "another_id"
        games = pd.concat([games, pd.DataFrame([duplicate])], ignore_index=True)
    elif case == "unknown_team":
        games.loc[0, "home_team"] = "UNKNOWN"
    elif case == "same_team":
        games.loc[0, "away_team"] = games.loc[0, "home_team"]
    else:
        games.loc[0, "week"] = 19
    with pytest.raises(ValueError):
        normalize(games)


def test_all_18_dashboard_weeks_and_cohorts_are_read_only(games, store):
    from streamlit.testing.v1 import AppTest
    extra = []
    for week in range(9, 19):
        row = games.iloc[-1].copy()
        row["game_id"] = f"2026_future_{week}"
        row["week"] = week
        row["kickoff"] = pd.Timestamp("2026-11-10T18:00:00Z") + pd.Timedelta(days=(week - 9) * 7)
        row["prediction_lock_at"] = row.kickoff
        row[["home_score", "away_score"]] = float("nan")
        row["status"] = "scheduled"
        extra.append(row)
    full = normalize(pd.concat([games, pd.DataFrame(extra)], ignore_index=True))
    full.to_csv(store / "games.csv", index=False)
    data.write_json(store / "source.json", {"fetched_at": "2026-11-05T19:00:00Z"})
    retrospective.backtest(full, {})
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in store.rglob("*") if p.is_file()}
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard/app.py").run(timeout=60)
    for week in range(1, 19):
        app.sidebar.selectbox[0].select(week).run(timeout=60)
        assert not app.exception, (week, app.exception)
        assert any(header.value.startswith(f"Week {week} ·") for header in app.subheader)
    app.sidebar.radio[0].set_value("Prediction history").run(timeout=60)
    app.selectbox[0].select("Retrospective backtests").run(timeout=60)
    assert not app.exception
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in store.rglob("*") if p.is_file()}
    assert before == after
