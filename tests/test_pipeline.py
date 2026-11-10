import json
from pathlib import Path
import pandas as pd
import pytest
from src.data import normalize
from src.pregame import features, training_rows
from src import data, modeling, predictions


def test_features_and_no_future_leakage(games):
    game = games.loc[(games.season == 2025) & (games.week == 3)].iloc[0]
    x = features(games, game, game.kickoff)
    assert x["home_win_pct"] == .5
    assert x["home_streak"] == -1
    assert x["home_avg_points"] == 14
    changed = games.copy()
    changed.loc[changed.kickoff >= game.kickoff, "home_score"] = 999
    assert x == features(changed, game, game.kickoff)
    early = features(games, game, pd.Timestamp("2025-09-04", tz="UTC"))
    assert early["home_win_pct"] == .5  # 12-hour embargo


def test_ties_week_one_missing_scores(games):
    first = games.iloc[0]
    assert features(games, first, first.kickoff)["home_win_pct"] == .5
    games.loc[0, ["home_score", "away_score"]] = 10
    second = games.iloc[1]
    assert features(games, second, second.kickoff)["home_streak"] == 0
    assert first.game_id not in training_rows(games, pd.Timestamp("2027", tz="UTC")).game_id.values
    games.loc[1, "away_score"] = float("nan")
    normalized = normalize(games)
    assert normalized.loc[normalized.game_id.eq(second.game_id), "status"].iloc[0] == "result_unavailable"


def test_delayed_completion_and_unlabeled_game(games):
    game = games.iloc[2].copy()
    game[["home_score", "away_score"]] = float("nan")
    games.loc[1, "available_at"] = game.kickoff + pd.Timedelta(hours=1)
    x = features(games, game, game.kickoff)
    assert x["home_win_pct"] == 1
    assert x["home_avg_points"] == 21
    assert "target" not in x


@pytest.mark.parametrize("gametime", [None, "", " ", "TBD", "25:99"])
def test_missing_kickoff_and_aliases(gametime):
    frame = pd.DataFrame([dict(game_id="x", season=2026, week=10, game_type="REG", gameday="2026-11-15", gametime=gametime,
                               home_team="LAR", away_team="JAC", home_score=None, away_score=None)])
    row = normalize(frame).iloc[0]
    assert pd.isna(row.kickoff)
    assert row.home_team == "LA" and row.away_team == "JAX"


def test_reconciliation_rejects_changed_earlier_kickoff(games, store):
    game = games.iloc[0]
    record = {"game_id": game.game_id, "kind": "live", "kickoff": "2023-09-05T17:00:00Z",
              "generated_at": "2023-09-04T17:00:00Z", "home_probability": .7, "predicted_winner": "A"}
    data.write_json(store / "predictions" / "test.json", record)
    summary = predictions.reconcile(games)
    assert summary["games"] == 0
    result = json.loads((store / "live-results.json").read_text())
    assert result[0]["result"] == "unavailable"


def test_train_predict_freeze_and_reconcile(games, store, monkeypatch):
    cutoff = pd.Timestamp("2026-10-01", tz="UTC")
    meta = modeling.train(games, cutoff, {"fetched_at": cutoff.isoformat()})
    assert set(meta["holdout_2025"]) >= {"A", "B", "C", "D", "home_baseline"}
    future = games.iloc[-1].copy()
    future["game_id"] = "2026_10_B_A"
    future["week"] = 10
    future["kickoff"] = pd.Timestamp("2026-11-10", tz="UTC")
    future["prediction_lock_at"] = future.kickoff
    future["available_at"] = future.kickoff + pd.Timedelta(hours=12)
    future["status"] = "scheduled"
    future[["home_score", "away_score"]] = float("nan")
    games = pd.concat([games, pd.DataFrame([future])], ignore_index=True)
    source = {"fetched_at": "2026-11-04T23:00:00Z", "stale": False}
    monkeypatch.setattr(predictions, "now", lambda: pd.Timestamp("2026-11-05", tz="UTC"))
    assert predictions.predict(games, source, week=10) == 1
    assert predictions.predict(games, source, week=10) == 0
    assert predictions.predict(games, source, week=10, rerun=True) == 1
    original = predictions.records()[0]
    assert original["home_probability"] + original["away_probability"] == pytest.approx(1)
    with pytest.raises(ValueError):
        predictions.predict(games, {"offline": True})
    monkeypatch.setattr(predictions, "now", lambda: pd.Timestamp("2026-11-11", tz="UTC"))
    source["fetched_at"] = "2026-11-10T23:00:00Z"
    assert predictions.predict(games, source, week=10, rerun=True) == 0
    mask = games.game_id.eq(future.game_id)
    games.loc[mask, ["home_score", "away_score", "status"]] = [30, 10, "completed"]
    summary = predictions.reconcile(games)
    assert summary["games"] == 1
    assert len(predictions.records()) == 2
    assert predictions.records()[0] == original
    assert predictions.reconcile(games) == summary


def test_refresh_failure_preserves_snapshot(games, store, monkeypatch):
    games.to_csv(store / "games.csv", index=False)
    data.write_json(store / "source.json", {"fetched_at": "2026-01-01T00:00:00Z"})
    previous = (store / "games.csv").read_bytes()
    def fail(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(data, "urlopen", fail)
    loaded, meta = data.load(refresh=True)
    assert len(loaded) == len(games)
    assert meta["stale"] and meta["error"] == "offline"
    assert previous == (store / "games.csv").read_bytes()


def test_dashboard(games, store, monkeypatch):
    from streamlit.testing.v1 import AppTest
    games.to_csv(store / "games.csv", index=False)
    data.write_json(store / "source.json", {"fetched_at": "2026-01-01T00:00:00Z"})
    modeling.train(games, pd.Timestamp("2026-10-01", tz="UTC"), {})
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard/app.py").run(timeout=30)
    assert not app.exception
    app.sidebar.selectbox[0].select(3).run()
    assert not app.exception
    assert any("Week 3" in header.value for header in app.subheader)


def live_record(game, **changes):
    record = dict(game_id=game.game_id, season=int(game.season), week=int(game.week),
                  kind="live", revision=1, home_team=game.home_team, away_team=game.away_team,
                  kickoff=game.kickoff.isoformat(), generated_at="2026-08-02T12:00:00Z",
                  data_cutoff="2026-08-02T11:00:00Z", training_cutoff="2026-08-01T00:00:00Z",
                  source_freshness="2026-08-02T10:00:00Z", home_probability=.7, away_probability=.3,
                  predicted_winner=game.home_team, model_version="test-v1", source_stale=False)
    return {**record, **changes}


@pytest.mark.parametrize("changes", [
    {"kickoff": pd.NaT}, {"status": "postponed"}, {"home_team": "CHANGED"},
])
def test_uncertain_or_changed_game_excluded(games, store, changes):
    game = games.loc[games.season.eq(2026)].iloc[0].copy()
    predictions.save_record(store / "predictions" / "pick.json", live_record(game))
    for key, value in changes.items():
        game[key] = value
    results, summary = predictions.reconciliation(pd.DataFrame([game]))
    assert results[0]["result"] == "unavailable"
    assert results[0]["integrity_error"]
    assert summary["games"] == 0


@pytest.mark.parametrize("changes", [
    {"generated_at": "not-a-date"}, {"generated_at": "2026-08-02"},
    {"data_cutoff": "2026-08-03T00:00:00Z"}, {"training_cutoff": "2026-08-03T00:00:00Z"},
    {"source_freshness": "2026-08-03T00:00:00Z"},
    {"home_probability": float("nan")}, {"away_probability": .9},
    {"home_probability": "0.7", "away_probability": "0.3"},
    {"predicted_winner": "OTHER"}, {"kind": "backtest"},
])
def test_invalid_record_metadata_rejected(games, changes):
    game = games.loc[games.season.eq(2026)].iloc[0]
    assert predictions.integrity_error(live_record(game, **changes), game)


def test_ties_missing_results_and_repeatability(games, store):
    game = games.loc[games.season.eq(2026)].iloc[0].copy()
    record = live_record(game)
    path = store / "predictions" / "pick.json"
    predictions.save_record(path, record)
    before = path.read_bytes()
    game[["home_score", "away_score"]] = 10
    results, summary = predictions.reconciliation(pd.DataFrame([game]))
    assert results[0]["result"] == "tie" and summary["games"] == 0
    game["home_score"] = float("nan")
    results, summary = predictions.reconciliation(pd.DataFrame([game]))
    assert results[0]["result"] == "pending" and summary["games"] == 0
    assert predictions.reconcile(pd.DataFrame([game])) == predictions.reconcile(pd.DataFrame([game]))
    assert before == path.read_bytes()


def test_atomic_create_only_and_writer_lock(store):
    path = store / "predictions" / "original.json"
    predictions.save_record(path, {"game_id": "x"})
    with pytest.raises(FileExistsError):
        predictions.save_record(path, {"game_id": "changed"})
    assert json.loads(path.read_text()) == {"game_id": "x"}
    with predictions.prediction_writer():
        with pytest.raises(FileExistsError):
            with predictions.prediction_writer():
                pytest.fail("Concurrent writer entered")
    assert not (store / ".prediction-writer.lock").exists()


@pytest.fixture
def future_prediction(games, store, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    game = games.iloc[-1].copy()
    game["game_id"] = "2026_future"
    game["kickoff"] = pd.Timestamp("2026-11-05T21:00:00Z")
    game["prediction_lock_at"] = game.kickoff
    game["status"] = "scheduled"
    game[["home_score", "away_score"]] = float("nan")
    bundle = {"model": SimpleNamespace(predict_proba=lambda x: np.array([[.3, .7]])),
              "metadata": {"training_cutoff": "2026-10-01T00:00:00Z", "features": ["win_pct_diff"], "version": "test"}}
    monkeypatch.setattr(predictions.joblib, "load", lambda path: bundle)
    return pd.DataFrame([game]).reset_index(drop=True), {"fetched_at": "2026-11-05T19:00:00Z"}, bundle


def test_computation_crossing_kickoff_never_persists(future_prediction, store, monkeypatch):
    games, source, bundle = future_prediction
    def slow_probability(x):
        import numpy as np
        monkeypatch.setattr(predictions, "now", lambda: pd.Timestamp("2026-11-05T21:00:01Z"))
        return np.array([[.3, .7]])
    bundle["model"].predict_proba = slow_probability
    assert predictions.predict(games, source) == 0
    assert predictions.records() == []


def test_successful_pick_has_distinct_generation_and_data_cutoff(future_prediction):
    games, source, bundle = future_prediction
    assert predictions.predict(games, source) == 1
    record = predictions.records()[0]
    assert predictions.integrity_error(record, games.iloc[0]) is None
    assert record["data_cutoff"] <= record["generated_at"] < record["kickoff"]


def test_reschedule_cannot_reopen_expired_window(future_prediction):
    games, source, _ = future_prediction
    games.loc[0, "prediction_lock_at"] = pd.Timestamp("2026-11-05T19:00:00Z")
    assert predictions.predict(games, source) == 0
    assert predictions.records() == []


def test_disk_flush_crossing_kickoff_never_publishes(future_prediction, store, monkeypatch):
    games, source, _ = future_prediction
    def delayed_flush(fd):
        monkeypatch.setattr(predictions, "now", lambda: pd.Timestamp("2026-11-05T21:00:01Z"))
    monkeypatch.setattr(predictions.os, "fsync", delayed_flush)
    assert predictions.predict(games, source) == 0
    assert predictions.records() == []


@pytest.mark.parametrize("source", [
    {"fetched_at": "2026-11-04T19:00:00Z"}, {"fetched_at": "2026-11-06T00:00:00Z"},
    {"fetched_at": "2026-11-05T19:00:00Z", "error": "failed"}, {},
])
def test_stale_or_future_source_cannot_create_live_pick(future_prediction, source):
    games, _, _ = future_prediction
    with pytest.raises(ValueError):
        predictions.predict(games, source)
    assert predictions.records() == []


def test_first_revision_is_never_replaced_by_a_better_rerun(games, store):
    game = games.loc[games.season.eq(2026)].iloc[0]
    predictions.save_record(store / "predictions" / "first.json", live_record(game, home_probability=.1, away_probability=.9, predicted_winner=game.away_team))
    predictions.save_record(store / "predictions" / "second.json", live_record(game, revision=2, generated_at="2026-08-03T00:00:00Z"))
    results, summary = predictions.reconciliation(pd.DataFrame([game]))
    assert results[0]["revision"] == 1 and summary["incorrect"] == 1


@pytest.mark.parametrize("score", [-1, 1.5, float("inf"), "not-a-score"])
def test_bad_scores_rejected(games, score):
    games["home_score"] = games.home_score.astype(object)
    games.loc[0, "home_score"] = score
    with pytest.raises((ValueError, TypeError)):
        normalize(games)


def test_naive_kickoff_cannot_be_guessed(games):
    games["kickoff"] = games.kickoff.astype(object)
    games.loc[0, "kickoff"] = "2026-11-05 20:00:00"
    assert pd.isna(normalize(games).set_index("game_id").loc[games.iloc[0].game_id, "kickoff"])


def test_refresh_retains_earliest_kickoff(games, store, monkeypatch):
    from io import BytesIO
    original_kickoff = games.iloc[-1].kickoff
    games.to_csv(store / "games.csv", index=False)
    updated = games.drop(columns="prediction_lock_at").copy()
    updated.loc[updated.index[-1], "kickoff"] = original_kickoff + pd.Timedelta(days=3)
    payload = updated.to_csv(index=False).encode()
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    loaded, meta = data.load(refresh=True)
    row = loaded.loc[loaded.game_id.eq(games.iloc[-1].game_id)].iloc[0]
    assert row.prediction_lock_at == original_kickoff
    assert row.kickoff == original_kickoff + pd.Timedelta(days=3)
    assert not meta["error"]


def test_truncated_source_preserves_cache(games, store, monkeypatch):
    from io import BytesIO
    games.to_csv(store / "games.csv", index=False)
    before = (store / "games.csv").read_bytes()
    payload = games.iloc[:1].to_csv(index=False).encode()
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    loaded, meta = data.load(refresh=True)
    assert len(loaded) == len(games) and meta["stale"] and "dropped" in meta["error"]
    assert before == (store / "games.csv").read_bytes()


def test_append_only_git_guard(tmp_path):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts/check_prediction_history.py"
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init")
    folder = tmp_path / "artifacts/predictions"
    folder.mkdir(parents=True)
    path = folder / "original.json"
    path.write_text('{"pick": 1}')
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "original")
    (folder / "new.json").write_text('{"pick": 2}')
    git("add", ".")
    assert subprocess.run([sys.executable, str(script), "HEAD"], cwd=tmp_path).returncode == 0
    path.write_text('{"pick": 0}')
    assert subprocess.run([sys.executable, str(script), "HEAD"], cwd=tmp_path, capture_output=True).returncode != 0
    path.unlink()
    assert subprocess.run([sys.executable, str(script), "HEAD"], cwd=tmp_path, capture_output=True).returncode != 0


def test_training_rejects_future_cutoff(games, store):
    with pytest.raises(ValueError, match="future"):
        modeling.train(games, pd.Timestamp("2027-01-01", tz="UTC"), {})


@pytest.mark.parametrize("state", ["pending", "completed", "tie", "postponed", "unknown", "missing", "bad_record", "corrupt_file"])
def test_dashboard_all_states_are_read_only(games, store, state):
    from streamlit.testing.v1 import AppTest
    game = games.loc[games.season.eq(2026)].iloc[0].copy()
    if state != "missing":
        record = live_record(game)
        if state == "bad_record":
            record["kickoff"] = "not-a-date"
        predictions.save_record(store / "predictions" / "pick.json", record)
        if state == "corrupt_file":
            (store / "predictions" / "pick.json").write_text("{")
    if state == "pending":
        game[["home_score", "away_score"]] = float("nan")
    elif state == "tie":
        game[["home_score", "away_score"]] = 10
    elif state == "postponed":
        game["status"] = "postponed"
    elif state == "unknown":
        game["kickoff"] = pd.NaT
    pd.DataFrame([game]).to_csv(store / "games.csv", index=False)
    data.write_json(store / "source.json", {"fetched_at": "2026-01-01T00:00:00Z", "error": "source offline"})
    before = {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard/app.py").run(timeout=30)
    app.sidebar.selectbox[0].select(1).run()
    assert not app.exception and len(app.sidebar.radio[0].options) == 4
    assert any("STALE" in warning.value for warning in app.warning)
    if state in ("unknown", "postponed", "missing"):
        assert any("Unavailable" in message.value for message in app.info)
    after = {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    assert before == after
