import json
from pathlib import Path
import pandas as pd
import pytest
from src.data import normalize
from src.pregame import features, training_rows
from src import data, modeling, predictions


@pytest.fixture
def store(tmp_path, monkeypatch):
    for module in (data, modeling, predictions):
        monkeypatch.setattr(module, "STORE", tmp_path)
    return tmp_path


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
    assert normalized.loc[normalized.game_id.eq(second.game_id), "status"].iloc[0] == "underway_or_unavailable"


def test_delayed_completion_and_unlabeled_game(games):
    game = games.iloc[2].copy()
    game[["home_score", "away_score"]] = float("nan")
    games.loc[1, "available_at"] = game.kickoff + pd.Timedelta(hours=1)
    x = features(games, game, game.kickoff)
    assert x["home_win_pct"] == 1
    assert x["home_avg_points"] == 21
    assert "target" not in x


def test_missing_kickoff_and_aliases():
    frame = pd.DataFrame([dict(game_id="x", season=2026, week=10, game_type="REG", gameday="2026-11-15", gametime=None,
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
    future["available_at"] = future.kickoff + pd.Timedelta(hours=12)
    future["status"] = "scheduled"
    future[["home_score", "away_score"]] = float("nan")
    games = pd.concat([games, pd.DataFrame([future])], ignore_index=True)
    source = {"fetched_at": cutoff.isoformat(), "stale": False}
    monkeypatch.setattr(predictions, "now", lambda: pd.Timestamp("2026-11-05", tz="UTC"))
    assert predictions.predict(games, source, week=10) == 1
    assert predictions.predict(games, source, week=10) == 0
    assert predictions.predict(games, source, week=10, rerun=True) == 1
    original = predictions.records()[0]
    assert original["home_probability"] + original["away_probability"] == pytest.approx(1)
    with pytest.raises(ValueError):
        predictions.predict(games, {"offline": True})
    monkeypatch.setattr(predictions, "now", lambda: pd.Timestamp("2026-11-11", tz="UTC"))
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
