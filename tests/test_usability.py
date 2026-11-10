"""User journeys and source-only team records, independent of model internals."""
from pathlib import Path
import hashlib
import pandas as pd
import pytest
from src.data import normalize
from src.teams import NAMES
from src.presentation import (matchup_rows, filter_matchups, team_schedule,
                              team_record, remaining_wins_estimate)

CURRENT = pd.Timestamp("2026-11-05T20:00:00Z")


@pytest.fixture
def matchups():
    games = pd.DataFrame([
        dict(game_id="recorded", week=1, home_team="BUF", away_team="MIA", status="completed",
             kickoff=pd.Timestamp("2026-09-01T18:00:00Z"), home_score=21., away_score=7.),
        dict(game_id="retro", week=2, home_team="MIA", away_team="BUF", status="completed",
             kickoff=pd.Timestamp("2026-09-08T18:00:00Z"), home_score=14., away_score=7.),
        dict(game_id="provisional", week=12, home_team="BUF", away_team="KC", status="scheduled",
             kickoff=pd.Timestamp("2026-12-01T18:00:00Z")),
        dict(game_id="missing", week=13, home_team="KC", away_team="BUF", status="kickoff_unavailable",
             kickoff=pd.NaT, gameday="2026-12-08"),
    ])
    def pick(kind, p, winner):
        return {"kind": kind, "home_probability": p, "predicted_winner": winner,
                "generated_at": CURRENT.isoformat()}
    return matchup_rows(games, {"recorded": pick("live", .7, "BUF")},
                        {"retro": pick("retrospective_backtest", .55, "MIA")},
                        {"provisional": pick("provisional", .8, "BUF")}, CURRENT)


@pytest.mark.parametrize("filters, expected", [
    ({"week": 12}, ["provisional"]),
    ({"team": "MIA"}, ["recorded", "retro"]),
    ({"query": "buffalo"}, ["recorded", "retro", "provisional", "missing"]),
    ({"query": " mia @ BUF "}, ["recorded", "retro"]),
    ({"query": "Kansas City at Buffalo"}, ["provisional", "missing"]),
    ({"statuses": ["kickoff_unavailable"]}, ["missing"]),
    ({"forecast_types": ["unavailable"]}, ["missing"]),
    ({"forecast_types": ["live", "provisional"], "winner": "BUF", "confidence": (.75, 1)}, ["provisional"]),
    ({"confidence": (.7, .8)}, ["recorded", "provisional"]),
    ({"winner": "MIA", "team": "KC"}, []),
    ({"query": "no such team"}, []),
])
def test_filters(matchups, filters, expected):
    assert [r["game_id"] for r in filter_matchups(matchups, **filters)] == expected


def test_sorting_confidence_tbd_and_no_input_mutation(matchups):
    assert [r["game_id"] for r in filter_matchups(matchups, sort="Confidence: high to low")] == ["provisional", "recorded", "retro", "missing"]
    assert [r["game_id"] for r in filter_matchups(matchups, sort="Confidence: low to high")] == ["retro", "recorded", "provisional", "missing"]
    assert filter_matchups(list(reversed(matchups)))[-1]["game_id"] == "missing"
    assert filter_matchups(matchups, sort="Matchup A-Z")[0]["away_team"] == "BUF"
    assert [r["game_id"] for r in matchups] == ["recorded", "retro", "provisional", "missing"]


def test_provenance_missing_forecasts_and_invalid_archive_precedence(matchups):
    assert [r["provenance"] for r in matchups] == ["Recorded pregame pick", "Retrospective backtest", "Provisional forecast", "Prediction unavailable"]
    assert matchups[-1]["confidence"] is None and matchups[-1]["winner"] is None
    game = pd.DataFrame([matchups[0]])
    invalid = {"kind": "live", "integrity_error": "late", "home_probability": .9}
    rows = matchup_rows(game, {"recorded": invalid}, {"recorded": {"kind": "retrospective_backtest", "home_probability": .8}}, {}, CURRENT)
    assert rows[0]["forecast_type"] == "unavailable"
    assert rows[0]["forecast"]["kind"] == "live"
    assert rows[0]["reason"] == "late"


@pytest.mark.parametrize("change", ["stale", "past_kickoff", "past_tbd_date", "past_lock", "completed"])
def test_cached_estimates_rechecked_in_both_views(matchups, change):
    game = dict(matchups[2])
    if change == "past_kickoff": game["kickoff"] = CURRENT
    if change == "past_tbd_date":
        game.update(kickoff=pd.NaT, gameday="2026-11-05", status="kickoff_unavailable")
    if change == "past_lock": game["prediction_lock_at"] = CURRENT
    if change == "completed": game["status"] = "completed"
    rows = matchup_rows(pd.DataFrame([game]), {}, {}, {"provisional": game["forecast"]}, CURRENT, stale=change == "stale")
    assert rows[0]["forecast_type"] == "unavailable"
    assert team_schedule(rows, "BUF")[0]["confidence"] is None


def test_team_record_uses_actual_scores_only(matchups):
    tie = {**matchups[0], "game_id": "tie", "week": 3, "home_score": 10., "away_score": 10.}
    postponed = {**tie, "game_id": "postponed", "status": "postponed"}
    incomplete = {**tie, "game_id": "incomplete", "away_score": float("nan")}
    schedule = team_schedule([*matchups, tie, postponed, incomplete], "BUF")
    assert team_record(schedule) == {"wins": 1, "losses": 1, "ties": 1}
    assert [(r["opponent"], r["actual_result"], r["score"]) for r in schedule[:2]] == [("MIA", "W", "21-7"), ("MIA", "L", "7-14")]
    # Changing or removing predictions has no effect on actual standings.
    for row in schedule:
        row.update(forecast=None, forecast_type="unavailable")
    assert team_record(schedule) == {"wins": 1, "losses": 1, "ties": 1}


def test_all_32_teams_have_17_source_games():
    path = Path(__file__).resolve().parents[1] / "artifacts/games.csv"
    games = normalize(pd.read_csv(path))
    games = games.loc[games.season.eq(2026)]
    rows = matchup_rows(games, {}, {}, {}, CURRENT)
    total_wins = total_losses = total_ties = 0
    for team in NAMES:
        schedule = team_schedule(rows, team)
        assert len(schedule) == 17
        assert len({r["game_id"] for r in schedule}) == 17
        assert all(r["opponent"] != team and r["opponent"] in NAMES for r in schedule)
        assert all(r["forecast_type"] == "unavailable" for r in schedule)
        actual = team_record(schedule)
        total_wins += actual["wins"]
        total_losses += actual["losses"]
        total_ties += actual["ties"]
    assert total_wins == total_losses
    assert total_wins + total_losses + total_ties == 2 * int(games.status.eq("completed").sum())


def test_estimates_never_fill_missing_probabilities_or_use_backtests(matchups):
    schedule = team_schedule(matchups, "BUF")
    estimate = remaining_wins_estimate(schedule, "BUF")
    assert estimate == {"remaining": 2, "covered": 1, "estimated_wins": None}
    covered = remaining_wins_estimate([schedule[2]], "BUF")
    assert covered["estimated_wins"] == .8
    assert remaining_wins_estimate(team_schedule([matchups[2]], "KC"), "KC")["estimated_wins"] == pytest.approx(.2)
    schedule[2]["forecast_type"] = "retrospective_backtest"
    assert remaining_wins_estimate(schedule, "BUF")["covered"] == 0
    assert remaining_wins_estimate([], "BUF")["estimated_wins"] is None


def test_weekly_and_explorer_user_journey():
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "artifacts").rglob("*") if p.is_file()}
    before = hashes()
    app = AppTest.from_file(str(root / "dashboard/app.py"), default_timeout=30).run()
    app.selectbox(key="week").select("All weeks").run()
    app.text_input(key="weekly_search").set_value("MIA @ BUF").run()
    assert not app.exception
    assert any("Showing 2 of 272" in c.value for c in app.caption)
    app.multiselect(key="weekly_type").set_value(["provisional"]).run()
    assert any("No games match" in e.value for e in app.info)
    app.button(key="weekly_clear").click().run()
    assert app.text_input(key="weekly_search").value == ""
    assert app.multiselect(key="weekly_type").value == []
    app.radio(key="view").set_value("Team Explorer").run()
    assert not app.exception
    assert len(app.selectbox(key="explorer_team").options) == 32
    app.selectbox(key="explorer_team").select("BUF").run()
    assert len(app.dataframe[0].value) == 17
    assert "Actual result" in app.dataframe[0].value
    app.checkbox(key="explorer_estimate").check().run()
    assert not app.exception
    assert any("Estimate unavailable" in e.value for e in app.info)
    assert before == hashes()


def test_unreadable_archive_withholds_estimates(games, store, monkeypatch):
    """A broken archive must not make an existing recorded pick look provisional."""
    from unittest.mock import Mock
    from streamlit.testing.v1 import AppTest
    from src import data
    import joblib
    future = games.iloc[-1].copy()
    future["kickoff"] = pd.Timestamp("2026-12-01T18:00:00Z")
    future["prediction_lock_at"] = future.kickoff
    future[["home_score", "away_score"]] = float("nan")
    pd.DataFrame([future]).to_csv(store / "games.csv", index=False)
    data.write_json(store / "source.json", {"fetched_at": CURRENT.isoformat()})
    (store / "predictions").mkdir()
    (store / "predictions/broken.json").write_text("{")
    (store / "model.joblib").write_bytes(b"test-only placeholder")
    model = Mock()
    import numpy as np
    model.predict_proba.return_value = np.array([[.2, .8]])
    monkeypatch.setattr(joblib, "load", lambda path: {"model": model, "metadata": {
        "training_cutoff": "2026-01-01T00:00:00Z", "version": "test-only",
        "features": ["win_pct_diff"]}})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), default_timeout=30).run()
    assert not app.exception
    assert model.predict_proba.called  # A viable estimate existed before the archive check.
    assert any("Estimates are withheld" in e.value for e in app.info)
    assert not any("Favored:" in m.value for m in app.markdown)
