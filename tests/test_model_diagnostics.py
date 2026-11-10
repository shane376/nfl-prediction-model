"""Research leakage checks; synthetic fixture, no artifact writes."""
import numpy as np
import pandas as pd

from scripts.model_diagnostics import engineered, paired, score
from src.pregame import features


def test_research_features_match_v1(games):
    research = engineered(games)
    for _, game in games.iterrows():
        expected = features(games, game, game.kickoff)
        actual = research.loc[game.game_id]
        assert actual.win_pct_diff == expected["win_pct_diff"]
        np.testing.assert_allclose(actual.point_diff_diff, expected["point_diff_diff"], atol=1e-12)


def test_target_and_future_outcomes_cannot_change_features(games):
    target = games.loc[games.season.eq(2024) & games.week.eq(4)].iloc[0]
    changed = games.copy()
    changed.loc[changed.kickoff >= target.kickoff, ["home_score", "away_score"]] = [99, 0]
    for window, shrink in ((None, 0), (4, 0), (8, 0), (None, 4)):
        a = engineered(games, window, shrink).loc[target.game_id]
        b = engineered(changed, window, shrink).loc[target.game_id]
        cols = ["win_pct_diff", "point_diff_diff", "elo_diff", "home_count", "away_count"]
        np.testing.assert_array_equal(a[cols], b[cols])


def test_embargo_and_cold_start(games):
    changed = games.copy()
    previous = changed.loc[changed.season.eq(2024) & changed.week.eq(3)].index[0]
    target = changed.loc[changed.season.eq(2024) & changed.week.eq(4)].iloc[0]
    changed.loc[previous, "available_at"] = target.kickoff + pd.Timedelta(hours=1)
    a = engineered(changed)
    assert a.loc[target.game_id].home_count == 2
    start = a.loc[a.week.eq(1)]
    assert start.home_count.eq(0).all()
    assert start.win_pct_diff.eq(0).all()
    assert start.point_diff_diff.eq(0).all()


def test_identical_paired_predictions_have_zero_difference():
    result = paired([0, 1, 0, 1], [.4, .7, .3, .8], [.4, .7, .3, .8], [1, 1, 2, 2])
    assert result == dict(log_loss_delta=0, low=0, high=0)
    assert score([], [])["games"] == 0


def test_full_research_replay_preserves_artifacts_and_exports_matched_scores(tmp_path, monkeypatch):
    from scripts import model_diagnostics as diagnostics
    monkeypatch.setattr(diagnostics, "OUT", tmp_path)
    diagnostics.run()  # Includes exact historical replay and artifact SHA checks.
    predictions = pd.read_csv(tmp_path / "selected_holdout_predictions.csv")
    table = pd.read_csv(tmp_path / "selected_holdout.csv")
    assert predictions.game_id.is_unique
    assert table.games.eq(len(predictions)).all()
    actual = score(predictions.target, predictions.selected_probability)
    np.testing.assert_allclose(table.iloc[1].log_loss, actual["log_loss"], atol=1e-12)
    # Verify the report's evidence tables still match a fresh, isolated replay.
    # Environment metadata can differ; do not rewrite the checked-in manifest.
    for filename in (
        "population_metrics.csv", "variant_reproduction.csv", "reliability.csv",
        "candidate_validation.csv", "selected_holdout.csv", "subgroups.csv",
        "validation_window_sensitivity.csv",
    ):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / filename),
            pd.read_csv(diagnostics.ROOT / "docs/research_v1_3" / filename),
            check_exact=False, rtol=1e-12, atol=1e-12,
        )
