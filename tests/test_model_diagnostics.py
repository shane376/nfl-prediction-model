"""Research leakage checks; synthetic fixture, no artifact writes."""
import numpy as np
import pandas as pd
import pytest

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
    # Exercise Unix CSV defaults even when the test is run on Windows.
    monkeypatch.setattr("os.linesep", "\n")
    # The review branch keeps receiving results. Reproduce the published report
    # at its own cutoff, rather than moving it with source.json's fetched_at.
    import json
    manifest = json.loads((diagnostics.ROOT / "docs/research_v1_3/manifest.json").read_text())
    # Regress the scheduled-run failure even on main's original snapshot:
    # advancing retrieval freshness must not advance a historical replay.
    original_read = diagnostics.Path.read_text
    def refreshed_source(path, *args, **kwargs):
        text = original_read(path, *args, **kwargs)
        if path == diagnostics.ROOT / "artifacts/source.json":
            source = json.loads(text)
            source["fetched_at"] = "2026-11-14T20:00:00Z"
            return json.dumps(source)
        return text
    monkeypatch.setattr(diagnostics.Path, "read_text", refreshed_source)
    diagnostics.run(as_of=manifest["as_of"])
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


@pytest.fixture
def windows_frozen_case(games, monkeypatch):
    """Create a synthetic artifact with the real historical Windows writer."""
    from src.retrospective import fit_frozen
    with monkeypatch.context() as context:
        context.setattr("os.linesep", "\r\n")
        bundle = fit_frozen(games)
    return games, bundle


@pytest.mark.parametrize("native_newline", ["\n", "\r\n"])
def test_frozen_replay_matches_windows_fingerprint_on_both_platforms(windows_frozen_case, monkeypatch, native_newline):
    from scripts.model_diagnostics import replay_frozen
    from src.retrospective import fit_frozen
    games, bundle = windows_frozen_case
    monkeypatch.setattr("os.linesep", native_newline)
    if native_newline == "\n":
        # This was the CI failure: identical ordered values, different bytes.
        assert fit_frozen(games)["metadata"]["training_data_sha256"] != bundle["metadata"]["training_data_sha256"]
    _, verification = replay_frozen(games, bundle)
    assert verification["status"] == "verified"
    assert verification["training_data_sha256"] == bundle["metadata"]["training_data_sha256"]
    assert verification["training_probability_max_error"] <= 1e-12


@pytest.mark.parametrize("change", ["score", "row_order", "feature"])
def test_frozen_replay_still_rejects_changed_historical_inputs(windows_frozen_case, monkeypatch, change):
    from scripts import model_diagnostics as diagnostics
    games, bundle = windows_frozen_case
    changed = games.copy()
    if change == "score":
        # Preserve winner/target while changing downstream historical features.
        changed.loc[changed.index[0], "home_score"] += 2
    elif change == "row_order":
        changed = changed.iloc[::-1]
    else:
        original = diagnostics.training_rows
        def changed_features(*args, **kwargs):
            rows = original(*args, **kwargs)
            rows.loc[rows.index[0], "point_diff_diff"] += .01
            return rows
        monkeypatch.setattr(diagnostics, "training_rows", changed_features)
    with pytest.raises(ValueError, match="training_data_sha256 mismatch"):
        diagnostics.replay_frozen(changed, bundle)


def test_frozen_replay_rejects_changed_model_with_unchanged_inputs(windows_frozen_case):
    from scripts.model_diagnostics import replay_frozen
    games, bundle = windows_frozen_case
    bundle["model"][-1].coef_[0, 0] += .01
    with pytest.raises(ValueError, match="probabilities differ despite matching training inputs"):
        replay_frozen(games, bundle)
