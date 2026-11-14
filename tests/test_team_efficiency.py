"""Research chronology, deterministic reconstruction, and production isolation."""
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from src.research.schedule_data import RESEARCH, normalize, read
from src.research.team_context import build_features, elo_probability, opening_ratings, rest_for
from src.research.efficiency import aggregate_batches, pregame_efficiency, training_priors
from scripts.v2_team_efficiency import evaluate_fold, hashes, run, split_fold


@pytest.fixture
def history():
    rows = []
    for year in range(2015, 2025):
        for week in range(1, 5):
            date = pd.Timestamp(f"{year}-09-01") + pd.Timedelta(days=7 * week)
            rows.append(dict(game_id=f"{year}_{week:02d}_MIA_BUF", season=year, week=week,
                             game_type="REG", gameday=str(date.date()), gametime="13:00",
                             home_team="BUF", away_team="MIA", home_score=21 if week % 2 else 10,
                             away_score=14, location="Home"))
    return normalize(pd.DataFrame(rows))


def test_target_and_future_outcomes_do_not_change_pregame_features(history):
    original = build_features(history)
    target = history.loc[history.season.eq(2021) & history.week.eq(2)].iloc[0]
    changed = history.copy()
    changed.loc[changed.kickoff >= target.kickoff, ["home_score", "away_score"]] = [77, 3]
    rebuilt = build_features(changed)
    columns = [c for c in original if c != "target"]
    pd.testing.assert_frame_equal(original.loc[original.forecast_cutoff <= target.kickoff, columns],
                                  rebuilt.loc[rebuilt.forecast_cutoff <= target.kickoff, columns])
    # A target-season result must never update its own frozen entering rating.
    elo_cols = ["home_prior_elo", "away_prior_elo", "prior_elo_diff"]
    pd.testing.assert_frame_equal(original.loc[original.season.eq(2021), elo_cols],
                                  rebuilt.loc[rebuilt.season.eq(2021), elo_cols])


def test_elo_warmup_regression_and_row_order_are_deterministic(history):
    one_per_year = history.loc[history.week.eq(1)].copy()
    ratings = opening_ratings(one_per_year)
    first_change = 20 * (1 - elo_probability(0))
    assert ratings[2016][0]["BUF"] == pytest.approx(1500 + .75 * first_change)
    second_change = 20 * (1 - elo_probability(1.5 * first_change))
    assert ratings[2017][0]["BUF"] == pytest.approx(1500 + .75 * (.75 * first_change + second_change))
    assert ratings == opening_ratings(one_per_year.iloc[::-1])
    assert ratings[2017][1]["BUF"] == 2
    assert ratings[2015] == ({}, {})
    a, b = build_features(history), build_features(history.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(a.sort_values("game_id").reset_index(drop=True), b.sort_values("game_id").reset_index(drop=True))


def test_missing_history_is_flagged_not_invented(history):
    history.loc[history.season.eq(2020), ["home_score", "away_score"]] = np.nan
    history.loc[history.season.eq(2020), "status"] = "unavailable"
    features = build_features(history)
    row = features.loc[features.season.eq(2021)].iloc[0]
    assert row.home_elo_history_incomplete == 1
    assert row.home_elo_history_games == 20  # 2015-19 only, never fabricated 2020 wins.
    first = features.iloc[0]
    assert first.home_prior_elo == 1500 and first.home_elo_initialized == 1


def test_rest_calendar_clipping_missing_kickoff_and_postponement(history):
    game = history.loc[history.season.eq(2021) & history.week.eq(3)].iloc[0].copy()
    cutoff = game.kickoff - pd.Timedelta(hours=24)
    assert rest_for(history, game, "home", cutoff) == (7, 0, 0)
    changed = history.copy()
    changed.loc[changed.season.eq(2021) & changed.week.eq(2), "status"] = "postponed"
    game["kickoff"] += pd.Timedelta(days=10)
    assert rest_for(changed, game, "home", game.kickoff - pd.Timedelta(hours=24))[0] == 14
    changed.loc[changed.season.eq(2021), "kickoff"] = pd.NaT
    assert rest_for(changed, game, "home", cutoff) == (7, 0, 1)
    opener = history.loc[history.season.eq(2021) & history.week.eq(1)].iloc[0]
    assert rest_for(history, opener, "home", opener.kickoff - pd.Timedelta(hours=24)) == (7, 1, 1)
    # Across DST, elapsed UTC days differ from New York calendar dates.
    prior = history.loc[history.season.eq(2021) & history.week.eq(1)].copy()
    prior["kickoff"] = pd.Timestamp("2021-11-01T03:30:00Z")  # Oct 31 NY
    prior["available_at"] = prior.kickoff + pd.Timedelta(hours=48)
    opener = opener.copy()
    opener["game_id"] = "target"
    opener["kickoff"] = pd.Timestamp("2021-11-08T04:30:00Z")  # Nov 7 NY
    assert rest_for(prior, opener, "home", opener.kickoff - pd.Timedelta(hours=24))[0] == 7


def test_venue_fallback_and_missing_scores(history):
    raw = pd.DataFrame([dict(game_id="test", season=2021, week=1, game_type="REG", gameday="2021-09-01",
                             gametime="TBD", home_team="STL", away_team="SD", home_score=None, away_score=None)])
    row = normalize(raw).iloc[0]
    assert pd.isna(row.kickoff) and row.status == "unavailable"
    assert (row.home_team, row.away_team) == ("LA", "LAC")
    history.loc[0, "location"] = "Unknown"
    result = build_features(history).iloc[0]
    assert result.neutral == 0 and result.venue_unavailable == 1 and not result.venue_enabled


def test_fold_fit_excludes_evaluation_labels_and_transforms(history):
    rows = build_features(history)
    train, evaluation, cutoff = split_fold(rows, 2021)
    assert train.season.max() == 2020 and (train.label_available_at < cutoff).all()
    a, _, provenance = evaluate_fold(rows, 2021)
    changed = rows.copy()
    changed.loc[changed.season.ge(2021), "target"] = 1 - changed.loc[changed.season.ge(2021), "target"]
    b, _, changed_provenance = evaluate_fold(changed, 2021)
    pd.testing.assert_frame_equal(a.drop(columns="target"), b.drop(columns="target"))
    assert provenance["fitted_parameters"] == changed_provenance["fitted_parameters"]
    assert set(train.game_id).isdisjoint(evaluation.game_id)


def plays(game_id="2021_01_MIA_BUF", n=20, missing=0):
    frame = pd.DataFrame(dict(game_id=[game_id]*n, play_id=range(n), season_type="REG", posteam="BUF", defteam="MIA",
                              play_type="pass", qb_kneel=0, qb_spike=0, epa=.2, success=1, qb_dropback=1, passer_player_id="qb"))
    if missing:
        frame.loc[:missing-1, ["epa", "success"]] = np.nan
    return frame


def test_bounded_efficiency_filters_missingness_and_shrinkage(history):
    p = plays(missing=2)  # 90% valid suppresses this game; missing is not zero EPA.
    extras = pd.concat([plays(n=1).assign(play_type="no_play"), plays(n=1).assign(qb_kneel=1), plays(n=1).assign(qb_spike=1)])
    table, audit = aggregate_batches([p.iloc[:10], p.iloc[10:], extras])
    assert audit["eligible"] == 20 and audit["missing_epa"] == 2
    assert not table.epa_valid.any() and not table.success_valid.any()
    game = history.loc[history.season.eq(2021) & history.week.eq(2)].iloc[0]
    priors = {f"{role}_{metric}": .05 if metric == "epa" else .4 for role in ("offense", "defense_allowed") for metric in ("epa", "success")}
    result = pregame_efficiency(table, history, game, game.kickoff - pd.Timedelta(hours=24), priors)
    assert result["home_offense_epa"] == .05 and result["home_offense_epa_unavailable"] == 1
    good, _ = aggregate_batches([plays()])
    result = pregame_efficiency(good, history, game, game.kickoff - pd.Timedelta(hours=24), priors)
    assert result["home_offense_epa"] == pytest.approx((20*.2 + 200*.05)/220)
    assert result["away_defense_allowed_epa"] == result["home_offense_epa"]
    assert result["home_offense_epa_plays"] == 20


def test_efficiency_excludes_target_and_evaluation_prior_labels(history):
    table, _ = aggregate_batches([plays(), plays("2020_01_MIA_BUF"), plays("2021_02_MIA_BUF").assign(epa=99)])
    game = history.loc[history.season.eq(2021) & history.week.eq(2)].iloc[0]
    cutoff = game.kickoff - pd.Timedelta(hours=24)
    # Both roles need training coverage; mirror offensive teams for prior test.
    mirror = table.copy()
    mirror["team"] = mirror.team.replace({"BUF": "MIA", "MIA": "BUF"})
    both = pd.concat([table, mirror], ignore_index=True)
    prior = training_priors(both, history, cutoff, evaluation_season=2021)
    assert all(v == pytest.approx(.2 if "epa" in k else 1) for k, v in prior.items())
    original = pregame_efficiency(table, history, game, cutoff, prior)
    changed = table.copy()
    changed.loc[changed.game_id.eq(game.game_id), "epa_sum"] = 1e6
    assert original == pregame_efficiency(changed, history, game, cutoff, prior)
    with pytest.raises(ValueError, match="prior"):
        training_priors(table.loc[table.game_id.str.startswith("2021")], history, cutoff, 2021)


def test_source_checksum_and_production_output_guard(tmp_path):
    (tmp_path / "games.csv.gz").write_bytes(gzip.compress(b"changed"))
    (tmp_path / "source.json").write_text(json.dumps({"raw_sha256": "not-matching"}))
    with pytest.raises(ValueError, match="checksum"):
        read(tmp_path)
    from scripts.v2_team_efficiency import ROOT
    with pytest.raises(ValueError, match="production"):
        run(output=ROOT / "artifacts/research")


def test_source_availability_is_explicitly_assumed_and_incomplete_seasons_fail(history):
    assert (history.available_at - history.kickoff).eq(pd.Timedelta(hours=52)).all()
    from src.research.schedule_data import audit
    assert not audit(history).complete.any()  # A tiny fixture cannot claim full coverage.


@pytest.mark.parametrize("problem", ["duplicate", "score", "kickoff_collision", "week"])
def test_schedule_rejects_invalid_inputs(problem):
    raw = pd.DataFrame([dict(game_id="one", season=2021, week=1, game_type="REG", gameday="2021-09-01",
                             gametime="13:00", home_team="BUF", away_team="MIA", home_score=21, away_score=14)])
    if problem == "duplicate":
        raw = pd.concat([raw, raw])
    elif problem == "score":
        raw["home_score"] = float("inf")
    elif problem == "week":
        raw["week"] = 1.5
    else:
        raw = pd.concat([raw, raw.assign(game_id="other")])
    with pytest.raises(ValueError):
        normalize(raw)


def test_recorded_experiment_reproduces_and_preserves_artifacts(tmp_path):
    from scripts.v2_team_efficiency import ROOT
    before = hashes(ROOT / "artifacts")
    manifest = run(output=tmp_path)
    for name in ("pooled_metrics.csv", "season_metrics.csv", "fallbacks.csv", "exclusions.csv", "predictions.csv", "paired_deltas.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / name), pd.read_csv(RESEARCH / "outputs" / name),
                                      check_exact=False, atol=1e-12, rtol=1e-12)
    assert hashes(ROOT / "artifacts") == before == manifest["production_artifact_sha256"]
    expected = json.loads((RESEARCH / "outputs/manifest.json").read_text())
    assert manifest["source"]["raw_sha256"] == expected["source"]["raw_sha256"]
    assert not manifest["promotion_allowed"]
