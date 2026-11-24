"""No live pushes: validate generation and publication in disposable repositories."""
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from scripts import publish_refresh as publish, refresh_validation as validation
from scripts.integrate_refresh_branch import integrate


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    store = tmp_path / "artifacts"
    for name in validation.FROZEN:
        target = store / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"reviewed-frozen-bytes")
    for name in ("predictions/original.json", "retrospective/predictions/original.json"):
        target = store / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"probability": 0.6}\n')
    for name in validation.GENERATED:
        value = {"source": "fixture", "fetched_at": "2026-11-05T19:00:00Z", "error": None}
        if name == "games.csv":
            (store / name).write_text("game_id,home_score\ngame,21.0\n")
        elif name == "season-report.json":
            (store / name).write_text(json.dumps(dict(source=value, generated_at="old", games=1, brier=.2)))
        else:
            (store / name).write_text(json.dumps(value if name == "source.json" else {"brier": .2}))
    (tmp_path / "code.py").write_text("reviewed = True\n")
    git("add", ".")
    git("commit", "-m", "reviewed")
    return store, git("rev-parse", "HEAD")


@pytest.mark.parametrize("change", ["model", "model_metadata", "live", "retro", "code"])
def test_guard_rejects_unreviewed_code_models_and_rewritten_history(repository, change):
    store, reviewed = repository
    names = {"model": store / "model.joblib", "model_metadata": store / "model.json",
             "live": store / "predictions/original.json", "retro": store / "retrospective/predictions/original.json",
             "code": store.parent / "code.py"}
    names[change].write_text("changed")
    with pytest.raises(ValueError):
        validation.guard(reviewed, reviewed)


def test_reviewed_merge_preserves_durable_prediction_and_models(repository):
    store, _ = repository
    git("switch", "-c", "durable")
    (store / "predictions/new.json").write_text('{"generated_at":"authentic-original"}')
    git("add", ".")
    git("commit", "-m", "append authentic history")
    history = git("rev-parse", "HEAD")
    git("switch", "main")
    (store.parent / "code.py").write_text("reviewed = 'new version'\n")
    git("add", ".")
    git("commit", "-m", "reviewed application change")
    reviewed = git("rev-parse", "HEAD")
    models = {name: (store / name).read_bytes() for name in validation.FROZEN}
    git("switch", "durable")
    integrate(reviewed, strict=True)
    validation.guard(reviewed, history)
    assert (store / "predictions/new.json").read_text() == '{"generated_at":"authentic-original"}'
    assert all((store / name).read_bytes() == content for name, content in models.items())
    assert "new version" in (store.parent / "code.py").read_text()


def test_scheduled_strict_merge_stops_on_generated_conflict(repository):
    store, _ = repository
    git("switch", "-c", "durable")
    (store / "source.json").write_text('{"durable":true}')
    git("add", ".")
    git("commit", "-m", "durable")
    git("switch", "main")
    (store / "source.json").write_text('{"main":true}')
    git("add", ".")
    git("commit", "-m", "main")
    git("switch", "durable")
    head = git("rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="human review"):
        integrate("main", strict=True)
    assert git("rev-parse", "HEAD") == head
    assert git("diff", "--name-only", "--diff-filter=U") == "artifacts/source.json"


def test_timestamp_and_float_noise_do_not_commit_but_results_and_forecasts_do(repository):
    store, base = repository
    source = json.loads((store / "source.json").read_text())
    source["fetched_at"] = "2026-11-05T20:00:00Z"
    (store / "source.json").write_text(json.dumps(source))
    report = json.loads((store / "season-report.json").read_text())
    report.update(generated_at="new", brier=.20000000000000004, source=source)
    (store / "season-report.json").write_text(json.dumps(report))
    (store / "games.csv").write_text("game_id,home_score\ngame,21\n")
    assert not validation.meaningful_changes(base)
    (store / "games.csv").write_text("game_id,home_score\ngame,22\n")
    assert validation.meaningful_changes(base)
    git("restore", "--worktree", "--", "artifacts/games.csv")
    (store / "predictions/new.json").write_text('{"generated_at":"new authentic record"}')
    assert validation.meaningful_changes(base)


@pytest.mark.parametrize(("new_stamp", "heartbeat_due"), [
    ("2026-11-05T20:00:00Z", False),
    ("2026-11-06T07:00:00Z", True),
])
def test_finalize_noop_refreshes_provenance_on_heartbeat(repository, monkeypatch, tmp_path, new_stamp, heartbeat_due):
    from src import data
    store, base = repository
    monkeypatch.setattr(data, "STORE", store)
    original = {name: (store / name).read_bytes() for name in validation.GENERATED}
    source = dict(source="fixture", fetched_at=new_stamp, error=None)
    (store / "source.json").write_text(json.dumps(source))
    monkeypatch.setattr(validation, "validate_state", lambda: dict(validated=True, source=source))
    result = validation.finalize(base, base, base, tmp_path / "receipt.json")
    assert not result["meaningful_change"]
    assert result["heartbeat_due"] == heartbeat_due
    assert result["source"]["fetched_at"] == new_stamp
    if heartbeat_due:
        assert json.loads((store / "source.json").read_text()) == source
    else:
        assert all((store / name).read_bytes() == original[name] for name in validation.GENERATED)
        assert not git("status", "--porcelain", "--", "artifacts")


def test_failed_validation_creates_no_publishable_receipt(repository, monkeypatch, tmp_path):
    from src import data
    store, base = repository
    monkeypatch.setattr(data, "STORE", store)
    def failed():
        raise ValueError("Source retrieval failed")
    monkeypatch.setattr(validation, "validate_state", failed)
    with pytest.raises(ValueError, match="Source retrieval"):
        validation.finalize(base, base, base, tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()


def test_publisher_rejects_unvalidated_or_changed_committed_tree(repository, monkeypatch, tmp_path):
    store, base = repository
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(dict(validated=False)))
    with pytest.raises(ValueError, match="successful"):
        publish.publish(receipt, "automation/nfl-artifacts")
    sealed = validation.artifact_blobs(store)
    (store / "predictions/new.json").write_text("unvalidated")
    git("add", "artifacts")
    git("commit", "-m", "unvalidated mutation")
    receipt.write_text(json.dumps(dict(validated=True, artifact_blobs=sealed)))
    with pytest.raises(ValueError, match="validated tree"):
        publish.publish(receipt, "automation/nfl-artifacts")


def test_publication_pushes_only_the_validated_snapshot(repository, monkeypatch, tmp_path):
    store, sha = repository
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(dict(validated=True, artifact_blobs=validation.artifact_blobs(store))))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    events = []
    run = subprocess.run

    def fake_publish_command(cmd, **kwargs):
        if cmd[:2] in (["gh", "auth"], ["git", "push"]):
            events.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return run(cmd, **kwargs)

    monkeypatch.setattr(publish.subprocess, "run", fake_publish_command)
    assert publish.publish(receipt, "automation/nfl-artifacts") == sha
    assert events == [
        ["gh", "auth", "setup-git"],
        ["git", "push", "origin", "HEAD:refs/heads/automation/nfl-artifacts"],
    ]


def test_workflow_validates_after_generation_before_publishing_with_no_pr_target_execution():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/refresh.yml").read_text()
    ci = (root / ".github/workflows/ci.yml").read_text()
    assert workflow.index("src.pipeline evaluate") < workflow.index("refresh_validation finalize") < workflow.index("git add artifacts") < workflow.index("scripts.publish_refresh")
    assert "--strict" in workflow and "refresh_validation.py\" guard" in workflow
    assert "pull-requests: write" not in workflow and "contents: read" in ci
    assert "pull_request_target" not in workflow + ci
    assert "GH_TOKEN:" not in workflow[:workflow.index("Publish validated production snapshot")]
    assert "persist-credentials: false" in workflow + ci


@pytest.fixture
def generated_candidate(store):
    from src import data, predictions, season
    from src.teams import NAMES
    teams = list(NAMES)
    rows = []
    for week in range(1, 18):
        for index in range(16):
            home, away = teams[index], teams[-index-1]
            kickoff = pd.Timestamp("2026-09-01T18:00:00Z") + pd.Timedelta(weeks=week-1)
            rows.append(dict(game_id=f"2026_{week:02d}_{away}_{home}", season=2026, week=week,
                             game_type="REG", kickoff=kickoff.isoformat(), home_team=home, away_team=away,
                             home_score=None, away_score=None))
        teams = [teams[0], teams[-1], *teams[1:-1]]
    rows[0]["kickoff"] = "2026-11-05T19:00:00Z"
    games = data.normalize(pd.DataFrame(rows))
    games.to_csv(store / "games.csv", index=False)
    data.write_json(store / "source.json", dict(source=data.URL, fetched_at="2026-11-05T19:30:00Z", error=None))
    target = games.loc[games.game_id.eq(rows[0]["game_id"])].iloc[0]
    record = dict(game_id=target.game_id, season=2026, week=1, home_team=target.home_team,
                  away_team=target.away_team, kickoff=target.kickoff.isoformat(), kind="live", revision=1,
                  model_version="synthetic-test-only", generated_at="2026-11-05T18:00:00Z",
                  data_cutoff="2026-11-05T18:00:00Z", training_cutoff="2026-08-01T00:00:00Z",
                  source_freshness="2026-11-05T17:00:00Z", home_probability=.6, away_probability=.4,
                  predicted_winner=target.home_team)
    data.write_json(store / "predictions/original.json", record)
    games, source = data.load(offline=True)
    predictions.reconcile(games)
    season.season_report(games, source, persist=True)
    return store


def test_post_generation_validation_keeps_started_missing_results_pending(generated_candidate):
    result = validation.validate_state()
    assert result["validated"] and result["metrics"]["games"] == 0
    assert result["coverage"]["missing_results_after_kickoff"] > 0
    assert json.loads((generated_candidate / "live-results.json").read_text())[0]["result"] == "pending"


@pytest.mark.parametrize("failure", ["missing_source", "stale_source", "source_error", "bad_report"])
def test_post_generation_validation_rejects_bad_candidates(generated_candidate, failure):
    from src.data import write_json
    if failure == "missing_source":
        (generated_candidate / "source.json").unlink()
    elif failure == "stale_source":
        write_json(generated_candidate / "source.json", dict(fetched_at="2026-01-01T00:00:00Z"))
    elif failure == "source_error":
        write_json(generated_candidate / "source.json", dict(fetched_at="2026-11-05T19:30:00Z", error="retrieval failed"))
    else:
        write_json(generated_candidate / "live-metrics.json", dict(games=999, accuracy=1))
    with pytest.raises(ValueError, match="Source|reconcile"):
        validation.validate_state()
