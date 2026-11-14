"""Real Git merge regressions in disposable repositories only."""
import json
import subprocess
import pytest
from scripts.integrate_refresh_branch import GENERATED, integrate


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


@pytest.fixture
def branches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    def snapshot(day, lock="2026-11-01T18:00:00Z", drop=False):
        for name in GENERATED:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(day))
        (tmp_path / "artifacts/source.json").write_text(json.dumps(dict(source="fixture", fetched_at=f"2026-11-{day:02d}T20:00:00Z")))
        (tmp_path / "artifacts/games.csv").write_text("game_id,prediction_lock_at\n" + ("" if drop else f"game,{lock}\n"))
        git("add", ".")
        git("commit", "-m", f"snapshot {day}")
    snapshot(10)
    git("switch", "-c", "durable")
    snapshot(11)
    git("switch", "main")
    return tmp_path, snapshot


@pytest.mark.parametrize("main_day", [12, 9])
def test_generated_conflicts_keep_newer_coherent_snapshot(branches, main_day):
    path, snapshot = branches
    snapshot(main_day)
    expected = {name: git("show", f'{"main" if main_day == 12 else "durable"}:{name}') for name in GENERATED}
    git("switch", "durable")
    integrate("main")
    assert not git("status", "--porcelain")
    for name in GENERATED:
        assert (path / name).read_text().strip() == expected[name]
    git("merge-base", "--is-ancestor", "main", "HEAD")


@pytest.mark.parametrize("unsafe", ["lock", "dropped_game", "forecast", "model", "code"])
def test_unsafe_merge_requires_manual_review(branches, unsafe):
    path, snapshot = branches
    snapshot(12, lock="2026-11-02T18:00:00Z" if unsafe == "lock" else "2026-11-01T18:00:00Z", drop=unsafe == "dropped_game")
    if unsafe in ("forecast", "model", "code"):
        filename = {"forecast": "artifacts/predictions/original.json", "model": "artifacts/model.joblib", "code": "code.py"}[unsafe]
        for branch in ("main", "durable"):
            git("switch", branch)
            target = path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(branch)
            git("add", filename)
            git("commit", "-m", branch)
    git("switch", "durable")
    before = git("rev-parse", "HEAD")
    with pytest.raises((ValueError, RuntimeError)):
        integrate("main")
    assert git("rev-parse", "HEAD") == before
    assert git("diff", "--name-only", "--diff-filter=U")
