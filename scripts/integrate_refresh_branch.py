"""Merge reviewed code; resolve only regenerated snapshot conflicts safely."""
import csv
import io
import json
import subprocess
import sys
from datetime import datetime


GENERATED = (
    "artifacts/games.csv", "artifacts/source.json", "artifacts/live-results.json",
    "artifacts/live-metrics.json", "artifacts/season-report.json",
)


def git(*args):
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8")


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Snapshot timestamps must include a timezone")
    return result


def integrate(reviewed):
    result = subprocess.run(["git", "merge", "--no-edit", reviewed], check=False)
    if result.returncode == 0:
        return
    conflicts = git("diff", "--name-only", "--diff-filter=U").splitlines()
    if not conflicts or set(conflicts) - set(GENERATED):
        raise RuntimeError("Merge requires manual review; conflicts outside generated snapshot files")
    sources = {ref: json.loads(git("show", f"{ref}:artifacts/source.json"))
               for ref in ("HEAD", reviewed)}
    if sources["HEAD"]["source"] != sources[reviewed]["source"]:
        raise ValueError("Source identity changed; manual review required")
    winner = max(sources, key=lambda ref: timestamp(sources[ref]["fetched_at"]))
    loser = reviewed if winner == "HEAD" else "HEAD"
    snapshots = {ref: {r["game_id"]: r for r in csv.DictReader(io.StringIO(
        git("show", f"{ref}:artifacts/games.csv")))} for ref in (winner, loser)}
    if set(snapshots[loser]) - set(snapshots[winner]):
        raise ValueError("Newer snapshot drops cached games; manual review required")
    for gid, old in snapshots[loser].items():
        old_lock = old.get("prediction_lock_at")
        new_lock = snapshots[winner][gid].get("prediction_lock_at")
        if old_lock and (not new_lock or timestamp(new_lock) > timestamp(old_lock)):
            raise ValueError("Newer snapshot relaxes a prediction lock; manual review required")
    # Restore the entire generated group from one coherent snapshot, including
    # files Git merged cleanly. Forecast and model conflicts are never resolved.
    subprocess.run(["git", "restore", f"--source={winner}", "--staged", "--worktree", "--", *GENERATED], check=True)
    subprocess.run(["git", "commit", "--no-edit"], check=True)


if __name__ == "__main__":
    integrate(sys.argv[1])
