"""Trusted refresh guards, post-generation validation, and no-op preparation."""
import argparse
import csv
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import io
import json
import math
from pathlib import Path
import subprocess

GENERATED = ("games.csv", "source.json", "live-results.json", "live-metrics.json", "season-report.json")
FROZEN = ("model.joblib", "model.json", "retrospective/model.joblib", "retrospective/model.json", "backtest.json")


def git(*args, text=True):
    return subprocess.check_output(["git", *args], text=text)


def artifact_blobs(store):
    store = Path(store)
    result = {}
    if store.is_symlink():
        raise ValueError("Artifact symlinks require manual review")
    for path in sorted(store.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact symlinks require manual review")
        if path.is_file():
            if path.suffix == ".tmp" or path.name == ".prediction-writer.lock":
                raise ValueError("Incomplete/concurrent artifact writer")
            name = "artifacts/" + path.relative_to(store).as_posix()
            result[name] = git("hash-object", "--path", name, str(path.resolve())).strip()
    return result


def guard(reviewed, history, store=Path("artifacts")):
    # This guard is copied from reviewed main before switching to the data branch.
    changed = git("diff", "--name-only", reviewed, "--", ".", ":(exclude)artifacts").strip()
    if changed:
        raise ValueError("Unreviewed code on artifact branch: " + changed)
    actual = artifact_blobs(store)
    for name in FROZEN:
        path = "artifacts/" + name
        if actual.get(path) != git("rev-parse", f"{reviewed}:{path}").strip():
            raise ValueError("Reviewed frozen artifact changed: " + path)
    for ref in dict.fromkeys([reviewed, history]):
        paths = git("ls-tree", "-r", "--name-only", ref, "--", "artifacts/predictions", "artifacts/retrospective/predictions").splitlines()
        for path in paths:
            if actual.get(path) != git("rev-parse", f"{ref}:{path}").strip():
                raise ValueError("Append-only prediction changed or deleted: " + path)
    return actual


def clean_report(value, filename):
    """Discard execution-only fields, never forecast timestamps or probabilities."""
    if filename == "source.json":
        return {k: v for k, v in value.items() if k != "fetched_at"}
    if filename == "season-report.json":
        value = dict(value)
        value.pop("generated_at", None)
        value["source"] = {k: v for k, v in value["source"].items() if k not in ("fetched_at", "offline")}
    return value


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    return a == b


def csv_value(payload):
    def number(value):
        try:
            parsed = Decimal(value)
            return parsed if parsed.is_finite() else value
        except InvalidOperation:
            return value
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    return sorted([{k: number(v) for k, v in row.items()} for row in rows], key=lambda r: r["game_id"])


def meaningful_changes(base, store=Path("artifacts")):
    actual = artifact_blobs(store)
    original = {}
    for line in git("ls-tree", "-r", base, "--", "artifacts").splitlines():
        meta, path = line.split("\t", 1)
        original[path] = meta.split()[2]
    if actual.keys() != original.keys():
        return True  # Added/deleted forecasts are never treated as timestamp noise.
    for path, blob in actual.items():
        if blob == original[path]:
            continue
        name = path.removeprefix("artifacts/")
        if name not in GENERATED:
            return True
        previous = git("show", f"{base}:{path}", text=False)
        current = (Path(store) / name).read_bytes()
        if name == "games.csv":
            if csv_value(previous) != csv_value(current):
                return True  # Exact numeric values; no source-data rounding tolerance.
        elif not equivalent(clean_report(json.loads(previous), name), clean_report(json.loads(current), name)):
            return True
    return False


def validate_state():
    # Lazy imports: the pre-execution guard itself requires only reviewed stdlib code.
    from src.data import STORE, load, now
    from src.predictions import audit, reconciliation
    from src.season import season_report
    games, source = load(offline=True)
    if source.get("error") or source["stale"]:
        raise ValueError("Source retrieval failed or source is stale: " + str(source))
    errors = [r for r in audit(games) if r["error"]]
    if errors:
        raise ValueError("Prediction integrity audit failed: " + str(errors))
    results, summary = reconciliation(games)
    report = season_report(games, source)
    if not report["schedule_validation"]["complete"]:
        raise ValueError("Incomplete 2026 schedule")
    for name, expected in (("live-results.json", results), ("live-metrics.json", summary), ("season-report.json", report)):
        saved = json.loads((STORE / name).read_text(encoding="utf-8"))
        if not equivalent(clean_report(saved, name), clean_report(expected, name)):
            raise ValueError("Generated output does not reconcile: " + name)
    return dict(validated=True, checked_at=now().isoformat(), source=source,
                predictions=len(results), coverage=report["coverage_totals"], metrics=summary)


def finalize(reviewed, history, base, receipt):
    from src.data import STORE
    if STORE.resolve() != Path("artifacts").resolve():
        raise ValueError("Finalization requires this checkout's artifacts; use an isolated checkout for verification")
    guard(reviewed, history, STORE)
    result = validate_state()
    changed = meaningful_changes(base, STORE)
    previous = json.loads(git("show", f"{base}:artifacts/source.json")).get("fetched_at")
    current = result["source"]["fetched_at"]
    try:
        age = (datetime.fromisoformat(current) - datetime.fromisoformat(previous))
        heartbeat_due = not timedelta(0) <= age < timedelta(hours=12)
    except (TypeError, ValueError):
        heartbeat_due = True
    if not changed and not heartbeat_due:
        # Preserve published provenance, recording this successful retrieval in
        # the separate receipt/run summary instead of inventing a new snapshot.
        subprocess.run(["git", "restore", f"--source={base}", "--staged", "--worktree", "--",
                        *["artifacts/" + name for name in GENERATED]], check=True)
    result.update(meaningful_change=changed, heartbeat_due=heartbeat_due,
                  reviewed_sha=reviewed, history_sha=history,
                  artifact_blobs=guard(reviewed, history, STORE))
    Path(receipt).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_blobs"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["guard", "finalize"])
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--base")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    if args.command == "guard":
        guard(args.reviewed, args.history)
    else:
        if not args.base or not args.receipt:
            parser.error("finalize requires --base and --receipt")
        finalize(args.reviewed, args.history, args.base, args.receipt)
