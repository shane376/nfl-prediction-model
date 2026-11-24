"""Publish a validated artifact tree to the production snapshot branch."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts.refresh_validation import git


def publish(receipt, branch):
    evidence = json.loads(Path(receipt).read_text(encoding="utf-8"))
    if evidence.get("validated") is not True:
        raise ValueError("Missing successful post-generation validation")
    committed = {}
    for line in git("ls-tree", "-r", "HEAD", "--", "artifacts").splitlines():
        meta, path = line.split("\t", 1)
        committed[path] = meta.split()[2]
    if committed != evidence["artifact_blobs"]:
        raise ValueError("Committed artifacts differ from validated tree; refuse publication")
    sha = git("rev-parse", "HEAD").strip()
    # A normal push protects concurrent updates and the append-only branch.
    subprocess.run(["gh", "auth", "setup-git"], check=True)
    subprocess.run(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], check=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
            summary.write(
                f"Published snapshot `{sha}` to `{branch}`.\n\n"
                f"Reviewed code: `{evidence['reviewed_sha']}`. "
                f"Source checked at {evidence['source']['fetched_at']}. "
                f"Audited records: {evidence['predictions']}.\n"
            )
    return sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--branch", default="automation/nfl-artifacts",
                        choices=["automation/nfl-artifacts"])
    args = parser.parse_args()
    publish(args.receipt, args.branch)
