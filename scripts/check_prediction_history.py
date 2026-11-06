"""Reject modifications/deletions of original predictions relative to a Git ref."""
import subprocess
import sys


def check(base):
    changes = subprocess.check_output(
        ["git", "diff", "--name-status", "--no-renames", base, "--", "artifacts/predictions"],
        text=True,
    ).splitlines()
    forbidden = [line for line in changes if not line.startswith("A\t")]
    if forbidden:
        raise SystemExit("Original predictions must never change:\n" + "\n".join(forbidden))


if __name__ == "__main__":
    check(sys.argv[1])
