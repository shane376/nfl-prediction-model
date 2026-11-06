"""Run with python -m src.pipeline refresh|train|predict|evaluate."""
import argparse
import json
import pandas as pd
from src.data import load, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["refresh", "train", "predict", "evaluate", "audit"])
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, choices=range(1, 19))
    parser.add_argument("--offline", action="store_true", help="Read cached schedule without network access")
    parser.add_argument("--rerun", action="store_true", help="Append a new pregame revision")
    args = parser.parse_args()
    if args.offline and args.command == "refresh":
        parser.error("refresh requires network; omit --offline")
    games, source = load(offline=args.offline, refresh=args.command == "refresh")
    if args.command == "refresh":
        print(json.dumps({**source, "games": len(games), "games_2026": int(games.season.eq(2026).sum())}, indent=2))
        if source.get("error"):
            raise SystemExit("Refresh failed; previous snapshot preserved")
    elif args.command == "train":
        from src.modeling import train
        print(json.dumps(train(games, now(), source), indent=2))
    elif args.command == "predict":
        from src.predictions import predict
        print(f"Recorded {predict(games, source, args.season, args.week, args.rerun)} predictions")
    elif args.command == "audit":
        from src.predictions import audit
        result = audit(games)
        print(json.dumps(result, indent=2))
        if any(row["error"] for row in result):
            raise SystemExit("Prediction integrity audit found unavailable/invalid records")
    else:
        from src.predictions import reconcile
        print(json.dumps(reconcile(games), indent=2))


if __name__ == "__main__":
    main()
