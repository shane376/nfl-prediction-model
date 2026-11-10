"""Season-wide coverage and strictly separated cohort evaluation."""
import numpy as np
import pandas as pd
from src.data import STORE, now, write_json
from src.modeling import metrics
from src.predictions import records, reconciliation
from src.retrospective import retrospective_results
from src.teams import NAMES


def score_records(rows):
    rows = [r for r in rows if r.get("kind") != "provisional"]
    scored = [r for r in rows if r.get("target") is not None and not r.get("integrity_error")]
    result = metrics([r["target"] for r in scored], [r["home_probability"] for r in scored])
    result.update(records=len(rows), correct=sum(r["result"] == "correct" for r in scored),
                  incorrect=sum(r["result"] == "incorrect" for r in scored),
                  ties=sum(r.get("result") == "tie" for r in rows),
                  unavailable=sum(r.get("result") == "unavailable" for r in rows),
                  home_baseline_accuracy=float(np.mean([r["target"] for r in scored])) if scored else None)
    return result


def weekly_performance(rows):
    result, total, correct = [], 0, 0
    for week in range(1, 19):
        values = score_records([r for r in rows if r.get("week") == week])
        total += values["games"]
        correct += values["correct"]
        result.append({"week": week, **{key: value for key, value in values.items() if key != "calibration"},
                       "cumulative_accuracy": correct / total if total else None,
                       "cumulative_games": total})
    return result


def coverage(games, live_rows, retro_rows, season=2026):
    selected = games.loc[games.season.eq(season)]
    archived = {r["game_id"] for r in live_rows if r.get("kind") == "live"}
    valid_live = {r["game_id"] for r in live_rows if not r.get("integrity_error")}
    retro = {r["game_id"] for r in retro_rows}
    valid_retro = {r["game_id"] for r in retro_rows if not r.get("integrity_error")}
    current, rows = now(), []
    for week in range(1, 19):
        frame = selected.loc[selected.week.eq(week)]
        completed = frame.status.eq("completed")
        unavailable_status = frame.status.isin(["postponed", "canceled", "suspended"])
        missing_result = (frame.kickoff < current) & (frame.home_score.isna() | frame.away_score.isna()) & ~unavailable_status
        eligible = completed & frame.kickoff.notna() & (frame.available_at < current)
        auth = frame.game_id.isin(archived)
        rows.append({"week": week, "scheduled_games": len(frame), "completed_games": int(completed.sum()),
                     "scheduled_unplayed": int(frame.status.eq("scheduled").sum()),
                     "ties": int((completed & frame.home_score.eq(frame.away_score)).sum()),
                     "postponed": int(frame.status.eq("postponed").sum()), "canceled": int(frame.status.eq("canceled").sum()),
                     "authentic_predictions": int(auth.sum()), "valid_authentic_predictions": int(frame.game_id.isin(valid_live).sum()),
                     "retrospective_predictions": int(frame.game_id.isin(retro).sum()),
                     "valid_retrospective_predictions": int(frame.game_id.isin(valid_retro).sum()),
                     "missing_authentic_predictions": int((~auth).sum()),
                     "missing_past_authentic": int((~auth & (frame.kickoff < current)).sum()),
                     "missing_future_authentic": int((~auth & (frame.kickoff >= current)).sum()),
                     "missing_retrospective_backtests": int((eligible & ~auth & ~frame.game_id.isin(valid_retro)).sum()),
                     "awaiting_result_embargo": int((completed & ~eligible).sum()),
                     "missing_results_after_kickoff": int(missing_result.sum()),
                     "missing_kickoff": int(frame.kickoff.isna().sum()),
                     "games_missing_required_data": int(((frame.kickoff.isna() & ~frame.status.eq("canceled")) | missing_result).sum())})
    return rows


def season_report(games, source, persist=False):
    live, _ = reconciliation(games)
    live = [r for r in live if r.get("season") == 2026]
    retro = retrospective_results(games)
    table = coverage(games, live, retro)
    selected = games.loc[games.season.eq(2026)]
    appearances = pd.concat([selected.home_team, selected.away_team]).value_counts()
    schedule_validation = {"expected_games": 272, "actual_games": len(selected),
                           "missing_schedule_rows": max(0, 272 - len(selected)),
                           "complete": len(selected) == 272 and set(appearances.index) == set(NAMES) and bool(appearances.eq(17).all()),
                           "games_per_team": {team: int(appearances.get(team, 0)) for team in NAMES}}
    report = {"season": 2026, "generated_at": now().isoformat(), "source": source,
              "schedule_validation": schedule_validation,
              "coverage_by_week": table,
              "coverage_totals": {key: sum(row[key] for row in table) for key in table[0] if key != "week"},
              "recorded": {"summary": score_records(live), "weekly": weekly_performance(live)},
              "retrospective_backtest": {"summary": score_records(retro), "weekly": weekly_performance(retro)}}
    if persist:
        write_json(STORE / "season-report.json", report)
    return report
