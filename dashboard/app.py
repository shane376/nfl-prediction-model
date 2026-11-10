"""Read-only dashboard: training and recording happen through the CLI."""
import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import STORE, load, now
from src.predictions import records, reconciliation
from src.teams import display
from src.retrospective import retrospective_results
from src.season import coverage, score_records, weekly_performance

st.set_page_config(page_title="NFL • 2026 Forecast", page_icon="🏈", layout="wide")


def read(name, default):
    path = STORE / name
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        if not isinstance(value, type(default)):
            raise ValueError("Unexpected artifact format")
        return value
    except (OSError, ValueError) as exc:
        st.warning(f"Unavailable artifact {name}: {exc}")
        return default


@st.cache_data(ttl=60, max_entries=2)
def snapshot(store_path, modified, metadata_modified):
    return load(offline=True)


@st.cache_data(ttl=60, max_entries=2)
def reconstructed_results(games, store_path, retro_modified, archive_modified):
    return retrospective_results(games)


st.caption("GRIDIRON LAB / V1 INTELLIGENCE")
st.title("2026 NFL Forecast")
st.write("Weekly matchups. Transparent probabilities. A record of every pick.")
try:
    schedule_path = STORE / "games.csv"
    metadata_path = STORE / "source.json"
    games, source = snapshot(str(STORE), schedule_path.stat().st_mtime_ns if schedule_path.exists() else 0,
                             metadata_path.stat().st_mtime_ns if metadata_path.exists() else 0)
except (OSError, ValueError, KeyError) as exc:
    st.warning(str(exc))
    st.code("python -m src.pipeline refresh\npython -m src.pipeline train\npython -m src.pipeline predict --season 2026")
    st.stop()

meta = read("model.json", {})
if source.get("stale"):
    st.warning(f"STALE SOURCE SNAPSHOT · {source.get('error') or 'Last successful fetch was over 24 hours ago.'}")
st.caption("Complete regular-season schedule · Recorded forecasts and retrospective backtests stay separate")
with st.expander("Source and model details"):
    st.write("Latest successful fetch", source.get("fetched_at", "unknown"))
    st.write("Recorded forecast model", meta.get("version", "not trained"))
    st.json(source)
season_games = games.loc[games.season.eq(2026)]
if len(season_games) != 272:
    st.warning(f"Incomplete schedule snapshot: {len(season_games)} of 272 expected regular-season games. Missing schedule rows are separate from missing forecasts.")
if season_games.empty:
    st.warning("The source snapshot contains no 2026 games. Refresh when the schedule is available; no games have been invented.")
upcoming = season_games.loc[season_games.kickoff >= now()]
default_week = int(upcoming.iloc[0].week) if len(upcoming) else 18
week = st.sidebar.selectbox("Regular season week", range(1, 19), index=default_week - 1, key="week")
st.sidebar.caption("Kickoffs shown in Eastern Time. Byes have no scheduled game. Predictions are binary; ties are excluded from accuracy.")
weekly, performance, history, info = st.tabs(["Weekly predictions", "Performance", "Prediction history", "Model information"])
try:
    saved = records()
    # Recompute in memory from the latest snapshot; page loads never write files.
    results, live = reconciliation(games)
except (OSError, ValueError) as exc:
    st.warning(f"Prediction history unavailable: {exc}")
    saved, results, live = [], [], {}
try:
    retro_dir, archive_dir = STORE / "retrospective/predictions", STORE / "predictions"
    retro_results = reconstructed_results(games, str(STORE), retro_dir.stat().st_mtime_ns if retro_dir.exists() else 0,
                                         archive_dir.stat().st_mtime_ns if archive_dir.exists() else 0)
except (OSError, ValueError) as exc:
    st.warning(f"Retrospective backtests unavailable: {exc}")
    retro_results = []
first = {row["game_id"]: row for row in results}
historical = {row["game_id"]: row for row in retro_results}
coverage_table = pd.DataFrame(coverage(games, results, retro_results))

with weekly:
    selected = season_games.loc[season_games.week.eq(week)]
    st.subheader(f"Week {week} · {len(selected)} games")
    counts = coverage_table.loc[coverage_table.week.eq(week)].iloc[0]
    st.caption(f"{int(counts.completed_games)} completed · {int(counts.authentic_predictions)} recorded picks · "
               f"{int(counts.retrospective_predictions)} retrospective · "
               f"{int(counts.missing_retrospective_backtests)} completed games awaiting backtest · "
               f"{int(counts.games_missing_required_data)} games missing required data")
    if selected.empty:
        st.info("No games in this source snapshot for the selected week.")
    for _, game in selected.iterrows():
        with st.container(border=True):
            left, right = st.columns([3, 2])
            left.write(f"**{game.away_team} @ {game.home_team}**")
            kickoff = game.kickoff.tz_convert("America/New_York").strftime("%a, %b %d · %I:%M %p %Z") if pd.notna(game.kickoff) else "Kickoff TBD"
            status_label = {"completed": "Final", "scheduled": "Upcoming", "kickoff_unavailable": "Kickoff TBD",
                            "underway_or_unavailable": "Underway / result pending", "result_unavailable": "Incomplete score",
                            "postponed": "Postponed", "canceled": "Canceled", "suspended": "Suspended"}.get(game.status, game.status)
            score = f" · {game.away_score:g}–{game.home_score:g}" if game.status == "completed" else ""
            left.caption(f"{kickoff} · {status_label}{score}")
            record = first.get(game.game_id) or historical.get(game.game_id)
            if record and not record.get("integrity_error"):
                p = record["home_probability"]
                label = "Retrospective backtest" if record["kind"] == "retrospective_backtest" else "Recorded pregame pick"
                outcome = f" · {record['result']}" if record.get("result") != "pending" else ""
                right.write(f"{label}: **{record['predicted_winner']}**{outcome}")
                right.progress(p, text=f"{game.home_team} {p:.1%} · {game.away_team} {1-p:.1%}")
                if record.get("source_stale"):
                    right.warning("Pick generated from a stale snapshot")
            else:
                right.info(f"Unavailable · {record['integrity_error'] if record else 'no recorded pregame prediction'}")
            if record and not record.get("integrity_error") and pd.notna(game.kickoff) and game.kickoff != pd.Timestamp(record.get("kickoff")):
                left.caption("Kickoff changed since this pick was recorded. The original record is retained.")
            with st.expander(f"Matchup details · {game.game_id}"):
                st.caption(f"{display(game.away_team)} at {display(game.home_team)}")
                if record:
                    st.json({key: value for key, value in record.items() if key != "features"})
                else:
                    st.write("No recorded forecast or eligible retrospective backtest. Future games are listed even without predictions.")

with performance:
    st.subheader("2026 season performance · separate cohorts")
    for title, cohort in (("Recorded pregame forecasts", results), ("Retrospective backtests · frozen pre-2026 model", retro_results)):
        st.write(f"**{title}**")
        summary = score_records(cohort)
        with st.container(horizontal=True):
            st.metric("Accuracy", f"{summary['accuracy']:.1%}" if summary["accuracy"] is not None else "Awaiting results", border=True)
            st.metric("Correct / incorrect", f"{summary['correct']} / {summary['incorrect']}", border=True)
            st.metric("Home baseline", f"{summary['home_baseline_accuracy']:.1%}" if summary["home_baseline_accuracy"] is not None else "Unavailable", border=True)
            st.metric("Scored games", summary["games"], border=True)
        st.caption(f"Brier: {summary['brier'] if summary['brier'] is not None else 'unavailable'} · "
                   f"Log loss: {summary['log_loss'] if summary['log_loss'] is not None else 'unavailable'} · "
                   f"Ties excluded: {summary['ties']} · Unavailable: {summary['unavailable']}")
        weekly_metrics = pd.DataFrame(weekly_performance(cohort)).set_index("week")
        if summary["games"]:
            st.line_chart(weekly_metrics[["accuracy", "cumulative_accuracy", "home_baseline_accuracy"]],
                          alt=f"{title}: weekly and cumulative accuracy against the home baseline")
        valid = [r["home_probability"] for r in cohort if not r.get("integrity_error")]
        if valid:
            distribution = pd.Series(pd.cut(valid, bins=[i / 10 for i in range(11)], include_lowest=True)).value_counts(sort=False)
            distribution.index = distribution.index.astype(str)
            st.bar_chart(distribution, alt=f"{title}: home-win probability distribution, including pending recorded games")
        with st.expander(f"Weekly evaluation details · {title}"):
            st.dataframe(weekly_metrics, alt=f"Weekly metrics for {title}")
    st.caption("Cohorts are never combined. Retrospective picks were generated after the games using historical cutoffs, not archived pregame timestamps. Baselines use the exact same scored non-tied games in each cohort.")
    st.subheader("Season data coverage")
    st.dataframe(coverage_table, hide_index=True, alt="Weeks 1–18 schedule, result and prediction coverage")
    st.caption("Missing authentic forecasts are not missing schedule rows. Future weeks do not require retrospective backtests; completed games with archived picks are excluded from retrospective generation.")
    with st.expander("Historical backtest · 2025 holdout (separate from 2026)"):
        comparisons = meta.get("holdout_2025", {})
        if comparisons:
            st.dataframe(pd.DataFrame(comparisons).T.drop(columns=["calibration"], errors="ignore"), alt="2025 model holdout comparison")
            calibration = comparisons.get(meta.get("variant"), {}).get("calibration", {})
            if calibration:
                st.dataframe(pd.DataFrame(calibration), hide_index=True, alt="2025 holdout calibration bins")

with history:
    cohort_name = st.selectbox("Prediction type", ["Recorded pregame forecasts", "Retrospective backtests"])
    history_rows = retro_results if cohort_name == "Retrospective backtests" else results
    query = st.text_input("Search team or matchup").upper()
    status = st.multiselect("Result", ["correct", "incorrect", "pending", "tie", "unavailable"])
    table = []
    for row in history_rows:
        matchup = f"{row.get('away_team', '?')} @ {row.get('home_team', '?')}"
        if query not in matchup or (status and row["result"] not in status):
            continue
        indicator = {"correct": "✓", "incorrect": "✕", "pending": "◷", "tie": "=", "unavailable": "—"}[row["result"]]
        table.append({"Week": row.get("week"), "Matchup": matchup, "Prediction": row.get("predicted_winner"), "Actual": row["actual"], "Result": f"{indicator} {row['result']}", "Type": row["kind"], "Generated": row.get("generated_at"), "Historical cutoff": row.get("feature_cutoff"), "Integrity": row.get("integrity_error")})
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    st.caption("First recorded pregame pick per game is scored. All revisions are retained below.")
    with st.expander("Versioned prediction records"):
        st.dataframe(pd.DataFrame(saved), width="stretch")

with info:
    st.subheader("Standardized logistic regression · V1 A–D")
    st.write(meta.get("policy", "Model not yet trained."))
    st.write("Features", meta.get("features", []))
    st.write("Training seasons", meta.get("training_seasons", []))
    st.write("Training cutoff", meta.get("training_cutoff", "unavailable"))
    st.write("Limitations: season-to-date statistics reset each year; Week 1 uses .5 win percentage and zero scoring/streak values. Ties count as half wins in features but are excluded from binary evaluation. No injuries, roster changes, betting lines or neutral-site adjustment. Probabilities have diagnostics, not a calibration correction.")
    st.write("The schedule has no final whistle timestamp. A conservative 12-hour post-kickoff embargo is used for feature availability and result reconciliation. Historical backtests assume final results were available after that embargo; live predictions use only the fetched snapshot.")
    st.write("2026 retrospective evaluation uses a separate frozen V1 A model fitted only on 2023–2025 regular seasons with a 2026-02-01 training cutoff. Model fitting and prediction generation happen now, not before those games. Each record keeps its actual generation time separately from its historical feature cutoff. Latest final-score reconstruction is not an archived source replay.")
    with st.expander("Frozen retrospective model provenance"):
        st.json(read("retrospective/model.json", {}))
    st.link_button("nflverse schedule documentation", "https://nflreadr.nflverse.com/reference/load_schedules.html")
