"""Read-only dashboard: training and recording happen through the CLI."""
import json
import joblib
from pathlib import Path
import sys
from typing import cast
import pandas as pd
import altair as alt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import STORE, load, now
from src.predictions import records, reconciliation
from src.teams import display, NAMES
from src.retrospective import retrospective_results
from src.season import coverage, score_records, weekly_performance
from src.provisional import forecast
from src.presentation import (kickoff_label, default_week as choose_week, data_status,
                              LABELS, STATUSES, matchup_rows, filter_matchups, team_schedule,
                              team_record, remaining_wins_estimate)

st.set_page_config(page_title="Gridiron Lab | 2026 NFL", page_icon=":material/sports_football:", layout="wide")


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


st.caption(":green[GRIDIRON LAB / NFL ANALYTICS]")
st.title("Every matchup. A transparent edge.")
st.caption("2026 REGULAR SEASON | V1.2.1 | Probabilities with provenance")
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
st.caption("Local regular-season snapshot | Recorded forecasts, retrospective backtests and provisional estimates stay separate")
with st.expander("Advanced: source and model details"):
    st.write("Latest successful fetch", source.get("fetched_at", "unknown"))
    st.write("Recorded forecast model", meta.get("version", "not trained"))
    st.json(source)
season_games = games.loc[games.season.eq(2026)]
if len(season_games) != 272:
    st.warning(f"Incomplete schedule snapshot: {len(season_games)} of 272 expected regular-season games. Missing schedule rows are separate from missing forecasts.")
if season_games.empty:
    st.warning("The source snapshot contains no 2026 games. Refresh when the schedule is available; no games have been invented.")
default_week = choose_week(season_games, now())
week = st.sidebar.selectbox("Regular season week", [*range(1, 19), "All weeks"], index=default_week - 1, key="week")
st.sidebar.caption("Kickoffs shown in Eastern Time. Byes have no scheduled game. Predictions are binary; ties are excluded from accuracy.")
st.sidebar.title("Gridiron Lab")
view = st.sidebar.radio("Explore", ["Weekly forecasts", "Team Explorer", "Season performance", "Prediction history", "Methodology"], key="view")
status = data_status(source, season_games, now())
selection_label = "All weeks" if week == "All weeks" else f"Week {week}"
st.caption(f"{selection_label} | {'STALE SNAPSHOT' if status['stale'] else 'SOURCE CURRENT'} | Last successful refresh: {status['refresh']} | {status['pending']} results pending")

@st.cache_data(ttl=300, max_entries=8)
def provisional_batch(games, source, model_path, model_stamp):
    try:
        bundle = joblib.load(model_path) if model_stamp else None
    except (OSError, ValueError, ImportError, EOFError):
        bundle = None
    generated = now()
    return {g.game_id: forecast(games, g, source, bundle, generated)
            for _, g in games.loc[games.season.eq(2026) & games.status.isin(["scheduled", "kickoff_unavailable"])].iterrows()}

model_path = STORE / "model.joblib"
provisional = provisional_batch(games, source, str(model_path), model_path.stat().st_mtime_ns if model_path.exists() else 0) if view in ("Weekly forecasts", "Team Explorer") else {}
try:
    saved = records()
    # Recompute in memory from the latest snapshot; page loads never write files.
    results, live = reconciliation(games)
    results = [row for row in results if row.get("season") == 2026]
except (OSError, ValueError) as exc:
    st.warning(f"Prediction history unavailable: {exc}")
    saved, results, live = [], [], {}
    provisional = {gid: {"kind": "unavailable", "reason": "Recorded history could not be read. Estimates are withheld until the archive is reviewed."}
                   for gid in provisional}
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

rows = matchup_rows(season_games, first, historical, provisional, now(), status["stale"])
team_options = sorted(NAMES, key=lambda t: NAMES[t])


def matchup_card(row):
    with st.container(border=True):
        left, right = st.columns([3, 2])
        left.write(f"**{display(row['away_team'])} at {display(row['home_team'])}**")
        left.caption(f"{kickoff_label(row['kickoff'])} | {row['status_label']}")
        if row["status"] == "completed" and pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            left.write(f"Final: **{row['away_team']} {row['away_score']:g} - {row['home_team']} {row['home_score']:g}**")
        record = row["forecast"]
        kind = row["forecast_type"]
        right.badge(row["provenance"], color={"live": "green", "retrospective_backtest": "violet", "provisional": "orange", "unavailable": "gray"}[kind])
        if kind != "unavailable":
            right.write(f"Favored: **{row['winner']} | {row['confidence']:.0%}**")
            result = record.get("result", "pending")
            if result in ("correct", "incorrect"):
                label = "Backtest" if kind == "retrospective_backtest" else "Recorded pick"
                color = "green" if result == "correct" else "red"
                right.markdown(f":{color}[{label} {result}]")
            elif result == "tie":
                right.caption("Tie - excluded from accuracy")
            if kind == "provisional":
                right.caption(f"Estimate generated {kickoff_label(record['generated_at'])} | Excluded from accuracy")
        else:
            right.info(f"Unavailable: {row['reason']}")
        with st.expander("Advanced forecast details"):
            if kind == "retrospective_backtest":
                st.write("Reconstructed after the game. This was not a recorded historical pregame pick.")
            if record:
                st.json({key: value for key, value in record.items() if key != "features"})
            else:
                st.write("No forecast artifact exists for this matchup.")


def clear_weekly_filters():
    for key, value in {"weekly_search": "", "weekly_team": None, "weekly_status": [],
                       "weekly_type": [], "weekly_winner": None,
                       "weekly_confidence": "Any confidence", "weekly_sort": "Kickoff"}.items():
        st.session_state[key] = value


if view == "Weekly forecasts":
    week_filter = None if week == "All weeks" else week
    selected = filter_matchups(rows, week=week_filter)
    heading = "All weeks" if week_filter is None else f"Week {week}"
    st.subheader(f"{heading} \u00b7 {len(selected)} games")
    st.caption("Find a matchup, see the favorite, then check how the forecast was made. Retrospective backtests were generated after the games; provisional estimates are excluded from accuracy.")
    query = st.text_input("Search team or matchup", placeholder="Buffalo, BUF, or MIA @ BUF", key="weekly_search")
    a, b, c = st.columns(3)
    team = a.selectbox("Team", [None, *team_options], format_func=lambda t: display(t) if t else "All teams", key="weekly_team")
    statuses = b.multiselect("Game status", list(STATUSES), format_func=lambda value: STATUSES[value], key="weekly_status")
    types = c.multiselect("Forecast type", list(LABELS), format_func=lambda value: LABELS[value], key="weekly_type")
    a, b, c = st.columns(3)
    winner = a.selectbox("Predicted winner", [None, *team_options], format_func=lambda t: display(t) if t else "Any winner", key="weekly_winner")
    confidence = b.selectbox("Confidence", ["Any confidence", "At least 60%", "At least 70%", "At least 80%", "At least 90%"], key="weekly_confidence")
    order = c.selectbox("Sort games", ["Kickoff", "Confidence: high to low", "Confidence: low to high", "Matchup A-Z"], key="weekly_sort")
    st.button("Clear list filters", on_click=clear_weekly_filters, key="weekly_clear")
    threshold = None if confidence == "Any confidence" else (int(confidence.split()[2].rstrip("%"))/100, 1.0)
    visible = filter_matchups(rows, week=week_filter, team=team, query=query, statuses=statuses,
                              forecast_types=types, winner=winner, confidence=threshold, sort=order)
    st.caption(f"Showing {len(visible)} of {len(selected)} games | Confidence means the favored team's win probability. Unavailable forecasts have no confidence.")
    if not visible:
        st.info("No games match these filters. Clear search or widen the team, status, forecast, winner or confidence filters; try All weeks for the full schedule.")
    for row in visible:
        matchup_card(row)
    with st.expander("Advanced: schedule and forecast coverage"):
        st.dataframe(coverage_table if week_filter is None else coverage_table.loc[coverage_table.week.eq(week_filter)], hide_index=True, alt="Schedule and prediction coverage")

if view == "Team Explorer":
    st.subheader("Team Explorer")
    team = st.selectbox("Choose a team", team_options, format_func=display, key="explorer_team")
    schedule = team_schedule(rows, team)
    actual = team_record(schedule)
    st.write(f"**{NAMES[team]} | {actual['wins']}-{actual['losses']}-{actual['ties']} W-L-T**")
    st.caption(f"Actual record from completed scores in this snapshot. {len(schedule)} of 17 regular-season matchups available. Scores below are from the selected team's perspective. The weekly filter does not limit this schedule.")
    if len(schedule) != 17:
        st.warning("Incomplete team schedule in the local snapshot. Missing matchups are not invented.")
    if st.checkbox("Show estimated remaining wins", key="explorer_estimate"):
        estimate = remaining_wins_estimate(schedule, team)
        if len(schedule) == 17 and estimate["remaining"] == 0:
            st.info("No remaining matchups to estimate. The actual record is shown above.")
        elif len(schedule) == 17 and estimate["estimated_wins"] is not None:
            st.metric("Estimated remaining wins", f"{estimate['estimated_wins']:.1f}")
            st.caption(f"Estimate only: sum of win probabilities across {estimate['covered']} remaining matchups, using available recorded and provisional forecasts. Not an actual record or an official accuracy metric. No intervening games are simulated.")
        else:
            st.info(f"Estimate unavailable: {estimate['covered']} of {estimate['remaining']} uncompleted matchups have eligible forecasts. A complete 17-game schedule and forecasts for every remaining matchup are required; missing probabilities are never filled in.")
    table = []
    for row in schedule:
        probability = f"{row['confidence']:.0%}" if row["confidence"] is not None else "Unavailable"
        record = row["forecast"] or {}
        table.append({"Week": row["week"], "Opponent": display(row["opponent"]), "Venue": row["venue"],
                      "Kickoff (ET)": kickoff_label(row["kickoff"]), "Status": row["status_label"],
                      "Actual result": f"{row['actual_result']} {row['score']}" if row["actual_result"] else "Not final",
                      "Favored team": row["winner"] or "Unavailable", "Favored win probability": probability,
                      "Forecast provenance": row["provenance"],
                      "Estimate generated (ET)": kickoff_label(record.get("generated_at")) if row["forecast_type"] == "provisional" else None})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", alt=f"Complete local regular-season schedule and forecast provenance for {NAMES[team]}")
    st.caption("Retrospective backtests were reconstructed after the game and are never historical pregame picks. Provisional forecasts are estimates excluded from official accuracy. Missing forecasts do not remove games from the schedule.")
    with st.expander("Advanced: unavailable forecasts and provenance"):
        for row in schedule:
            st.write(f"Week {row['week']}: {row['provenance']}")
            if row["forecast_type"] == "unavailable":
                st.caption(row["reason"])
            if row["forecast"]:
                st.json({key: value for key, value in row["forecast"].items() if key != "features"})

if view == "Season performance":
    st.subheader("2026 season performance · separate cohorts")
    for title, cohort in (("Recorded pregame forecasts", results), ("Retrospective backtests · frozen pre-2026 model", retro_results)):
        st.write(f"**{title}**")
        summary = score_records(cohort)
        with st.container(horizontal=True):
            st.metric("Accuracy", f"{summary['accuracy']:.1%}" if summary["accuracy"] is not None else "Awaiting results", border=True)
            st.metric("Correct / incorrect", f"{summary['correct']} / {summary['incorrect']}", border=True)
            st.metric("Scored games", summary["games"], border=True)
        with st.expander(f"Advanced performance analysis | {title}"):
            st.caption(f"Brier: {summary['brier'] if summary['brier'] is not None else 'unavailable'} · "
                       f"Log loss: {summary['log_loss'] if summary['log_loss'] is not None else 'unavailable'} · "
                       f"Ties excluded: {summary['ties']} · Unavailable: {summary['unavailable']}")
            weekly_metrics = pd.DataFrame(weekly_performance(cohort)).set_index("week")
            if summary["games"]:
                last_scored = weekly_metrics.loc[weekly_metrics.games.gt(0)].index.max()
                chart_data = weekly_metrics.loc[:last_scored].copy()
                chart_data["cumulative_home"] = (chart_data.home_baseline_accuracy.fillna(0) * chart_data.games).cumsum() / chart_data.cumulative_games.replace(0, float("nan"))
                for chart_title, columns in (("Cumulative accuracy", {"cumulative_accuracy": "Model", "cumulative_home": "Always home"}),
                                             ("Weekly accuracy", {"accuracy": "Model", "home_baseline_accuracy": "Always home"})):
                    st.caption(chart_title)
                    plot = chart_data[list(columns)].rename(columns=columns).reset_index().melt("week", var_name="Series", value_name="Accuracy").dropna()
                    chart = cast(alt.Chart, alt.Chart(plot).mark_line(point=True)).encode(
                        x=alt.X("week:O", title="Regular season week"),
                        y=alt.Y("Accuracy:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Series:N", scale=alt.Scale(domain=["Model", "Always home"], range=["#50d890", "#8ca4bc"])),
                        tooltip=["week:O", "Series:N", alt.Tooltip("Accuracy:Q", format=".1%")]).properties(height=230)
                    st.altair_chart(chart, width="stretch", alt=f"{title}: {chart_title.lower()} against the home-team baseline")
            else:
                st.info("No eligible scored games in this cohort yet. Accuracy appears after a valid prediction has a non-tied, eligible final result.")
            valid = [r["home_probability"] for r in cohort if not r.get("integrity_error")]
            if valid:
                distribution = pd.Series(pd.cut(valid, bins=[i / 10 for i in range(11)], include_lowest=True)).value_counts(sort=False)
                distribution.index = distribution.index.astype(str)
                st.bar_chart(distribution, alt=f"{title}: home-win probability distribution, including pending recorded games")
            with st.expander(f"Weekly evaluation details · {title}"):
                st.dataframe(weekly_metrics, alt=f"Weekly metrics for {title}")
    st.caption("Cohorts are never combined. Retrospective picks were generated after the games using historical cutoffs, not archived pregame timestamps. Baselines use the exact same scored non-tied games in each cohort.")
    with st.expander("Advanced: season data coverage"):
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

if view == "Prediction history":
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
        table.append({"Week": row.get("week"), "Matchup": matchup, "Prediction": row.get("predicted_winner"), "Actual": row["actual"], "Result": f"{indicator} {row['result']}", "Type": LABELS.get(row["kind"], row["kind"])})
    if not table:
        st.info("No predictions match these filters. Recorded picks appear only after the pipeline archives an eligible pregame forecast.")
    else:
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    st.caption("First recorded pregame pick per game is scored. All revisions are retained below.")
    with st.expander("Versioned prediction records"):
        st.dataframe(pd.DataFrame(saved), width="stretch")

if view == "Methodology":
    st.info("Provisional forecasts use the saved production model and only observed season-to-date results available in the source snapshot. Intervening future games are not simulated. Estimates are cached in memory for up to five minutes, carry their actual generation time, and never enter recorded history or official metrics. TBD kickoff estimates require a published future game date. Stale sources fail closed.")
    st.subheader("Standardized logistic regression · V1 A–D")
    st.write(meta.get("policy", "Model not yet trained."))
    with st.expander("Advanced: production model metadata"):
        st.write("Features", meta.get("features", []))
        st.write("Training seasons", meta.get("training_seasons", []))
        st.write("Training cutoff", meta.get("training_cutoff", "unavailable"))
    st.write("Limitations: season-to-date statistics reset each year; Week 1 uses .5 win percentage and zero scoring/streak values. Ties count as half wins in features but are excluded from binary evaluation. No injuries, roster changes, betting lines or neutral-site adjustment. Probabilities have diagnostics, not a calibration correction.")
    st.write("The schedule has no final whistle timestamp. A conservative 12-hour post-kickoff embargo is used for feature availability and result reconciliation. Historical backtests assume final results were available after that embargo; live predictions use only the fetched snapshot.")
    st.write("2026 retrospective evaluation uses a separate frozen V1 A model fitted only on 2023–2025 regular seasons with a 2026-02-01 training cutoff. Model fitting and prediction generation happen now, not before those games. Each record keeps its actual generation time separately from its historical feature cutoff. Latest final-score reconstruction is not an archived source replay.")
    with st.expander("Frozen retrospective model provenance"):
        st.json(read("retrospective/model.json", {}))
    st.link_button("nflverse schedule documentation", "https://nflreadr.nflverse.com/reference/load_schedules.html")
