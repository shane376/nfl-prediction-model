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
                              team_record, remaining_wins_estimate, adjacent_week, football_display,
                              matchup_html, schedule_html)

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


st.html(Path(__file__).with_name("style.css"))
header = st.container(horizontal=True, vertical_alignment="center", gap="small")
header.markdown("### Gridiron Lab", width="content")
header.caption("2026 NFL", width="content")
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
season_games = games.loc[games.season.eq(2026)]
if len(season_games) != 272:
    st.warning(f"Incomplete schedule snapshot: {len(season_games)} of 272 expected regular-season games. Missing schedule rows are separate from missing forecasts.")
if season_games.empty:
    st.warning("The source snapshot contains no 2026 games. Refresh when the schedule is available; no games have been invented.")
default_week = choose_week(season_games, now())
if "week" not in st.session_state:
    st.session_state["week"] = default_week
st.sidebar.subheader("Gridiron Lab")
week = st.sidebar.selectbox("Regular season week", [*range(1, 19), "All weeks"], index=None, required=True, key="week")
view = st.sidebar.radio("Explore", ["Weekly forecasts", "Team Explorer", "Season performance", "Prediction history", "Methodology"], format_func=lambda name: "Weekly predictions" if name == "Weekly forecasts" else name, key="view")
status = data_status(source, season_games, now())
header.badge("Stale source" if status["stale"] else "Source current",
             color="orange" if status["stale"] else "green",
             help=f"Last successful refresh: {status['refresh']}. {status['pending']} results pending. Local snapshot only.")
with st.sidebar.expander("Data & forecast guide"):
    st.caption(f"V1.2.2 | Last successful refresh: {status['refresh']} | {status['pending']} results pending")
    if status["stale"]:
        st.warning(f"STALE SOURCE SNAPSHOT: {source.get('error') or 'Last successful refresh was over 24 hours ago.'}")
    st.write("Recorded picks were archived before kickoff. Retrospective backtests were reconstructed after games. Provisional forecasts are estimates excluded from official accuracy.")
    st.caption("Kickoffs are Eastern Time. Byes have no game. Ties are excluded from prediction accuracy.")
    st.write("Recorded forecast model", meta.get("version", "unavailable"))
    st.json(source)

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
    content = football_display(row)
    kind = row["forecast_type"]
    record = row["forecast"] or {}
    with st.container(border=True, gap="xsmall", height="stretch", key=f"game_{row['game_id']}"):
        st.html(matchup_html(row))
        with st.container(horizontal=True, vertical_alignment="center", gap="xsmall"):
            st.badge(content["provenance"], color={"live": "green", "retrospective_backtest": "violet", "provisional": "orange", "unavailable": "gray"}[kind], help=LABELS[kind])
            if content["outcome"]:
                color = "green" if record["result"] == "correct" else "red"
                st.markdown(f":{color}[{content['outcome']}]")
            with st.popover("Details", type="tertiary", key=f"details_{row['game_id']}"):
                st.write(f"{display(row['away_team'])} at {display(row['home_team'])}")
                if kind == "retrospective_backtest":
                    st.info("Reconstructed after the game; not an archived pregame pick.")
                if kind == "unavailable":
                    st.info(f"Unavailable: {row['reason']}")
                if record:
                    st.json({key: value for key, value in record.items() if key != "features"})
                else:
                    st.write("No forecast artifact exists for this matchup.")
        if kind == "provisional":
            st.caption(f"Estimated {kickoff_label(record['generated_at'])} | Not scored")


def change_week(direction):
    st.session_state["week"] = adjacent_week(st.session_state["week"], direction)


def clear_weekly_filters():
    for key, value in {"weekly_search": "", "weekly_team": None, "weekly_status": [],
                       "weekly_type": [], "weekly_winner": None,
                       "weekly_confidence": "Any confidence", "weekly_sort": "Kickoff"}.items():
        st.session_state[key] = value


if view == "Weekly forecasts":
    week_filter = None if week == "All weeks" else week
    selected = filter_matchups(rows, week=week_filter)
    heading = "All weeks" if week_filter is None else f"Week {week}"
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.subheader(f"{heading} \u00b7 {len(selected)} games")
        with st.container(horizontal=True, width="content", wrap=False, gap="xsmall"):
            st.button("Previous", icon=":material/chevron_left:", on_click=change_week, args=(-1,), disabled=week == 1, key="previous_week")
            st.button("Next", icon=":material/chevron_right:", on_click=change_week, args=(1,), disabled=week == 18, key="next_week")
    with st.grid(2, min_column_width=240, gap="small", key="weekly_search_controls"):
        query = st.text_input("Search team or matchup", placeholder="Search: Buffalo, BUF or MIA @ BUF", key="weekly_search", label_visibility="collapsed")
        team = st.selectbox("Team", [None, *team_options], format_func=lambda t: display(t) if t else "All teams", key="weekly_team", label_visibility="collapsed")
    with st.expander("Advanced filters"):
        a, b = st.columns(2)
        statuses = a.multiselect("Game status", list(STATUSES), format_func=lambda value: STATUSES[value], key="weekly_status")
        types = b.multiselect("Forecast type", list(LABELS), format_func=lambda value: LABELS[value], key="weekly_type")
        a, b, c = st.columns(3)
        winner = a.selectbox("Predicted winner", [None, *team_options], format_func=lambda t: display(t) if t else "Any winner", key="weekly_winner")
        confidence = b.selectbox("Confidence", ["Any confidence", "At least 60%", "At least 70%", "At least 80%", "At least 90%"], key="weekly_confidence")
        order = c.selectbox("Sort games", ["Kickoff", "Confidence: high to low", "Confidence: low to high", "Matchup A-Z"], key="weekly_sort")
        st.caption("Confidence is the favored team's probability. Games without forecasts have no confidence.")
    threshold = None if confidence == "Any confidence" else (int(confidence.split()[2].rstrip("%"))/100, 1.0)
    visible = filter_matchups(rows, week=week_filter, team=team, query=query, statuses=statuses,
                              forecast_types=types, winner=winner, confidence=threshold, sort=order)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"Showing {len(visible)} of {len(selected)} games", width="stretch")
        st.button("Clear filters", on_click=clear_weekly_filters, key="weekly_clear", type="tertiary")
    if not visible:
        st.info("No games match. Clear filters, choose another team or week, or try All weeks.")
    with st.grid(2, min_column_width=380, gap="small", key="weekly_grid"):
        for row in visible:
            matchup_card(row)
    with st.expander("Advanced: schedule and forecast coverage"):
        st.dataframe(coverage_table if week_filter is None else coverage_table.loc[coverage_table.week.eq(week_filter)], hide_index=True, alt="Schedule and prediction coverage")

if view == "Team Explorer":
    st.subheader("Team Explorer")
    team = st.selectbox("Choose a team", team_options, format_func=display, key="explorer_team")
    schedule = team_schedule(rows, team)
    actual = team_record(schedule)
    played = sum(actual.values())
    remaining = sum(row["actual_result"] is None and row["status"] != "canceled" for row in schedule)
    with st.container(horizontal=True):
        st.metric("Actual W-L-T", f"{actual['wins']}-{actual['losses']}-{actual['ties']}", border=True, help="Completed scores in the local snapshot; independent of forecasts.")
        st.metric("Played", played, border=True)
        st.metric("Remaining", remaining, border=True)
    if len(schedule) != 17:
        st.warning("Incomplete team schedule in the local snapshot. Missing matchups are not invented.")
    with st.popover("Model estimate", type="tertiary"):
        if st.checkbox("Show estimated remaining wins", key="explorer_estimate"):
            estimate = remaining_wins_estimate(schedule, team)
            if len(schedule) == 17 and estimate["remaining"] == 0:
                st.info("No remaining matchups to estimate.")
            elif len(schedule) == 17 and estimate["estimated_wins"] is not None:
                st.metric("Estimated remaining wins", f"{estimate['estimated_wins']:.1f}")
                st.caption(f"Model estimate only: sum of probabilities for {estimate['covered']} remaining games. Not an actual record or an accuracy metric.")
            else:
                st.info(f"Estimate unavailable: {estimate['covered']} of {estimate['remaining']} uncompleted matchups have eligible forecasts. A full 17-game schedule and every remaining probability are required.")
    st.caption(f"{len(schedule)} regular-season matchups | Scores from {team}'s perspective | Eastern Time")
    st.html(schedule_html(schedule))
    with st.popover("Forecast guide & details", type="tertiary"):
        st.write("Recorded = archived pregame pick. Retrospective = reconstructed after the game. Provisional = model estimate, excluded from official accuracy.")
        if schedule:
            detail_id = st.selectbox("Game details", [row["game_id"] for row in schedule],
                                     format_func=lambda gid: next(f"Week {r['week']}: {r['opponent']}" for r in schedule if r['game_id'] == gid), key="explorer_details")
            detail = next(row for row in schedule if row["game_id"] == detail_id)
            if detail["forecast_type"] == "unavailable":
                st.info(detail["reason"])
            if detail["forecast"]:
                st.json({key: value for key, value in detail["forecast"].items() if key != "features"})

if view == "Season performance":
    st.subheader("2026 season performance · separate cohorts")
    for title, cohort in (("Recorded pregame forecasts", results), ("Retrospective backtests · frozen pre-2026 model", retro_results)):
        st.write(f"**{title}**")
        summary = score_records(cohort)
        with st.container(horizontal=True):
            st.metric("Accuracy", f"{summary['accuracy']:.1%}" if summary["accuracy"] is not None else "Awaiting results", border=True)
            st.metric("Correct predictions", summary["correct"], border=True)
            st.metric("Total evaluated", summary["games"], border=True)
        weekly_metrics = pd.DataFrame(weekly_performance(cohort)).set_index("week")
        if summary["games"]:
            st.caption("Weekly results")
            evaluated = weekly_metrics.loc[weekly_metrics.games.gt(0), ["correct", "incorrect"]].rename(columns={"correct": "Correct", "incorrect": "Incorrect"})
            st.bar_chart(evaluated, color=["#50d890", "#ffabab"], height=180, x_label="Week", y_label="Evaluated predictions", alt=f"{title}: correct and incorrect predictions by week")
        else:
            st.info("No eligible evaluated predictions in this cohort yet.")
        with st.expander(f"Technical metrics & model evaluation | {title}"):
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
