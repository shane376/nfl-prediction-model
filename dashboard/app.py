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
st.caption(f"Latest successful fetch: {source.get('fetched_at', 'unknown')} · Model: {meta.get('version', 'not trained')}")
season_games = games.loc[games.season.eq(2026)]
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
first = {row["game_id"]: row for row in results}

with weekly:
    selected = season_games.loc[season_games.week.eq(week)]
    st.subheader(f"Week {week} · {len(selected)} games")
    if selected.empty:
        st.info("No games in this source snapshot for the selected week.")
    for _, game in selected.iterrows():
        with st.container(border=True):
            left, right = st.columns([3, 2])
            left.subheader(f"{game.away_team}  @  {game.home_team}")
            left.write(f"{display(game.away_team)} at {display(game.home_team)}")
            kickoff = game.kickoff.tz_convert("America/New_York").strftime("%a, %b %d · %I:%M %p %Z") if pd.notna(game.kickoff) else "Kickoff TBD"
            left.caption(f"{kickoff} · {game.status}")
            record = first.get(game.game_id)
            if record and not record.get("integrity_error"):
                p = record["home_probability"]
                right.write(f"Recorded pregame pick: **{record['predicted_winner']}**")
                right.progress(p, text=f"{game.home_team} {p:.1%} · {game.away_team} {1-p:.1%}")
                left.caption(f"{record['model_version']} · Recorded {record['generated_at']} · Data cutoff {record['data_cutoff']} · Source {record['source_freshness']}")
                if record.get("source_stale"):
                    right.warning("Pick generated from a stale snapshot")
            else:
                right.info(f"Unavailable · {record['integrity_error'] if record else 'no recorded pregame prediction'}")
            if record and not record.get("integrity_error") and pd.notna(game.kickoff) and game.kickoff != pd.Timestamp(record.get("kickoff")):
                left.caption("Kickoff changed since this pick was recorded. The original record is retained.")
            if game.status == "completed":
                left.write(f"Final score: {game.away_team} {game.away_score:g} — {game.home_team} {game.home_score:g}")

with performance:
    st.subheader("Historical backtest · 2025 holdout")
    comparisons = meta.get("holdout_2025", {})
    if comparisons:
        st.dataframe(pd.DataFrame(comparisons).T.drop(columns=["calibration"], errors="ignore"), width="stretch")
        st.caption("Retrospective predictions. Variants selected on 2024 validation, never 2025 results. Home baseline always picks the home team; the probability baseline uses the training home-win rate.")
        chosen = comparisons.get(meta.get("variant"), {})
        calibration = chosen.get("calibration", {})
        if calibration:
            st.write("Holdout calibration diagnostics")
            st.dataframe(pd.DataFrame(calibration), hide_index=True)
    else:
        st.info("Run training to create historical evaluation records.")
    st.subheader("Live pregame record")
    a, b, c = st.columns(3)
    a.metric("Live accuracy", f"{live['accuracy']:.1%}" if live.get("accuracy") is not None else "Awaiting results")
    b.metric("Correct picks", live.get("correct", 0))
    c.metric("Incorrect picks", live.get("incorrect", 0))
    st.write({key: live.get(key) for key in ("games", "log_loss", "brier")})
    scored = pd.DataFrame([r for r in results if r.get("target") is not None])
    if not scored.empty:
        scored["correct"] = scored.result.eq("correct").astype(int)
        st.line_chart(scored.groupby("week").correct.mean(), y_label="Live accuracy")
    backtest = pd.DataFrame(read("backtest.json", []))
    if not backtest.empty:
        backtest["correct"] = (backtest.home_probability.ge(.5).astype(int) == backtest.target).astype(int)
        st.write("Historical weekly accuracy (separate backtest)")
        st.line_chart(backtest.groupby("week").correct.mean())

with history:
    query = st.text_input("Search team or matchup").upper()
    status = st.multiselect("Result", ["correct", "incorrect", "pending", "tie", "unavailable"])
    table = []
    for row in results:
        matchup = f"{row.get('away_team', '?')} @ {row.get('home_team', '?')}"
        if query not in matchup or (status and row["result"] not in status):
            continue
        indicator = {"correct": "✓", "incorrect": "✕", "pending": "◷", "tie": "=", "unavailable": "—"}[row["result"]]
        table.append({"Week": row.get("week"), "Matchup": matchup, "Prediction": row.get("predicted_winner"), "Actual": row["actual"], "Result": f"{indicator} {row['result']}", "Recorded": row.get("generated_at"), "Integrity": row.get("integrity_error")})
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
    st.link_button("nflverse schedule documentation", "https://nflreadr.nflverse.com/reference/load_schedules.html")
