# NFL prediction model · 2026 dashboard

Local Streamlit dashboard built on the existing V1 logistic regression models. Displays real nflverse schedules, saved pregame probabilities, a separate 2025 historical backtest, and persistent live prediction history. No retraining or prediction writes occur on dashboard page loads.

## Run locally (PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

Open http://localhost:8501. The committed `artifacts/` snapshot and fitted model support immediate offline execution. On Linux/macOS substitute `.venv/bin/python` for `.venv/Scripts/python.exe`. `requirements.txt` includes `requirements.lock`, which records the tested environment, including the exact scikit-learn version required by the joblib artifact. Load only trusted repository joblib files.

## Pipeline

From the repository root:

```powershell
.venv/Scripts/python.exe -m src.pipeline refresh
.venv/Scripts/python.exe -m src.pipeline train
.venv/Scripts/python.exe -m src.pipeline predict --season 2026 --week 10
.venv/Scripts/python.exe -m src.pipeline evaluate
.venv/Scripts/python.exe -m pytest -q
```

`refresh` downloads the [public nflverse schedule](https://nflreadr.nflverse.com/reference/load_schedules.html) from [nfldata games.csv](https://github.com/nflverse/nfldata/blob/master/data/games.csv). Supports regular seasons 2023–2026, UTC kickoffs converted from source Eastern time, normalized abbreviations, scores, and inferred scheduled/completed status. Missing kickoff times remain unavailable. Byes do not create games. A source error preserves the last successful schedule, records the error, marks it stale, and exits nonzero. Fetches older than 24 hours are also stale. Fetch timestamp measures retrieval freshness, not a guarantee that the upstream provider has updated every result.

`train` is explicit. It builds shared pregame features, selects A–D by **2024 validation log loss after fitting 2023**, evaluates on **2025**, then separately refits the selected production model using all eligible completed games through the current cutoff. C=1 and decision threshold=.5 are fixed; the 2025 test is never used for selection. Training on a cached source is supported but its provenance is retained in metadata. This is a pragmatic reused historical holdout: the original scripts previously experimented on 2025, so it is not an untouched research test set.

`predict --week N` records every eligible upcoming game in that week. Without `--week`, it records upcoming kickoffs within seven days, so automation covers Thursday, Sunday, Monday and exceptional dates. Repeated commands skip games already recorded. `--rerun` appends a revision before kickoff. Original predictions remain immutable; once either the recorded kickoff or current kickoff has passed, a new revision is prohibited. Unknown/underway/completed games cannot receive a new pregame prediction. Every record includes model version, prediction time, data cutoff, training cutoff, source fetch time and stale-source flag.

`evaluate` recomputes live metrics from the **first authentic pregame record per game** and completed results. It checks changed kickoff times and excludes late records. Ties are shown separately and excluded from binary accuracy/log loss/Brier. Records with no completed result remain pending. No retrospective pick is inserted for an already-started game.

`--offline` on train/evaluate reads the committed snapshot without a network call. Live `predict --offline` is intentionally rejected to prevent fixtures from becoming live picks; normal `predict` already reads the cached source without networking. Tests use explicitly synthetic deterministic fixtures in `tests/conftest.py`, isolated in temporary storage. `NFL_STORE` can select another artifact directory; set it before starting Python. Use a single writer per store.

## Architecture and saved records

| Path | Purpose |
|---|---|
| `src/data.py` | Source retrieval, normalized schedule, atomic snapshot writes and freshness |
| `src/pregame.py` | Shared season-to-date pregame features, historical training rows |
| `src/modeling.py` | Reuses `train_v1.MODEL_FEATURES` and model builder; validation and production fit |
| `src/predictions.py` | Immutable versioned pregame JSON records and first-pick reconciliation |
| `src/pipeline.py` | CLI orchestration |
| `dashboard/app.py` | Read-only weekly, performance, searchable history and model information views |
| `artifacts/games.csv`, `source.json` | Public schedule snapshot and retrieval metadata |
| `artifacts/model.joblib`, `model.json` | Trusted fitted model and evaluation/training metadata |
| `artifacts/backtest.json` | Clearly retrospective 2025 prediction records |
| `artifacts/predictions/*.json` | Original live pregame records, retained across reruns |
| `artifacts/live-results.json`, `live-metrics.json` | Reproducible first-pick evaluation |

Team statistics reset each season. Without prior games, win percentage=.5 and scoring/streak values=0, matching the legacy convention. Ties count as half wins and reset streaks. A game's own outcome never enters its features. The source has no completion timestamp, so scores become feature-eligible **12 hours after kickoff** and only if present in the loaded snapshot. This conservative embargo deliberately omits some recent finished games. Historical backtests assume final scores were known after that embargo; they are not an archive of historically fetched snapshots. Exceptional games lasting over 12 hours cannot be precisely reconstructed from this source. The dashboard states this limitation.

## Evaluation (snapshot fetched November 5, 2026)

271 non-tied 2025 holdout games:

| Variant | Accuracy | Log loss | Brier |
|---|---:|---:|---:|
| A · selected on 2024 | 61.25% | 0.64434 | 0.22681 |
| B | 60.52% | 0.64874 | 0.22907 |
| C | 63.10% | 0.64060 | 0.22508 |
| D | 61.25% | 0.64985 | 0.22963 |
| Always home | 53.87% | — | — |
| Training home-rate probabilities | 53.87% | 0.69020 | 0.24853 |

Model A is retained despite C's better test accuracy: changing selection after observing 2025 would contaminate the test. Calibration-bin diagnostics are saved and displayed; no probability calibration correction is claimed. Live accuracy starts unavailable until genuine saved pregame picks resolve. Dashboard metrics come from artifacts, not these documentation figures.

The original training run is preserved in `baseline-results.txt`: A 61.62%, B 60.52%, C 63.10%, D 61.25%. Differences reflect the shared feature policy (ties and kickoff embargo). The original scripts and original ignored datasets are preserved.

The source labels November 5, 2026 as **Week 9**, rather than the sprint request's Week 10. The dashboard follows source kickoffs to choose its default; Week 10 remains available. No schedule dates or results were fabricated to match the request.

## Automation and deployment

`.github/workflows/ci.yml` runs deterministic offline unit/integration/UI tests. `.github/workflows/refresh.yml` refreshes every 30 minutes, records upcoming games, reconciles results and commits public artifacts to `feature/2026-dashboard-mvp`. It serializes writers, preserves earlier predictions, and uploads a diagnostic artifact even on failure. Failed refresh blocks prediction generation in that run. Training happens only if the artifact is absent; retraining is a separate explicit maintenance operation.

**GitHub scheduled workflows execute from the default branch.** This workflow will not run on a schedule until reviewed and merged into the default branch. After an approved merge, change its checkout/push branch from `feature/2026-dashboard-mvp` to the chosen deployment branch. Enable Actions and workflow content-write permissions. Branch protections may require a bot PR instead of direct pushes. Manual workflow dispatch also requires the workflow to be present on the default branch. Scheduled jobs can be delayed or disabled for inactivity; prediction timestamps always enforce kickoff eligibility. Monitor job failures; this free scheduler provides no punctuality guarantee.

For Streamlit Community Cloud:

1. Push this feature branch and public artifacts to your GitHub repository after review. Do not merge without approval.
2. Create a Community Cloud app for `shane376/nfl-prediction-model`, select the branch, set entrypoint `dashboard/app.py`, and choose Python 3.12.
3. The root `requirements.txt` installs the tested lockfile automatically. Joblib models require matching scikit-learn versions; update the lockfile whenever you deliberately retrain with changed dependencies.
4. Verify the deployed URL, selected week, model metadata and freshness before sharing it.
5. Enable the workflow after the approved default-branch integration and confirm its first artifact commit.

Durability comes from versioned Git snapshots, not Community Cloud's ephemeral disk. The hosted dashboard only reads committed artifacts; Actions owns updates. Repo updates trigger hosted refresh/redeployment. Do not store live prediction writes only on the hosted filesystem. No paid infrastructure or credentials are required for public nflverse data. No public deployment has been performed or verified.

## Limits and next sprint

No injury, roster, rest, travel, odds or neutral-site features; season-to-date cold starts; binary probabilities do not model ties. The data source exposes final scores rather than a live game-status API; unscored started games are unavailable for new predictions, not falsely treated as completed. Postponements use updated source kickoff times; prior records are retained and checked against revised kickoffs at evaluation.

Next sprint: final-status/completion timestamps with archived source snapshots, monitor freshness and scheduled-run failures, then evaluate additional features on a new forward validation window. Keep live picks separate from historical model research.
