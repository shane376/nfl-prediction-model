# NFL prediction model · 2026 dashboard

Local Streamlit dashboard built on the existing V1 logistic regression models. Represents the entire 2026 regular season with real nflverse schedules, saved pregame probabilities, separate 2026 retrospective backtests, a separate 2025 holdout, and persistent prediction history. No retraining or prediction writes occur on dashboard page loads.

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
.venv/Scripts/python.exe -m src.pipeline audit --offline
.venv/Scripts/python.exe -m src.pipeline backtest --offline
.venv/Scripts/python.exe -m src.pipeline coverage --offline
.venv/Scripts/python.exe -m pytest -q
```

`refresh` downloads the [public nflverse schedule](https://nflreadr.nflverse.com/reference/load_schedules.html) from [nfldata games.csv](https://github.com/nflverse/nfldata/blob/master/data/games.csv). Supports regular seasons 2023–2026, UTC kickoffs converted from source Eastern time, normalized abbreviations, scores, and inferred scheduled/completed status. Missing kickoff times remain unavailable. Byes do not create games. A source error preserves the last successful schedule, records the error, marks it stale, and exits nonzero. Fetches older than 24 hours are also stale. Fetch timestamp measures retrieval freshness, not a guarantee that the upstream provider has updated every result.

`train` is explicit. It builds shared pregame features, selects A–D by **2024 validation log loss after fitting 2023**, evaluates on **2025**, then separately refits the selected production model using all eligible completed games through the current cutoff. C=1 and decision threshold=.5 are fixed; the 2025 test is never used for selection. Training on a cached source is supported but its provenance is retained in metadata. This is a pragmatic reused historical holdout: the original scripts previously experimented on 2025, so it is not an untouched research test set.

`predict --week N` records every eligible upcoming game in that week. Without `--week`, it records upcoming kickoffs within seven days, so automation covers Thursday, Sunday, Monday and exceptional dates. It requires a successful source fetch within 24 hours, never a future-dated snapshot. Repeated commands skip games already recorded. `--rerun` appends a revision before kickoff. A create-only atomic write prevents replacing any original, including after a revision-number collision. A per-store writer lock rejects concurrent prediction runs. Computation and disk flush are followed by clock checks with a 30-second kickoff safety margin.

Once the earliest kickoff seen in the store, the recorded kickoff, or the current kickoff has passed, a new prediction is prohibited. Refresh preserves the earliest known kickoff in `prediction_lock_at`; a later postponement cannot reopen that window. This conservative rule may omit legitimate postponed games, which is preferable to generating a pick after an uncertain start. Unknown/naive-time/underway/completed/postponed/partially scored games cannot receive a new pick. Every record includes model version, generation time, feature-data cutoff, training cutoff and source fetch time. Refresh also rejects truncated snapshots that drop cached game IDs; upstream ID changes require manual review.

`evaluate` recomputes live metrics from the **first recorded live revision per game** and completed results. It validates timezone-aware timestamps (`training cutoff <= data cutoff <= generation < recorded/latest/earliest-known kickoff`), source provenance, matchup, winner, and finite complementary probabilities. Missing/uncertain kickoffs, changed matchups and invalid metadata become explicitly unavailable, without promoting a later revision over the original. Ties are shown separately and excluded from binary accuracy/log loss/Brier. Missing scores remain pending. No retrospective pick is inserted for an already-started game.

`audit` is read-only: it validates every saved revision against the latest source schedule and exits nonzero on invalid or unavailable records. The provider supplies published kickoff times, not independent verified actual-start timestamps; this audit cannot prove an actual start time the provider does not publish. Git publication times provide an additional check for the existing records, but are not signed external timestamp attestations.

## V1.1 · Season coverage and retrospective evaluation

Missing forecasts do not mean missing schedule data. The initial snapshot already contained all 272 games, but only Weeks 9–10 had saved forecasts because those were the weeks explicitly recorded after the MVP was created. Earlier games could not receive an authentic pick after kickoff. `SEASON_COVERAGE_BEFORE.md` preserves the required pre-implementation audit; `V1_1_REPORT.md` contains the final week-by-week coverage and separate performance tables.

`coverage` prints a read-only JSON report for Weeks 1–18, including schedule rows, completed results, ties, postponed/canceled games, authentic/retrospective records, missing past/future authentic picks, missing eligible retrospective backtests, result embargo counts and missing required data. It also checks the full season has 272 rows, 32 known teams and 17 scheduled appearances per team. Unknown teams, duplicate IDs/matchups, teams playing simultaneous games, bad weeks, invalid scores and uncertain timestamps are guarded during ingestion. Cancellation and suspension are preserved separately from postponement when supplied by the source; a missing upstream game ID fails refresh rather than inventing cancellation information.

`backtest` adds only completed 2026 games **without any archived prediction record** and past the result embargo. It creates/reuses a separate frozen V1 **A** artifact trained on **2023–2025 regular seasons only**, with a **2026-02-01 UTC training cutoff**. This includes the January 2026 games belonging to the 2025 season; it excludes every 2026-season result. A was fixed from the existing pre-sprint policy; no 2026 outcomes select features or hyperparameters. C=1 and the decision threshold=.5 remain fixed. The production model is not used or modified for retrospective evaluation.

The artifact is reconstructed and fitted **now**, then frozen; its training-data cutoff is not a claim that the file existed before 2026. Every new record has `kind: retrospective_backtest`, its actual `generated_at`, separate historical `feature_cutoff` equal to that game's kickoff, model generation time/version, training seasons/cutoff and the feature values used. Target-game and future results are excluded. No original timestamp is invented, and no archived pick is overwritten or reclassified. Repeated runs skip existing immutable retrospective files. The frozen model is reused without retraining on refresh.

Retrospective features reconstruct season-to-date information from the latest final-score snapshot, applying the same 12-hour post-kickoff availability assumption as training. They are **not point-in-time archived source replays**; later score corrections cannot be proven known at the original cutoff. Changed historical feature inputs or kickoffs are flagged unavailable at reconciliation and require review, rather than silently rewriting an old backtest. A newly imported authentic archive takes precedence over an existing retrospective entry, which remains on disk but is excluded from retrospective metrics.

`evaluate` now saves `artifacts/season-report.json` with separate recorded and retrospective summaries and weekly/cumulative metrics. The dashboard recomputes these cohorts read-only. Each gets its own accuracy, correct/incorrect counts, home baseline on the same non-tied games, Brier score, log loss and probability distribution. Ties are displayed and excluded from binary scoring. Pending forecasts never enter accuracy. The separate 2025 holdout remains under an expandable section. Future weeks stay visible without invented forecasts or results.

New artifacts:

| Path | Meaning |
|---|---|
| `artifacts/retrospective/model.joblib`, `model.json` | Frozen prior-season model and provenance; separate from production |
| `artifacts/retrospective/predictions/*.json` | Immutable, explicitly retrospective 2026 predictions |
| `artifacts/season-report.json` | Timestamped coverage and separate cohort metrics |

To initialize a new clone without those artifacts, explicitly run `backtest --offline`, then `evaluate --offline`. Normal deployments use the reviewed checked-in frozen model. An incomplete or stale source remains visibly labeled; retrospective records retain its freshness flag. No retrospective computation occurs during dashboard page loads beyond read-only validation/scoring of saved records.

`--offline` on train/evaluate/audit reads the committed snapshot without a network call. Live `predict --offline` is intentionally rejected to prevent fixtures from becoming live picks; normal `predict` already reads the cached source without networking. Tests use explicitly synthetic deterministic fixtures in `tests/conftest.py`, isolated in temporary storage. `NFL_STORE` can select another artifact directory; set it before starting Python. Use a single writer per store. If a process crashes while holding `artifacts/.prediction-writer.lock`, confirm no prediction writer is running before manually removing only that lock. Never delete or rewrite prediction JSON files to recover a run. Create-only publication requires a filesystem supporting hard links (NTFS on Windows, standard GitHub Linux runner filesystems).

To verify CLI changes without touching authentic artifacts, copy `artifacts/` to a new directory beneath `.review/`, set `$env:NFL_STORE` to its absolute path, and run refresh/train/predict/evaluate there. Remove the environment variable afterward (`Remove-Item Env:NFL_STORE`). Tests already isolate their storage. Do not commit verification-only picks.

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

The dashboard uses the same pure reconciliation function to compute live results in memory against its loaded schedule; this avoids displaying obsolete derived metrics and never writes on page loads. Missing artifacts and malformed prediction files produce unavailable states. `evaluate` persists these derived views for automation and external consumers.

Team statistics reset each season. Without prior games, win percentage=.5 and scoring/streak values=0, matching the legacy convention. Ties count as half wins and reset streaks. A game's own outcome never enters its features. The source has no completion timestamp, so scores become feature-eligible **12 hours after kickoff** and only if present in the loaded snapshot. This conservative embargo deliberately omits some recent finished games. Historical backtests assume final scores were known after that embargo; they use the latest final values, including any later corrections, and are not an archive of historically fetched snapshots. Exceptional games lasting over 12 hours cannot be precisely reconstructed from this source. The dashboard states this limitation.

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

GitHub Actions runs **Refresh pregame predictions** from `main` at minutes 07 and 37 UTC every hour, on pushes to `main`, and on manual dispatch. GitHub can delay or skip scheduled runs. Each run starts from the durable `automation/nfl-artifacts` history, integrates reviewed code from `main`, runs tests, fetches nflverse results, records eligible pregame picks within seven days of kickoff, fills the separate retrospective cohort, and reconciles results. Post-generation validation checks source freshness, schedule completeness, prediction history, frozen models, and reconciliation before allowing a normal non-force push to the production branch.

Deploy Streamlit Community Cloud from **`automation/nfl-artifacts`**, with entry point **`dashboard/app.py`** and **Python 3.12**. Community Cloud watches the deployment branch for Git updates. Code still goes through PR review on `main`. A successful refresh publishes directly to the deployment branch, so recurring artifact PRs are no longer needed. Keep that branch and restrict manual writes, force pushes and deletion. The workflow needs the built-in GitHub token with `contents: write` and permission to push to this branch. The nflverse feed and the public Actions status endpoint require no credentials. [Streamlit deployment and Git updates](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/edit-your-app).

After the fix is reviewed and merged, wait for a successful refresh on `main` before binding production to the artifact branch. If an existing app's branch cannot be changed, stage a deployment on the artifact branch and plan the URL cutover with the app owner. Set `NFL_MONITOR_REFRESH = "1"` as a top-level Streamlit secret or host environment variable to enable the public Actions status check. Verify the deployed URL, latest snapshot timestamp, default week, forecast timestamps and Team Explorer. Leave the old artifact PR open until the cutover is verified, then close it. Keep code changes on `main` and append-only forecast history on the artifact branch.

The dashboard shows **Last Updated** using the published source retrieval time in Eastern time. A successful run with unchanged results and predictions retains the existing snapshot for up to 12 hours before publishing a fresh source timestamp. The dashboard flags snapshots older than 24 hours and pauses provisional forecasts while stale. With monitoring enabled, it separately flags failed runs, overdue successes, and unavailable status. The status request is cached for five minutes. Pending result counts measure source coverage, since a fresh fetch can still lack a final score.

On source, test, validation, merge, or push failure, the last published snapshot remains available. Inspect the [refresh workflow](https://github.com/shane376/nfl-prediction-model/actions/workflows/refresh.yml), job summary and diagnostic artifacts, retained for up to 90 days. Resolve conflicts while preserving original forecast files and reviewed model artifacts, then dispatch a retry from `main`.

## Limits and next sprint

No injury, roster, rest, travel, odds or neutral-site features; season-to-date cold starts; binary probabilities do not model ties. The data source exposes final scores rather than a live game-status API; unscored started games are unavailable for new predictions, not falsely treated as completed. Postponements use updated source kickoff times; prior records are retained and checked against revised kickoffs at evaluation.

Next sprint: final-status/completion timestamps with archived source snapshots, monitor freshness and scheduled-run failures, then evaluate additional features on a new forward validation window. Keep live picks separate from historical model research.
