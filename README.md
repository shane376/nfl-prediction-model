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

`.github/workflows/ci.yml` runs deterministic offline unit/integration/UI tests, audits checked-in records, and rejects changes/deletions of original prediction files in PRs. `.github/workflows/refresh.yml` runs every 30 minutes **from the default branch**, expected to be `main`, after approved integration. Feature-branch dispatch is intentionally skipped. No obsolete feature-branch push target remains.

The refresh job resumes the long-lived `automation/nfl-artifacts` branch, merges reviewed default-branch code, tests, refreshes, records upcoming pregame picks, backfills missing retrospective games and evaluates. It requires both reviewed production and frozen retrospective models; the backtest step reuses the frozen artifact. It checks that original predictions are unchanged before and after processing, then makes a normal (never force) push **only to the artifact branch** and opens/reuses a PR into the default branch. It never pushes directly to `main`, approves/merges a PR, trains a production model implicitly, or deploys. **Public dashboard updates require review and merge of the artifact PR**; unmerged new picks remain durably recorded in Git on the artifact branch. Review its latest commit, not an earlier snapshot. Keep the artifact branch and use merge commits for recurring PRs so Git ancestry remains intact; squash/rebase merges or deleting the branch require checking all unmerged prediction records have been preserved before recovery.

Repository settings required after approval:

- Enable Actions, allow the workflow's `contents: write` and `pull-requests: write` scopes, and enable **Allow GitHub Actions to create and approve pull requests**. This workflow only creates PRs; it does not approve them. No PAT is required.
- Protect `main` with review requirements and do not grant the bot a protection bypass. Permit normal bot pushes to `automation/nfl-artifacts`; do not auto-delete it. Do not enable auto-merge for the artifact PR.
- `GITHUB_TOKEN`-created pushes/PRs generally do not trigger another workflow. The refresh job therefore runs pytest itself. If branch protections require a CI check on the PR head, manually run the **CI** workflow using its `workflow_dispatch` on `automation/nfl-artifacts`, then review/merge. Do not bypass a missing required check.
- Schedules only start once the workflow is on the default branch; they can be delayed, and inactive public-repository schedules may be disabled. Enable failure notifications and inspect failed runs. There is no punctuality guarantee; late predictions are skipped.

On source/validation/test failure the job stops before pushing artifacts; previous Git records remain. Diagnostics are uploaded for up to 90 days (subject to repository retention limits). Source failure metadata may exist only in those diagnostics; the deployed dashboard continues showing its last reviewed snapshot and flags it stale after 24 hours. Merge conflicts, a rejected push, missing model, history-guard rejection or PR-permission errors fail visibly rather than resetting or force-pushing history. Resolve conflicts by preserving both sets of original prediction files, rerun tests/audit/history checks, then resume. If a push succeeds but PR creation fails, the picks are already durable on the artifact branch; fix permissions and rerun or open the PR manually. Never replace the branch with `main` while it contains unmerged picks.

For Streamlit Community Cloud:

1. Review the pushed `feature/v1.1-season-coverage` changes, CI and `V1_1_REPORT.md`; obtain approval before merging into `main` or deploying.
2. After approval, create a Community Cloud app for `shane376/nfl-prediction-model`, select `main` (or an explicitly approved staging branch), set entrypoint **`dashboard/app.py`**, open Advanced settings and choose **Python 3.12**, then deploy.
3. The root `requirements.txt` installs the tested lockfile automatically. Confirm both `artifacts/model.joblib` and `artifacts/retrospective/model.joblib`, the archived picks and retrospective records are tracked. Joblib models require matching scikit-learn versions; update the lockfile whenever you deliberately retrain with changed dependencies. V1.1 adds no package or framework requirement.
4. Verify the deployed URL, selected week, model metadata and freshness before sharing it.
5. Apply the Actions permissions above, dispatch **Refresh pregame predictions** from `main`, inspect its artifact branch/PR and run CI on the PR head if needed. Review and merge the artifact PR to publish the latest data.
6. Verify all four tabs, current-week kickoffs in Eastern time, recorded model versions and source freshness at the actual public URL. Confirm the next scheduled run succeeds and newly recorded picks precede kickoff. Do not claim deployment complete before that URL check.

Durability comes from versioned Git snapshots, not Community Cloud's ephemeral disk. The hosted dashboard only reads committed artifacts; Actions owns updates. Repo updates trigger hosted refresh/redeployment. Do not store live prediction writes only on the hosted filesystem. No paid infrastructure or credentials are required for public nflverse data. No public deployment has been performed or verified.

Compatibility was verified in a fresh Python 3.12 environment using root `requirements.txt`, the checked-in joblib model and `dashboard/app.py`; no external database, OS package or Streamlit secret is needed. Community Cloud installs the root requirements file even though the entrypoint is under `dashboard/`. See [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency placement](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [GitHub scheduled-event behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule) and [workflow permissions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository).

## Limits and next sprint

No injury, roster, rest, travel, odds or neutral-site features; season-to-date cold starts; binary probabilities do not model ties. The data source exposes final scores rather than a live game-status API; unscored started games are unavailable for new predictions, not falsely treated as completed. Postponements use updated source kickoff times; prior records are retained and checked against revised kickoffs at evaluation.

Next sprint: final-status/completion timestamps with archived source snapshots, monitor freshness and scheduled-run failures, then evaluate additional features on a new forward validation window. Keep live picks separate from historical model research.
