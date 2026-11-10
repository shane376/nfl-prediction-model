# V1.1 season-data completion · November 9, 2026

Branch: `feature/v1.1-season-coverage`. Prepared for review; no merge or deployment performed. The original 29 prediction JSON files are unchanged byte-for-byte. The production model and separate 2025 backtest are unchanged.

## Audit and ingestion

The pre-implementation report is preserved in `SEASON_COVERAGE_BEFORE.md`. It found a full 272-game schedule but only 29 recorded forecasts in Weeks 9–10, because those were the weeks explicitly recorded after launch. Earlier games could not legitimately receive original forecasts. There were no 2026 retrospective records. The November 5 snapshot had 121 completed results and was stale, with 14 passed kickoffs lacking results.

A successful refresh at **2026-11-09 22:29 UTC** supplied **135 completed results**. The schedule contains all **272 games**, all **32 teams**, and **17 scheduled appearances per team**. No missing schedule rows, missing required timestamps, missing results after kickoff, or missing eligible retrospective backtests remain at this snapshot. One game is tied. A passed kickoff without a score is reported separately from future unplayed games; it is never automatically assumed completed.

## Coverage after implementation

Games means schedule rows, including completed and future games. Missing data means required kickoff/result data, not absent forecasts. Future games do not require a retrospective backtest. Completed games with recorded picks do not get duplicate retrospective picks.

| Week | Games | Completed | Recorded | Retrospective | Missing results | Missing eligible backtests | Missing data |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 0 | 16 | 0 | 0 | 0 |
| 2 | 16 | 16 | 0 | 16 | 0 | 0 | 0 |
| 3 | 16 | 16 | 0 | 16 | 0 | 0 | 0 |
| 4 | 16 | 16 | 0 | 16 | 0 | 0 | 0 |
| 5 | 15 | 15 | 0 | 15 | 0 | 0 | 0 |
| 6 | 14 | 14 | 0 | 14 | 0 | 0 | 0 |
| 7 | 14 | 14 | 0 | 14 | 0 | 0 | 0 |
| 8 | 14 | 14 | 0 | 14 | 0 | 0 | 0 |
| 9 | 15 | 14 | 15 | 0 | 0 | 0 | 0 |
| 10 | 14 | 0 | 14 | 0 | 0 | 0 | 0 |
| 11 | 13 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 | 14 | 0 | 0 | 0 | 0 | 0 | 0 |
| 14 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| 15 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **272** | **135** | **29** | **121** | **0** | **0** | **0** |

Missing authentic predictions: **243**, comprising **121 historical gaps** now represented by clearly labeled retrospective records and **122 future games** without recorded picks yet. There are **137 future/unplayed games**, of which 15 already have archived pregame picks. No past gap was filled by backdating an original forecast.

## Performance · cohorts never combined

| Cohort | Records | Scored non-tied games | Correct | Incorrect | Accuracy | Always-home accuracy on same games | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Authentic recorded pregame | 29 | 14 | 12 | 2 | 85.71% | 35.71% | 0.18221 | 0.55466 |
| `retrospective_backtest` | 121 | 120 | 67 | 53 | 55.83% | 51.67% | 0.25691 | 0.71302 |

The retrospective tie is displayed but excluded from binary accuracy/Brier/log loss. Fifteen recorded picks are pending. The 14-game recorded sample is small and differs from the retrospective population; these figures are not an overall season accuracy or a controlled comparison of models. No calibration correction is claimed. The 2025 holdout remains separate and unchanged.

`artifacts/season-report.json` contains the actual timestamped coverage, weekly/cumulative accuracy and separate metrics. The dashboard computes the same results read-only and adds separate probability distributions. Reproduce with `python -m src.pipeline coverage --offline` or `evaluate --offline`.

## Frozen historical setup and persistence

The separate artifact **`v1A-pre2026-64572cfed775`** was fitted on **815 non-tied 2023–2025 regular-season games**, using fixed V1 A features, C=1 and threshold=.5. Training eligibility cutoff: **2026-02-01 UTC**, before all 2026-season kickoffs. This includes January 2026 games belonging to the 2025 season; no 2026-season labels enter fitting, feature selection or hyperparameter choice. A follows the existing pre-sprint variant policy, rather than being selected from 2026 results.

The model file was reconstructed on November 9, not falsely timestamped before the season. Each immutable retrospective record separately stores its actual generation time, historical feature cutoff at kickoff, frozen model generation time/version, prior training seasons/cutoff and exact feature values. Changing target or future outcomes does not change that game's feature vector or prediction. The target game is excluded from its own features. Existing frozen artifacts are reused, and reruns add zero duplicate records.

The existing conservative 12-hour result-availability assumption is retained. nflverse provides published kickoff times/final scores, not a complete archive of actual completion timestamps or historical corrections. These backtests reconstruct past information from the latest source; they are **not an independently verified point-in-time source replay**. Changed historical feature inputs or revised kickoffs are surfaced as unavailable and require review. Original archived records are never rewritten, reclassified or replaced by a retrospective result.

## Verification and delivery

- Full pytest suite: **72 passed** (53 existing checks plus new frozen-model, leakage, coverage, cancellation/tie, idempotence, archive-precedence, missing-model recovery, source-correction and 18-week dashboard checks).
- Target-only, future-only and combined outcome perturbations leave the target retrospective probability and features unchanged. Changing all 2026 outcomes leaves frozen coefficients/model fingerprint unchanged. Incorrect 2026 training provenance is rejected.
- Archived-pick and production-model byte preservation tests pass; 29 original file hashes match the pre-sprint manifest. Git's original-history guard remains enabled.
- Root CLI refresh, backtest, evaluate, coverage, predict and audit verified; prediction/backtest reruns add zero records. Explicit production `train --offline` and all CLI commands also pass in an ignored isolated copy, leaving the checked-in production model unchanged.
- Type checking and compilation pass. Streamlit smoke verification covers Weeks 1–18, both history cohorts, all four tabs and future weeks with no forecasts; page loads do not mutate artifacts. No public deployment or browser pixel-level review is claimed.
- GitHub refresh now reuses both reviewed model artifacts, records upcoming live picks, adds missing retrospective games and evaluates through the existing review-PR workflow. CI also checks coverage. No additional framework or dependency is required.

Remaining deployment actions: approve/merge this branch, enable the documented Actions content/PR permissions, dispatch refresh on the default branch, review the artifact PR, and verify the Community Cloud URL with entrypoint `dashboard/app.py` and Python 3.12. Both joblib files and all public JSON/CSV artifacts must remain tracked. Git, not ephemeral hosting disk, remains the durable record. Scheduler delays, source failures, review backlog and source timing limitations remain operational risks; failures preserve original records. See README for exact setup and recovery steps.
