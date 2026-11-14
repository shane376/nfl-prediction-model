# Refresh recovery baseline

Prepared 2026-11-14. No forecasts were generated, revised, or backdated. No model was retrained or promoted; existing test-only in-memory research fits remain part of the required suite.

## Verified failure and minimal repair

Actual job logs for scheduled runs [43616296687](https://github.com/shane376/nfl-prediction-model/actions/runs/43616296687) (job 132511982207) and [43583641649](https://github.com/shane376/nfl-prediction-model/actions/runs/43583641649) (job 132413437706), both on November 14, show the same failure: `test_full_research_replay_preserves_artifacts_and_exports_matched_scores` compares 15 scored recorded forecasts against the published research table's 14. Each run reports 127 passed, 1 failed. Refresh and subsequent persistence steps were skipped.

`refresh.yml` checks out main, switches to the durable review branch, merges main, checks append-only history, then tests before retrieving data. Both logs show merge success ("Already up to date") and a successful history check. `automation/nfl-artifacts` at `9b4200534ac488283a942545a30ae36eb2ac00f7` contains main `2ee913860e57ced737cea6d1f4ff3105943fee38` through merge `4c1fbc0`; its later source snapshot resolves BUF at MIN as an incorrect Week 9 forecast. The research test used this moving source time instead of its published November 10 12:55 UTC cutoff. The failure is population drift, not a verified source or merge failure.

The repair adds an explicit research evaluation cutoff and pins the report reproduction test to the existing manifest cutoff. A regression advances source freshness to November 14 while retaining exact historical evidence-table comparisons and model/input fingerprint checks. Default diagnostics behavior stays source-based. No workflow bypass or test removal is needed.

The pending [artifact review PR #6](https://github.com/shane376/nfl-prediction-model/pull/6) targets main. It is left untouched. Its five changing derived/source files overlap this recovery snapshot; review conflicts if that PR and this one are both merged. Future workflow runs merge reviewed code from main into the durable branch; no reset, force-push, automatic merge, or deployment is introduced.

## Source and coverage

Successfully retrieved [nflverse games.csv](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv) at **2026-11-14T20:48:23.525869+00:00** into `.review/recovery/store` using the real clock. Source error: none; fresh at evaluation. All 272 regular-season 2026 games are present (17 per team). This timestamp is retrieval freshness, not an upstream publication timestamp.

Only the verified games/source and derived reconciliation/season-report files were copied from the isolated store. All 29 authentic prediction files, both model bundles and metadata, 121 retrospective records, and the 2025 backtest remain byte-identical to main and the automation branch.

Missing authentic coverage: 121 past games (Weeks 1-8), plus 122 future games (Weeks 11-18). These are unrecorded forecasts, not losses or new backtests. Weeks 9 and 10 have 29/29 authentic records, zero integrity errors, and no missing source scores after kickoff. Week 10 has only one completed game; 13 scheduled games remain pending.

## Metrics

Binary metrics exclude pending results and ties. Brier/log loss use saved home-win probabilities; accuracy uses the first authentic pregame revision. Small live samples do not establish model quality.

| Cohort | Scored | Correct | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Authentic Week 9 | 15 | 12 | 0.800000 | 0.189325 | 0.569099 |
| Authentic Week 10 | 1 | 0 | 0.000000 | 0.375787 | 0.949368 |
| Authentic combined | 16 | 12 | 0.750000 | 0.200979 | 0.592866 |
| Existing retrospective Weeks 1-8 (separate) | 120 | 67 | 0.558333 | 0.256907 | 0.713022 |

Retrospective cohort: 121 existing backtests, 120 binary outcomes and one tie. No additional retrospective records were generated. The 2025 holdout stays frozen (271 games, accuracy 0.612546, Brier 0.226805, log loss 0.644337).

## Per-game authentic results

Scores are away-home. A dash means no final score is available yet; pending games are excluded from metrics. Generation timestamps and full precision probabilities remain in the unchanged prediction JSON files.

| Week | Game | Pick | P(home) | Final away-home | Result |
|---:|---|---|---:|---|---|
| 9 | ARI at SEA | SEA | 0.639945 | 3-17 | correct |
| 9 | BUF at MIN | MIN | 0.537535 | 23-6 | incorrect |
| 9 | CIN at ATL | CIN | 0.422358 | 34-17 | correct |
| 9 | CLE at NO | CLE | 0.485491 | 22-23 | incorrect |
| 9 | DAL at IND | DAL | 0.461831 | 27-23 | correct |
| 9 | DEN at CAR | DEN | 0.468851 | 26-21 | correct |
| 9 | DET at MIA | DET | 0.363556 | 19-9 | correct |
| 9 | GB at NE | NE | 0.556160 | 45-27 | incorrect |
| 9 | HOU at LAC | HOU | 0.423494 | 44-22 | correct |
| 9 | JAX at BAL | JAX | 0.399559 | 33-10 | correct |
| 9 | LA at WAS | LA | 0.472241 | 20-13 | correct |
| 9 | LV at SF | SF | 0.662802 | 17-38 | correct |
| 9 | NYG at PHI | NYG | 0.418825 | 19-11 | correct |
| 9 | NYJ at KC | KC | 0.665224 | 12-35 | correct |
| 9 | TB at CHI | CHI | 0.625313 | 24-27 | correct |
| 10 | BUF at NYJ | BUF | 0.376770 | - | pending |
| 10 | CAR at NO | NO | 0.502019 | - | pending |
| 10 | HOU at CLE | CLE | 0.571605 | - | pending |
| 10 | JAX at TEN | JAX | 0.318034 | - | pending |
| 10 | KC at ATL | KC | 0.434544 | - | pending |
| 10 | LAC at BAL | BAL | 0.673330 | - | pending |
| 10 | LA at ARI | LA | 0.442060 | - | pending |
| 10 | MIA at IND | IND | 0.570688 | - | pending |
| 10 | MIN at GB | MIN | 0.462869 | - | pending |
| 10 | NE at DET | DET | 0.560841 | - | pending |
| 10 | PIT at CIN | CIN | 0.553900 | - | pending |
| 10 | SEA at LV | LV | 0.511037 | - | pending |
| 10 | SF at DAL | SF | 0.410944 | - | pending |
| 10 | WAS at NYG | NYG | 0.613014 | 24-21 | incorrect |

## Validation

Full suite: **128 passed** against the refreshed snapshot (`python -m pytest -q --basetemp .pytest-tmp-recovery-final -o cache_dir=.review/recovery/pytest-cache --tb=short`). Isolated reconciliation and all 29 integrity audits passed. Append-only history checks passed against both `origin/main` and `origin/automation/nfl-artifacts`; Git byte comparisons confirm all original predictions, retrospective records, frozen model files/metadata, and holdout backtest are unchanged. `git diff --check` passed. No live workflow was dispatched.
