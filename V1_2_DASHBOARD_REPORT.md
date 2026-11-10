# V1.2 dashboard sprint

## Audit and scope

The original dashboard already loaded local snapshots with `load(offline=True)`,
reconciled the earliest recorded pick in memory, and kept retrospective results
separate. It showed basic matchup containers, four eagerly rendered tabs, coverage,
and a combined weekly/cumulative chart. These integrity behaviors are retained.

Reusable modules:

- `src/data.py`: normalized schedule, timezone-aware timestamps, earliest kickoff
  locks, 12-hour result availability embargo, source freshness, offline loading.
- `src/pregame.py`: identical training/inference season-to-date feature definitions.
- `src/predictions.py`: immutable archives, timestamp validation, read-only reconciliation.
- `src/retrospective.py`: frozen pre-2026 provenance and read-only historical scoring.
- `src/season.py`: coverage, cohort metrics and weekly performance.
- CI already runs tests, offline audits and coverage, and protects archived picks.
  Refresh automation stays on its review branch; neither workflow needed changes.

No training, refresh, prediction persistence, or evaluation-write command was added
to page rendering. Existing production model and prediction artifacts are unchanged.

## Before and after

| Area | Before | V1.2 |
| --- | --- | --- |
| Identity | Season title and caption | Product header, season, selected week, refresh timestamp, stale and pending-result status |
| Navigation | All four tabs computed on rerun | Sidebar routes render only the chosen view |
| Matchups | Basic text | Compact away/home hierarchy, full names, kickoff, score, winner, complementary probabilities |
| Provenance | Text labels | Green recorded, violet retrospective, amber provisional badges plus explicit unavailable explanations |
| Outcomes | Result appended to text | Separate green correct/red incorrect labels; pending and ties remain explicit |
| Performance | Combined accuracy chart | Separate weekly and cumulative percentage charts, each compared with always-home on the same scored games |
| Empty states | Missing pick notice | Specific eligibility reasons, filtered-history empty state, unscored-cohort explanation |
| Presentation | Basic dark theme | Restrained dark surfaces, green accent, stronger heading weights, compact padding and consistent borders |

Uses native Streamlit layouts and theme tokens, without external fonts or assets.
Metric rows wrap and matchup columns stack on narrow screens. Raw provenance,
revisions, evaluation tables, and source metadata remain in expanders.

## Provisional policy

`src/provisional.py` performs deterministic inference from the saved reviewed
production model. It never fits or saves a model and never writes predictions.
The page caches estimates in memory for up to five minutes with the original
generation timestamp. Snapshot and model modification changes invalidate the cache.
Visible estimates are rechecked for source staleness and a closed kickoff window.

Only observed current-season results with availability and kickoff before the
earlier of generation and source-refresh time enter features. Future intervening
games are never simulated. Existing model definitions for teams without prior games
are reused; no team statistics are fabricated. Future games with TBD kickoff are
eligible only when their published game date is strictly after today in Eastern
Time and their earliest known kickoff has not passed. No date means unavailable.

Stale/failed/future-dated sources, unavailable models, future training cutoffs,
closed windows, final/underway games, invalid probabilities and missing required
metadata fail closed. An existing recorded or retrospective row always takes
precedence, including an integrity failure. Provisional rows have no scored target,
never appear in history, and `score_records` explicitly excludes them defensively.

## Verification

- Full suite: **83 passed**, final run in 16.51 seconds, with pytest cache disabled.
- `ty check` passed for the dashboard and both new helper modules; `git diff --check`
  passed. Git comparison confirms no persistent artifacts or production inference,
  training, data-loading, retrospective, or CLI implementations changed.
- Read-only UI tests hash artifacts before/after navigation, all 18 weeks, and both
  historical cohorts. Tests cover missing kickoff, corrupt history, changed
  schedule, ties, postponed games, stale sources and immutable revisions.
- New tests cover provisional inference, cutoff filtering, future-date TBD support,
  missing dates, invalid source/model timestamps, locked windows, missing model,
  invalid probabilities, presentation helpers and defensive metric exclusion.
- Offline page tests prohibit network access, training and persistence calls.
- Real headless Streamlit agent API: all four views and Week 18 returned `ready`;
  server console showed no new warnings or tracebacks.
- Actual local production model inference produced an in-memory provisional result.
- CLI `audit --offline`: all 29 archived picks valid; `coverage --offline` and
  `--help` completed successfully. Existing CLI paths are also covered by the suite.
- Initial Windows pytest temporary-directory permissions required a workspace
  basetemp; a later run with elevated permission passed. No application change was
  needed for that environment issue. One repeat AppTest run timed out while the
  development server was active; after stopping it, the isolated check and the
  full 83-test suite both passed.

## Deployment

From the repository root with Python 3.12:

```sh
python -m venv .venv
# Activate .venv using your platform's activation command.
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run dashboard/app.py
```

For Streamlit hosting, select `dashboard/app.py` as the entrypoint and use the
repository root for dependencies and `.streamlit/config.toml`. Deploy the reviewed
branch only after review. Local committed artifacts are sufficient for page loads;
no credentials or network refresh are required. Keep the refresh workflow and CLI
as the only writers, outside the serving process. Never enable the development
agent API on the public deployment. No new dependencies or infrastructure.

## Known limitations

- Browser screenshot and device rendering were not verified in this environment;
  AppTest and the real headless runtime verified behavior, not pixel appearance.
- Forecasts omit injuries, roster changes, betting markets and neutral-site
  adjustments. Distant provisional estimates use today's observed inputs and can
  change as real results arrive. They are not archived pregame picks.
- Historical backtests reconstruct latest final scores under the existing 12-hour
  embargo; they are not archived point-in-time source replays.
- The product remains scoped to the 2026 regular season. Source status is checked
  on reruns; it does not poll while the browser is idle. Snapshot caching lasts up
  to 60 seconds and provisional caching up to five minutes.
- Incomplete schedules remain visible with warnings; absent rows and ineligible
  forecasts are never synthesized. Missing or incompatible model artifacts yield
  unavailable estimates, not implicit training.
