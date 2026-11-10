# V1.2.1 product usability pass

## Scope and audit

Continues `feature/v1.2-dashboard` and its existing PR. The audit covered the
current dashboard, presentation and provisional helpers, season scoring, team
names, and existing read-only AppTest and cohort tests. The V1.2 architecture was
reused: offline artifact loading, in-memory reconciliation, cached provisional
inference and cohort scoring are unchanged. No model, data source, forecast
persistence, CLI, dependency or infrastructure changes.

## User journeys

- **Find a game:** select a week or All weeks, search full team names or team
  codes (including `MIA @ BUF`), then narrow by team, game status, forecast type,
  predicted winner or minimum confidence. Filters combine; no active confidence
  filter means games without forecasts remain visible. Sort by kickoff, confidence
  in either direction, or matchup name. Missing times/probabilities sort last.
  Clear list filters resets search and list controls while retaining the week.
- **Read a matchup:** teams, kickoff/status, final score when available, favored
  team and its win probability, forecast provenance, and correct/incorrect result.
  Recorded picks, retrospective backtests, provisional estimates and unavailable
  forecasts have separate labels. Backtest outcomes say “Backtest,” not recorded
  pick. Estimates retain generation time, shown in Eastern Time, and an exclusion
  from accuracy notice. Raw metadata remains in advanced expanders.
- **Explore a team:** select any of the 32 NFL teams. The table includes all 17
  matchups present in the committed schedule, independent of weekly filters,
  with week, opponent, venue, kickoff, actual score/result and available forecasts.
  Scores and W-L-T are from the selected team's perspective. Missing forecasts
  do not hide schedule rows; incomplete snapshots produce a warning.
- **Optional remaining wins:** disabled by default. An estimate is shown only
  with a complete 17-game source schedule and eligible recorded/provisional
  probabilities for every uncompleted, non-canceled matchup. It sums the selected
  team's win probabilities, never substitutes a missing probability, never uses
  retrospective predictions and never changes the actual record or accuracy.
  Partial coverage produces an unavailable explanation and coverage count.
- **Check performance:** basic cohort accuracy, correct/incorrect and scored-game
  counts remain visible. Loss metrics, probability distributions, baseline charts,
  detailed evaluation and coverage tables are behind advanced expanders. History
  keeps human-readable provenance while technical timestamps/revisions stay in
  expandable detail.

## Integrity

`src/presentation.py` holds pure joining, filtering, sorting and team-summary
helpers shared by weekly cards and Team Explorer. Existing recorded rows always
take precedence over retrospective rows and estimates, including invalid recorded
rows. Cached provisional eligibility is checked again in both views, including TBD
game dates, stale sources and earliest kickoff locks. An unreadable archive
withholds estimates instead of suggesting that an existing recorded pick is missing.

Actual records count only completed games with both source scores, including ties;
forecast correctness and forecast coverage never affect W-L-T. Actual results can
appear before the existing prediction-scoring embargo expires. This is intentional:
the actual record describes published scores, while official forecast accuracy
continues to follow its existing eligibility policy. No division standings,
tiebreakers, missing opponents, scores or predictions are synthesized.

## Verification

The focused tests cover combined filters, search, confidence boundaries, sorting,
missing forecasts, cache expiry/window checks, actual home/away W-L-T including
ties, excluded nonfinal scores, all 32 committed 17-game schedules, source-level
league W/L conservation, estimate coverage, provenance precedence, an unreadable
archive and UI journeys. The read-only UI tests hash artifacts before and after
navigation, filtering, team selection and the optional estimate.

- Full suite: **106 passed in 19.15 seconds**, with pytest cache disabled and a
  workspace-local temporary directory.
- `ty check dashboard/app.py src/presentation.py` and `git diff --check` passed.
- Git comparison confirms all persistent artifacts and the existing data, model,
  prediction, retrospective, provisional, season-scoring and CLI modules are unchanged.
- A real headless Streamlit runtime returned `ready` for weekly
  search/confidence/sort, Team Explorer, its estimate control and the other views;
  no new server warnings or tracebacks were emitted.

## Deployment and limitations

Deployment is unchanged: install `requirements.txt` with Python 3.12 and run
`python -m streamlit run dashboard/app.py` from the repository root. Local artifacts
are sufficient; no refresh, network access or retraining occurs during rendering.

- The local snapshot may be stale. Future estimates can therefore be unavailable;
  the dashboard does not fetch replacement data or persist estimates.
- A missing team matchup remains missing, with a schedule warning. W-L-T reflects
  the available completed source scores rather than an independently verified
  official standings feed.
- Confidence filters use the favored team's probability, not necessarily the
  selected team's probability. In Team Explorer that column is explicitly named
  “Favored win probability.” Probabilities are displayed rounded for readability.
- Estimated remaining wins are a model expectation, not a standings forecast or
  guarantee. Existing model omissions and current-statistics-only policy remain.
- Wide team schedules scroll horizontally on small screens. Automated runtime and
  AppTest verification do not establish pixel-level desktop/mobile appearance;
  browser/device visual review remains outstanding.
- Retrospective reconstruction and the 12-hour scoring embargo retain all existing
  limitations. Provisional estimates never enter official accuracy or history.

Do not enable the development agent API on a public deployment. The existing PR
targets `main`; this pass does not merge or deploy it.
