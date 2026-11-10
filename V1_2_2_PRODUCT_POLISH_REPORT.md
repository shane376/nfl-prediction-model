# V1.2.2 — football first, analytics on demand

## Audit and scope

Continues `feature/v1.2-dashboard` and PR #5. Reviewed `dashboard/app.py`, the
presentation, provisional and season helpers, current UI tests, theme and the
V1.2.1 report. The existing offline loader, forecast cache, immutable record
reconciliation, team schedules, filters and cohort scoring are reused.

The starting desktop screenshot confirmed that branding, explanations and seven
visible controls pushed the first matchup to the bottom of a 1000px viewport.
Team Explorer exposed an eleven-column table. This pass changes presentation,
not the model, its inputs, forecast eligibility or persistence.

## Product changes

- A compact Gridiron Lab / 2026 header replaces the marketing headline. A small
  freshness badge has refresh/pending-result details on hover. The sidebar guide
  holds the detailed source state and provenance explanations.
- Weekly predictions remains the default landing view. Bounded Previous/Next
  buttons navigate Weeks 1–18; the existing week selector and All weeks remain.
- Search and team selection stay visible. Status, type, predicted winner,
  confidence and sorting live in one collapsed Advanced filters section. Clear
  filters and the showing/total count remain available. Existing AND-filtering,
  deterministic ordering, missing-probability handling and empty states survive.
- A native responsive grid lays out two matchup cards when space allows and one
  on narrow screens. Text-based team identities show full names and abbreviations
  without external images or logo licensing/network dependencies.
- Upcoming cards lead with the favorite and probability, then complementary
  probabilities. Finals emphasize actual scores and the actual winner. Eligible
  recorded picks receive correct/incorrect labels; backtests never receive an
  authentic recorded-pick outcome label.
- Compact green Recorded, violet Retrospective and amber Provisional labels keep
  provenance visible. Provisional cards retain generation time and Not scored.
  Tertiary Details popovers replace full-width per-card expanders.
- Team Explorer leads with the selected team, actual W-L-T, played and remaining
  counts. Its semantic schedule list retains every available game, including
  missing forecasts and TBD kickoffs. Desktop rows show week, venue, opponent,
  kickoff, actual result and forecast; narrow rows stack without horizontal scroll.
  W/L/T colors and upcoming borders supplement the visible text labels.
- Remaining-win estimates remain optional, coverage-gated model expectations in
  a Model estimate popover. They never affect actual records or accuracy.
- Performance leads with accuracy, correct predictions, total evaluated and a
  weekly correct/incorrect chart for each separate cohort. Brier score, log loss,
  baseline/cumulative analysis, calibration and raw tables remain on demand.
- The existing dark theme uses less top padding and consistent typography. A
  small scoped stylesheet styles only the football cards and schedule list;
  native Streamlit controls, grid reflow, popovers and navigation are retained.

## Integrity

Production data/model artifacts, recorded and retrospective predictions, source
refresh automation, model training, provisional inference and season scoring were
not modified. The page reads the local snapshot and produces only existing
in-memory estimates. It performs no network refresh, training or artifact writes.
Provisional estimates remain excluded from history and official accuracy.

New helpers format already-reconciled results. They do not infer prediction
correctness from scores: a recorded outcome requires the reconciler's eligible
target. Finals and W-L-T still come from actual completed scores, independently
of forecast coverage. Missing predictions, scores and schedule rows are never
invented. Existing invalid-record precedence and unreadable-archive safeguards
remain intact.

## Verification

- Full test suite: **116 passed in 19.50 seconds**. Focused additions cover bounded week
  navigation, final/upcoming display, actual winners, complementary probabilities,
  eligible recorded correctness, provenance, safe HTML escaping, stale source
  presentation, collapsed filters and the native wrapping two-column structure.
- Existing tests continue to cover combined filters/sorts, all 32 teams' complete
  17-game local schedules, missing forecasts, estimates, independent actual W-L-T,
  cohort separation and artifact hashes before/after UI journeys.
- Static type checks and Git whitespace checks passed. Git comparison confirms
  protected artifacts and production data/model/scoring/automation modules are
  unchanged.
- The app starts with `python -m streamlit run dashboard/app.py` (temporary
  loopback port/headless options used for verification). The development agent
  API returned `ready` for the landing view, navigation, advanced filtering,
  Team Explorer, performance, history and methodology.
- Headless Chrome checked 1440×1000 and 390×844 layouts. Weekly cards occupy two
  columns on desktop and one on mobile. Both mobile views measured document width
  exactly 390px with no horizontal overflow; all 17 team schedule rows remained in
  the DOM. The desktop's first card moved from roughly 859px to 347px from the top,
  allowing six upcoming cards to be read in the first 1000px viewport.
- A runtime warning from giving week selection both a default and callback state
  was fixed by initializing session state once and using a required selector.
  The final startup/navigation smoke test emitted no new warnings or tracebacks.

## Screenshots

Captured from the real local committed-artifact snapshot, without invented scores
or forecasts:

- [Weekly desktop](docs/screenshots/v1.2.2-weekly-desktop.png)
- [Weekly mobile](docs/screenshots/v1.2.2-weekly-mobile.png)
- [Recorded finals](docs/screenshots/v1.2.2-finals-desktop.png)
- [Provisional forecasts](docs/screenshots/v1.2.2-provisional-desktop.png)
- [Team Explorer desktop](docs/screenshots/v1.2.2-team-desktop.png)
- [Team Explorer mobile](docs/screenshots/v1.2.2-team-mobile.png)

## Deployment and limitations

Deployment is unchanged: Python 3.12, `pip install -r requirements.txt`, then
`python -m streamlit run dashboard/app.py` from the repository root. Include
`dashboard/style.css` with the app. No new dependencies, services or credentials.
Do not enable the temporary development agent API on public hosting.

- Browser checks cover Chrome at two widths, not every device, browser, font
  scaling setting or assistive technology. Text wrapping can change card height.
- No team logos are fetched. Clean text identities preserve the offline product.
- The season remains 2026. Actual records reflect the committed source snapshot,
  not a live standings feed. A stale source is visible and can withhold estimates.
- Probability display is rounded. Long-range provisional forecasts still use
  only current observed inputs and remain estimates, not archived predictions.
- Retrospective reconstruction, missing kickoff eligibility and the existing
  12-hour result-scoring embargo retain their prior limitations. Some published
  final scores can appear before the corresponding pick becomes score-eligible.
- Popovers disclose complete metadata on demand. Their contents are read-only;
  this pass does not add a writer or change refresh automation.

Commits are pushed to the existing PR #5 branch for review. No automatic merge or
deployment is performed.
