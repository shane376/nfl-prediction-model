# V2 feature feasibility — bounded research proposal

Reviewed November 10, 2026. This is a design note, not a feature implementation,
new performance result, or authorization to replace V1. The [V1 evaluation](model_diagnostics_v1.md)
still supports retaining production: selected Elo raises 2025 accuracy to 64.21%
but worsens Brier (.227689 versus .226805) and log loss (.649154 versus .644337).
Prioritize cheap schedule/rating features first; add play-level features only
after their provenance passes an availability audit.

## Evidence and availability contract

**Documentation verified** below means official provider documentation was read;
it does not mean every season file was downloaded or audited. No external data
files were imported in this extension. **Unverified** means completeness, exact
historical publication times, or model vintages remain to be checked. Historical
coverage is not proof of availability before a historical forecast.

For a proposed forecast time `t = kickoff - 24 hours`, require source publication
and local ingestion before `t`, and prior games to have finished before `t`.
Persist URL/release or commit, raw SHA-256, retrieved/published timestamps, schema,
transform version, player/team mapping, underlying statistical-model version and
feature cutoff. Use the latest revision actually observed before `t`; never
backfill it from a later corrected file. A kickoff change must retain the earlier
forecast lock; uncertain timestamps cannot create a new prediction window.

For historical files without publication history, label evaluation **latest-data
reconstruction**. A 48-hour postgame lag can be a conservative sensitivity
assumption, not evidence that the file or revision existed then. Compare 24/48/72
hour assumptions as prespecified robustness checks; do not pick the best lag on
test outcomes. Strong availability claims require archived snapshots or forward
collection. This supersedes treating V1's kickoff-plus-12-hour convention as
proof of availability.

The provider describes cleaned play-by-play as updated after game days and notes
subsequent stat corrections. Such schedules describe intended processing, not a
delivery guarantee for an individual record. [nflverse availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)

## Feature decisions

| Feature | Initial priority | Historical evidence | Main blocker |
|---|---|---|---|
| Prior-season Elo | First experiment | Local scores: 2023–2026; earlier schedule files unverified here | Warm-up history and historically correct scores |
| Rest days | First experiment | Local scheduled kickoffs: 2023–2026; documented upstream rest fields | Reschedules and Week 1 convention |
| Home field / neutral venue | First experiment, conditional on venue audit | Documented schedule location field; local non-null coverage unverified | Designated home team is not always playing at home |
| Offensive/defensive EPA per play | Second stage | PBP loader documents data since 1999 | Revision history and EP model vintage |
| Offensive/defensive success rate | EPA ablation | Same potential PBP coverage | EPA-derived definition shares EP vintage risk |
| QB efficiency | Later ablation | PBP; weekly player stats; NGS documented since 2016 | Pregame QB identity and sparse history |
| Injury / roster availability | Later stage only | Documented injury history since 2009; roster/depth-chart sources | Historical report revisions and current feed continuity |

### Prior-season Elo

Dataset: derive ratings from nflverse game scores and kickoff times, rather than
depending on a third-party Elo file. The [schedule loader](https://nflreadr.nflverse.com/reference/load_schedules.html)
identifies Lee Sharpe's maintained game/schedule data. This repository verifies
2023–2026 schedule coverage; earliest upstream season and complete older kickoff
coverage are **unverified in this review**. Earlier seasons must be audited before
expanding training. The existing research's 2023 all-1500 initialization is a
warm-up limitation, not an estimate of actual entering team strength.

Availability/provenance: compute season-opening ratings from eligible results of
strictly earlier seasons; freeze that opening rating for the whole target season
in the minimal experiment. Use K=20, a 50-point home adjustment (zero at verified
neutral sites), 25% offseason regression, and no margin-of-victory multiplier.
Fix regular-season-only updates, matching V1's scope; playoff updates would be a
separate hypothesis. Never use a target season's ending rating or tune K on it.

Nulls/cost/validation: initialize genuinely unseen franchises at 1500, record
history counts and initialization flags, reconcile aliases before updating, and
do not invent missing scores. Sequential score processing is O(games), with a
small per-team state and no GPU. Reserve at least two earlier seasons for warm-up
where coverage permits. Ablate the frozen prior-season home-minus-away rating
against V1 on identical season folds; inspect Week 1 without tuning on its small
sample. This differs from the diagnostics' continuously updated Elo and has no
measured advantage yet.

### Offensive and defensive EPA per play

Dataset/coverage: [nflreadr load_pbp](https://nflreadr.nflverse.com/reference/load_pbp.html)
documents available PBP since 1999 and Parquet/CSV access. The [PBP dictionary](https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_pbp.csv)
defines possession/defense teams and EPA from the possession team's perspective.
Exact non-null coverage of required fields across proposed seasons, including
current 2026 completeness, is **unverified**; inspect files and missingness before
claiming a usable span.

Definition: aggregate offensive EPA and EPA allowed on defense as separate
team-game sums and counts. Positive defensive EPA allowed means worse defense;
do not accidentally reverse this sign twice. Predeclare eligible scrimmage plays
as pass/run, excluding kneels, spikes, no-play rows and special teams, with sacks
and scrambles retained under documented classifications. Keep overtime; avoid
outcome-dependent garbage-time filters in the first experiment. Form two matchup
features: home-minus-away offensive EPA/play and home-minus-away defensive EPA
allowed/play. Shrink each cumulative rate toward the training-fold league rate
with a fixed 200-play prior; retain valid-play counts and missingness flags.

Availability/leakage: use prior completed games only, from snapshots eligible at
the forecast cutoff. Latest season summaries and target-game PBP are forbidden.
An additional risk is **upstream EP model training**: lagging EPA rows does not
remove future information if EP was retrospectively fit using later seasons.
The [model authors' explanation](https://opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/)
describes learned EP models and leave-one-season-out assessment; that is not proof
of chronological fitness for our folds. Exact training end dates of current
released EP models are **unverified**. Require a pinned model with training ending
before the evaluation fold, or recompute EP within chronological training data;
otherwise keep EPA exploratory and exclude it from promotion evidence.

Nulls/cost/validation: missing EPA is not zero EPA. Count excluded rows, suppress
an incomplete game aggregate if less than 95% of its eligible plays have valid
EPA, and use the prior plus an unavailable flag; report affected game coverage.
Estimate the league prior and every transform on training only. Read selected
Parquet columns one season at a time and cache team-game aggregates; aggregation
is O(plays), and downstream logistic fits are small. EP refitting would require
additional tooling and substantially more compute, outside the minimal scope.
Test EPA as a separate addition after the schedule candidate, on matched games
with the same fallbacks and an ablation of offense versus defense.

### Success rate

Dataset/coverage: the same [PBP dictionary](https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_pbp.csv)
defines `success` as positive EPA. Use that definition consistently, not an
unannounced yards-to-go heuristic. Potential history follows PBP's documented
1999 start; per-season field completeness is **unverified**.

Availability/provenance: use the EPA cohort, play filter, cutoff, release hash and
model-vintage gate above. A positive-EPA success flag inherits EP model leakage
risk; it is not a model-free escape from the EPA provenance problem. Exclude
target-game success counts and full-season rates computed after the forecast.

Nulls/cost/validation: unknown success stays missing, not failure. Aggregate
successes/valid plays separately for offense and defense, shrink with 200 prior
plays at training-fold league success rates, and record denominators and excluded
plays. Reuse the PBP scan and aggregates, adding negligible marginal storage.
Test as a replacement for EPA first to measure redundancy, then consider combining
them only on an inner validation window. Use the same chronological folds and
coverage requirements as EPA; do not select whichever definition wins a holdout.

### QB efficiency

Datasets/coverage: derive prior QB EPA/dropback from PBP; the dictionary includes
dropback and player-ID fields. [Weekly player stats](https://nflreadr.nflverse.com/reference/load_player_stats.html)
provide a plausible box-score alternative, but their full historical field span
was **not verified** here. [Next Gen Stats](https://nflreadr.nflverse.com/reference/load_nextgen_stats.html)
documents weekly data from 2016, minimum-attempt eligibility, and season summaries
at week zero. A passing CPOE variant is plausible, but its exact column coverage
and underlying model vintage remain **unverified**. Never use week-zero
end-of-season summaries to predict games within that season.

Availability/provenance: carry forward only the expected starter known at the
24-hour cutoff, joined by stable player ID. The final schedule's starting-QB
field or the player with most target-game attempts is postgame knowledge.
[Depth charts](https://nflreadr.nflverse.com/reference/load_depth_charts.html)
document history back to 2001; weekly historical entries alone do not prove
pregame timestamps. The provider [describes timestamped appended depth-chart
updates from 2025](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html),
but their completeness and QB forecast accuracy are unverified here. Archive
starter assumptions and revision times; an expected starter can still be wrong.

Nulls/cost/validation: shrink identified QB history toward a training-only league
dropback prior (fixed 200 dropbacks); rookies/no history receive that prior and a
flag. If the starter is unknown, retain an unknown-starter flag and team-level
prior efficiency; do not recover the eventual starter after kickoff. Reconcile
trades, duplicate IDs and scramble attribution explicitly. This adds a modest
player-game aggregation to PBP; mapping and timestamp maintenance dominate cost.
Ablate QB efficiency after team EPA, with separate coverage and performance for
rookies, known/unknown starters and starter changes. Exclude from the minimal V2
experiment until historical starter assignments pass the availability audit.

### Rest days

Dataset/coverage: the [schedule dictionary](https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_schedules.csv)
documents rest fields and kickoff time in Eastern time. Local 2023–2026 kickoff
coverage is verified by V1 diagnostics; older years and semantics/completeness of
the provided rest values are **unverified**. Prefer deriving our own reproducible
value from schedule snapshots.

Availability/provenance: define rest as calendar-day difference between current
and most recent actually played regular-season game, using America/New_York
dates. It is a schedule interval, not measured recovery time. Identify the prior
played game from status known at cutoff; a canceled/postponed listing is not a
played game. Do not use a later rescheduled kickoff retroactively. Compute
home-minus-away rest, clipping each side at 14 days by fixed design.

Nulls/cost/validation: use 7 days plus an explicit unavailable/season-opener flag
when no prior regular-season game exists; avoid treating the offseason as months
of useful rest. No future backfill or arbitrary midnight for TBD kickoffs. The
small game table sorts in O(games log games); no additional large download or GPU
is needed. Ablate rest on matched seasonal folds and describe short-rest/byes
and Week 1 separately; do not tune clipping on their outcomes.

### Home field

Dataset/coverage: the [schedule dictionary](https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_schedules.csv)
distinguishes designated home teams from Home/Neutral location, including
international games. Local designated teams are available for 2023–2026; actual
venue/location completeness and earlier-year coverage are **unverified**.

Availability/provenance: use venue designation known at forecast time; archive
relocations and schedule corrections. V1's logistic intercept already models an
average home advantage: a constant `home_field=1` column adds no new information.
With an intercept, add only a verified-neutral indicator (1 neutral, 0 Home) and
its unavailable flag; avoid team-specific stadium coefficients in the first run.
Do not infer venue from final weather, attendance or outcome fields.

Nulls/cost/validation: unknown location is not a confirmed home venue. Use zero
neutral indicator plus an explicit unknown flag and report its rate; if location
cannot be audited, omit this feature family rather than silently assert Home.
Compute cost is a small schedule join. Compare V1's intercept-only home treatment
with the neutral adjustment on identical chronological folds; neutral games are
sparse, so subgroup accuracy cannot justify a new coefficient by itself.

### Injury and roster information — deferred

The [injury loader](https://nflreadr.nflverse.com/reference/load_injuries.html)
documents API-derived history since 2009; this does **not verify continuous
2025/2026 availability** or complete historical revisions. [Season rosters](https://nflreadr.nflverse.com/reference/load_rosters.html)
document records back to 1920, which is not evidence of historical daily active
rosters. Archived team/NFL injury reports are plausible corroboration; their
machine-readable history, access terms and collection costs are unverified.

Require the last published status before the forecast cutoff and distinguish
out/questionable/unknown; postgame inactives, actual snaps, season-ending IR and
final roster membership cannot reconstruct an earlier decision. Missing reports
mean unknown, not healthy. Use stable player IDs and training-only player impact
weights, with missing-report/coverage flags and explicit roster churn handling.
Compute after joins is cheap, but snapshot collection, identity resolution,
possible licensing and report revisions impose substantial maintenance. Validate
as a later ablation on the same eligible games, including a full-population
fallback analysis so report coverage does not select easier games. Do not use
postseason-only participation files for an in-season availability feature; the
provider explicitly describes that delayed release for 2023 onward.
[Participation availability](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)

## Minimal experiment and chronological protocol

This extension implements **none** of the following. Propose one small research
run after a source audit, with at most three fixed configurations:

1. V1 A features, StandardScaler and L2 logistic regression, C=1, threshold .5.
2. Configuration 1 plus frozen prior-season Elo difference.
3. Configuration 2 plus rest difference, opener/unavailable flags, and a neutral
   adjustment only if venue provenance passes. Predeclare that omission before
   inspecting labels; report the resulting feature set.

No calibration or hyperparameter sweep in this first run. Compare against a .5
probability, training-only home-win rate, and a fixed Elo-only probability
`1 / (1 + 10**(-(rating_diff + home_adjustment)/400))`. Hard always-home is an
accuracy reference, not a useful log-loss comparator. Refit the V1 **research
reference** on exactly the same earlier training games as each candidate; never
score earlier folds with the deployed artifact that has seen later outcomes.

Subject to verified older data, use 2015–2016 for rating warm-up and earlier-than-
test data for model fitting, with expanding season test windows 2021–2024. For
example, a 2021 test fit ends with the 2020 season; the 2022 fit may use 2021
outcomes because it predicts later games. All input labels must be available
before the relevant fold's first forecast. Freeze coefficients for each season;
V1's within-season statistics may update from eligible prior games, while this
candidate's season-opening Elo stays fixed. With the current 2023-only history,
this protocol is **blocked on the historical data audit**, not already feasible.

Keep identical eligible non-tied game IDs, forecast horizon and missing-data
fallbacks for all candidates/baselines; report excluded games and reasons. Do not
restrict the main score to games with complete premium features. Scalers, priors,
imputation and any later calibration fit on preceding data only. If parameters
are later tuned, add inner chronological windows entirely before each test fold;
random folds and leave-one-season-out fits that include later seasons are invalid
for this purpose. No target/future-outcome mutation may change pregame features.

Use 2021–2024 results for development and configuration selection, with selection
bias acknowledged. The already-inspected 2025 and observed 2026 outcomes remain
descriptive and cannot become a fresh final holdout. Freeze one configuration and
the gates below before a complete prospective 2027 regular season, with archived
24-hour-cutoff snapshots. Treat that season as confirmation; do not tune after
interim results. If more independent forward evidence is needed, continue
collecting rather than relabel an old holdout as unused.

EPA, success rate and QB efficiency are separate future ablations, not extra
configurations hidden in this initial grid. Admit EPA only after the model-vintage
and release audit; success rate can then replace EPA on inner validation, and QB
features require timestamped expected starters. No large feature combination
search is warranted by the present 271/120/14-game evidence.

## Prespecified no-go gate

The following are proposed engineering thresholds, **not measured effects or a
power calculation**. Define each delta as candidate score minus comparator score,
so negative is better. Score Brier and log loss without outcome-dependent
probability clipping; use one documented numerical epsilon identically for log
loss. Accuracy is secondary. Reuse the same games and paired resamples.

- **Historical screen:** require at least two complete out-of-time seasons and
  500 non-tied matched games. Against V1 and each of the .5, training home-rate
  and fixed Elo baselines, require pooled Brier delta <= −.002 and log-loss delta
  <= −.005, plus improvements in both scores versus V1 in at least two seasons.
- **Confirmation:** on the frozen prospective season require those same absolute
  improvements versus every comparator. Require upper 95% paired week-block
  bootstrap limits below zero for both score deltas versus V1, using at least
  5,000 draws and `(season, week)` blocks. Report per-season results and a
  contiguous multiweek-block sensitivity because repeated teams span weeks.
- **No-go:** a failed score threshold, either interval including zero,
  insufficient sample, broken chronology/provenance, or dependence on dropping
  missing-feature games means no promotion. Mark inadequate precision
  inconclusive and collect more data; do not call it a proven absence of signal.
  Require 100% of published experimental predictions to have auditable cutoffs
  and explicit fallback state. Any forward data used to redesign the candidate
  becomes development data and requires a later untouched confirmation window.

For scale only, applying the V1-relative margin to the old 2025 reference would
require Brier <= .224805 and log loss <= .639337; these are **not** the thresholds
for a future population. Recompute all baselines on each matched out-of-time
population. Current selected Elo fails even the direction-of-improvement check
on both 2025 proper scores, so this proposal cannot support deploying it now.

## Resource bound and remaining verification

Planning estimates, **not benchmarks**: the schedule-only experiment should fit
within a few CPU minutes and under 1 GB RAM for a modest historical span. For a
later PBP audit, budget one season per scan, 2–4 CPU cores, up to 8 GB RAM and
10 GB scratch initially; compressed input sizes and elapsed time remain
unverified. Project only required columns, cache team/player-game sums, measure
one season before expanding, and stop if the budget is exceeded. No GPU, tracking
data or paid feed is required for the proposed schedule experiment. Verify
redistribution/access terms and actual coverage before adopting any new source.

The next engineering decision is whether the audited schedule/Elo experiment
clears the gate. A broad PBP/QB/injury implementation should wait for both data
provenance and incremental out-of-time evidence.
