# V1.3 model diagnostics — research only

V1 A has useful discrimination on 2025, but that result does not establish reliable
2026 probabilities. The frozen 2026 retrospective model loses to constant home-rate
probabilities on both proper scoring rules. No tested candidate establishes a
production replacement: the validation winner, Elo, improves 2025 accuracy but
slightly worsens Brier score and log loss. Keep production unchanged and prioritize
data provenance, early-season shrinkage research, and additional forward evaluation.

This report uses the **committed snapshot fetched November 10, 2026 at 07:55:36
Eastern (12:55:36 UTC)**. It makes no claim about later results. All inputs remain
unchanged; research outputs live under `docs/research_v1_3/`.

## Reproduce

From the repository root, using Python 3.12 and the existing locked environment:

```powershell
python -m pip install -r requirements.txt
python -m scripts.model_diagnostics
python -m pytest -q
```

Unset `NFL_STORE` first if it points to a different store. No refresh, training CLI,
prediction writer, or production evaluation writer is invoked. The script fits
research models in memory and writes CSV/JSON only to its research directory. It
hashes every artifact before execution and asserts those hashes remain unchanged.
`manifest.json` records input SHA-256 values, source cutoff, script digest, library
versions, cohort inventory, and selection policy. No network access is needed for
the diagnostic run. Exact probability comparisons use tolerance 1e-12 and fail on
unreproducible historical outputs.

## Existing pipeline and populations

`src/pregame.py` constructs identical training and inference season-to-date
features. `src/modeling.py` selects A–D by 2024 log loss after fitting 2023, scores
2025 after fitting 2023–2024, and separately refits the production artifact on all
eligible data. That production artifact includes 2026 outcomes and **must not be
used to reconstruct early 2026 games**.

`src/retrospective.py` instead fits fixed A on 2023–2025 results available before
February 1, 2026. Historical features can use earlier eligible 2026 results;
coefficients never use 2026 outcomes. The script reproduces its training-data
fingerprint and every scored archived retrospective probability. These records
were reconstructed after games, not published pregame forecasts.

`src/predictions.py` reconciles only the first authentic archived live revision,
enforces timestamp/matchup/probability checks, and excludes ties, unavailable
records, and pending results. Research calls its read-only reconciliation and the
retrospective integrity checks at the snapshot timestamp. There are no integrity
errors: retrospective inventory is 121 records (120 scored, one tie), while live
inventory is 29 records (14 scored, 15 pending). Pending includes the remaining
Week 9 game under the 12-hour result embargo, as well as Week 10 forecasts.

The populations remain separate throughout; they are neither pooled nor treated
as matched comparisons against each other.

| Population | Non-tied scored games | V1 accuracy (95% Wilson interval) | Brier | Log loss | Always-home accuracy | Prior home-rate Brier / log loss |
|---|---:|---:|---:|---:|---:|---:|
| 2025 holdout | 271 | 61.25% (55.34–66.86%) | .226805 | .644337 | 53.87% | .248528 / .690200 |
| 2026 retrospective, Weeks 1–8 | 120 | 55.83% (46.90–64.40%) | .256907 | .713022 | 51.67% | .250381 / .693915 |
| Authentic 2026 recorded, scored Week 9 | 14 | 85.71% (60.06–95.99%) | .182209 | .554665 | 35.71% | .262677 / .718557 |

Prior home rates are fitted only on the corresponding eligible training
population: .544118 for 2025, .542331 for retrospective, and .539037 for recorded
production forecasts. A constant .5 has Brier .25 and log loss .693147 in every
population. With the project's `p >= .5` decision rule, .5 picks home; its observed
accuracy equals always-home accuracy, not a simulated random-coin accuracy.
Always-home is primarily a classification baseline. Its hard probability of 1
has infinite mathematical log loss on away wins; the CSV includes sklearn's
finite machine-clipped value for completeness, not as a meaningful probability
benchmark. Always-away accuracies are the complements of the home rates in the
table; class imbalance is modest, and the 14-game away-heavy live sample is noisy.

The historical A–D metrics also reproduce to 1e-12 (`variant_reproduction.csv`):

| 2025 variant | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| A, selected on 2024 | 61.25% | .226805 | .644337 |
| B | 60.52% | .229074 | .648742 |
| C | 63.10% | .225079 | .640596 |
| D | 61.25% | .229627 | .649851 |

C's published holdout advantage is descriptive; selecting C now would reuse the
holdout for model selection. The older standalone `train_v1.py` main routine
selects its displayed best model on 2025 accuracy; that legacy research behavior
is unsuitable for a clean holdout protocol. Its ignored processed input datasets
are not available in this checkout, so the old 61.62% A run cannot be independently
reconstructed here. The active shared-feature pipeline gives 61.25%, matching its
saved records. Legacy week-based feature builders lack the active kickoff/embargo
checks; do not interchange their data with the current pipeline.

## Calibration, confidence, and error slices

`reliability.csv` contains equal-width home-probability bins, counts, mean
probabilities, observed home frequencies, and Wilson intervals for those
frequencies. Empty bins are omitted. On 2025, the largest bin (164 games) averages
.5195 and observes .4878 home wins; the .6–.8 bin (65 games) averages .6788 and
observes .7538. There is no uniform evidence of overconfidence in that holdout.
The apparent perfection above .8 comprises only four games.

In retrospective 2026, the .6–.8 bin averages .6988 against .5185 observed home
wins (27 games; interval .340–.693). Its global mean probability .5455 exceeds
the .5167 observed rate. Combined with worse-than-constant proper scores, this
is a monitoring signal for poor probability quality, not proof of a stable
calibration function. Live reliability bins contain 2–8 games and cannot support
a calibration recommendation.

Confidence means `max(p_home, 1-p_home)`; close is below .55, high is at least .70.
These are pregame probability definitions, not postgame score-margin categories.

| Confidence | 2025 n / accuracy | Retrospective 2026 n / accuracy | Recorded 2026 n / accuracy |
|---|---:|---:|---:|
| .50–.55 | 97 / 53.61% | 44 / 56.82% | 4 / 75.00% |
| .55–.60 | 67 / 53.73% | 25 / 52.00% | 4 / 75.00% |
| .60–.70 | 78 / 69.23% | 23 / 43.48% | 6 / 100% |
| .70–.80 | 25 / 80.00% | 16 / 68.75% | 0 / — |
| .80–1.00 | 4 / 100% | 12 / 66.67% | 0 / — |

The 2025 high-confidence aggregate is 24/29, versus 19/28 in retrospective 2026.
Both are small; their uncertainty is too large to infer reliable season-to-season
change. Close-game accuracy is near chance, as expected for probabilities near
.5. Increasing the decision threshold or dropping close games after seeing these
outcomes would create selection bias and would not fix probability quality.

Home picks versus away picks show little descriptive asymmetry: 2025 is
110/179 (61.45%) versus 56/92 (60.87%); retrospective is 44/79 (55.70%) versus
23/41 (56.10%). Authentic picks are 4/5 versus 8/9. The subgroup file also separates
actual home/away outcomes, distinguishing recall from accuracy conditional on a
pick. A fixed home intercept naturally produces more home picks than away picks.

That similar pick accuracy masks asymmetric recall: V1 identifies 110/146 actual
home wins (75.34%) but only 56/125 actual away wins (44.80%) on 2025. Retrospective
recall is 44/62 (70.97%) versus 23/58 (39.66%). This is a measurable weakness for
identifying away winners, consistent with the home-favoring decision rule. It
does not justify a new threshold without chronological validation against the
chosen objective; a threshold change alone cannot improve Brier or log loss.

`subgroups.csv` provides every scored week and team with sample counts and metric
intervals. For example, retrospective Week 2 is 6/16 with log loss .9718, much
worse than its overall .7130; 2025 weekly accuracy ranges from 6/15 in Week 4 to
11/14 in Week 12. Neither supports week-specific engineering. Team rows measure
forecast correctness on games involving that team, not the team's win rate. Each
game appears in two team rows; team rows must never be summed as independent
observations. Full-season team samples are at most 17 (ATL's 6/17 is descriptive),
and 2026 samples are smaller. No team-specific model is justified.

Week 1 has no current-season history: every A matchup has two zero difference
features and receives the same intercept-only probability. That is .54644 for
2025 and .54748 for retrospective 2026, with 9/16 and 10/16 home wins respectively.
These accuracies say nothing about team discrimination: there is none. When
only one team has played, the zero scoring fallback also gives different amounts
of information the same numeric treatment. Prior-season ratings, shrinkage, and
explicit game counts are plausible remedies; the 16-game slices cannot validate
them alone.

## Features, instability, and data correctness

A's standardized win-percentage coefficient changes from −.0946 on 2023 to +.1687
on 2023–2024 and +.2115 through 2025. Point-difference coefficients remain
positive (.4598, .3760, .3892). This is evidence of fit sensitivity, not evidence
that winning causes worse future performance: the two signals overlap, and each
coefficient is conditional on the other. `coefficients.csv` records standardized
and raw slopes plus standardized-space intercepts for every A–D fit. No 2026
coefficient fit is used for historical scoring.

B contains exact linear dependencies: home point difference equals points scored
minus points allowed, and each difference feature duplicates a home/away
combination already present. L2 regularization makes predictions estimable, but
individual coefficients are not unique causal explanations. More features do not
guarantee improvement: B loses to A on both 2024 and 2025 log loss.
`feature_correlations.csv` and `feature_distributions.csv` expose overlap, ranges,
missing counts, and season shifts, including partial 2026. For example, the
standard deviation of win-percentage difference rises from .337 in 2023 to .365
in 2025 while point-difference dispersion falls from 10.89 to 9.55. Partial 2026
is early-season weighted and must not be interpreted as comparable full-season
drift without week matching. No significance or causal claim is made from these
distribution summaries.

The source has 272 regular-season rows in each season, no missing kickoff values,
one tied game each in 2025 and 2026, and 136 unscored 2026 rows. Missing future
scores are expected and are never imputed as outcomes. Active normalization
validates IDs, scores, teams and duplicate matchups; unknown kickoff values remain
unavailable. Training excludes ties and only uses completed results before its
cutoff. Ties in prior history count as half wins and reset streaks. Cold starts
use .5 win percentage and zero scoring/streak values; this is a modeling choice,
not missing-value estimation learned from future games. Scalers fit training only.

Current-game and future results are excluded by strict kickoff and availability
filters. The new tests perturb target/future outcomes and verify invariance for
rolling, shrinkage, and Elo features; they also test delayed availability and
feature agreement with V1. A result at exactly the cutoff remains excluded.
**No target/future leakage was found under the implemented availability policy.**
That statement does not establish real point-in-time data fidelity: `available_at`
is kickoff plus 12 hours, not an observed completion/publication timestamp. Latest
final scores can incorporate corrections; suspended games and reschedules may not
be reconstructible. Authentic records validate timestamp chains but do not archive
their feature vectors or entire source snapshots. Their probability calculations
therefore cannot be fully replayed from the original inputs; integrity scoring is
reproducible against the current snapshot, which is a narrower guarantee.

## Controlled experiments

The fixed research grid uses 2023 for fitting; 2024 Weeks 1–9 (138 games) are used
only for calibrator fitting, and Weeks 10–18 (134 matched games) select the lowest
log loss. Noncalibrated candidates do not consume early-2024 labels for fitting.
All candidates see the same late-2024 game IDs. Hyperparameters never use 2025 or
2026 outcomes. This is a retrospective research protocol with previously published
2025 results, not an untouched prospective trial.

| Candidate | Validation accuracy | Brier | Log loss |
|---|---:|---:|---:|
| A, C=.01 | 58.21% | .232055 | .656716 |
| A, C=.1 | 68.66% | .222273 | .636520 |
| A, C=1 reference | 68.66% | .221352 | .634531 |
| A, C=10 | 68.66% | .221320 | .634449 |
| A, C=100 | 68.66% | .221318 | .634443 |
| Last 4 team games | 61.19% | .233766 | .660009 |
| Last 8 team games | 64.93% | .223876 | .639785 |
| Season-to-date, four-game prior | 70.15% | .212806 | .616035 |
| Elo | 74.63% | .200087 | .588365 |
| Sigmoid calibration | 64.93% | .232586 | .658007 |
| Isotonic calibration | 72.39% | .215579 | .622683 |
| Fit only Weeks 10–18 of 2023 | 69.40% | .208513 | .606333 |

Rolling candidates replace A's inputs with the last 4/8 eligible current-season
team results. Shrinkage uses `(wins + .5*ties + 2)/(n+4)` and total point margin
divided by `n+4`, equivalent to four neutral prior games. Elo starts all teams at
1500, K=20, fixed 50-point home adjustment in expected-score updates, and regresses
ratings 25% toward 1500 between seasons. It updates only on embargo-eligible
results, including ties, and feeds home-minus-away rating to scaled logistic
regression (C=1). Elo's intercept absorbs home advantage in the final forecast.
These are deliberately small fixed experiments, not optimized Elo parameters.

Sigmoid uses C=1 logistic regression on uncalibrated logits; isotonic clips outside
its calibration range. Both fit their calibrator chronologically after the base
model. For a final calibrated candidate the base would remain fitted on 2023 and
calibration would use 2024; it is not silently refitted underneath its calibrator.
This trades training data for honest separation and limits general conclusions
about calibration methods.

Validation-window sensitivity (`validation_window_sensitivity.csv`) scores every
noncalibrated configuration on early and full 2024 without changing selection.
Elo remains strongest on full-2024 log loss (.62475 versus A .66074), but its early
2024 accuracy is only 59.42%. Recent-history fitting improves late-2024 loss yet
worsens full-year loss (.66797); this is a strong warning against optimizing a
single favorable season segment. Early/full windows overlap the selection window
and are sensitivity checks, not independent confirmation. Calibrated candidates
are excluded because their early-2024 labels were consumed for calibration.

Only the selected Elo candidate receives a new 2025 holdout experiment, alongside
the fixed A reference; the published A–D reproduction is not new tuning.

| Matched 2025 games (n=271) | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Fixed V1 A | 61.25% | .226805 | .644337 |
| Selected Elo | 64.21% | .227689 | .649154 |

Elo wins eight additional classifications, but loses on both proper scores.
Its paired log-loss difference is **+.00482** (95% week-block bootstrap interval
−.03531 to +.04469; negative favors Elo). Its late-2024 difference was −.04617,
with interval −.08574 to +.01152. Neither establishes a stable log-loss advantage.
The script uses 2,000 paired week-block draws with seed 130, preserving identical
games for each comparison. Only nine validation blocks and 18 holdout blocks are
available. These descriptive intervals do not correct for testing 12 candidates,
cross-week team dependence, or prior awareness of the published holdout. Wilson
accuracy intervals also assume independent Bernoulli trials and are approximate
for repeated teams. No significance claim or deployment recommendation follows.

No research candidate is fitted or selected using 2026 outcomes. No new candidate
scores are reported on 2026; those outcomes are used only to diagnose the existing
frozen/recorded populations, avoiding another tuning loop.

## Ranked engineering recommendations

1. **V1: improve provenance and monitoring first.** Archive timestamped source
   snapshots, actual availability where obtainable, and feature vectors for future
   authentic forecasts. Report proper scores versus training-only constant
   baselines separately for recorded and retrospective cohorts. This addresses
   demonstrated replay limitations and the retrospective probability deficit.
2. **V1 research: test season-to-date shrinkage and game-count indicators in shadow
   evaluation.** The four-game prior lowers late-2024 loss by .01850 and full-2024
   loss by .01131, and addresses the known cold-start design. It did not win the
   predeclared selection, so it has no new 2025 confirmation here. Freeze a protocol
   before collecting additional forward data; no production change yet.
3. **V2: retain Elo as a benchmark and possible feature.** It offers cross-season
   information and better validation discrimination, but its selected holdout
   probabilities are worse than A. Evaluate a prespecified Elo-plus-shrunken-form
   model over additional chronological seasons; do not engineer a replacement
   from the accuracy gain alone. Earlier seasons would add independent validation
   windows; a future season is needed for genuinely fresh confirmation.
4. **V1: retain C=1 and current window pending more evidence.** Raising C improves
   late-validation loss by less than .0001 with identical accuracy; this is not
   worth an engineering change. Four-game rolling loses .02548 and eight-game
   rolling loses .00525 in validation loss. Shorter training is window sensitive.
5. **V2: investigate calibration after more chronological data.** Isotonic's
   validation improvement is uncertain and sigmoid worsens loss. Small reliability
   bins and the conflicting 2025/2026 pattern do not justify automatic correction.
   Use a separate later calibration window or chronological out-of-fold scores,
   never random folds or a calibrator fitted on the evaluation outcomes.
6. **V2: evaluate opponent strength, prior-season carryover, rest, neutral sites,
   roster/QB availability and injuries only with timestamped inputs.** These are
   hypotheses, not measured improvements in this sprint. Start with cheap
   schedule-derived features and ablate each family on matched chronological
   folds. Market probabilities can be an external benchmark only if timestamped
   and available under a suitable data agreement; no odds data were used here.

Before any replacement, freeze the feature/selection/calibration protocol and
evaluate it on multiple additional chronological windows with a final unused
forward period, matched game IDs, proper-score deltas and uncertainty intervals.
The current evidence supports research priorities, not a V2 performance promise.

## Output index and verification

All linked CSV files are under [research_v1_3](research_v1_3/):

- `population_metrics.csv`, `variant_reproduction.csv`: baseline and historical reproduction.
- `*_predictions.csv`: row-level scored populations and selected holdout predictions.
- `subgroups.csv`, `reliability.csv`: calibration, confidence, week, team and side slices.
- `candidate_validation.csv`, `selected_holdout.csv`, `validation_window_sensitivity.csv`: matched experiments.
- `coefficients.csv`, `feature_distributions.csv`, `feature_correlations.csv`, `data_audit.csv`: feature and source audits.
- `manifest.json`: provenance, cohort exclusions, hashes and environment.

Verification: **77 tests passed** using the locked existing Python 3.12.7
environment. Pytest emitted one Windows cache-permission warning; test execution
and isolated temporary research exports completed successfully. The diagnostic
CLI also completed with zero holdout reproduction error and unchanged artifact
hashes.

Tests cover production behavior plus research chronology, score exports and exact
historical replay. The end-to-end research test writes to pytest's isolated
temporary directory and asserts original artifact hashes remain unchanged. All
production code, models, prediction archives and published metrics are outside
this change. This branch is intended for a research-only PR into `main`, with no
automatic merge.
