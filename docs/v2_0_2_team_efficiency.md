# V2.0.2 — team-context research

Frozen prior-season Elo improves the reconstructed 2021–2024 comparison against
V1 A: pooled Brier falls from **0.238954 to 0.232946** and log loss from
**0.671689 to 0.658900** on 1,084 identical non-tied games. Rest adds little pooled
benefit and worsens both proper scores in 2024. Investigate Elo further; retain
production V1. These are **latest-data reconstructions**, not independently
verified historical pregame forecasts or a fresh holdout.

PR #7 was verified merged at `85aca3215969f12ff9e65daf007185e110f82583`
before creating `feature/v2.0.2-team-efficiency` from latest main. This sprint
changes no production models, archived predictions, dashboard, refresh workflow,
or durable artifact branch. Research fits remain in memory; fitted coefficients
and training IDs are research JSON, never production joblib replacements.

## Audited inputs and availability

Schedules are attributed to **Lee Sharpe / nflverse**. The committed compressed
input preserves the exact downloaded CSV, not an expanded production store.
[Source metadata](../research/v2_0_2/source.json) records its schema, URL,
retrieval time **2026-11-14 21:19:11 UTC**, source commit
`68962ec7c9beab9dc3f449744ebea3b1a3a97079`, and raw SHA-256
`44f6ba50a76303869229be8fdad82d124e77f24f50c90c319fbc245dd34120a0`.
[Pinned upstream source](https://raw.githubusercontent.com/nflverse/nfldata/68962ec7c9beab9dc3f449744ebea3b1a3a97079/data/games.csv).

| Seasons | REG games | Purpose | Missing kickoffs/scores | Teams |
|---|---:|---|---:|---:|
| 2015–2016 | 512 | Elo warm-up only | 0 / 0 | 32 each |
| 2017–2020 | 1,024 | Initial training; later expands | 0 / 0 | 32 each |
| 2021 | 272 | Evaluation; 271 binary outcomes | 0 / 0 | 32 |
| 2022 | 271 | Evaluation; 269 binary outcomes | 0 / 0 | 32 |
| 2023 | 272 | Evaluation | 0 / 0 | 32 |
| 2024 | 272 | Evaluation | 0 / 0 | 32 |

All 2,623 audited games have unique IDs, valid scores and known Eastern kickoff
times. ID season/week/matchup checks pass after alias reconciliation. Original
source IDs are retained: 124 use historical franchise tokens rather than the
normalized modern tokens. `STL/LAR→LA`, `SD→LAC`, `OAK→LV`, `JAC→JAX`, and
`WSH→WAS` preserve franchise rating history. Team counts match 16 games before
2021 and 17 thereafter, with BUF/CIN at 16 in 2022. The
[NFL canceled their Week 17 game](https://www.nfl.com/_amp/week-17-buffalo-cincinnati-game-will-not-be-resumed-neutral-afc-championship-gam);
no row or result is invented for it. Per-season details:
[coverage.csv](../research/v2_0_2/outputs/coverage.csv).

The source commit/retrieval proves what was downloaded now, not what was
published or ingested before historical forecasts. Scores, kickoffs, and venues
may have been revised. Forecast cutoff is kickoff minus 24 hours. Since actual
final-whistle times are unavailable, availability is explicitly *assumed* to be
kickoff plus a fixed four-hour game-duration proxy plus 24/48/72 postgame hours.
The primary assumption is 48 hours. Neither that duration nor any lag certifies
actual completion or publication. Latest rescheduled kickoffs are not a verified
archive of earlier locks. Independent point-in-time evaluation remains blocked
until archived revisions and ingestion timestamps exist.

The upstream [schedule documentation](https://nflreadr.nflverse.com/reference/load_schedules.html)
identifies the maintained source; the
[availability documentation](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
also describes subsequent PBP corrections. Data terms were inspected: the
[nflverse-data distribution license](https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md)
is CC BY 4.0; the direct nfldata repository does not expose an equivalent explicit
license in the inspected README. Attribution is retained here; broader commercial
redistribution rights are not asserted by this research.

## Fixed design and fallbacks

[protocol.json](../research/v2_0_2/protocol.json) fixed the configurations and lag
comparisons before candidate scoring. Exactly three configurations use StandardScaler and L2 logistic
regression, C=1, threshold 0.5, with no tuning or calibration search:

1. **V1 A:** season-to-date win percentage and mean point differential differences,
   using the existing pregame feature function at the research cutoff.
2. **A + prior Elo:** configuration 1 plus frozen entering-season Elo difference.
3. **A + prior Elo + rest:** configuration 2 plus rest difference and each side's
   opener/unavailable flags. Venue features are omitted.

Elo starts genuinely unseen franchises at 1500, uses K=20 and 50 home points,
regresses 25% toward 1500 each offseason, and has no margin multiplier. Only
eligible previous regular-season results update the entering rating; ties update
as 0.5. Ratings freeze for the whole target season. Two complete warm-up seasons
precede the first 2017 training rows. Counts and initialization/incomplete-history
flags are exported; evaluation rows have no initialized or incomplete Elo sides.

Rest uses the most recent eligible played REG game in the same season and New
York calendar dates, clips each side at 14 days, and defaults to 7 with explicit
flags if no eligible history exists. Season openers reset instead of treating
offseason months as useful rest. Postponed/unscored/unknown-kickoff games never
become played history. Missing target kickoffs are excluded, not assigned midnight.
At the primary lag, rest fallbacks are the 32 team opener appearances per season;
there are no other gaps. The 72-hour assumption additionally withholds four
team-game histories across 2021–2023, explicitly flagging those fallbacks.

Venue metadata contains only Home/Neutral with no unknowns, including 19 neutral
games in 2021–2024. Historical *pregame* publication/revision provenance is still
unverified, so neutral coefficients and neutral Elo adjustments were disabled
before scoring. The reusable feature layer can expose a neutral indicator and
unknown flag, but the initial comparison does not enable them. No constant
home-field column is added to an intercept model.

For each 2021–2024 fold, training uses 2017 through the preceding season only;
all training labels must precede that fold's first forecast. Coefficients and
scalers freeze for the evaluation season. Earlier evaluation seasons may enter
later training windows. All candidates and comparators use identical evaluation
IDs and fallback rules. Three ties are excluded: DET–PIT 2021, IND–HOU 2022,
WAS–NYG 2022; other exclusions are zero. Those tied games remain eligible history.
2025 and observed 2026 were previously inspected and receive no new candidate
scores. Training V1 A on these earlier windows and 24-hour cutoffs is a matched
research reference, not a replay of the deployed model or its original metrics.

The methodological clarification beyond the feasibility note is the explicit
four-hour end-time proxy, needed to interpret its postgame lag without inventing
final-whistle timestamps. Implementation review corrected an initial kickoff-plus-
lag pass after that initial run; the four-hour proxy was not searched or selected
on performance. All final lag comparisons were rerun with this shared assumption.

## Matched results

Primary 48-hour postgame assumption; pooled across four expanding folds:

| Method | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| V1 A | 1,084 | 0.602399 | 0.238954 | 0.671689 |
| A + prior Elo | 1,084 | 0.613469 | 0.232946 | 0.658900 |
| A + prior Elo + rest | 1,084 | 0.617159 | 0.232907 | 0.658746 |
| Constant 50% | 1,084 | 0.541513 | 0.250000 | 0.693147 |
| Training-only home rate | 1,084 | 0.541513 | 0.248362 | 0.689868 |
| Fixed frozen Elo probability | 1,084 | 0.607011 | 0.236958 | 0.666822 |

Constant 50% predicts designated home under the fixed `p >= .5` convention;
its accuracy is therefore not necessarily 50%. The fixed Elo comparator uses
`1/(1+10**(-(prior_elo_diff+50)/400))` without fitting on evaluation labels.

Season differences versus matched V1 (negative proper-score deltas favor candidate):

| Season | Binary games | Elo Δaccuracy | Elo ΔBrier | Elo Δlog loss | Elo/rest ΔBrier | Elo/rest Δlog loss |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | 271 | -0.003690 | -0.012248 | -0.026343 | -0.012233 | -0.026286 |
| 2022 | 269 | +0.022305 | -0.000409 | -0.000404 | -0.001029 | -0.001786 |
| 2023 | 272 | +0.003676 | -0.003877 | -0.008375 | -0.005328 | -0.011427 |
| 2024 | 272 | +0.022059 | -0.007457 | -0.015949 | -0.005563 | -0.012201 |

Elo's pooled changes versus V1 are Brier **−0.006008**, log loss **−0.012789**.
Against fixed Elo-only probabilities they are −0.004012 and −0.007922.
Both learned candidates clear the *numerical historical screen*: four complete
folds, >500 matched games, pooled Brier improvement ≥0.002 and log-loss improvement
≥0.005 versus V1 and every probability comparator, and improvement in both scores
versus V1 in all four seasons. This does not clear the chronology/provenance or
prospective confirmation requirements.

Rest adds only **−0.000039 Brier / −0.000154 log loss** versus Elo alone, despite
slightly higher pooled accuracy. It worsens both scores in 2021 and 2024. At a
72-hour lag it also worsens pooled proper scores versus Elo alone. These results
do not justify choosing rest based on accuracy.

All three lag assumptions were run, not selected after scoring:

| Assumed postgame lag | V1 Brier / loss | Elo Brier / loss | Elo/rest Brier / loss |
|---|---|---|---|
| 24 hours | 0.238954 / 0.671689 | 0.232946 / 0.658900 | 0.232907 / 0.658746 |
| 48 hours | 0.238954 / 0.671689 | 0.232946 / 0.658900 | 0.232907 / 0.658746 |
| 72 hours | 0.239490 / 0.672806 | 0.233320 / 0.659710 | 0.233612 / 0.660596 |

Descriptive paired resampling uses 5,000 shared `(season, week)` block draws,
seed 20202. For Elo versus V1 at the primary lag, 95% intervals are
**[−0.010653, −0.001178]** Brier and **[−0.022794, −0.002234]** log loss.
Grouping contiguous two-week pairs instead gives [−0.010111, −0.001808] and
[−0.021826, −0.003697]. These are development-population uncertainty summaries,
not prospective confirmation or correction for prior research/selection.
Cross-block repeated-team dependence remains. Comparator/candidate intervals and
all lag/season metrics are exported without selective reporting.

## EPA, success rate, and QB feasibility

One bounded 2021 PBP audit downloaded **20,249,925 bytes** from the official
[nflverse release](https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.parquet),
retrieved 2026-11-14 21:22:43 UTC, SHA-256
`333ad34378e5339d5172717cc83378e908daf02c8699416ab3e17c2ec10f78d8`.
The response's Last-Modified is January 8, 2026, not evidence of 2021 availability.
The raw Parquet stays in ignored scratch; the 1,088 offense/defense team-game
aggregate rows and [audit provenance](../research/v2_0_2/efficiency_audit.json) are
committed. No claim of a complete multi-season PBP audit is made.

The reusable aggregator projects 12 fields in batches of 65,536, one season at a
time, with a 128 MiB download ceiling. It retains REG pass/run plays, sacks and
scrambles under provider classifications, excludes kneels/spikes/no-play and
special teams, keeps overtime, and applies no score-dependent garbage-time filter.
Of 49,922 raw rows, **34,139** are eligible; their team IDs, EPA and success fields
are complete. Offense EPA and defense EPA allowed remain separate with the same
possession-team sign; positive defensive EPA allowed is worse defense. A game
requires ≥95% valid eligible plays for each metric. Unknown efficiency is never
zero: suppress incomplete game contributions, use a training-only league prior
with fixed 200-play shrinkage, and retain counts/unavailable flags. Prior and
pregame helpers exclude target games and evaluation-season prior labels.

**EPA/success remain exploratory and are not model-comparison inputs.** The
[EP training source](https://raw.githubusercontent.com/nflverse/fastrmodels/master/data-raw/MODELS.R)
loads a calibration dataset without a verifiable training-end date or lineage
mapping to this PBP release. Its
[serialization history](https://raw.githubusercontent.com/nflverse/fastrmodels/master/data-raw/DATASET.R)
documents later re-exports; serialization dates do not prove model training
cutoffs. Neither that source nor the authors' model description establishes a
chronologically eligible EP model for these folds. Success defined as positive
EPA inherits the same risk. No EP refitting or additional efficiency candidates
are run until a pre-fold trained model and historical releases can be pinned.

The audit finds 20,907 dropbacks, with 910 missing `passer_player_id` values
(4.35%). That single field does not identify every sack/scramble dropback;
additional passer/rusher and stable player-ID mapping would be needed. Final
schedule QB IDs or observed attempts identify participants, not expected starters.
[Historical depth charts](https://nflreadr.nflverse.com/reference/load_depth_charts.html)
document weekly entries back to 2001, while timestamped append-only updates are
documented from 2025. Weekly historical membership does not establish the expected
starter at the 24-hour cutoff. Trades, starter switches, rookies, and uncertain
assignments require player-ID joins and timestamped starter assumptions. Keep QB
efficiency blocked until that provenance exists; an unknown starter must retain
an unknown flag/team prior, never be filled using who ultimately played.

## Reproduce and verify

Using the existing locked Python 3.12 environment, offline from committed inputs:

```powershell
python -m src.research.schedule_data
python -m scripts.v2_team_efficiency --output .review/v2_0_2/reproduction
python -m pytest -q --basetemp .pytest-tmp-v202-full -p no:cacheprovider
```

Fetching is explicit: `python -m src.research.schedule_data --fetch` refuses to
replace a changed pinned snapshot; use a new research directory for a new source.
`python -m scripts.audit_v2_efficiency` audits documentation and retrieves/reuses
one bounded PBP file with checksum validation. Mutable upstream releases must
match the recorded checksum to reproduce this audit. No network is used by the
experiment runner or tests. Nothing writes to `artifacts/` or invokes production
training, prediction, refresh, or evaluation writers.

[Experiment manifest](../research/v2_0_2/outputs/manifest.json) includes the protocol,
raw source checksum, transform/code hashes, environment, output checksums and
production-artifact hashes. [folds.json](../research/v2_0_2/outputs/folds.json)
records every training/evaluation game ID, latest training label availability,
first forecast, scalers and fitted coefficients. CSV outputs contain row-level
features/cutoffs/fallbacks, predictions, exclusions, all scores and paired deltas.
Numerical reproduction is tested to 1e-12; raw source checksum is exact. Git text
line-ending conversion may alter file hashes across checkouts, so use the
manifest generated by that run for byte-level output verification.

Explicit `--directory research/v2_0_2` and absolute external input directories
are supported. Manifest `code_sha256` keys stay repository-relative for files
inside the repository. External files use canonical absolute `file:` URIs,
including percent-escaped spaces, so distinct files with the same basename stay
unambiguous. Relative path aliases resolve to the same identifier; relocating an
external input changes its URI but not its content checksum. Source SHA-256
validation and complete per-file hashing remain mandatory. Use `--output` to
write a reproduction separately and preserve the frozen historical outputs.
Path-handling follow-up validation: **154 tests passed**, including real default,
relative-directory and external-directory CLI reproductions, unchanged frozen
outputs/production bytes, and rejection of a corrupted external input checksum.

Tests cover target/future-outcome mutation, frozen Elo/warm-up determinism,
evaluation-label exclusion from fitting/scaling/priors, missing score/kickoff and
rest/venue fallbacks, DST/calendar behavior, play filtering, 95% suppression,
200-play shrinkage, input checksum rejection, production output-path rejection,
and complete recorded-input replay with unchanged production bytes.

Verification: **150 tests passed** in 200.67 seconds with the full command above.
The recorded-input reproduction matches exported predictions, pooled/season
scores, exclusions, fallbacks and paired intervals to 1e-12. All production
artifacts remain byte-identical to latest main; the dashboard, workflows and
production source modules also have zero diff. No production model was retrained
or promoted and no archived forecast was created, modified or backdated.

## Recommendation

Prior-season Elo deserves a prespecified forward shadow experiment: it improves
proper scores consistently in the historical reconstruction. Rest has weak
incremental evidence and should remain a secondary ablation. Venue, EPA/success
and QB inputs need stronger historical provenance before comparison/promotion.
Neither candidate is promoted: archive actual pregame revisions and ingestion
times, freeze the design before a complete prospective season, and apply the
feasibility report's confirmation gates without interim tuning. Historical
reconstruction alone cannot answer whether the same advantage survives a real
point-in-time deployment.
