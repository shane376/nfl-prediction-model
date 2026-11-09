# Pre-merge review · November 6, 2026

**Ready for reviewed integration and deployment setup; no merge or deployment performed.** This pass preserves V1 A–D and the existing architecture. No authentic artifact or original prediction was changed. Feature branch: `feature/2026-dashboard-mvp`.

## Confirmed issues and fixes

| Issue | Fix |
|---|---|
| Generation time captured before work could backdate a slow prediction across kickoff | Separate feature-data cutoff and generation time; recheck real clock after computation and disk flush; require 30 seconds remaining before atomic publication |
| Revision filename collisions could overwrite originals | Atomic create-only hard-link publication, fsync, per-store exclusive writer lock, revision numbering from highest saved revision |
| Stale/future source metadata could produce live forecasts | Require successful timezone-aware fetch within 24 hours; fail closed on error/offline fixture metadata |
| Missing or revised kickoffs could leave invalid picks pending or displayed as valid | Shared validation of timestamp chain, both published kickoffs, matchup, probability pair and winner; mark uncertain/late records unavailable |
| Later postponements could reopen prediction eligibility without an earlier pick | Persist earliest known kickoff across refreshes and never reopen an expired window |
| Partial/invalid scores and guessed naive kickoff times were insufficiently guarded | Reject negative/fractional/nonfinite/nonnumeric scores; partial scores remain unavailable; naive/unknown kickoffs stay unknown |
| A truncated source could replace a good cached schedule | Reject snapshots dropping cached game IDs, preserve prior bytes and report failure |
| Dashboard relied on possibly obsolete derived live metrics and could fail on corrupt artifacts | Pure read-only reconciliation shared with CLI; explicit unavailable states, metadata-aware cache invalidation and defensive artifact reads |
| Refresh workflow pushed to obsolete feature branch | Read reviewed code from dynamic default branch, append records on `automation/nfl-artifacts`, open/reuse a review PR; no main push, force push, auto-merge, implicit training or deployment |
| Automation lacked a history gate | Reject changes/deletions of original prediction files before publication and in PR CI; keep failures visible and diagnostics downloadable |

## Integrity and verification evidence

- **53 pytest checks passed** in a fresh Python **3.12.7** environment installed from `requirements.txt`/`requirements.lock`; dependency check reported no broken requirements. Regression coverage includes compute/flush crossing kickoff, concurrent writers, filename collisions, reschedules, timezone/score errors, blank/TBD kickoff times, numeric probability types, timestamp chains, source truncation, immutable Git history and all dashboard states.
- Checked-in model `v1A-1aa64cb4b1` loaded successfully in that environment. `ty` checks passed for the dashboard, reviewed pipeline modules and history guard. Workflow YAML parsed successfully. Linux CI status is reported separately by GitHub after push; the live scheduled job cannot be exercised before approved default-branch integration.
- CLI **refresh, train, predict (Weeks 10 and 11), evaluate, offline evaluate and audit** all exited successfully in an ignored isolated copy. It held 42 records after generating verification-only Week 11 picks; none were copied back into production artifacts.
- All **29 original picks** passed full integrity checks against a fresh nflverse snapshot fetched **2026-11-06 05:30 UTC**. Their first Git publication commits also precede the provider's corresponding kickoff timestamps. SHA-256 comparisons confirmed **all original artifacts unchanged**, including model, backtests, live records and derived results.
- Live Streamlit agent API: health `ok`, all four tabs present, no exceptions across Weeks 1/9/10/18 and search/result-filter changes. Artifact hashes unchanged after page interaction. Unit UI coverage includes missing model metadata, stale source, pending/completed/tied/postponed/TBD games, malformed records and corrupt JSON. No browser pixel-level or public URL verification is claimed.
- Temporal review: shared train/inference features exclude the current game, missing scores, non-completed games and results unavailable before the cutoff. Selection remains 2023 fit / 2024 validation; 2025 is retrospective holdout, separate from production refit and timestamped live picks. The original playground's prior 2025 experimentation remains disclosed. Model/artifact policy and historical evaluation figures were not changed by this review.

## Remaining risks and launch actions

1. **Approval required:** review/merge this feature into `main`; separately approve Community Cloud deployment. Select repository `shane376/nfl-prediction-model`, branch `main`, entrypoint `dashboard/app.py`, Python 3.12. Root requirements install pinned model-compatible dependencies. No secrets or external persistence service required.
2. **Actions setup:** allow contents and pull-request write scopes and Actions PR creation. Protect `main` with reviews; bot only pushes to the artifact branch. Keep that branch and use merge commits for recurring data PRs. Dispatch the refresh after integration and verify its first successful artifact PR. Review/merge artifact updates promptly; the public app shows only the last reviewed snapshot.
3. **Required CI checks:** token-created PRs do not generally trigger another workflow. Tests run within refresh; manually dispatch CI on the artifact branch if required by protection rules. Never bypass a missing required check.
4. **Source limitation:** nflverse supplies published kickoff times and scores, not independently verified actual-start/final-whistle timestamps. Existing picks precede all available published kickoffs; actual wall-clock starts cannot be independently proven from this dataset. The 12-hour result embargo remains a documented conservative assumption, not a guarantee for an exceptionally long/suspended game. Final status/timestamps would require another source, outside this focused pass.
5. **Recovery:** source failures, ID changes, merge conflicts, rejected pushes and immutable-history violations stop publication without deleting prior Git records. Diagnose uploaded artifacts/logs, preserve all original predictions, resolve normally and rerun. A crash can leave a writer lock; confirm the process is gone before removing only the lock. No force-push/reset recovery. See README for PR-permission failures after a successful data-branch push.
6. **Operational limits:** free schedules may be late or disabled for inactivity. Turn on failure notifications, monitor source age and review backlog. Uploaded diagnostics expire (up to 90 days); Git is the durable record. No public deployment, hosted Linux runtime or scheduler execution is claimed until actually verified after approval.

Exact local, isolation, deployment and workflow-permission instructions are in `README.md`. Next action is review of the feature-branch commits and approval of integration/deployment, not V2 feature work.
