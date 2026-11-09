# Coverage before V1.1 implementation · November 9, 2026

Read-only audit of the existing local schedule **before implementation or refresh**. Snapshot fetched November 5, 2026 at 22:03 UTC; stale at audit. All 272 regular-season games are present. The 29 archived picks exist because the initial CLI was run for Weeks 9–10; earlier games had already started and were correctly excluded from live recording. Future weeks are present in the schedule but have not yet received picks. No 2026 retrospective pipeline existed; `backtest.json` contains the separate 2025 holdout.

| Week | Schedule rows | Completed results | Authentic picks | Missing results after kickoff* | Missing authentic picks | Retrospective | Completed games needing retrospective |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 0 | 0 | 16 | 0 | 16 |
| 2 | 16 | 16 | 0 | 0 | 16 | 0 | 16 |
| 3 | 16 | 16 | 0 | 0 | 16 | 0 | 16 |
| 4 | 16 | 16 | 0 | 0 | 16 | 0 | 16 |
| 5 | 15 | 15 | 0 | 0 | 15 | 0 | 15 |
| 6 | 14 | 14 | 0 | 0 | 14 | 0 | 14 |
| 7 | 14 | 14 | 0 | 0 | 14 | 0 | 14 |
| 8 | 14 | 14 | 0 | 0 | 14 | 0 | 14 |
| 9 | 15 | 0 | 15 | 14 | 0 | 0 | 0 |
| 10 | 14 | 0 | 14 | 0 | 0 | 0 | 0 |
| 11 | 13 | 0 | 0 | 0 | 13 | 0 | 0 |
| 12 | 16 | 0 | 0 | 0 | 16 | 0 | 0 |
| 13 | 14 | 0 | 0 | 0 | 14 | 0 | 0 |
| 14 | 15 | 0 | 0 | 0 | 15 | 0 | 0 |
| 15 | 16 | 0 | 0 | 0 | 16 | 0 | 0 |
| 16 | 16 | 0 | 0 | 0 | 16 | 0 | 0 |
| 17 | 16 | 0 | 0 | 0 | 16 | 0 | 0 |
| 18 | 16 | 0 | 0 | 0 | 16 | 0 | 0 |
| **Total** | **272** | **121** | **29** | **14** | **243** | **0** | **121** |

*A passed kickoff without both scores is not proof the game has finished. This count includes potential underway/postponed games and stale-source omissions. Future unplayed games are not missing results. Missing authentic forecasts are permanent historical gaps, not missing schedule rows; future missing forecasts are not yet overdue. Completed games with authentic picks do not require a second retrospective pick. Refresh and the new reproducible coverage command will produce the final report without rewriting this baseline.
