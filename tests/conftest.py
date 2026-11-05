import pandas as pd
import pytest
from src.data import normalize


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    from src import data, predictions
    fixed = lambda: pd.Timestamp("2026-11-05T20:00:00Z")
    monkeypatch.setattr(data, "now", fixed)
    monkeypatch.setattr(predictions, "now", fixed)


@pytest.fixture
def games():
    # Explicitly synthetic test fixture, never used as production source data.
    rows = []
    for season in (2023, 2024, 2025, 2026):
        for week in range(1, 9):
            rows.append(dict(game_id=f"{season}_{week:02d}_A_B", season=season, week=week,
                             game_type="REG", kickoff=f"{season}-09-{week*3:02d}T17:00:00Z",
                             home_team="A", away_team="B", home_score=21 if week % 2 else 7,
                             away_score=14))
    return normalize(pd.DataFrame(rows))
