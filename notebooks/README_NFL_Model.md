# NFL Prediction Model — Data Setup

## Generate the schedule CSV

Run:

```bash
python create_nfl_schedule.py
```

The script creates:

```text
2023-2025 nfl schedule.csv
```

The CSV has:

- 32 teams as rows
- 54 weekly columns: 2023 W1-W18, 2024 W1-W18, 2025 W1-W18
- `W` = win
- `L` = loss
- `T` = tie
- `BYE` = no game that week

This is the simple starting dataset for the NFL prediction model.

The script uses nflverse's game/schedule dataset, which contains historical NFL games and schedule information.
