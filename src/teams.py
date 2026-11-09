"""Current team display names, keyed by nflverse abbreviations."""
NAMES = dict(zip(
    "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split(),
    ["Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills", "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns", "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers", "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs", "Los Angeles Rams", "Los Angeles Chargers", "Las Vegas Raiders", "Miami Dolphins", "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants", "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "Seattle Seahawks", "San Francisco 49ers", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"]))


def display(team):
    return f"{NAMES.get(team, team)} ({team})"
