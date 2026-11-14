"""Explicit bounded network audit: one PBP season and provider documentation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from src.research.efficiency import aggregate_parquet
from src.research.schedule_data import RESEARCH

DOCS = {
    "schedule_dictionary.csv": "https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_schedules.csv",
    "pbp_dictionary.csv": "https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_pbp.csv",
    "availability.Rmd": "https://raw.githubusercontent.com/nflverse/nflreadr/main/vignettes/articles/nflverse_data_schedule.Rmd",
    "depth_charts.R": "https://raw.githubusercontent.com/nflverse/nflreadr/main/R/load_depth_charts.R",
    "ep_models.html": "https://opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/",
    "model_training.R": "https://raw.githubusercontent.com/nflverse/fastrmodels/master/data-raw/MODELS.R",
    "model_serialization.R": "https://raw.githubusercontent.com/nflverse/fastrmodels/master/data-raw/DATASET.R",
    "nflverse_data_license.md": "https://raw.githubusercontent.com/nflverse/nflverse-data/main/LICENSE.md",
    "nfldata_README.md": "https://raw.githubusercontent.com/nflverse/nfldata/master/README.md",
}


def download(url, path, limit):
    path.parent.mkdir(parents=True, exist_ok=True)
    sha, size = hashlib.sha256(), 0
    started = datetime.now(timezone.utc).isoformat()
    with urlopen(url, timeout=60) as response, path.open("wb") as target:
        modified = response.headers.get("Last-Modified")
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError("Bounded audit download limit exceeded; stop affected audit")
            sha.update(chunk)
            target.write(chunk)
    return dict(url=url, retrieved_at=started, last_modified=modified, bytes=size, sha256=sha.hexdigest())


def run():
    evidence = {}
    old_path = RESEARCH / "efficiency_audit.json"
    previous = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
    for name, url in DOCS.items():
        path = Path(".review/v2_0_2/provider_docs") / name
        try:
            evidence[name] = download(url, path, 2 * 1024**2)
        except Exception as exc:
            evidence[name] = dict(url=url, error=str(exc))
    path = Path(".review/v2_0_2/play_by_play_2021.parquet")
    url = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.parquet"
    try:
        if path.exists() and "pbp_2021" in previous:
            if hashlib.sha256(path.read_bytes()).hexdigest() != previous["pbp_2021"]["sha256"]:
                raise ValueError("Cached PBP checksum differs from recorded input")
            evidence["pbp_2021"] = previous["pbp_2021"].copy()
        else:
            metadata = download(url, path, 128 * 1024**2)
            if "pbp_2021" in previous and metadata["sha256"] != previous["pbp_2021"]["sha256"]:
                raise ValueError("Released PBP changed; stop affected experiment and use a new audit")
            evidence["pbp_2021"] = metadata
        table, audit = aggregate_parquet(path)
        table.to_csv(RESEARCH / "efficiency_team_games_2021.csv", index=False, lineterminator="\n")
        evidence["pbp_2021"].update(audit=audit, team_game_rows=len(table),
                                   epa_incomplete_team_games=int((~table.epa_valid).sum()),
                                   success_incomplete_team_games=int((~table.success_valid).sum()),
                                   ep_model_training_end=None, historical_publication_verified=False,
                                   decision="exploratory only; excluded from candidate comparisons")
    except Exception as exc:
        evidence["pbp_error"] = dict(url=url, error=str(exc), decision="affected experiments stopped")
    evidence["qb_decision"] = "Blocked: final passer IDs are not pregame expected-starter assignments. No verified historical starter revision archive."
    (RESEARCH / "efficiency_audit.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    run()
