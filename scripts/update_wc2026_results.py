"""Fetch completed WC 2026 results and push to HuggingFace storage.

Run this script any time after matches have been played to update the
leaderboard in the WorldcupPredictionGame HuggingFace Space.

Requirements:
  - FOOTBALL_DATA_API_KEY in .env or environment
  - HF_TOKEN in .env or environment (with write access to
    Koddenbrock/WorldcupPredictionGame-storage)

Usage:
    python scripts/update_wc2026_results.py               # clears cache + fetches live
    python scripts/update_wc2026_results.py --dry-run     # print only, no upload
    python scripts/update_wc2026_results.py --local       # save locally only
    python scripts/update_wc2026_results.py --no-refresh  # reuse cached responses

By default the script clears the cached WC responses first (the football-data.org
cache has no TTL, so a stale cache otherwise replays the matches finished at the
time of the first fetch). Pass --no-refresh to reuse the cache.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()
app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPETITION = "WC"
SEASON = 2026
HF_REPO   = "Koddenbrock/WorldcupPredictionGame-storage"
HF_FILE   = "results/match_results.json"
LOCAL_OUT = Path("data/predictions/wc2026_results.json")

# football-data.org responses are cached permanently (no TTL), so a stale cache
# keeps replaying the matches finished at first fetch. Clear it before fetching.
from footyml.config import RAW_DIR  # noqa: E402

FD_CACHE_DIR = RAW_DIR / "football_data_api"

# football-data.org stage → scoring.py stage key
_STAGE_MAP: dict[str, str] = {
    "GROUP_STAGE":    "GROUP_STAGE",
    "LAST_32":        "ROUND_OF_32",
    "LAST_16":        "ROUND_OF_16",
    "QUARTER_FINALS": "QUARTER_FINALS",
    "SEMI_FINALS":    "SEMI_FINALS",
    "THIRD_PLACE":    "THIRD_PLACE",
    "FINAL":          "FINAL",
}

# football-data.org group string → GROUP_X key used in predictions
# The API returns "GROUP_A", "GROUP_B", etc. — should match directly.
# Add overrides here if the API returns something unexpected.
_GROUP_MAP: dict[str, str] = {}

# Team name normalization: football-data.org name → prediction file name
_TEAM_NAME_MAP: dict[str, str] = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Côte d'Ivoire":          "Ivory Coast",
    "Congo DR":               "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",
    "Republic of Ireland":    "Ireland",
    "Korea Republic":         "South Korea",
    "IR Iran":                "Iran",
    "United States":          "United States",
    "USA":                    "United States",
    "Czechia":                "Czechia",
    "Czech Republic":         "Czechia",
    "Cape Verde":             "Cape Verde Islands",
    "Curaçao":                "Curaçao",
}


def normalize_team(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)


def normalize_group(group: str | None) -> str | None:
    if group is None:
        return None
    return _GROUP_MAP.get(group, group)


# ---------------------------------------------------------------------------
# Result encoding
# ---------------------------------------------------------------------------

def encode_result(home_goals: int | None, away_goals: int | None) -> str | None:
    """Convert goals to 'A Win' / 'Draw' / 'B Win' (A = home, B = away)."""
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "A Win"
    elif home_goals < away_goals:
        return "B Win"
    return "Draw"


def ko_winner_from_api(match_row: dict, home_name: str, away_name: str) -> str:
    """For KO matches, return the advancing team name (may be decided by pens)."""
    # football-data.org returns score.winner = "HOME_TEAM" | "AWAY_TEAM" | "DRAW"
    # In KO rounds, there is no draw — winner is always set.
    api_winner = str(match_row.get("score", {}).get("winner", "") or "")
    if api_winner == "HOME_TEAM":
        return home_name
    elif api_winner == "AWAY_TEAM":
        return away_name
    # Fallback: use goal score
    home_g = match_row.get("home_goals")
    away_g = match_row.get("away_goals")
    if home_g is not None and away_g is not None:
        return home_name if home_g >= away_g else away_name
    return ""


# ---------------------------------------------------------------------------
# Main fetch + transform
# ---------------------------------------------------------------------------

def clear_cache() -> int:
    """Delete cached WC match responses so the next fetch hits the live API."""
    if not FD_CACHE_DIR.exists():
        return 0
    files = list(FD_CACHE_DIR.glob(f"fd_matches_{COMPETITION}*.parquet"))
    for f in files:
        f.unlink()
    return len(files)


async def fetch_results() -> list[dict]:
    """Fetch all completed WC 2026 matches from football-data.org."""
    from footyml.providers.football_data import FootballDataProvider

    async with FootballDataProvider() as provider:
        df = await provider.get_completed(COMPETITION, season=SEASON)

    if len(df) == 0:
        console.print("[yellow]No completed matches found.[/yellow]")
        return []

    results: list[dict] = []
    for row in df.iter_rows(named=True):
        stage_raw = str(row.get("stage") or "")
        stage     = _STAGE_MAP.get(stage_raw, stage_raw)
        group_id  = normalize_group(str(row.get("group_id") or "")) or None

        home_name = normalize_team(str(row.get("home_club_name") or ""))
        away_name = normalize_team(str(row.get("away_club_name") or ""))

        home_goals = row.get("home_goals")
        away_goals = row.get("away_goals")
        actual_result = encode_result(home_goals, away_goals)
        if actual_result is None:
            continue

        is_group = stage == "GROUP_STAGE"
        winner = "" if is_group else ko_winner_from_api(
            {"score": {"winner": None}, "home_goals": home_goals, "away_goals": away_goals},
            home_name, away_name
        )

        results.append({
            "game_id":       str(row.get("game_id") or ""),
            "stage":         stage,
            "group_id":      group_id,
            "home_team":     home_name,
            "away_team":     away_name,
            "home_goals":    home_goals,
            "away_goals":    away_goals,
            "actual_result": actual_result,
            "winner":        winner,  # empty for group stage, team name for KO
        })

    return results


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------

def upload_to_hf(results: list[dict]) -> None:
    """Upload results JSON to HuggingFace dataset repo."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        console.print("[red]HF_TOKEN not set — cannot upload.[/red]")
        console.print("Set it with: export HF_TOKEN=hf_...")
        raise typer.Exit(1)

    from huggingface_hub import HfApi
    api = HfApi()
    payload = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
    api.upload_file(
        path_or_fileobj=payload,
        path_in_repo=HF_FILE,
        repo_id=HF_REPO,
        repo_type="dataset",
        token=hf_token,
        commit_message=f"Update WC 2026 results — {len(results)} matches",
    )
    console.print(f"  [green]✓[/green] Uploaded to [bold]{HF_REPO}[/bold] / {HF_FILE}")


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    table = Table(title=f"WC 2026 Completed Matches ({len(results)} total)", show_lines=False)
    table.add_column("Date", style="dim", width=10)
    table.add_column("Stage", style="cyan", width=14)
    table.add_column("Group", style="dim", width=8)
    table.add_column("Home", width=20)
    table.add_column("Score", justify="center", width=7)
    table.add_column("Away", width=20)
    table.add_column("Result", justify="center", width=8)

    result_icon = {"A Win": "🟢", "Draw": "🟡", "B Win": "🔴"}

    for r in sorted(results, key=lambda x: (x.get("stage", ""), x.get("group_id", ""))):
        hg = r.get("home_goals")
        ag = r.get("away_goals")
        score = f"{hg}–{ag}" if hg is not None else "?"
        table.add_row(
            "",
            r.get("stage", ""),
            r.get("group_id") or "",
            r.get("home_team", ""),
            score,
            r.get("away_team", ""),
            result_icon.get(r.get("actual_result", ""), "?"),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print results only, don't upload"),
    local:   bool = typer.Option(False, "--local",   help="Save locally without uploading to HF"),
    season:  int  = typer.Option(SEASON, "--season", help="WC season year"),
    refresh: bool = typer.Option(
        True, "--refresh/--no-refresh",
        help="Clear the cached WC responses before fetching (default: on)",
    ),
) -> None:
    console.rule("[bold]WC 2026 Results Updater")

    # Clear stale cache so we fetch live results (cache has no TTL).
    if refresh:
        n = clear_cache()
        console.print(f"Cleared [bold]{n}[/bold] cached WC response(s).")

    # Fetch
    console.print("Fetching completed matches from football-data.org…")
    results = asyncio.run(fetch_results())

    if not results:
        console.print("[yellow]No finished matches to update.[/yellow]")
        raise typer.Exit(0)

    # Display
    print_summary(results)

    if dry_run:
        console.print("[yellow]--dry-run: no files written.[/yellow]")
        raise typer.Exit(0)

    # Always save locally as a backup
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.print(f"  [green]✓[/green] Saved locally → {LOCAL_OUT}")

    if local:
        console.print("[yellow]--local: skipping HuggingFace upload.[/yellow]")
        raise typer.Exit(0)

    # Upload to HF
    console.print("Uploading to HuggingFace…")
    upload_to_hf(results)

    console.rule(f"[bold green]Done — {len(results)} results updated")


if __name__ == "__main__":
    app()
