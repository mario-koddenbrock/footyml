"""Store national team squad market values into player_market_values.

Reads national_team_mv.parquet (built by fetch_national_team_mv.py),
maps team names to their fd_ club IDs via the matches table, and inserts
one synthetic player record per team so that MarketValueBuilder can compute
home_mv_total_eur / away_mv_total_eur / mv_ratio / mv_diff_eur for WC fixtures.

Usage:
    python scripts/store_national_team_mv.py
    python scripts/store_national_team_mv.py --season 2026
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from footyml.config import PROCESSED_DIR
from footyml.data.store import DataStore

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    competition: str = typer.Option("WC", help="Competition code to look up fd_ IDs from"),
    season: int = typer.Option(2026, help="Season to store values under (must match matches.season)"),
) -> None:
    mv_path = PROCESSED_DIR / "national_team_mv.parquet"
    if not mv_path.exists():
        console.print(f"[red]Missing {mv_path} — run fetch_national_team_mv.py first[/red]")
        raise typer.Exit(1)

    mv_df = pl.read_parquet(mv_path)
    mv_lookup: dict[str, int] = {
        str(r["team"]): int(r["total_market_value_eur"])
        for r in mv_df.iter_rows(named=True)
    }

    store = DataStore()
    id_df = store.query(f"""
        SELECT DISTINCT home_club_id as fd_id, home_club_name as name
        FROM matches WHERE competition_id = '{competition}'
        AND home_club_name IS NOT NULL AND home_club_name != ''
        UNION
        SELECT DISTINCT away_club_id, away_club_name FROM matches
        WHERE competition_id = '{competition}'
        AND away_club_name IS NOT NULL AND away_club_name != ''
    """)

    season_id = str(season)
    rows: list[dict] = []
    missing: list[str] = []

    for r in id_df.iter_rows(named=True):
        fd_id = str(r["fd_id"])
        name = str(r["name"])
        mv = mv_lookup.get(name)
        if mv is None:
            missing.append(name)
            continue
        # Store under fd_ ID (for WC 2026 upcoming match lookup)
        rows.append({
            "player_id":        f"squad_total_{fd_id}",
            "club_id":          fd_id,
            "season_id":        season_id,
            "player_name":      f"{name} (squad total)",
            "market_value_eur": mv,
            "date_of_birth":    "",
            "nationality":      name,
        })
        # Also store under team name (for historical martj42 training data lookup)
        # Use a broad season range so MarketValueBuilder finds it for any training season
        for s in ["2000", "2002", "2004", "2006", "2008", "2010", "2012",
                  "2014", "2016", "2018", "2019", "2020", "2021", "2022",
                  "2023", "2024", "2025", "2026"]:
            rows.append({
                "player_id":        f"squad_total_{name}_{s}",
                "club_id":          name,
                "season_id":        s,
                "player_name":      f"{name} (squad total)",
                "market_value_eur": mv,
                "date_of_birth":    "",
                "nationality":      name,
            })

    if missing:
        console.print(f"[yellow]No MV data for: {', '.join(missing)}[/yellow]")

    if not rows:
        console.print("[red]No rows to insert[/red]")
        raise typer.Exit(1)

    df = pl.DataFrame(rows, schema={
        "player_id":        pl.String,
        "club_id":          pl.String,
        "season_id":        pl.String,
        "player_name":      pl.String,
        "market_value_eur": pl.Int64,
        "date_of_birth":    pl.String,
        "nationality":      pl.String,
    })
    store.upsert_squad(df)

    t = Table(title=f"Stored national team MV ({competition}, season {season})")
    t.add_column("Team", width=28)
    t.add_column("fd_id", width=10)
    t.add_column("Squad Value (€M)", justify="right")
    for row in sorted(rows, key=lambda x: -x["market_value_eur"]):
        t.add_row(
            row["nationality"],
            row["club_id"],
            f"{row['market_value_eur']/1e6:,.0f}",
        )
    console.print(t)
    console.print(f"[bold green]Inserted {len(rows)} squad value records[/bold green]")


if __name__ == "__main__":
    app()
