"""Simulate the full FIFA World Cup 2026 and predict the winner.

Usage:
    python scripts/simulate_tournament.py
    python scripts/simulate_tournament.py --n-sims 100000
    python scripts/simulate_tournament.py --predictions data/predictions/wc2026_upcoming.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from footyml.config import DATA_DIR
from footyml.prediction.simulate import TournamentSimulator

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    predictions: Path = typer.Option(
        DATA_DIR / "predictions" / "wc2026_upcoming.json",
        help="Path to wc2026_upcoming.json from predict_tournament.py",
    ),
    n_sims: int = typer.Option(50_000, "--n-sims", help="Number of Monte Carlo simulations"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    export: Path | None = typer.Option(None, help="Save simulation results as JSON"),
    top_n: int = typer.Option(16, "--top", help="Show top-N teams in the probability table"),
) -> None:
    console.rule("[bold]WC 2026 — Tournament Simulation")
    console.print(f"[dim]Predictions: {predictions}[/dim]")
    console.print(f"[dim]Simulations: {n_sims:,}   Seed: {seed}[/dim]\n")

    sim = TournamentSimulator(predictions_path=Path(predictions))

    with console.status(f"Running {n_sims:,} Monte Carlo simulations..."):
        probs = sim.run(n=n_sims, seed=seed)

    # Sort by champion probability
    ranked = sorted(probs.items(), key=lambda x: x[1]["champion"], reverse=True)

    # --- Most likely bracket ---
    console.rule("[bold]Most Likely Bracket")
    bracket = sim.most_likely_bracket()

    _print_bracket_summary(bracket)

    # --- Probability table ---
    console.rule(f"[bold]Win Probabilities (top {top_n})")
    _print_prob_table(ranked[:top_n])

    if export:
        export.parent.mkdir(parents=True, exist_ok=True)
        with open(export, "w") as f:
            json.dump({"probabilities": probs, "bracket": bracket}, f, indent=2, default=str)
        console.print(f"\n[dim]Results saved → {export}[/dim]")
    else:
        # Auto-save alongside predictions
        out = predictions.parent / "wc2026_simulation.json"
        with open(out, "w") as f:
            json.dump({"probabilities": probs, "bracket": bracket}, f, indent=2, default=str)
        console.print(f"\n[dim]Results saved → {out}[/dim]")


def _print_bracket_summary(bracket: dict) -> None:
    champion = bracket.get("champion", "?")
    finalist = bracket.get("finalist", "?")
    third    = bracket.get("third", "?")

    console.print(Panel(
        f"[bold gold1]🏆 Champion:  {champion}[/bold gold1]\n"
        f"[bold silver]🥈 Finalist:  {finalist}[/bold silver]\n"
        f"[bold]🥉 3rd Place: {third}[/bold]",
        title="Most Likely Outcome",
        border_style="gold1",
        expand=False,
    ))

    for i, path in enumerate(bracket.get("paths", []), 1):
        qf = path["qf"]
        console.print(
            f"  Path {i} [{', '.join(path['groups'])}]  "
            f"QF: [cyan]{qf['home']}[/cyan] vs [cyan]{qf['away']}[/cyan] → [bold]{qf['winner']}[/bold]"
        )

    sf1, sf2 = bracket.get("sf1", {}), bracket.get("sf2", {})
    fin = bracket.get("final", {})
    console.print()
    console.print(f"  SF1: [cyan]{sf1.get('home','?')}[/cyan] vs [cyan]{sf1.get('away','?')}[/cyan] → [bold]{sf1.get('winner','?')}[/bold]")
    console.print(f"  SF2: [cyan]{sf2.get('home','?')}[/cyan] vs [cyan]{sf2.get('away','?')}[/cyan] → [bold]{sf2.get('winner','?')}[/bold]")
    console.print(f"  [bold gold1]FINAL: {fin.get('home','?')} vs {fin.get('away','?')} → {fin.get('winner','?')}[/bold gold1]")
    console.print()


def _print_prob_table(ranked: list[tuple[str, dict[str, float]]]) -> None:
    t = Table(show_header=True, header_style="bold")
    t.add_column("#",          style="dim", justify="right", width=3)
    t.add_column("Team",       style="bold", min_width=20)
    t.add_column("Champion",   style="gold1",  justify="right")
    t.add_column("Finalist",   style="cyan",   justify="right")
    t.add_column("Top 4",      style="green",  justify="right")
    t.add_column("Top 8",      style="dim",    justify="right")
    t.add_column("Advance",    style="dim",    justify="right")

    for rank, (team, p) in enumerate(ranked, 1):
        t.add_row(
            str(rank),
            team,
            f"{p['champion']:.1f}%",
            f"{p['finalist']:.1f}%",
            f"{p['top4']:.1f}%",
            f"{p['top8']:.1f}%",
            f"{p['group_advance']:.1f}%",
        )

    console.print(t)


if __name__ == "__main__":
    app()
