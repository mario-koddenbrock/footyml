"""Export WC 2026 tournament figures to PNG and HTML files.

Reads from data/predictions/wc2026_simulation.json and generates:
  - group_standings.png   — 4×3 grid of group tables with advance %
  - bracket.png           — full knockout bracket tree
  - champion_probs.png    — top-N champion probability bar chart
  - reach_chart.png       — stacked bar chart (top 16 teams)
  - podium.html           — interactive HTML summary

Usage:
    python scripts/export_wc2026_figures.py [--out-dir data/figures] [--top 20]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
import plotly.graph_objects as go
import plotly.subplots as sp
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Country flags
# ---------------------------------------------------------------------------

_FLAG_OVERRIDES: dict[str, str] = {
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
}

try:
    import countryflag

    def flag(name: str) -> str:
        if name in _FLAG_OVERRIDES:
            return _FLAG_OVERRIDES[name]
        try:
            f = countryflag.getflag([name])
            return f.strip() if f.strip() else ""
        except Exception:
            return ""

except ImportError:
    def flag(_: str) -> str:
        return ""


def fl(name: str) -> str:
    """Return 'FLAG Name' or just 'Name' if no flag found."""
    f = flag(name)
    return f"{f} {name}" if f else name


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

GOLD   = "#FFD700"
SILVER = "#C0C0C0"
BRONZE = "#CD7F32"
GREEN  = "#2ecc71"
BLUE   = "#3498db"
RED    = "#e74c3c"
DARK   = "#1a1a2e"
MID    = "#16213e"
LIGHT  = "#e8e8e8"

GROUP_LETTERS = list("ABCDEFGHIJKL")


# ---------------------------------------------------------------------------
# Figure 1 — Group standings
# ---------------------------------------------------------------------------

def _make_group_standings(bracket: dict, probs: dict) -> go.Figure:
    group_standings = bracket["group_standings"]

    fig = sp.make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"Group {g}" for g in GROUP_LETTERS],
        vertical_spacing=0.12,
        horizontal_spacing=0.04,
    )

    for idx, letter in enumerate(GROUP_LETTERS):
        row = idx // 4 + 1
        col = idx % 4 + 1
        gid = f"GROUP_{letter}"
        teams_in_group = group_standings.get(gid, [])

        team_names   = [fl(t["team"]) for t in teams_in_group]
        advance_pcts = [round(probs.get(t["team"], {}).get("group_advance", 0), 1) for t in teams_in_group]
        colors = [GREEN if i < 2 else RED for i in range(len(teams_in_group))]

        fig.add_trace(
            go.Bar(
                x=advance_pcts,
                y=team_names,
                orientation="h",
                marker_color=colors,
                text=[f"{p:.0f}%" for p in advance_pcts],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ),
            row=row, col=col,
        )
        fig.update_xaxes(range=[0, 110], showticklabels=False, row=row, col=col)
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_layout(
        title=dict(text="⚽ FIFA World Cup 2026 — Group Stage Advancement Probabilities", font_size=18),
        height=750, width=1400,
        paper_bgcolor=DARK, plot_bgcolor=MID,
        font=dict(color=LIGHT, size=11),
        margin=dict(t=80, b=20, l=10, r=10),
    )
    fig.update_annotations(font=dict(color=LIGHT, size=13))
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Bracket tree
# ---------------------------------------------------------------------------

def _make_bracket(bracket: dict, probs: dict) -> go.Figure:
    """Draw the full knockout bracket as a Plotly figure."""
    shapes: list[dict] = []
    annotations: list[dict] = []

    def _prob(team: str) -> float:
        return probs.get(team, {}).get("champion", 0.0)

    def _box_color(team: str) -> str:
        p = _prob(team)
        if p >= 15:
            return "#c0392b"
        if p >= 8:
            return "#e67e22"
        if p >= 4:
            return "#27ae60"
        if p >= 1:
            return "#2980b9"
        return "#555577"

    def add_box(x: float, y: float, text: str, w: float = 1.4, h: float = 0.35, color: str = "#555577") -> None:
        shapes.append(dict(
            type="rect",
            x0=x - w / 2, y0=y - h / 2,
            x1=x + w / 2, y1=y + h / 2,
            line=dict(color="#aaaaaa", width=1),
            fillcolor=color,
            layer="below",
        ))
        annotations.append(dict(
            x=x, y=y, text=text,
            showarrow=False,
            font=dict(color="white", size=8),
            xanchor="center", yanchor="middle",
        ))

    def add_line(x0: float, y0: float, x1: float, y1: float) -> None:
        shapes.append(dict(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#aaaaaa", width=1),
        ))

    # ---- Layout constants ------------------------------------------------
    # X: left side  path 1→0..4.5, path 2→4.5..9 | right side path3→11..15.5, path4→15.5..20
    # Y: 12 slots in R32 per path, spacing 1.0

    paths = bracket.get("paths", [])

    def _draw_path(path: dict, x_r32: float, x_r16: float, x_qf: float, direction: int) -> tuple[str, float]:
        """Draw one path's R32/R16/QF. direction: +1=left side, -1=right side."""
        r32 = path.get("r32", [])
        r16 = path.get("r16", [])
        qf  = path.get("qf") or {}

        # R32 — 8 matches → 16 teams in 8 pairs
        r32_y: list[float] = []
        for i, m in enumerate(r32):
            y_top = i * 2.5 + 1.25
            y_bot = y_top + 1.0
            mid_y = (y_top + y_bot) / 2
            for team, y in [(m.get("home", ""), y_top), (m.get("away", ""), y_bot)]:
                add_box(x_r32, y, fl(team), color=_box_color(team))
            # bracket line to R16
            add_line(x_r32 + direction * 0.7, y_top, x_r32 + direction * 0.7, y_bot)
            add_line(x_r32 + direction * 0.7, mid_y, x_r32 + direction * 1.4, mid_y)
            r32_y.append(mid_y)

        # R16 — 4 matches
        r16_y: list[float] = []
        for i, m in enumerate(r16):
            if i * 2 + 1 >= len(r32_y):
                break
            y1 = r32_y[i * 2]
            y2 = r32_y[i * 2 + 1]
            mid_y = (y1 + y2) / 2
            winner = m.get("winner", "")
            add_box(x_r16, mid_y, fl(winner), color=_box_color(winner))
            add_line(x_r16 + direction * 0.7, y1, x_r16 + direction * 0.7, y2)
            add_line(x_r16 + direction * 0.7, mid_y, x_r16 + direction * 1.4, mid_y)
            r16_y.append(mid_y)

        # QF
        qf_y: float | None = None
        if r16_y:
            qf_winner = qf.get("winner", "")
            if len(r16_y) >= 2:
                qf_y = (r16_y[0] + r16_y[-1]) / 2
            else:
                qf_y = r16_y[0]
            add_box(x_qf, qf_y, fl(qf_winner), color=_box_color(qf_winner))
            if len(r16_y) >= 2:
                add_line(x_qf + direction * 0.7, r16_y[0], x_qf + direction * 0.7, r16_y[-1])
                add_line(x_qf + direction * 0.7, qf_y, x_qf + direction * 1.4, qf_y)

        return (qf.get("winner", ""), qf_y if qf_y is not None else 10.0)

    # Left side: paths 0 and 1
    # Right side: paths 2 and 3
    # Center: SF, Final, Champion

    sf_ys: list[tuple[str, float]] = []

    # --- Left side ---
    x_offsets_left  = [(0.8, 3.0, 5.2), (7.0, 9.2, 11.4)]
    x_offsets_right = [(20.0, 17.8, 15.6), (12.4, 14.6, 16.8)]

    for pi, (x32, x16, xqf) in enumerate(x_offsets_left):
        if pi < len(paths):
            w, wy = _draw_path(paths[pi], x32, x16, xqf, +1)
            sf_ys.append((w, wy))

    for pi, (x32, x16, xqf) in enumerate(x_offsets_right):
        actual_pi = pi + 2
        if actual_pi < len(paths):
            w, wy = _draw_path(paths[actual_pi], x32, x16, xqf, -1)
            sf_ys.append((w, wy))

    # --- Semifinals ---
    x_sf = 10.4
    sf1 = bracket.get("sf1") or {}
    sf2 = bracket.get("sf2") or {}

    sf1_y = (sf_ys[0][1] + sf_ys[1][1]) / 2 if len(sf_ys) >= 2 else 10.0
    sf2_y = (sf_ys[2][1] + sf_ys[3][1]) / 2 if len(sf_ys) >= 4 else 10.0

    add_box(x_sf, sf1_y, fl(sf1.get("winner", "")), w=1.8, color=_box_color(sf1.get("winner", "")))
    add_box(x_sf, sf2_y, fl(sf2.get("winner", "")), w=1.8, color=_box_color(sf2.get("winner", "")))

    # Lines: QF left → SF
    if len(sf_ys) >= 2:
        add_line(x_sf - 0.9, sf_ys[0][1], x_sf - 0.9, sf_ys[1][1])
        add_line(x_sf - 0.9, sf1_y, x_sf - 1.6, sf1_y)
    if len(sf_ys) >= 4:
        add_line(x_sf + 0.9, sf_ys[2][1], x_sf + 0.9, sf_ys[3][1])
        add_line(x_sf + 0.9, sf2_y, x_sf + 1.6, sf2_y)

    # --- Final ---
    final = bracket.get("final") or {}
    fin_y = (sf1_y + sf2_y) / 2
    add_line(x_sf - 0.9, sf1_y, x_sf - 0.9, fin_y)
    add_line(x_sf - 0.9, fin_y, x_sf - 0.0, fin_y)
    add_line(x_sf + 0.9, sf2_y, x_sf + 0.9, fin_y)
    add_line(x_sf + 0.9, fin_y, x_sf + 0.0, fin_y)

    champion = bracket.get("champion", "")
    finalist = bracket.get("finalist", "")

    # Show both finalist and champion
    add_box(x_sf, fin_y + 1.2, fl(finalist), w=1.8, h=0.4, color=_box_color(finalist))
    add_box(x_sf, fin_y - 1.2, fl(finalist[:6] + "…" if len(finalist) > 10 else finalist), w=1.8, h=0.4, color=_box_color(finalist))

    # Champion box (gold)
    add_box(x_sf, fin_y, f"🏆 {fl(champion)}", w=2.4, h=0.6, color="#8B7500")

    # 3rd place
    third_match = bracket.get("third_place") or {}
    third = bracket.get("third", "")
    add_box(x_sf, fin_y - 2.8, f"3rd: {fl(third)}", w=2.0, h=0.4, color=BRONZE)

    # Annotations for rounds
    for label, xpos in [("R32", 0.8), ("R16", 3.0), ("QF", 5.2),
                         ("R32", 20.0), ("R16", 17.8), ("QF", 15.6),
                         ("R32", 7.0), ("R16", 9.2), ("SF/QF", 10.4),
                         ("R32", 12.4), ("R16", 14.6), ("QF", 16.8)]:
        annotations.append(dict(
            x=xpos, y=-0.5, text=f"<b>{label}</b>",
            showarrow=False, font=dict(color="#aaaaaa", size=9),
            xanchor="center",
        ))

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        title=dict(text="⚽ FIFA World Cup 2026 — Most Likely Bracket", font_size=18),
        xaxis=dict(range=[-0.5, 21.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 22], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=0.8),
        height=900, width=1600,
        paper_bgcolor=DARK,
        plot_bgcolor=DARK,
        font=dict(color=LIGHT),
        margin=dict(t=60, b=30, l=10, r=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Champion probability bar chart
# ---------------------------------------------------------------------------

def _make_champion_chart(probs: dict, top: int = 20) -> go.Figure:
    teams_sorted = sorted(probs.items(), key=lambda x: -x[1].get("champion", 0))[:top]
    names  = [fl(t) for t, _ in teams_sorted]
    champ  = [v.get("champion", 0) for _, v in teams_sorted]
    colors = [GOLD if i == 0 else SILVER if i == 1 else BRONZE if i == 2 else BLUE for i in range(len(names))]

    fig = go.Figure(go.Bar(
        x=champ, y=names, orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in champ],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"⚽ WC 2026 — Champion Probability (Top {top})", font_size=18),
        xaxis=dict(title="Probability (%)", range=[0, max(champ) * 1.2]),
        yaxis=dict(autorange="reversed"),
        height=max(400, top * 28), width=900,
        paper_bgcolor=DARK, plot_bgcolor=MID,
        font=dict(color=LIGHT, size=12),
        margin=dict(t=60, b=40, l=160, r=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Reach chart (stacked bars)
# ---------------------------------------------------------------------------

def _make_reach_chart(probs: dict, top: int = 16) -> go.Figure:
    teams_sorted = sorted(probs.items(), key=lambda x: -x[1].get("champion", 0))[:top]
    names = [fl(t) for t, _ in teams_sorted]

    stages = [
        ("Champion",      "champion",      GOLD),
        ("Finalist",      "finalist",      SILVER),
        ("Top 4",         "top4",          BLUE),
        ("Top 8",         "top8",          GREEN),
        ("Group Advance", "group_advance", "#888888"),
    ]

    fig = go.Figure()
    prev = {t: 0.0 for t, _ in teams_sorted}

    # Cumulative → differential
    cum_keys = ["group_advance", "top8", "top4", "finalist", "champion"]
    cum_keys_rev = list(reversed(cum_keys))

    for label, key, color in reversed(stages):
        vals = [probs.get(t, {}).get(key, 0) for t, _ in teams_sorted]
        # subtract higher round to get incremental bar
        if key != "group_advance":
            higher_key = cum_keys_rev[cum_keys_rev.index(key) - 1] if cum_keys_rev.index(key) > 0 else None
            if higher_key:
                higher_vals = [probs.get(t, {}).get(higher_key, 0) for t, _ in teams_sorted]
                vals = [max(0.0, v - h) for v, h in zip(vals, higher_vals)]

        fig.add_trace(go.Bar(
            x=vals, y=names, orientation="h",
            name=label, marker_color=color,
            hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text=f"⚽ WC 2026 — Tournament Reach Probabilities (Top {top})", font_size=18),
        xaxis=dict(title="Probability (%)", range=[0, 105]),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=max(400, top * 30), width=1000,
        paper_bgcolor=DARK, plot_bgcolor=MID,
        font=dict(color=LIGHT, size=12),
        margin=dict(t=80, b=40, l=160, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.command()
def main(
    out_dir: Path = typer.Option(Path("data/figures"), "--out-dir"),
    sim_file: Path = typer.Option(Path("data/predictions/wc2026_simulation.json"), "--sim"),
    top: int = typer.Option(20, "--top", help="Top N teams for champion/reach charts"),
    png: bool = typer.Option(True, "--png/--no-png", help="Export PNG (requires kaleido)"),
    html: bool = typer.Option(True, "--html/--no-html", help="Export interactive HTML"),
) -> None:
    console.rule("[bold]WC 2026 Figure Export")

    if not sim_file.exists():
        console.print(f"[red]Simulation file not found: {sim_file}[/red]")
        console.print("Run: python scripts/simulate_tournament.py --export")
        raise typer.Exit(1)

    with open(sim_file) as f:
        sim = json.load(f)

    probs   = sim["probabilities"]
    bracket = sim["bracket"]

    out_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "group_standings": _make_group_standings(bracket, probs),
        "bracket":         _make_bracket(bracket, probs),
        "champion_probs":  _make_champion_chart(probs, top=top),
        "reach_chart":     _make_reach_chart(probs, top=top),
    }

    for name, fig in figures.items():
        if html:
            path = out_dir / f"{name}.html"
            fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
            console.print(f"  [green]✓[/green] {path}")

        if png:
            path = out_dir / f"{name}.png"
            try:
                fig.write_image(str(path), scale=2)
                console.print(f"  [green]✓[/green] {path}")
            except Exception as e:
                console.print(f"  [yellow]PNG skipped ({e.__class__.__name__}: {e})[/yellow]")
                console.print("  Install kaleido: uv add kaleido")

    console.rule(f"[bold green]Done — {len(figures)} figures exported to {out_dir}/")


if __name__ == "__main__":
    app()
