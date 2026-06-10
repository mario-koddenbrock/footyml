"""Export WC 2026 tournament figures as PNG files.

Reads from data/predictions/wc2026_simulation.json and generates:
  - group_standings.png   — 4×3 grid of group tables with advance %
  - bracket.png           — full knockout bracket tree
  - champion_probs.png    — top-N champion probability bar chart
  - reach_chart.png       — stacked bar chart (top 16 teams)

Usage:
    python scripts/export_wc2026_figures.py [--out-dir data/figures] [--top 20]
"""
from __future__ import annotations

import json
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
    f = flag(name)
    return f"{f} {name}" if f else name


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

GOLD   = "#FFD700"
SILVER = "#C0C0C0"
BRONZE = "#CD7F32"
GREEN  = "#2ecc71"
BLUE   = "#3498db"
TRANS  = "rgba(0,0,0,0)"

GROUP_LETTERS = list("ABCDEFGHIJKL")

_CHAMP_COLORS = [
    (15, "#c0392b"),   # strong red — top favorites
    (8,  "#e67e22"),   # orange
    (4,  "#27ae60"),   # green
    (1,  "#2980b9"),   # blue
    (0,  "#4a5068"),   # neutral dark
]

def team_color(team: str, probs: dict) -> str:
    p = probs.get(team, {}).get("champion", 0.0)
    for threshold, color in _CHAMP_COLORS:
        if p >= threshold:
            return color
    return "#4a5068"


# ---------------------------------------------------------------------------
# Figure 1 — Group standings
# ---------------------------------------------------------------------------

def _make_group_standings(bracket: dict, probs: dict) -> go.Figure:
    group_standings = bracket["group_standings"]

    fig = sp.make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"Group {g}" for g in GROUP_LETTERS],
        vertical_spacing=0.10,
        horizontal_spacing=0.05,
    )

    for idx, letter in enumerate(GROUP_LETTERS):
        row = idx // 4 + 1
        col = idx % 4 + 1
        gid = f"GROUP_{letter}"
        teams = group_standings.get(gid, [])

        names   = [fl(t["team"]) for t in teams]
        adv_pct = [round(probs.get(t["team"], {}).get("group_advance", 0), 1) for t in teams]
        colors  = [GREEN if i < 2 else "#e74c3c" for i in range(len(teams))]

        fig.add_trace(
            go.Bar(
                x=adv_pct, y=names, orientation="h",
                marker_color=colors,
                text=[f"{p:.0f}%" for p in adv_pct],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ),
            row=row, col=col,
        )
        fig.update_xaxes(range=[0, 115], showticklabels=False, row=row, col=col)
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10), row=row, col=col)

    fig.update_layout(
        title=dict(text="FIFA World Cup 2026 — Group Stage Advancement Probabilities",
                   font=dict(size=17, color="#333333")),
        height=700, width=1400,
        paper_bgcolor=TRANS, plot_bgcolor=TRANS,
        font=dict(color="#333333", size=11),
        margin=dict(t=70, b=10, l=10, r=10),
    )
    fig.update_annotations(font=dict(color="#333333", size=13))
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Bracket tree (full knockout bracket)
# ---------------------------------------------------------------------------

def _make_bracket(bracket: dict, probs: dict) -> go.Figure:
    """
    Layout (horizontal, left→right for left half, right→left for right half):

    Left side (paths 0 & 1):
        R32 col → R16 col → QF col → SF col
                                        ↓
                                    FINAL (center)
                                        ↑
    Right side (paths 2 & 3):
        R32 col → R16 col → QF col → SF col

    Y layout (shared by both sides):
        Path 0 (upper): 4 R32 matches, y ≈ 0..10
        Path 1 (lower): 4 R32 matches, y ≈ 14..24
        SF midpoint ≈ 12
    """
    paths = bracket.get("paths", [])

    shapes: list[dict] = []
    annotations: list[dict] = []

    # ---- Y coordinate system ----
    # Slots per side: 2 teams per match × 4 matches × 2 paths = 16 slots
    # Within each match: home at y, away at y+1.0
    # Between matches (same path): 0.6 gap after each pair → next match at y+2.6
    # Between paths: 3.0 gap

    def _build_slot_ys() -> list[float]:
        ys: list[float] = []
        y = 0.0
        for path_idx in range(2):
            for match_idx in range(4):
                ys.append(y)        # home slot
                ys.append(y + 1.0)  # away slot
                if match_idx < 3:
                    y += 1.0 + 0.8  # within-path gap between matches
                elif path_idx == 0:
                    y += 1.0 + 3.0  # between-path gap
        return ys

    slot_ys = _build_slot_ys()

    # R32 match midpoints (8 per side)
    r32_y = [(slot_ys[2*i] + slot_ys[2*i + 1]) / 2 for i in range(8)]

    # R16 midpoints: pairs of R32 midpoints (4 per side)
    r16_y = [(r32_y[2*i] + r32_y[2*i + 1]) / 2 for i in range(4)]

    # QF midpoints: pairs of R16 midpoints (2 per side)
    qf_y = [(r16_y[2*i] + r16_y[2*i + 1]) / 2 for i in range(2)]

    # SF midpoint
    sf_y = (qf_y[0] + qf_y[1]) / 2

    MAX_Y = slot_ys[-1] + 1.5  # figure height

    # ---- X positions (left side; right side is mirrored) ----
    BW = 1.40   # box half-width
    BH = 0.36   # box half-height (winner boxes)
    TEAM_BH = 0.30  # box half-height (R32 individual team rows)

    COL_SPACING = 3.8
    X_L_R32 = BW + 0.1                  # ~1.5
    X_L_R16 = X_L_R32 + COL_SPACING     # ~5.3
    X_L_QF  = X_L_R16 + COL_SPACING     # ~9.1
    X_L_SF  = X_L_QF  + COL_SPACING     # ~12.9
    X_FIN   = X_L_SF  + COL_SPACING     # ~16.7 (center)
    X_R_SF  = X_FIN   + COL_SPACING     # ~20.5
    X_R_QF  = X_R_SF  + COL_SPACING     # ~24.3
    X_R_R16 = X_R_QF  + COL_SPACING     # ~28.1
    X_R_R32 = X_R_R16 + COL_SPACING     # ~31.9

    TOTAL_W = X_R_R32 + BW + 0.1

    def add_box(x: float, y: float, label: str,
                bw: float = BW, bh: float = BH,
                fill: str = "#4a5068", border: str = "#888888",
                fsize: int = 9, bold: bool = False) -> None:
        shapes.append(dict(
            type="rect", x0=x - bw, y0=y - bh, x1=x + bw, y1=y + bh,
            line=dict(color=border, width=1), fillcolor=fill, layer="below",
        ))
        text = f"<b>{label}</b>" if bold else label
        annotations.append(dict(
            x=x, y=y, text=text, showarrow=False,
            font=dict(color="white", size=fsize),
            xanchor="center", yanchor="middle",
        ))

    def add_line(x0: float, y0: float, x1: float, y1: float,
                 color: str = "#aaaaaa", width: float = 1.0) -> None:
        shapes.append(dict(
            type="line", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=color, width=width),
        ))

    def connect_left(x_from: float, y_a: float, y_b: float, x_to: float) -> None:
        """Connect two boxes at (x_from, y_a) and (x_from, y_b) to one box at (x_to, midpoint)."""
        y_mid = (y_a + y_b) / 2
        jx = (x_from + BW + x_to - BW) / 2
        add_line(x_from + BW, y_a, jx, y_a)
        add_line(x_from + BW, y_b, jx, y_b)
        add_line(jx, y_a, jx, y_b)
        add_line(jx, y_mid, x_to - BW, y_mid)

    def connect_right(x_from: float, y_a: float, y_b: float, x_to: float) -> None:
        """Right-side version: boxes at x_from connect leftward to box at x_to."""
        y_mid = (y_a + y_b) / 2
        jx = (x_from - BW + x_to + BW) / 2
        add_line(x_from - BW, y_a, jx, y_a)
        add_line(x_from - BW, y_b, jx, y_b)
        add_line(jx, y_a, jx, y_b)
        add_line(jx, y_mid, x_to + BW, y_mid)

    # ---- Draw one bracket side ----
    def draw_side(path0: dict, path1: dict, side: str) -> None:
        is_left = (side == "left")
        x_r32 = X_L_R32 if is_left else X_R_R32
        x_r16 = X_L_R16 if is_left else X_R_R16
        x_qf  = X_L_QF  if is_left else X_R_QF
        x_sf  = X_L_SF  if is_left else X_R_SF
        connect = connect_left if is_left else connect_right

        for path_idx, path in enumerate([path0, path1]):
            r32_matches = path.get("r32", [])
            r16_matches = path.get("r16", [])
            qf_match    = path.get("qf") or {}
            groups_label = "  ".join(g.replace("GROUP_", "") for g in path.get("groups", []))

            # Path label (small, centered on the path's y range)
            path_start = slot_ys[path_idx * 8]
            path_end   = slot_ys[path_idx * 8 + 7]
            path_mid   = (path_start + path_end) / 2
            x_label = (x_r32 - BW - 0.1) if is_left else (x_r32 + BW + 0.1)
            annotations.append(dict(
                x=x_label, y=path_mid,
                text=f"<b>Groups<br>{groups_label}</b>",
                showarrow=False, textangle=-90 if is_left else 90,
                font=dict(color="#666666", size=8),
                xanchor="center", yanchor="middle",
            ))

            # ---- R32: 4 matches per path, each with home+away team rows ----
            for mi, m in enumerate(r32_matches[:4]):
                global_mi = path_idx * 4 + mi
                home = m.get("home", "?")
                away = m.get("away", "?")
                winner = m.get("winner", "")

                y_home = slot_ys[global_mi * 2]
                y_away = slot_ys[global_mi * 2 + 1]

                home_fill = team_color(home, probs) if home == winner else "#2c2c40"
                away_fill = team_color(away, probs) if away == winner else "#2c2c40"

                add_box(x_r32, y_home, fl(home), bw=BW, bh=TEAM_BH, fill=home_fill, fsize=8,
                        border="#888888" if home == winner else "#555566")
                add_box(x_r32, y_away, fl(away), bw=BW, bh=TEAM_BH, fill=away_fill, fsize=8,
                        border="#888888" if away == winner else "#555566")

                # Bracket connector from winner row to junction for R16
                winner_y = y_home if home == winner else y_away
                jx_r32_r16 = (x_r32 + BW + x_r16 - BW) / 2
                if is_left:
                    add_line(x_r32 + BW, winner_y, jx_r32_r16, winner_y)
                else:
                    add_line(x_r32 - BW, winner_y, jx_r32_r16, winner_y)

            # ---- R16: 2 matches per path ----
            for mi, m in enumerate(r16_matches[:2]):
                global_mi = path_idx * 2 + mi
                winner = m.get("winner", "")
                y = r16_y[global_mi]
                add_box(x_r16, y, fl(winner), fill=team_color(winner, probs), fsize=9)

                # Vertical bracket line gathering the two R32 winners
                i_a = global_mi * 2
                i_b = global_mi * 2 + 1
                y_a_r32 = r32_y[i_a]
                y_b_r32 = r32_y[i_b]
                jx = (x_r32 + BW + x_r16 - BW) / 2 if is_left else (x_r32 - BW + x_r16 + BW) / 2
                add_line(jx, y_a_r32, jx, y_b_r32)
                add_line(jx, y, x_r16 - BW if is_left else x_r16 + BW, y)

            # ---- QF ----
            qf_winner = qf_match.get("winner", "")
            y = qf_y[path_idx]
            add_box(x_qf, y, fl(qf_winner), fill=team_color(qf_winner, probs), fsize=10, bh=BH + 0.04)
            connect(x_r16, r16_y[path_idx * 2], r16_y[path_idx * 2 + 1], x_qf)

        # ---- SF box (where path0 QF winner meets path1 QF winner) ----
        sf_match = bracket.get("sf1") if is_left else bracket.get("sf2")
        sf_match = sf_match or {}
        sf_winner = sf_match.get("winner", "")
        add_box(x_sf, sf_y, fl(sf_winner), fill=team_color(sf_winner, probs),
                bh=BH + 0.06, fsize=11, bold=True)
        connect(x_qf, qf_y[0], qf_y[1], x_sf)

        # Horizontal line from SF to Final
        if is_left:
            add_line(x_sf + BW, sf_y, X_FIN - BW, sf_y, color="#dddddd", width=1.5)
        else:
            add_line(x_sf - BW, sf_y, X_FIN + BW, sf_y, color="#dddddd", width=1.5)

    # ---- Draw both sides ----
    if len(paths) >= 2:
        draw_side(paths[0], paths[1], "left")
    if len(paths) >= 4:
        draw_side(paths[2], paths[3], "right")

    # ---- Final ----
    final = bracket.get("final") or {}
    home_f = final.get("home", "")
    away_f = final.get("away", "")
    champion = bracket.get("champion", "")

    # Show both finalists as small boxes flanking the Final label
    add_box(X_FIN, sf_y + 0.72, fl(home_f), bw=BW, bh=TEAM_BH,
            fill=team_color(home_f, probs), fsize=9, border="#dddddd")
    add_box(X_FIN, sf_y - 0.72, fl(away_f), bw=BW, bh=TEAM_BH,
            fill=team_color(away_f, probs), fsize=9, border="#dddddd")

    # Champion box
    add_box(X_FIN, sf_y, f"🏆  {fl(champion)}", bw=BW + 0.35, bh=BH * 0.6,
            fill="#7a5c00", border=GOLD, fsize=12, bold=True)

    # ---- Round labels (top) ----
    label_y = MAX_Y + 0.8
    for label, x in [
        ("R32", X_L_R32), ("R16", X_L_R16), ("QF", X_L_QF), ("SF", X_L_SF),
        ("FINAL", X_FIN),
        ("SF", X_R_SF), ("QF", X_R_QF), ("R16", X_R_R16), ("R32", X_R_R32),
    ]:
        is_final_col = label == "FINAL"
        annotations.append(dict(
            x=x, y=label_y, text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color=GOLD if is_final_col else "#888888", size=11 if is_final_col else 10),
            xanchor="center", yanchor="bottom",
        ))

    # ---- 3rd place ----
    third = bracket.get("third", "")
    third_y = -2.5
    add_box(X_FIN, third_y, f"3rd  {fl(third)}", bw=BW, bh=BH * 0.8,
            fill="#5a3e1b", border=BRONZE, fsize=10, bold=True)
    annotations.append(dict(
        x=X_FIN, y=third_y - BH - 0.5,
        text="<b>3rd Place</b>",
        showarrow=False, font=dict(color="#888888", size=9),
        xanchor="center", yanchor="top",
    ))

    # ---- Title ----
    annotations.append(dict(
        x=X_FIN, y=label_y + 1.2,
        text="<b>FIFA World Cup 2026 — Most Likely Bracket</b>",
        showarrow=False,
        font=dict(color="#333333", size=16),
        xanchor="center", yanchor="bottom",
    ))

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(range=[-1.5, TOTAL_W + 1.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[third_y - 2, label_y + 2.5], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=0.9),
        height=1000, width=1800,
        paper_bgcolor=TRANS,
        plot_bgcolor=TRANS,
        font=dict(color="#333333"),
        margin=dict(t=20, b=20, l=10, r=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Champion probability bar chart
# ---------------------------------------------------------------------------

def _make_champion_chart(probs: dict, top: int = 20) -> go.Figure:
    teams_sorted = sorted(probs.items(), key=lambda x: -x[1].get("champion", 0))[:top]
    names  = [fl(t) for t, _ in teams_sorted]
    champ  = [v.get("champion", 0) for _, v in teams_sorted]
    colors = [GOLD if i == 0 else SILVER if i == 1 else BRONZE if i == 2
              else team_color(t, probs) for i, (t, _) in enumerate(teams_sorted)]

    fig = go.Figure(go.Bar(
        x=champ, y=names, orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in champ],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"FIFA World Cup 2026 — Champion Probability (Top {top})",
                   font=dict(size=17, color="#333333")),
        xaxis=dict(title="Champion probability (%)", range=[0, max(champ) * 1.22],
                   color="#333333"),
        yaxis=dict(autorange="reversed", color="#333333"),
        height=max(400, top * 28), width=900,
        paper_bgcolor=TRANS, plot_bgcolor=TRANS,
        font=dict(color="#333333", size=12),
        margin=dict(t=60, b=40, l=160, r=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Reach chart (stacked bars)
# ---------------------------------------------------------------------------
# (see below)

# ---------------------------------------------------------------------------
# Figure 5 — Podium (top 3 without probabilities)
# ---------------------------------------------------------------------------

def _make_podium(bracket: dict) -> go.Figure:
    """Classic podium — champion, finalist, 3rd place. No numbers."""
    champion = bracket.get("champion", "")
    finalist = bracket.get("finalist", "")
    third    = bracket.get("third", "")

    shapes: list[dict] = []
    anns: list[dict] = []

    # Podium block positions: 2nd left, 1st center, 3rd right
    # x centers at 2.0, 5.5, 9.0 — y from 0 up to block height
    PODIUM = [
        # (x_center, block_top, fill, border, medal, label, team)
        (2.0, 2.2, "#c0c0c0", "#a0a0a0", "🥈", "Runner-up",      finalist),
        (5.5, 3.6, "#ffd700", "#c8a800", "🥇", "World Champion",  champion),
        (9.0, 1.4, "#cd7f32", "#a0621a", "🥉", "Third Place",     third),
    ]

    BLOCK_W = 1.8  # half-width of each podium block
    FLOOR_Y = 0.0

    for x, top, fill, border, medal, label, team in PODIUM:
        # Podium block
        shapes.append(dict(
            type="rect",
            x0=x - BLOCK_W, y0=FLOOR_Y,
            x1=x + BLOCK_W, y1=top,
            fillcolor=fill,
            line=dict(color=border, width=2),
            layer="below",
        ))
        # Rank number inside the block
        rank_num = {"🥇": "1", "🥈": "2", "🥉": "3"}[medal]
        anns.append(dict(
            x=x, y=top / 2, text=f"<b>{rank_num}</b>",
            showarrow=False, font=dict(size=38, color="rgba(255,255,255,0.35)"),
            xanchor="center", yanchor="middle",
        ))
        # Flag + team name above block
        f = flag(team)
        anns.append(dict(
            x=x, y=top + 0.25, text=f"<b>{f}</b>" if f else "",
            showarrow=False, font=dict(size=48),
            xanchor="center", yanchor="bottom",
        ))
        anns.append(dict(
            x=x, y=top + 1.15, text=f"<b>{team}</b>",
            showarrow=False, font=dict(size=20, color="#222222"),
            xanchor="center", yanchor="bottom",
        ))
        anns.append(dict(
            x=x, y=top + 1.6, text=label,
            showarrow=False, font=dict(size=12, color="#666666"),
            xanchor="center", yanchor="bottom",
        ))
        # Medal emoji above name
        anns.append(dict(
            x=x, y=top + 1.85, text=medal,
            showarrow=False, font=dict(size=20),
            xanchor="center", yanchor="bottom",
        ))

    # Title
    anns.append(dict(
        x=5.5, y=6.2, text="<b>FIFA World Cup 2026</b>",
        showarrow=False, font=dict(size=26, color="#111111"),
        xanchor="center", yanchor="bottom",
    ))
    anns.append(dict(
        x=5.5, y=5.75, text="Final Standings",
        showarrow=False, font=dict(size=14, color="#888888"),
        xanchor="center", yanchor="bottom",
    ))

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes, annotations=anns,
        xaxis=dict(range=[0, 11], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.4, 7.0], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=0.6),
        height=600, width=900,
        paper_bgcolor=TRANS, plot_bgcolor=TRANS,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Group overview (all 12 groups, clean table layout, no probabilities)
# ---------------------------------------------------------------------------

def _make_group_overview(bracket: dict) -> go.Figure:
    """12-group overview grid: team rank, flag, name. Top 2 highlighted."""
    group_standings = bracket["group_standings"]

    # Row colors by rank
    ROW_COLORS = {
        1: "rgba(26,107,58,0.15)",
        2: "rgba(26,107,58,0.15)",
        3: "rgba(122,92,0,0.15)",
        4: "rgba(90,26,26,0.10)",
    }
    ROW_BORDERS = {1: "rgba(46,204,113,0.4)", 2: "rgba(46,204,113,0.4)",
                   3: "rgba(243,156,18,0.4)", 4: "rgba(192,57,43,0.3)"}

    shapes: list[dict] = []
    anns: list[dict] = []

    # Card layout: 4 columns × 3 rows
    CARD_W  = 4.8
    CARD_H  = 5.8   # header + 4 rows
    COL_GAP = 0.4
    ROW_GAP = 0.6
    HEADER_H = 0.85
    ROW_H    = (CARD_H - HEADER_H) / 4  # ~1.24

    TOTAL_W = 4 * CARD_W + 3 * COL_GAP
    TOTAL_H = 3 * CARD_H + 2 * ROW_GAP

    for idx, letter in enumerate(GROUP_LETTERS):
        col = idx % 4
        row = idx // 4
        # Card origin (top-left), y inverted (y=0 at top)
        card_x = col * (CARD_W + COL_GAP)
        card_y = TOTAL_H - row * (CARD_H + ROW_GAP)  # top y of this card

        gid = f"GROUP_{letter}"
        teams = group_standings.get(gid, [])

        # Card background
        shapes.append(dict(
            type="rect",
            x0=card_x, y0=card_y - CARD_H,
            x1=card_x + CARD_W, y1=card_y,
            fillcolor="rgba(240,244,255,0.7)",
            line=dict(color="#cccccc", width=1),
            layer="below",
        ))

        # Group header
        shapes.append(dict(
            type="rect",
            x0=card_x, y0=card_y - HEADER_H,
            x1=card_x + CARD_W, y1=card_y,
            fillcolor="#1a2744",
            line=dict(color="#1a2744", width=0),
            layer="below",
        ))
        anns.append(dict(
            x=card_x + CARD_W / 2, y=card_y - HEADER_H / 2,
            text=f"<b>Group {letter}</b>",
            showarrow=False,
            font=dict(size=15, color="white"),
            xanchor="center", yanchor="middle",
        ))

        # Team rows
        for ti, team_data in enumerate(teams[:4]):
            team = team_data["team"]
            rank = team_data["rank"]
            row_y_top = card_y - HEADER_H - ti * ROW_H
            row_y_bot = row_y_top - ROW_H
            row_mid   = (row_y_top + row_y_bot) / 2

            # Row background
            shapes.append(dict(
                type="rect",
                x0=card_x, y0=row_y_bot,
                x1=card_x + CARD_W, y1=row_y_top,
                fillcolor=ROW_COLORS.get(rank, "rgba(80,80,80,0.1)"),
                line=dict(color=ROW_BORDERS.get(rank, "rgba(136,136,136,0.3)"), width=0.5),
                layer="below",
            ))

            # Rank indicator dot on the left
            dot_colors = {1: "#2ecc71", 2: "#2ecc71", 3: "#f39c12", 4: "#e74c3c"}
            shapes.append(dict(
                type="circle",
                x0=card_x + 0.10, y0=row_mid - 0.18,
                x1=card_x + 0.46, y1=row_mid + 0.18,
                fillcolor=dot_colors.get(rank, "#888888"),
                line=dict(color=dot_colors.get(rank, "#888888"), width=0),
                layer="above",
            ))
            # Rank number in dot
            anns.append(dict(
                x=card_x + 0.28, y=row_mid,
                text=f"<b>{rank}</b>",
                showarrow=False,
                font=dict(size=9, color="white"),
                xanchor="center", yanchor="middle",
            ))

            # Flag emoji
            f = flag(team)
            anns.append(dict(
                x=card_x + 0.70, y=row_mid,
                text=f,
                showarrow=False,
                font=dict(size=14),
                xanchor="center", yanchor="middle",
            ))

            # Team name
            anns.append(dict(
                x=card_x + 0.95, y=row_mid,
                text=team,
                showarrow=False,
                font=dict(size=11, color="#111111"),
                xanchor="left", yanchor="middle",
            ))

    # Title
    anns.append(dict(
        x=TOTAL_W / 2, y=TOTAL_H + 0.7,
        text="<b>FIFA World Cup 2026 — Group Stage Overview</b>",
        showarrow=False, font=dict(size=20, color="#111111"),
        xanchor="center", yanchor="bottom",
    ))

    # Legend
    for lx, color, label in [
        (0.0, "#2ecc71", "Advances (Top 2)"),
        (3.8, "#f39c12", "Potential wild card (3rd)"),
        (9.0, "#e74c3c", "Eliminated (4th)"),
    ]:
        shapes.append(dict(
            type="circle",
            x0=lx, y0=TOTAL_H + 0.08, x1=lx + 0.28, y1=TOTAL_H + 0.38,
            fillcolor=color, line=dict(color=color, width=0), layer="above",
        ))
        anns.append(dict(
            x=lx + 0.42, y=TOTAL_H + 0.23,
            text=label,
            showarrow=False, font=dict(size=11, color="#444444"),
            xanchor="left", yanchor="middle",
        ))

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes, annotations=anns,
        xaxis=dict(range=[-0.3, TOTAL_W + 0.3], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.4, TOTAL_H + 1.4], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=0.9),
        height=950, width=1400,
        paper_bgcolor=TRANS, plot_bgcolor=TRANS,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig


# ---------------------------------------------------------------------------
# (Figure 4 — Reach chart resumes below)
# ---------------------------------------------------------------------------

def _make_reach_chart(probs: dict, top: int = 16) -> go.Figure:
    teams_sorted = sorted(probs.items(), key=lambda x: -x[1].get("champion", 0))[:top]
    names = [fl(t) for t, _ in teams_sorted]

    stages_ordered = ["champion", "finalist", "top4", "top8", "group_advance"]
    stage_labels   = ["Champion", "Finalist", "Top 4", "Top 8", "Group exit"]
    stage_colors   = [GOLD, SILVER, BLUE, GREEN, "#aaaaaa"]

    fig = go.Figure()

    # Build incremental (differential) values per stage
    # Stages are cumulative (champion ⊂ finalist ⊂ top4 ⊂ top8 ⊂ group_advance)
    # Draw from bottom stage upward so stacking looks right
    prev_vals = [0.0] * len(teams_sorted)

    for stage, label, color in zip(
        reversed(stages_ordered), reversed(stage_labels), reversed(stage_colors)
    ):
        cum_vals = [probs.get(t, {}).get(stage, 0) for t, _ in teams_sorted]
        incremental = [max(0.0, c - p) for c, p in zip(cum_vals, prev_vals)]
        fig.add_trace(go.Bar(
            x=incremental, y=names, orientation="h",
            name=label, marker_color=color,
            hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>",
        ))
        prev_vals = cum_vals

    fig.update_layout(
        barmode="stack",
        title=dict(text=f"FIFA World Cup 2026 — Tournament Reach Probabilities (Top {top})",
                   font=dict(size=17, color="#333333")),
        xaxis=dict(title="Probability (%)", range=[0, 105], color="#333333"),
        yaxis=dict(autorange="reversed", color="#333333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#333333")),
        height=max(400, top * 30), width=1000,
        paper_bgcolor=TRANS, plot_bgcolor=TRANS,
        font=dict(color="#333333", size=12),
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
    scale: int = typer.Option(2, "--scale", help="PNG scale factor (2 = 2× pixel density)"),
) -> None:
    console.rule("[bold]WC 2026 Figure Export (PNG)")

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
        "podium":          _make_podium(bracket),
        "group_overview":  _make_group_overview(bracket),
    }

    for name, fig in figures.items():
        path = out_dir / f"{name}.png"
        try:
            fig.write_image(str(path), scale=scale)
            console.print(f"  [green]✓[/green] {path}")
        except Exception as e:
            console.print(f"  [red]✗ {name}.png failed: {e}[/red]")

    console.rule(f"[bold green]Done — {len(figures)} PNGs exported to {out_dir}/")


if __name__ == "__main__":
    app()
