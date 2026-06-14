"""Export per-model and combined WC 2026 figures.

Per-model (data/figures/{model}/):
  group_overview.png   — group standings based on that model's predictions
  bracket.png          — most-likely knockout bracket

Combined (data/figures/combined/):
  podium_comparison.png  — all models' predicted top-3 side by side
  champion_votes.png     — heatmap / bar showing which model picks which champion

Usage:
    python scripts/export_wc2026_multimodel.py [--out-dir data/figures] [--scale 2]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
import plotly.graph_objects as go
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "tabpfn":          "TabPFN-3",
    "tabicl":          "TabICL v2",
    "random_forest":   "Random Forest",
    "knn":             "KNN (k=15)",
    "catboost":        "CatBoost",
    "random_baseline": "Random Baseline",
    "home_always_wins": "Home-Always-Wins",
}

MODEL_COLORS: dict[str, str] = {
    "tabpfn":          "#00d25b",
    "tabicl":          "#4c9fff",
    "random_forest":   "#ff9f40",
    "knn":             "#a855f7",
    "catboost":        "#f43f5e",
    "random_baseline": "#6b7280",
    "home_always_wins": "#374151",
}

# ---------------------------------------------------------------------------
# Flag helpers (same as export_wc2026_figures.py)
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
# Colors / constants
# ---------------------------------------------------------------------------

GOLD   = "#FFD700"
SILVER = "#C0C0C0"
BRONZE = "#CD7F32"
TRANS  = "rgba(0,0,0,0)"
GROUP_LETTERS = list("ABCDEFGHIJKL")


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB + alpha float → rgba(r,g,b,a) for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"

# ---------------------------------------------------------------------------
# Fixed WC 2026 R32 bracket seeding
# ---------------------------------------------------------------------------
# Each tuple: (path_idx, (home_group_letter, home_rank), (away_group_letter, away_rank))
# 3rd-place team slots are filled by the best 3rd-place team from designated groups.

_R32_SEEDING: list[tuple[int, tuple[str, int], tuple[str, int]]] = [
    # Path 0 — Groups A, B, C
    (0, ("A", 1), ("B", 2)),
    (0, ("B", 1), ("A", 2)),
    (0, ("C", 1), ("F", 3)),
    (0, ("C", 2), ("D", 3)),
    # Path 1 — Groups D, E, F
    (1, ("D", 1), ("E", 2)),
    (1, ("E", 1), ("D", 2)),
    (1, ("F", 1), ("J", 3)),
    (1, ("F", 2), ("B", 3)),
    # Path 2 — Groups G, H, I
    (2, ("G", 1), ("H", 2)),
    (2, ("H", 1), ("G", 2)),
    (2, ("I", 1), ("L", 3)),
    (2, ("I", 2), ("C", 3)),
    # Path 3 — Groups J, K, L
    (3, ("J", 1), ("K", 2)),
    (3, ("K", 1), ("J", 2)),
    (3, ("L", 1), ("K", 3)),
    (3, ("L", 2), ("E", 3)),
]

_PATH_GROUPS: dict[int, list[str]] = {
    0: ["GROUP_A", "GROUP_B", "GROUP_C"],
    1: ["GROUP_D", "GROUP_E", "GROUP_F"],
    2: ["GROUP_G", "GROUP_H", "GROUP_I"],
    3: ["GROUP_J", "GROUP_K", "GROUP_L"],
}

# ---------------------------------------------------------------------------
# Build group standings from model match predictions
# ---------------------------------------------------------------------------

def build_group_standings(preds: list[dict]) -> dict[str, list[dict]]:
    """Parse group-stage predictions → ranked group standings."""
    # {group_id: {team: {pts, gd, gf, wins, draws, losses}}}
    groups: dict[str, dict[str, dict]] = {}

    for m in preds:
        gid = m.get("group_id", "")
        if not gid:
            continue
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        result = m.get("predicted_result", "")

        for team in (home, away):
            groups.setdefault(gid, {}).setdefault(
                team, {"pts": 0, "gd": 0, "gf": 0, "wins": 0, "draws": 0, "losses": 0}
            )

        stats = groups[gid]
        # For expected goals we approximate from probabilities
        h_win_p = m.get("home_win_prob", 33.3) / 100
        d_p = m.get("draw_prob", 33.3) / 100
        a_win_p = m.get("away_win_prob", 33.3) / 100

        # Expected goals: rough proxy — 1.5 goals per team avg, adjusted by win prob
        exp_home_gf = 1.5 * (1 + h_win_p - a_win_p)
        exp_away_gf = 1.5 * (1 + a_win_p - h_win_p)

        if result == "A Win":
            stats[home]["pts"] += 3
            stats[home]["wins"] += 1
            stats[away]["losses"] += 1
            gd = max(1, round(exp_home_gf - exp_away_gf))
            stats[home]["gf"] += round(exp_home_gf)
            stats[away]["gf"] += round(exp_away_gf)
            stats[home]["gd"] += gd
            stats[away]["gd"] -= gd
        elif result == "Draw":
            stats[home]["pts"] += 1
            stats[away]["pts"] += 1
            stats[home]["draws"] += 1
            stats[away]["draws"] += 1
            stats[home]["gf"] += 1
            stats[away]["gf"] += 1
        elif result == "B Win":
            stats[away]["pts"] += 3
            stats[away]["wins"] += 1
            stats[home]["losses"] += 1
            gd = max(1, round(exp_away_gf - exp_home_gf))
            stats[home]["gf"] += round(exp_home_gf)
            stats[away]["gf"] += round(exp_away_gf)
            stats[away]["gd"] += gd
            stats[home]["gd"] -= gd

    # Rank each group: pts desc, gd desc, gf desc
    standings: dict[str, list[dict]] = {}
    for gid, team_stats in groups.items():
        ranked = sorted(
            team_stats.items(),
            key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]),
        )
        standings[gid] = [
            {"team": team, "rank": i + 1, **stats}
            for i, (team, stats) in enumerate(ranked)
        ]
    return standings


# ---------------------------------------------------------------------------
# Team strength score (for bracket winner prediction)
# ---------------------------------------------------------------------------

def team_strength(gs: dict[str, list[dict]]) -> dict[str, float]:
    """Return {team: strength} — higher = stronger. Used to predict knockout winners."""
    strength: dict[str, float] = {}
    for teams in gs.values():
        for t in teams:
            pts = t["pts"]
            gd = t["gd"]
            gf = t["gf"]
            rank = t["rank"]  # 1 = group winner (bonus)
            # Score: pts*10 + gd + gf*0.1 + rank_bonus
            rank_bonus = {1: 5, 2: 2, 3: -5, 4: -10}.get(rank, 0)
            strength[t["team"]] = pts * 10 + gd + gf * 0.1 + rank_bonus
    return strength


# ---------------------------------------------------------------------------
# Simulate full bracket deterministically
# ---------------------------------------------------------------------------

def simulate_bracket(
    group_standings: dict[str, list[dict]],
    preds: list[dict],
) -> dict:
    """Build bracket dict (same shape as simulation JSON) from group standings."""
    # Lookup table: (group_letter, rank) -> team name
    def get_team(letter: str, rank: int) -> str:
        gid = f"GROUP_{letter}"
        teams = group_standings.get(gid, [])
        for t in teams:
            if t["rank"] == rank:
                return t["team"]
        return f"{gid}_{rank}"

    strength = team_strength(group_standings)

    def winner(home: str, away: str) -> str:
        return home if strength.get(home, 0) >= strength.get(away, 0) else away

    # Build R32 matches per path
    paths_r32: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    for path_idx, (hl, hr), (al, ar) in _R32_SEEDING:
        home = get_team(hl, hr)
        away = get_team(al, ar)
        w = winner(home, away)
        paths_r32[path_idx].append({"home": home, "away": away, "winner": w})

    # R16: pair consecutive R32 winners
    paths_r16: dict[int, list[dict]] = {}
    for pi, matches in paths_r32.items():
        r16: list[dict] = []
        for i in range(0, len(matches), 2):
            h = matches[i]["winner"]
            a = matches[i + 1]["winner"]
            w = winner(h, a)
            r16.append({"home": h, "away": a, "winner": w})
        paths_r16[pi] = r16

    # QF: pair the 2 R16 winners in each path
    qf_winners: dict[int, dict] = {}
    for pi, r16 in paths_r16.items():
        h = r16[0]["winner"]
        a = r16[1]["winner"]
        w = winner(h, a)
        qf_winners[pi] = {"home": h, "away": a, "winner": w}

    # SF: path0 vs path1, path2 vs path3
    sf1_h, sf1_a = qf_winners[0]["winner"], qf_winners[1]["winner"]
    sf1_w = winner(sf1_h, sf1_a)
    sf1_l = sf1_a if sf1_w == sf1_h else sf1_h

    sf2_h, sf2_a = qf_winners[2]["winner"], qf_winners[3]["winner"]
    sf2_w = winner(sf2_h, sf2_a)
    sf2_l = sf2_a if sf2_w == sf2_h else sf2_h

    # Final
    fin_w = winner(sf1_w, sf2_w)
    fin_l = sf2_w if fin_w == sf1_w else sf1_w

    # 3rd place
    third_w = winner(sf1_l, sf2_l)

    # Build paths list (same structure as simulation)
    paths = []
    for pi in range(4):
        r32_m = paths_r32[pi]
        r16_m = paths_r16[pi]
        qf_m = qf_winners[pi]
        groups = [f"GROUP_{g}" for g in "ABCDEFGHIJKL"[pi * 3: pi * 3 + 3]]
        paths.append({
            "groups": groups,
            "r32": r32_m,
            "r16": r16_m,
            "qf": qf_m,
        })

    return {
        "group_standings": group_standings,
        "paths": paths,
        "sf1": {"home": sf1_h, "away": sf1_a, "winner": sf1_w},
        "sf2": {"home": sf2_h, "away": sf2_a, "winner": sf2_w},
        "third_place": {"home": sf1_l, "away": sf2_l, "winner": third_w},
        "final": {"home": sf1_w, "away": sf2_w, "winner": fin_w},
        "champion": fin_w,
        "finalist": fin_l,
        "third": third_w,
    }


# ---------------------------------------------------------------------------
# Pseudo-probability dict (for box coloring in bracket figure)
# ---------------------------------------------------------------------------

def make_pseudo_probs(bracket: dict) -> dict[str, dict]:
    """Assign pseudo-champion-probabilities from bracket stage reached."""
    champion  = bracket.get("champion", "")
    finalist  = bracket.get("finalist", "")
    third     = bracket.get("third", "")
    sf_losers = {bracket["sf1"]["home"], bracket["sf1"]["away"],
                 bracket["sf2"]["home"], bracket["sf2"]["away"]}

    qf_teams: set[str] = set()
    for path in bracket["paths"]:
        for m in path.get("r16", []):
            qf_teams.add(m["home"])
            qf_teams.add(m["away"])

    r16_teams: set[str] = set()
    for path in bracket["paths"]:
        for m in path.get("r32", []):
            r16_teams.add(m["home"])
            r16_teams.add(m["away"])

    probs: dict[str, dict] = {}

    def assign(team: str, champ_val: float) -> None:
        probs[team] = {"champion": champ_val, "group_advance": 100.0}

    assign(champion, 100.0)
    assign(finalist, 50.0)
    assign(third, 20.0)
    for t in sf_losers - {champion, finalist, third}:
        assign(t, 12.0)
    for t in qf_teams - sf_losers:
        assign(t, 5.0)
    for t in r16_teams - qf_teams:
        assign(t, 2.0)
    return probs


# ---------------------------------------------------------------------------
# Imports from existing export script
# ---------------------------------------------------------------------------

from scripts.export_wc2026_figures import (
    _make_bracket,
    _make_group_overview,
    fl as _fl,
    flag as _flag,
    GOLD,
    SILVER,
    BRONZE,
    TRANS,
    GROUP_LETTERS,
)


# ---------------------------------------------------------------------------
# Per-model figures
# ---------------------------------------------------------------------------

def make_group_overview_for_model(bracket: dict, model_label: str) -> go.Figure:
    """Wrap _make_group_overview with model-specific subtitle."""
    fig = _make_group_overview(bracket)
    # Update subtitle annotation (2nd-to-last annotation by convention)
    for ann in fig.layout.annotations:
        if ann.text == "TabPFN-3 Predictions":
            ann.text = f"{model_label} Predictions"
            break
    return fig


def make_bracket_for_model(bracket: dict, pseudo_probs: dict, model_label: str) -> go.Figure:
    """Wrap _make_bracket with model-specific title."""
    fig = _make_bracket(bracket, pseudo_probs)
    for ann in fig.layout.annotations:
        if "Most Likely Bracket" in str(ann.text):
            ann.text = f"<b>World Cup 2026 — {model_label} Bracket</b>"
            break
    return fig


# ---------------------------------------------------------------------------
# Combined: podium comparison (all models, top 3 side by side)
# ---------------------------------------------------------------------------

def make_podium_comparison(model_brackets: dict[str, dict]) -> go.Figure:
    """7-column table showing Champion / Runner-up / 3rd per model."""
    model_names = list(model_brackets.keys())
    n = len(model_names)

    shapes: list[dict] = []
    anns: list[dict] = []

    # Layout
    COL_W  = 3.2
    HEADER_H = 1.0
    ROW_H  = 2.4
    GAP    = 0.3
    TOTAL_W = n * (COL_W + GAP) - GAP
    TOTAL_H = HEADER_H + 3 * ROW_H

    medal_labels = ["🥇 Champion", "🥈 Runner-up", "🥉 3rd Place"]
    medal_bg    = ["rgba(255,215,0,0.12)", "rgba(192,192,192,0.10)", "rgba(205,127,50,0.08)"]
    medal_border = [GOLD, SILVER, BRONZE]

    # Row labels (left)
    ROW_LABEL_W = 1.5
    for ri, (label, bg, border) in enumerate(zip(medal_labels, medal_bg, medal_border)):
        y_top = TOTAL_H - ri * ROW_H
        y_bot = y_top - ROW_H
        shapes.append(dict(
            type="rect",
            x0=-ROW_LABEL_W, y0=y_bot, x1=0, y1=y_top,
            fillcolor=bg,
            line=dict(color=border, width=1.5),
            layer="below",
        ))
        anns.append(dict(
            x=-ROW_LABEL_W / 2, y=(y_top + y_bot) / 2,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=13, color="#222222"),
            xanchor="center", yanchor="middle",
        ))

    for ci, model_name in enumerate(model_names):
        bracket = model_brackets[model_name]
        champion = bracket.get("champion", "?")
        finalist = bracket.get("finalist", "?")
        third    = bracket.get("third", "?")
        label    = MODELS.get(model_name, model_name)
        color    = MODEL_COLORS.get(model_name, "#888888")

        x_left  = ci * (COL_W + GAP)
        x_right = x_left + COL_W

        # Column header
        shapes.append(dict(
            type="rect",
            x0=x_left, y0=TOTAL_H, x1=x_right, y1=TOTAL_H + HEADER_H,
            fillcolor=_hex_rgba(color, 0.13),
            line=dict(color=color, width=2),
            layer="below",
        ))
        anns.append(dict(
            x=(x_left + x_right) / 2, y=TOTAL_H + HEADER_H / 2,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color=color),
            xanchor="center", yanchor="middle",
        ))

        # Cells: champion, runner-up, 3rd
        teams = [champion, finalist, third]
        for ri, (team, bg, border) in enumerate(zip(teams, medal_bg, medal_border)):
            y_top = TOTAL_H - ri * ROW_H
            y_bot = y_top - ROW_H
            y_mid = (y_top + y_bot) / 2

            shapes.append(dict(
                type="rect",
                x0=x_left, y0=y_bot, x1=x_right, y1=y_top,
                fillcolor=bg,
                line=dict(color=border, width=0.8),
                layer="below",
            ))

            f = _flag(team)
            # Flag emoji
            if f:
                anns.append(dict(
                    x=(x_left + x_right) / 2, y=y_mid + 0.5,
                    text=f,
                    showarrow=False,
                    font=dict(size=30),
                    xanchor="center", yanchor="middle",
                ))
            # Team name
            anns.append(dict(
                x=(x_left + x_right) / 2, y=y_mid - 0.5,
                text=f"<b>{team}</b>",
                showarrow=False,
                font=dict(size=11, color="#111111"),
                xanchor="center", yanchor="middle",
            ))

    # Title
    title_y = TOTAL_H + HEADER_H + 1.0
    anns.append(dict(
        x=(TOTAL_W - GAP) / 2, y=title_y,
        text="<b>World Cup 2026 — Model Podium Predictions</b>",
        showarrow=False,
        font=dict(size=24, color="#111111"),
        xanchor="center", yanchor="bottom",
    ))
    anns.append(dict(
        x=(TOTAL_W - GAP) / 2, y=title_y,
        text="Which AI model picks which Champion, Runner-up & 3rd Place",
        showarrow=False,
        font=dict(size=14, color="#888888"),
        xanchor="center", yanchor="top",
    ))

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes, annotations=anns,
        xaxis=dict(range=[-ROW_LABEL_W - 0.3, TOTAL_W + 0.3],
                   showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.3, title_y + 1.5],
                   showgrid=False, zeroline=False, showticklabels=False),
        height=750,
        width=max(1400, n * 220 + 300),
        paper_bgcolor=TRANS, plot_bgcolor=TRANS,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Combined: champion votes heatmap
# ---------------------------------------------------------------------------

def make_champion_votes(model_brackets: dict[str, dict]) -> go.Figure:
    """Which team is picked as Champion / Runner-up / 3rd by how many models?"""
    from collections import Counter

    stages = {
        "Champion": "champion",
        "Runner-up": "finalist",
        "3rd Place": "third",
    }
    stage_colors = {
        "Champion":   GOLD,
        "Runner-up":  SILVER,
        "3rd Place":  BRONZE,
    }

    model_names = list(model_brackets.keys())
    n_models = len(model_names)

    # Collect predictions per stage
    all_teams: set[str] = set()
    stage_preds: dict[str, dict[str, list[str]]] = {}  # stage → {team: [models...]}

    for stage_label, stage_key in stages.items():
        stage_preds[stage_label] = {}
        for model_name, bracket in model_brackets.items():
            team = bracket.get(stage_key, "")
            if team:
                all_teams.add(team)
                stage_preds[stage_label].setdefault(team, []).append(model_name)

    # Sort teams: champion votes desc, then runner-up, then 3rd
    def sort_key(team: str) -> tuple:
        return (
            -len(stage_preds["Champion"].get(team, [])),
            -len(stage_preds["Runner-up"].get(team, [])),
            -len(stage_preds["3rd Place"].get(team, [])),
        )

    sorted_teams = sorted(all_teams, key=sort_key)

    shapes: list[dict] = []
    anns: list[dict] = []

    CELL_W = 3.8
    CELL_H = 1.6
    LEFT_W = 2.5
    TOP_H  = 0.8
    STAGE_GAP = 0.3

    stage_list = list(stages.keys())
    x_starts = [i * (CELL_W + STAGE_GAP) for i in range(3)]

    # Draw one block per (team, stage)
    for ti, team in enumerate(sorted_teams):
        y_bot = -(ti + 1) * CELL_H
        y_top = -ti * CELL_H
        y_mid = (y_top + y_bot) / 2

        # Team label
        f = _flag(team)
        anns.append(dict(
            x=-LEFT_W / 2, y=y_mid,
            text=f"{''.join([f, ' ']) if f else ''}{team}",
            showarrow=False,
            font=dict(size=12, color="#111111"),
            xanchor="center", yanchor="middle",
        ))

        for si, stage_label in enumerate(stage_list):
            x0 = x_starts[si]
            x1 = x0 + CELL_W
            x_mid = (x0 + x1) / 2
            picks = stage_preds[stage_label].get(team, [])
            n_picks = len(picks)

            # Color intensity by vote count
            alpha = 0.08 + 0.20 * n_picks if n_picks > 0 else 0.04
            border_col = stage_colors[stage_label] if n_picks > 0 else "#dddddd"
            border_w   = 1.5 if n_picks > 0 else 0.5

            shapes.append(dict(
                type="rect",
                x0=x0, y0=y_bot, x1=x1, y1=y_top,
                fillcolor=f"rgba(0,0,0,{alpha:.2f})" if n_picks == 0 else
                          _hex_rgba(stage_colors[stage_label], alpha),
                line=dict(color=border_col, width=border_w),
                layer="below",
            ))

            if n_picks > 0:
                # Model name abbreviations
                model_abbr = " · ".join(
                    MODELS.get(m, m)[:4] for m in picks
                )
                anns.append(dict(
                    x=x_mid, y=y_mid + 0.25,
                    text=f"<b>{n_picks} model{'s' if n_picks > 1 else ''}</b>",
                    showarrow=False,
                    font=dict(size=12, color="#111111"),
                    xanchor="center", yanchor="middle",
                ))
                anns.append(dict(
                    x=x_mid, y=y_mid - 0.35,
                    text=model_abbr,
                    showarrow=False,
                    font=dict(size=8, color="#555555"),
                    xanchor="center", yanchor="middle",
                ))

    # Column headers (stage labels)
    for si, stage_label in enumerate(stage_list):
        x0 = x_starts[si]
        x1 = x0 + CELL_W
        shapes.append(dict(
            type="rect",
            x0=x0, y0=0, x1=x1, y1=TOP_H,
            fillcolor=_hex_rgba(stage_colors[stage_label], 0.20),
            line=dict(color=stage_colors[stage_label], width=2),
            layer="below",
        ))
        anns.append(dict(
            x=(x0 + x1) / 2, y=TOP_H / 2,
            text=f"<b>{stage_label}</b>",
            showarrow=False,
            font=dict(size=14, color="#222222"),
            xanchor="center", yanchor="middle",
        ))

    # Row header
    shapes.append(dict(
        type="rect",
        x0=-LEFT_W, y0=0, x1=0, y1=TOP_H,
        fillcolor="#f0f0f0",
        line=dict(color="#cccccc", width=1),
        layer="below",
    ))
    anns.append(dict(
        x=-LEFT_W / 2, y=TOP_H / 2,
        text="<b>Team</b>",
        showarrow=False, font=dict(size=13, color="#222222"),
        xanchor="center", yanchor="middle",
    ))

    total_height = len(sorted_teams) * CELL_H + TOP_H
    total_width  = x_starts[-1] + CELL_W

    # Title
    title_y = TOP_H + 1.0
    anns.append(dict(
        x=(total_width - CELL_W / 2) / 2, y=title_y,
        text="<b>World Cup 2026 — Podium Consensus Across Models</b>",
        showarrow=False,
        font=dict(size=22, color="#111111"),
        xanchor="center", yanchor="bottom",
    ))
    anns.append(dict(
        x=(total_width - CELL_W / 2) / 2, y=title_y,
        text=f"How {n_models} AI models vote on the top 3 finishers",
        showarrow=False,
        font=dict(size=13, color="#888888"),
        xanchor="center", yanchor="top",
    ))

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes, annotations=anns,
        xaxis=dict(range=[-LEFT_W - 0.2, total_width + 0.2],
                   showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-total_height - 0.3, title_y + 1.8],
                   showgrid=False, zeroline=False, showticklabels=False),
        height=max(600, len(sorted_teams) * 55 + 200),
        width=1100,
        paper_bgcolor=TRANS, plot_bgcolor=TRANS,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.command()
def main(
    out_dir: Path  = typer.Option(Path("data/figures"), "--out-dir"),
    data_dir: Path = typer.Option(Path("data/predictions"), "--data-dir"),
    scale: int     = typer.Option(2, "--scale", help="PNG scale factor"),
) -> None:
    console.rule("[bold]WC 2026 Multi-Model Figure Export")

    # Load all model predictions
    model_preds: dict[str, list[dict]] = {}
    for model_name in MODELS:
        path = data_dir / f"wc2026_{model_name}.json"
        if path.exists():
            with open(path) as f:
                model_preds[model_name] = json.load(f)
            console.print(f"  [green]✓[/green] Loaded {model_name} ({len(model_preds[model_name])} predictions)")
        else:
            console.print(f"  [yellow]⚠[/yellow] Missing: {path}")

    if not model_preds:
        console.print("[red]No prediction files found. Run predict_multi_model.py first.[/red]")
        raise typer.Exit(1)

    # Build brackets for each model
    model_brackets: dict[str, dict] = {}
    model_standings: dict[str, dict] = {}

    console.print("\n[bold]Building brackets...")
    for model_name, preds in model_preds.items():
        standings = build_group_standings(preds)
        bracket   = simulate_bracket(standings, preds)
        model_brackets[model_name]  = bracket
        model_standings[model_name] = standings
        champion = bracket["champion"]
        finalist = bracket["finalist"]
        third    = bracket["third"]
        console.print(
            f"  {MODELS[model_name]:28s}  "
            f"🥇 {champion:18s}  🥈 {finalist:18s}  🥉 {third}"
        )

    # ── Per-model figures ──────────────────────────────────────────────────
    console.print("\n[bold]Generating per-model figures...")
    for model_name, bracket in model_brackets.items():
        model_label = MODELS[model_name]
        pseudo_probs = make_pseudo_probs(bracket)
        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        figures = {
            "group_overview": make_group_overview_for_model(bracket, model_label),
            "bracket":        make_bracket_for_model(bracket, pseudo_probs, model_label),
        }

        for name, fig in figures.items():
            path = model_out / f"{name}.png"
            try:
                fig.write_image(str(path), scale=scale)
                console.print(f"    [green]✓[/green] {path}")
            except Exception as e:
                console.print(f"    [red]✗ {name}.png failed: {e}[/red]")

    # ── Combined figures ───────────────────────────────────────────────────
    console.print("\n[bold]Generating combined figures...")
    combined_out = out_dir / "combined"
    combined_out.mkdir(parents=True, exist_ok=True)

    combined_figs = {
        "podium_comparison": make_podium_comparison(model_brackets),
        "champion_votes":    make_champion_votes(model_brackets),
    }

    for name, fig in combined_figs.items():
        path = combined_out / f"{name}.png"
        try:
            fig.write_image(str(path), scale=scale)
            console.print(f"  [green]✓[/green] {path}")
        except Exception as e:
            console.print(f"  [red]✗ {name}.png failed: {e}[/red]")

    console.rule(
        f"[bold green]Done — "
        f"{len(model_preds) * 2} per-model + {len(combined_figs)} combined PNGs"
    )


if __name__ == "__main__":
    app()
