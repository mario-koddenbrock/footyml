from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from footyml.data.store import DataStore


def team_form_timeline(
    club_id: str,
    store: DataStore,
    n_matches: int = 10,
    club_name: str | None = None,
) -> go.Figure:
    """Line chart of rolling points per game over the last n_matches."""
    matches = store.query(f"""
        SELECT date, result,
               CASE WHEN home_club_id = '{club_id}' THEN 1 ELSE 0 END AS is_home,
               home_goals, away_goals
        FROM matches
        WHERE (home_club_id = '{club_id}' OR away_club_id = '{club_id}')
          AND result IS NOT NULL
        ORDER BY date DESC
        LIMIT {n_matches}
    """).sort("date")

    if len(matches) == 0:
        return go.Figure().update_layout(title="No data available")

    dates = matches["date"].to_list()
    points = []
    results_label = []

    for row in matches.iter_rows(named=True):
        is_home = row["is_home"] == 1
        result = row["result"]
        if (is_home and result == 2) or (not is_home and result == 0):
            points.append(3)
            results_label.append("W")
        elif result == 1:
            points.append(1)
            results_label.append("D")
        else:
            points.append(0)
            results_label.append("L")

    rolling_ppg = [
        round(sum(points[max(0, i - 4): i + 1]) / min(i + 1, 5), 2)
        for i in range(len(points))
    ]

    colors = {"W": "#2ecc71", "D": "#f39c12", "L": "#e74c3c"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=rolling_ppg,
        mode="lines+markers",
        name="Rolling PPG (5)",
        line=dict(color="#3498db", width=2),
        marker=dict(
            color=[colors.get(r, "gray") for r in results_label],
            size=10,
        ),
        text=results_label,
        hovertemplate="%{x}<br>PPG: %{y}<br>Result: %{text}<extra></extra>",
    ))

    title = f"{club_name or club_id} — Form (last {len(matches)} matches)"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rolling PPG (5 games)",
        yaxis=dict(range=[-0.1, 3.1]),
        showlegend=False,
    )
    return fig


def elo_timeline(store: DataStore, club_ids: list[str], labels: list[str] | None = None) -> go.Figure:
    """Line chart of Elo ratings over time for one or more clubs."""
    fig = go.Figure()
    for i, club_id in enumerate(club_ids):
        elo_df = store.get_elo(club_id)
        if len(elo_df) == 0:
            continue
        label = (labels[i] if labels and i < len(labels) else club_id)
        fig.add_trace(go.Scatter(
            x=elo_df["date"].to_list(),
            y=elo_df["elo"].to_list(),
            mode="lines",
            name=label,
        ))

    fig.update_layout(
        title="Elo Rating History",
        xaxis_title="Date",
        yaxis_title="Elo Rating",
    )
    return fig
