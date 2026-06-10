from __future__ import annotations

import plotly.graph_objects as go


def match_prediction_bar(
    prediction: dict[str, float],
    home_team: str,
    away_team: str,
) -> go.Figure:
    """Horizontal stacked bar showing home win / draw / away win probabilities."""
    home_win = prediction.get("home_win_prob", 0.0)
    draw = prediction.get("draw_prob", 0.0)
    away_win = prediction.get("away_win_prob", 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Home Win",
        x=[home_win],
        y=[""],
        orientation="h",
        marker_color="#2ecc71",
        text=[f"{home_win:.1f}%"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Draw",
        x=[draw],
        y=[""],
        orientation="h",
        marker_color="#95a5a6",
        text=[f"{draw:.1f}%"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Away Win",
        x=[away_win],
        y=[""],
        orientation="h",
        marker_color="#e74c3c",
        text=[f"{away_win:.1f}%"],
        textposition="inside",
    ))

    fig.update_layout(
        barmode="stack",
        title=f"{home_team} vs {away_team}",
        xaxis=dict(title="Probability (%)", range=[0, 100]),
        height=200,
        showlegend=True,
    )
    return fig


def matchday_prediction_table(predictions: list[dict]) -> go.Figure:
    """Plotly table with all upcoming matches and probability columns."""
    headers = ["Date", "Home Team", "Away Team", "Home Win %", "Draw %", "Away Win %", "Prediction"]
    rows: list[list[str]] = [[], [], [], [], [], [], []]

    for pred in predictions:
        rows[0].append(str(pred.get("date", ""))[:10])
        rows[1].append(str(pred.get("home_team", "")))
        rows[2].append(str(pred.get("away_team", "")))
        rows[3].append(f"{pred.get('home_win_prob', 0):.1f}%")
        rows[4].append(f"{pred.get('draw_prob', 0):.1f}%")
        rows[5].append(f"{pred.get('away_win_prob', 0):.1f}%")
        rows[6].append(str(pred.get("predicted_result", "")))

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color="#2c3e50",
            font=dict(color="white", size=13),
            align="center",
        ),
        cells=dict(
            values=rows,
            fill_color=[["#f8f9fa" if i % 2 == 0 else "white" for i in range(len(rows[0]))]],
            align=["center"] * 7,
        ),
    )])
    fig.update_layout(title="Upcoming Match Predictions", margin=dict(t=50, b=10))
    return fig
