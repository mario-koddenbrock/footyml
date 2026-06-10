"""FootyML World Cup Dashboard — Streamlit app.

Launch:
    streamlit run app/dashboard.py

Requires:
    pip install streamlit   (or: uv add streamlit)
    FOOTBALL_DATA_API_KEY environment variable
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="FootyML — World Cup 2026",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data(ttl=300)  # refresh every 5 minutes
def _load_predictions(competition: str, season: int) -> list[dict]:
    from footyml.config import MODELS_DIR
    from footyml.prediction.tournament import TournamentPredictor

    candidates = sorted(MODELS_DIR.glob(f"tabpfn_{competition}_*.pkl"))
    model_path = candidates[-1] if candidates else None

    async def _run() -> list[dict]:
        async with TournamentPredictor(model_path=model_path) as predictor:
            return await predictor.predict_all_upcoming(competition=competition, season=season, sync=True)

    try:
        return asyncio.run(_run())
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")
        return []


@st.cache_data(ttl=3600)
def _load_standings(competition: str, season: int) -> "pl.DataFrame | None":
    import polars as pl
    from footyml.providers.football_data import FootballDataProvider

    async def _run() -> "pl.DataFrame":
        async with FootballDataProvider() as p:
            return await p.get_standings(competition, season=season)

    try:
        return asyncio.run(_run())
    except Exception:
        return None


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("⚽ FootyML")
    st.markdown("**World Cup 2026 Prediction Dashboard**")
    st.divider()

    competition = st.selectbox(
        "Competition",
        options=["WC", "EC", "CL"],
        index=0,
    )
    season = st.number_input("Season", value=2026, min_value=2000, max_value=2030, step=1)

    if not os.environ.get("FOOTBALL_DATA_API_KEY"):
        st.warning("Set FOOTBALL_DATA_API_KEY environment variable to load live data.")

    refresh = st.button("🔄 Refresh Data", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

# ------------------------------------------------------------------
# Main content
# ------------------------------------------------------------------

st.title(f"⚽ {competition} {season} — Upcoming Match Predictions")

predictions = _load_predictions(competition, int(season))

if not predictions:
    st.info("No upcoming matches found, or no API key configured.")
    st.stop()

import polars as pl

df = pl.DataFrame([{k: v for k, v in p.items()} for p in predictions])

# ------------------------------------------------------------------
# Group filter
# ------------------------------------------------------------------

groups = sorted({str(p.get("group_id", "")) for p in predictions if p.get("group_id")})
if groups:
    selected_groups = st.multiselect("Filter by Group", options=groups, default=groups)
    filtered = [p for p in predictions if not p.get("group_id") or str(p.get("group_id")) in selected_groups]
else:
    filtered = predictions

# ------------------------------------------------------------------
# Match cards
# ------------------------------------------------------------------

cols = st.columns(2)
for i, pred in enumerate(filtered):
    col = cols[i % 2]
    with col:
        home = str(pred.get("home_team", "Team A"))
        away = str(pred.get("away_team", "Team B"))
        date = str(pred.get("date", ""))[:10]
        stage = str(pred.get("stage", ""))
        group = str(pred.get("group_id", ""))
        hw = pred.get("home_win_prob")
        dw = pred.get("draw_prob")
        aw = pred.get("away_win_prob")
        predicted = str(pred.get("predicted_result", "—"))
        method = str(pred.get("method", ""))

        label = f"{group} — " if group else ""
        st.markdown(f"**{label}{home} vs {away}**")
        st.caption(f"{date} | {stage}")

        if hw is not None:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Bar(x=[hw], y=[""], orientation="h", name=f"{home} Win",
                                 marker_color="#2ecc71", text=[f"{hw:.1f}%"], textposition="inside"))
            fig.add_trace(go.Bar(x=[dw], y=[""], orientation="h", name="Draw",
                                 marker_color="#95a5a6", text=[f"{dw:.1f}%"], textposition="inside"))
            fig.add_trace(go.Bar(x=[aw], y=[""], orientation="h", name=f"{away} Win",
                                 marker_color="#e74c3c", text=[f"{aw:.1f}%"], textposition="inside"))
            fig.update_layout(
                barmode="stack",
                xaxis=dict(range=[0, 100], showticklabels=False),
                height=70,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"bar_{i}")
            badge = "🔮" if method == "tabpfn" else "📊"
            st.caption(f"{badge} Prediction: **{predicted}**")
        else:
            st.caption("No model loaded — showing fixtures only")

        st.divider()

# ------------------------------------------------------------------
# Group standings
# ------------------------------------------------------------------

if groups:
    st.subheader("Group Standings")
    standings = _load_standings(competition, int(season))
    if standings is not None and len(standings) > 0:
        for group in sorted(standings["group_id"].unique().to_list()):
            grp_df = standings.filter(pl.col("group_id") == group).sort("position")
            with st.expander(f"**{group}**", expanded=True):
                st.dataframe(
                    grp_df.select(["position", "team_name", "played", "won", "draw", "lost", "goal_difference", "points"]),
                    hide_index=True,
                    use_container_width=True,
                )
    else:
        st.info("Standings not available yet (group stage may not have started).")

# ------------------------------------------------------------------
# Raw prediction table
# ------------------------------------------------------------------

with st.expander("📊 Full Prediction Table"):
    display_df = df.select([
        c for c in ["date", "stage", "group_id", "home_team", "away_team",
                    "home_win_prob", "draw_prob", "away_win_prob", "predicted_result", "venue_type"]
        if c in df.columns
    ])
    st.dataframe(display_df, hide_index=True, use_container_width=True)
