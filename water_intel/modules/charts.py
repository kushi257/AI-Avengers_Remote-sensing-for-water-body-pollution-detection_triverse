"""
Visualisation helpers — Plotly charts used across dashboard pages.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DARK_BG   = "#0a0f1e"
CARD_BG   = "#0e1628"
ACCENT    = "#4a90d9"
ACCENT2   = "#00c896"
WARN      = "#f5a623"
DANGER    = "#e8412e"
TEXT_COL  = "#c8d6e5"


def gauge_chart(score: float, label: str, colour: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": colour}, "suffix": ""},
        title={"text": f"Water Health Score<br><span style='font-size:16px;color:{colour}'>{label}</span>",
               "font": {"color": TEXT_COL, "size": 14}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": TEXT_COL,
                      "tickfont": {"color": TEXT_COL}},
            "bar":   {"color": colour, "thickness": 0.25},
            "bgcolor": CARD_BG,
            "bordercolor": "#1e2d4a",
            "steps": [
                {"range": [0,  20], "color": "#3d0c0c"},
                {"range": [20, 40], "color": "#5c1a1a"},
                {"range": [40, 60], "color": "#4a3000"},
                {"range": [60, 80], "color": "#1a3d1a"},
                {"range": [80, 100],"color": "#0d2d1f"},
            ],
            "threshold": {"line": {"color": colour, "width": 3},
                          "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG, font_color=TEXT_COL,
        height=280, margin=dict(l=20, r=20, t=60, b=10),
    )
    return fig


def source_pie(source_scores: dict) -> go.Figure:
    colours = {"Industrial": "#e8412e", "Sewage": "#f5a623", "Agricultural": "#52be80"}
    labels  = list(source_scores.keys())
    values  = list(source_scores.values())
    cols    = [colours[k] for k in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=cols, line=dict(color=DARK_BG, width=2)),
        textfont=dict(color="white", size=13),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font_color=TEXT_COL,
        legend=dict(font=dict(color=TEXT_COL)),
        title=dict(text="Pollution Source Attribution", font=dict(color=TEXT_COL, size=14)),
        height=280, margin=dict(l=10, r=10, t=50, b=10),
        annotations=[dict(text="Source<br>Type", x=0.5, y=0.5,
                          font_size=13, font_color=TEXT_COL, showarrow=False)],
    )
    return fig


def band_radar(band_analysis: dict, sub_scores: dict) -> go.Figure:
    categories = list(sub_scores.keys())
    values     = [sub_scores[c] for c in categories]
    values_loop = values + [values[0]]
    cats_loop   = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_loop, theta=cats_loop,
        fill="toself",
        fillcolor=f"rgba(74,144,217,0.25)",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=ACCENT, size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(range=[0, 100], tickfont=dict(color=TEXT_COL, size=9),
                            gridcolor="#1e2d4a", linecolor="#1e2d4a"),
            angularaxis=dict(tickfont=dict(color=TEXT_COL, size=10),
                             gridcolor="#1e2d4a", linecolor="#1e2d4a"),
        ),
        paper_bgcolor=DARK_BG,
        font_color=TEXT_COL,
        title=dict(text="Water Quality Sub-scores", font=dict(color=TEXT_COL, size=14)),
        height=300, margin=dict(l=30, r=30, t=50, b=10),
        showlegend=False,
    )
    return fig


def forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    risk_col_map = {"High": DANGER, "Medium": WARN, "Low": ACCENT2}
    bar_cols = [risk_col_map[r] for r in forecast_df["risk"]]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=["Predicted Pollution Index", "Forecast Rainfall (mm)"],
    )

    # Pollution line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["predicted_poll"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=bar_cols, size=8, line=dict(color="white", width=1)),
        name="Pollution Index",
        hovertemplate="Day %{x|%b %d}: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Confidence band (mock ±5%)
    fig.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
        y=list(np.clip(forecast_df["predicted_poll"] + 0.07, 0, 1)) +
          list(np.clip(forecast_df["predicted_poll"] - 0.07, 0, 1))[::-1],
        fill="toself", fillcolor="rgba(74,144,217,0.10)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="CI",
    ), row=1, col=1)

    # Risk threshold lines
    for val, lbl, col in [(0.65, "High", DANGER), (0.40, "Medium", WARN)]:
        fig.add_hline(y=val, line_dash="dot", line_color=col,
                      annotation_text=lbl, annotation_font_color=col, row=1, col=1)

    # Rainfall bars
    fig.add_trace(go.Bar(
        x=forecast_df["date"], y=forecast_df["rainfall_mm"],
        marker_color=ACCENT, opacity=0.7, name="Rainfall",
        hovertemplate="%{x|%b %d}: %{y}mm<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font_color=TEXT_COL, height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(font=dict(color=TEXT_COL), bgcolor=CARD_BG),
        xaxis2=dict(showgrid=False, gridcolor="#1e2d4a", linecolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", range=[0, 1]),
        yaxis2=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_COL
    return fig


def historical_chart(hist_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["pollution_idx"],
        name="Pollution Index", line=dict(color=DANGER, width=2),
        fill="toself", fillcolor="rgba(232,65,46,0.10)",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=hist_df["date"], y=hist_df["rainfall_mm"],
        name="Rainfall (mm)", marker_color=ACCENT, opacity=0.5,
    ), secondary_y=True)

    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font_color=TEXT_COL, height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(font=dict(color=TEXT_COL), bgcolor=CARD_BG),
        title=dict(text="90-Day Historical Record", font=dict(color=TEXT_COL, size=14)),
        xaxis=dict(gridcolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a", title="Pollution Index"),
        yaxis2=dict(title="Rainfall mm", gridcolor="#1e2d4a"),
    )
    return fig


def wqi_bar(wqi_data: dict, thresholds: dict | None = None) -> go.Figure:
    """Horizontal bar chart for WQI parameters."""
    default_thresh = {
        "pH": 8.5, "DO (mg/L)": 6.0, "BOD (mg/L)": 3.0,
        "Turbidity (NTU)": 5.0, "Nitrates (mg/L)": 10.0,
    }
    thresh = thresholds or default_thresh

    labels, values, cols = [], [], []
    for k, v in wqi_data.items():
        if isinstance(v, str):
            continue
        labels.append(k)
        values.append(float(v))
        t = thresh.get(k)
        if t:
            cols.append(DANGER if float(v) > t * 1.5
                        else WARN if float(v) > t else ACCENT2)
        else:
            cols.append(ACCENT)

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=cols,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT_COL, size=11),
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font_color=TEXT_COL, height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(gridcolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a"),
        showlegend=False,
    )
    return fig
