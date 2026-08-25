"""
AquaIntel — AI-Powered Water Intelligence & Restoration Platform
Main Streamlit Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from modules.pollution_detector      import detect_pollution, WATER_BODIES
from modules.health_score            import compute_health_score, get_wqi_breakdown
from modules.predictor               import generate_historical_data, predict_risk, risk_summary
from modules.spread_simulator        import run_diffusion, build_spread_animation, compute_spread_stats
from modules.alerts_recommendations  import get_recommendations, send_alert, get_alert_log
from modules.charts                  import (
    gauge_chart, source_pie, band_radar,
    forecast_chart, historical_chart, wqi_bar,
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaIntel — Water Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0f1e;
    color: #c8d6e5;
  }
  .main { background-color: #0a0f1e; }
  .stApp { background-color: #0a0f1e; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1628 0%, #0a1020 100%);
    border-right: 1px solid #1e2d4a;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #0e1628 0%, #121e35 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4a90d9, #00c896);
  }
  .metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6b8cae;
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-sub {
    font-size: 12px;
    color: #4a90d9;
    margin-top: 4px;
  }

  /* Alert card */
  .alert-card {
    background: linear-gradient(135deg, #1a0808 0%, #2d0d0d 100%);
    border: 1px solid #e8412e;
    border-left: 4px solid #e8412e;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }
  .alert-dispatched {
    background: linear-gradient(135deg, #0a1a0d 0%, #0d2410 100%);
    border: 1px solid #00c896;
    border-left: 4px solid #00c896;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
  }

  /* Action pills */
  .action-pill {
    display: inline-block;
    background: #0e1e35;
    border: 1px solid #1e3a5a;
    border-radius: 20px;
    padding: 6px 14px;
    margin: 4px;
    font-size: 13px;
    color: #c8d6e5;
  }
  .action-critical { border-color: #e8412e; }
  .action-medium   { border-color: #f5a623; }
  .action-good     { border-color: #00c896; }

  /* Section headers */
  .section-header {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4a90d9;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 8px;
    margin: 20px 0 14px 0;
    font-weight: 600;
  }

  /* Tab styling */
  [data-baseweb="tab-list"] {
    background: #0d1628;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }
  [data-baseweb="tab"] {
    border-radius: 8px;
    color: #6b8cae;
    font-weight: 600;
  }
  [aria-selected="true"] {
    background: #1e3a5a !important;
    color: #4a90d9 !important;
  }

  /* Satellite image border */
  .sat-image {
    border: 2px solid #1e2d4a;
    border-radius: 12px;
    overflow: hidden;
  }

  /* Hide Streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}

  .stSelectbox > div > div {
    background: #0e1628;
    border: 1px solid #1e2d4a;
    color: #c8d6e5;
  }
  .stSlider > div { color: #c8d6e5; }
  div[data-testid="stMetricValue"] { color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
if "analysis" not in st.session_state:
    st.session_state.analysis    = None
if "alert_sent" not in st.session_state:
    st.session_state.alert_sent  = False
if "alert_data" not in st.session_state:
    st.session_state.alert_data  = None
if "sim_frames" not in st.session_state:
    st.session_state.sim_frames  = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
      <div style='font-size:38px'>🌊</div>
      <div style='font-size:20px; font-weight:700; color:#4a90d9; letter-spacing:2px'>AQUAINTEL</div>
      <div style='font-size:11px; color:#6b8cae; letter-spacing:3px; margin-top:4px'>WATER INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📍 Target Location</div>', unsafe_allow_html=True)
    location = st.selectbox(
        "Water Body",
        list(WATER_BODIES.keys()),
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-header">⚙️ Analysis Parameters</div>', unsafe_allow_html=True)
    manual_poll = st.slider(
        "Override Pollution Level", 0.0, 1.0,
        value=WATER_BODIES[location]["base_pollution"],
        step=0.01,
        help="Simulate different pollution scenarios",
    )
    sim_days = st.slider("Simulation Days", 5, 30, 20, 1)
    forecast_days = st.slider("Forecast Horizon (days)", 7, 21, 14, 1)

    st.markdown("---")
    run_btn = st.button("🔬  RUN FULL ANALYSIS", use_container_width=True, type="primary")
    if st.button("🔄  Reset", use_container_width=True):
        st.session_state.analysis   = None
        st.session_state.alert_sent = False
        st.session_state.sim_frames = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#3a4a5a; text-align:center; line-height:1.8'>
      Powered by AI · Sentinel-2 Simulation<br>
      Diffusion Modelling · Risk Forecasting<br><br>
      <span style='color:#1e3a5a'>© AquaIntel Prototype 2025</span>
    </div>
    """, unsafe_allow_html=True)


# ── Run Analysis ───────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("🛰️  Processing satellite data…"):
        detection   = detect_pollution(location, manual_override=manual_poll)

    with st.spinner("🧠  Computing health metrics…"):
        health      = compute_health_score(
            detection["pollution_level"],
            detection["band_analysis"],
            spread_area_pct=manual_poll * 60,
        )
        wqi         = get_wqi_breakdown(detection["pollution_level"])

    with st.spinner("📈  Running forecast model…"):
        hist_df     = generate_historical_data(location)
        forecast_df = predict_risk(detection["pollution_level"], hist_df, forecast_days)
        risk_sum    = risk_summary(forecast_df)

    with st.spinner("🌊  Simulating pollution spread…"):
        frames      = run_diffusion(
            steps=sim_days,
            pollution_level=detection["pollution_level"],
            source_type=detection["source_type"],
        )
        spread_stats= compute_spread_stats(frames)

    recs = get_recommendations(detection["source_type"], health["score"])

    st.session_state.analysis = {
        "detection": detection, "health": health, "wqi": wqi,
        "hist_df": hist_df, "forecast_df": forecast_df, "risk_sum": risk_sum,
        "frames": frames, "spread_stats": spread_stats, "recs": recs,
        "location": location,
    }
    st.session_state.alert_sent = False
    st.session_state.alert_data = None


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 10px 0 20px 0;'>
  <h1 style='font-size:28px; font-weight:700; color:#ffffff; margin:0; letter-spacing:1px'>
    🌊 AquaIntel — Water Intelligence & Restoration Platform
  </h1>
  <p style='color:#6b8cae; font-size:14px; margin:6px 0 0 0'>
    AI-Powered Detection · Spread Simulation · Predictive Risk · Action Intelligence
  </p>
</div>
""", unsafe_allow_html=True)


# ── No analysis yet ────────────────────────────────────────────────────────────
if st.session_state.analysis is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding:60px 40px; background:linear-gradient(135deg,#0e1628,#121e35);
                    border:1px solid #1e2d4a; border-radius:16px; margin-top:40px'>
          <div style='font-size:64px; margin-bottom:20px'>🛰️</div>
          <h2 style='color:#4a90d9; margin:0 0 12px 0'>Ready for Analysis</h2>
          <p style='color:#6b8cae; font-size:14px; line-height:1.7'>
            Select a water body in the sidebar and click<br>
            <strong style='color:#c8d6e5'>RUN FULL ANALYSIS</strong> to begin.<br><br>
            The platform will:<br>
            🛰️ Process simulated satellite imagery<br>
            🤖 Detect and classify pollution<br>
            📊 Score water health (0–100)<br>
            🔮 Forecast 14-day risk<br>
            🌊 Simulate spread dynamics<br>
            🚨 Generate alerts & action plans
          </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ── Pull analysis from state ───────────────────────────────────────────────────
A          = st.session_state.analysis
detection  = A["detection"]
health     = A["health"]
wqi        = A["wqi"]
hist_df    = A["hist_df"]
forecast_df= A["forecast_df"]
risk_sum   = A["risk_sum"]
frames     = A["frames"]
spread_stats=A["spread_stats"]
recs       = A["recs"]

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def metric_card(col, label, value, sub="", colour="#4a90d9"):
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{colour}'>{value}</div>
      <div class='metric-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

metric_card(k1, "💧 Health Score", f"{health['score']}", health['label'], health['colour'])
metric_card(k2, "☣️ Pollution Level",
            f"{round(detection['pollution_level']*100)}%",
            detection["source_type"], "#e8412e" if detection["pollution_level"] > 0.6 else "#f5a623")
metric_card(k3, "🌊 Spread Area",
            f"{spread_stats['final_area_pct']}%",
            f"{spread_stats['affected_km2']} km² affected", "#f5a623")
metric_card(k4, "⚡ 14-Day Risk",
            risk_sum["worst_risk"],
            f"{risk_sum['high_days']} high-risk days",
            "#e8412e" if risk_sum["worst_risk"] == "High" else "#f5a623")
metric_card(k5, "🏭 Primary Source",
            detection["source_type"][:3].upper(),
            f"{round(max(detection['source_scores'].values())*100)}% confidence", "#4a90d9")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🛰️ Satellite View",
    "🔬 Pollution Analysis",
    "📈 Prediction",
    "🌊 Spread Simulation",
    "💡 Recommendations",
    "🚨 Alert System",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — SATELLITE VIEW
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown('<div class="section-header">🛰️ Simulated Sentinel-2 Composite</div>',
                    unsafe_allow_html=True)
        img_array = detection["image_array"]
        pil_img   = Image.fromarray(img_array)

        # Add scan-line overlay effect
        st.image(pil_img, caption=f"{detection['location']} — False-colour composite (B4/B3/B2)",
                 use_container_width=True)

        st.markdown(f"""
        <div style='background:#0e1628;border:1px solid #1e2d4a;border-radius:8px;
                    padding:12px 16px;margin-top:10px;font-size:13px;'>
          <span style='color:#6b8cae'>Coordinates:</span>
          <span style='font-family:JetBrains Mono,monospace;color:#4a90d9'>
            {detection['lat']}°N, {detection['lon']}°E
          </span> &nbsp;|&nbsp;
          <span style='color:#6b8cae'>Resolution:</span>
          <span style='color:#c8d6e5'>10m/px (simulated)</span> &nbsp;|&nbsp;
          <span style='color:#6b8cae'>Pass:</span>
          <span style='color:#c8d6e5'>Sentinel-2A</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">📡 Band Analysis</div>',
                    unsafe_allow_html=True)
        ba = detection["band_analysis"]

        for name, val, unit, thresh, better in [
            ("NDWI (Water Index)", ba["ndwi_mean"], "", 0, "higher"),
            ("Turbidity Index",    ba["turbidity"],  "", 0.3, "lower"),
            ("Chlorophyll Proxy",  ba["chlorophyll"],"", 0.4, "lower"),
            ("Water Coverage",     ba["water_pixel_pct"], "%", 50, "higher"),
        ]:
            good  = (val > thresh) if better == "higher" else (val < thresh)
            col   = "#00c896" if good else "#e8412e"
            pct   = min(100, abs(val) * 100 if unit == "" else val)
            st.markdown(f"""
            <div style='margin:10px 0'>
              <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                <span style='font-size:12px;color:#c8d6e5'>{name}</span>
                <span style='font-size:12px;font-family:JetBrains Mono,monospace;color:{col}'>
                  {val:.3f}{unit}
                </span>
              </div>
              <div style='height:6px;background:#1e2d4a;border-radius:3px'>
                <div style='height:6px;width:{min(100,pct):.0f}%;background:{col};border-radius:3px'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:20px">🗺️ Location Overview</div>',
                    unsafe_allow_html=True)

        # Simple Plotly map
        fig_map = go.Figure(go.Scattermapbox(
            lat=[detection["lat"]], lon=[detection["lon"]],
            mode="markers",
            marker=dict(
                size=20, color=health["colour"],
                symbol="circle",
            ),
            text=[f"{detection['location']}<br>Score: {health['score']}"],
            hoverinfo="text",
        ))
        fig_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=detection["lat"], lon=detection["lon"]),
                zoom=8,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=230,
            paper_bgcolor="#0a0f1e",
        )
        st.plotly_chart(fig_map, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — POLLUTION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.plotly_chart(gauge_chart(health["score"], health["label"], health["colour"]),
                        use_container_width=True)

    with c2:
        st.plotly_chart(source_pie(detection["source_scores"]),
                        use_container_width=True)

    with c3:
        st.plotly_chart(band_radar(detection["band_analysis"], health["sub_scores"]),
                        use_container_width=True)

    st.markdown('<div class="section-header">🧪 Water Quality Index Parameters</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(wqi_bar(wqi), use_container_width=True)

    with c2:
        st.markdown("**WHO / IS 10500 Benchmarks**")
        for param, val in wqi.items():
            if isinstance(val, str):
                colour = "#e8412e" if val in ("Detected",) else "#f5a623" if val == "Trace" else "#00c896"
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:7px 0;
                            border-bottom:1px solid #1e2d4a;font-size:13px'>
                  <span style='color:#c8d6e5'>{param}</span>
                  <span style='color:{colour};font-family:JetBrains Mono,monospace'>{val}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                thresholds = {
                    "pH": (6.5, 8.5), "DO (mg/L)": (6.0, None),
                    "BOD (mg/L)": (None, 3.0), "Turbidity (NTU)": (None, 5.0),
                    "Nitrates (mg/L)": (None, 10.0), "Coliform (MPN)": (None, 10),
                }
                t = thresholds.get(param)
                if t:
                    ok = True
                    if t[0] and val < t[0]: ok = False
                    if t[1] and val > t[1]: ok = False
                    colour = "#00c896" if ok else "#e8412e"
                    status = "✓" if ok else "✗"
                else:
                    colour, status = "#4a90d9", "·"

                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:7px 0;
                            border-bottom:1px solid #1e2d4a;font-size:13px'>
                  <span style='color:#c8d6e5'>{param}</span>
                  <span style='color:{colour};font-family:JetBrains Mono,monospace'>{val} {status}</span>
                </div>
                """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    # Risk headline
    rs = risk_sum
    headline_col = "#e8412e" if rs["high_days"] >= 5 else "#f5a623" if rs["high_days"] >= 1 else "#00c896"
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0e1628,#121e35);border:1px solid {headline_col};
                border-left:4px solid {headline_col};border-radius:10px;padding:18px 24px;margin-bottom:16px'>
      <div style='font-size:18px;font-weight:700;color:{headline_col}'>{rs["headline"]}</div>
      <div style='font-size:13px;color:#6b8cae;margin-top:6px'>
        High Risk: {rs["high_days"]} days &nbsp;|&nbsp;
        Medium Risk: {rs["medium_days"]} days &nbsp;|&nbsp;
        Low Risk: {rs["low_days"]} days &nbsp;|&nbsp;
        Peak Pollution Day: {rs["peak_day"]} ({round(rs["peak_poll"]*100)}%)
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(forecast_chart(forecast_df), use_container_width=True)

    st.markdown('<div class="section-header">📜 90-Day Historical Record</div>',
                unsafe_allow_html=True)
    st.plotly_chart(historical_chart(hist_df), use_container_width=True)

    # Forecast table
    st.markdown('<div class="section-header">📋 Day-by-Day Forecast</div>',
                unsafe_allow_html=True)
    display_df = forecast_df[["day","date","predicted_poll","rainfall_mm","risk","confidence_pct"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%b %d")
    display_df.columns = ["Day","Date","Pollution Index","Rainfall (mm)","Risk","Confidence %"]
    st.dataframe(
        display_df.style.applymap(
            lambda v: f"color: #e8412e; font-weight:bold" if v == "High"
                      else f"color: #f5a623" if v == "Medium"
                      else f"color: #00c896" if v == "Low"
                      else "",
            subset=["Risk"]
        ),
        use_container_width=True, height=300,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — SPREAD SIMULATION
# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, unit in [
        (c1, "Initial Coverage",  spread_stats["initial_area_pct"], "%"),
        (c2, "Final Coverage",    spread_stats["final_area_pct"],   "%"),
        (c3, "Affected Area",     spread_stats["affected_km2"],     " km²"),
        (c4, "Growth Factor",     spread_stats["growth_factor"],    "×"),
    ]:
        col.markdown(f"""
        <div class='metric-card' style='text-align:center'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value' style='font-size:24px'>{val}{unit}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    sim_fig = build_spread_animation(
        frames, detection["source_type"], detection["location"]
    )
    st.plotly_chart(sim_fig, use_container_width=True)

    st.markdown("""
    <div style='background:#0e1628;border:1px solid #1e2d4a;border-radius:8px;
                padding:14px 18px;font-size:13px;color:#6b8cae'>
      <strong style='color:#4a90d9'>Simulation Model:</strong>
      2-D Fickian diffusion with advection (river current) and first-order decay.
      Parameters calibrated per pollution source type.
      Press <strong style='color:#c8d6e5'>▶ Play</strong> to animate spread over time,
      or drag the slider to inspect any time step.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    urgency_colours = {"CRITICAL": "#e8412e", "HIGH": "#f5a623", "MODERATE": "#4a90d9"}
    urg_col = urgency_colours.get(recs["urgency"], "#4a90d9")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0e1628,#121e35);
                border:1px solid {urg_col};border-radius:10px;
                padding:16px 22px;margin-bottom:20px;display:flex;
                justify-content:space-between;align-items:center'>
      <div>
        <div style='font-size:12px;color:#6b8cae;letter-spacing:2px'>POLLUTION SOURCE</div>
        <div style='font-size:22px;font-weight:700;color:#ffffff'>{recs["source_type"]}</div>
      </div>
      <div style='text-align:right'>
        <div style='font-size:12px;color:#6b8cae;letter-spacing:2px'>URGENCY LEVEL</div>
        <div style='font-size:22px;font-weight:700;color:{urg_col}'>{recs["urgency"]}</div>
      </div>
      <div style='text-align:right'>
        <div style='font-size:12px;color:#6b8cae;letter-spacing:2px'>EST. RECOVERY</div>
        <div style='font-size:22px;font-weight:700;color:#c8d6e5'>{recs["est_recovery_days"]} days</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    def action_section(col, title, actions, border_col, icon):
        items = "".join([f"""
        <div style='display:flex;align-items:flex-start;gap:10px;padding:10px 0;
                    border-bottom:1px solid #1e2d4a'>
          <span style='font-size:16px;flex-shrink:0'>{a.split()[0]}</span>
          <span style='font-size:13px;color:#c8d6e5;line-height:1.5'>{' '.join(a.split()[1:])}</span>
        </div>
        """ for a in actions])
        col.markdown(f"""
        <div style='background:#0e1628;border:1px solid {border_col};
                    border-top:3px solid {border_col};border-radius:10px;
                    padding:16px 18px;height:100%'>
          <div style='font-size:12px;text-transform:uppercase;letter-spacing:2px;
                      color:{border_col};margin-bottom:12px'>{icon} {title}</div>
          {items}
        </div>
        """, unsafe_allow_html=True)

    action_section(col1, "Immediate Actions (0–48h)",
                   recs["immediate"], "#e8412e", "🚨")
    action_section(col2, "Short-Term (1–4 weeks)",
                   recs["short_term"], "#f5a623", "⚡")
    action_section(col3, "Long-Term (1–6 months)",
                   recs["long_term"],  "#00c896", "🌱")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — ALERT SYSTEM
# ──────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    c1, c2 = st.columns([1.5, 1])

    with c1:
        st.markdown('<div class="section-header">🚨 Alert Dispatch Panel</div>',
                    unsafe_allow_html=True)

        poll_pct  = round(detection["pollution_level"] * 100)
        hs        = health["score"]
        sev_text  = "CRITICAL 🔴" if hs < 40 else "HIGH 🟠" if hs < 60 else "MODERATE 🟡"
        sev_col   = "#e8412e"  if hs < 40 else "#f5a623" if hs < 60 else "#f5a623"

        st.markdown(f"""
        <div class='alert-card'>
          <div style='font-size:15px;font-weight:700;color:#e8412e;margin-bottom:12px'>
            📋 DRAFT ALERT — PENDING DISPATCH
          </div>
          <div style='line-height:2;color:#c8d6e5'>
            <b style='color:#6b8cae'>Location   :</b> {detection["location"]}<br>
            <b style='color:#6b8cae'>Severity   :</b>
              <span style='color:{sev_col}'>{sev_text}</span><br>
            <b style='color:#6b8cae'>Source     :</b> {detection["source_type"]}<br>
            <b style='color:#6b8cae'>Pollution  :</b> {poll_pct}%<br>
            <b style='color:#6b8cae'>Health Score:</b> {hs}/100<br>
            <b style='color:#6b8cae'>Risk (14d) :</b> {risk_sum["worst_risk"]}<br>
            <b style='color:#6b8cae'>Channels   :</b> SMS · Email · Dashboard · GIS
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.alert_sent:
            if st.button("🚨  DISPATCH ALERT TO AUTHORITIES", use_container_width=True, type="primary"):
                alert_data = send_alert(
                    location=detection["location"],
                    pollution_level=detection["pollution_level"],
                    source_type=detection["source_type"],
                    health_score=health["score"],
                    risk_level=risk_sum["worst_risk"],
                )
                st.session_state.alert_sent = True
                st.session_state.alert_data = alert_data
                st.rerun()
        else:
            ad = st.session_state.alert_data
            st.markdown(f"""
            <div class='alert-dispatched'>
              <div style='font-size:16px;font-weight:700;color:#00c896;margin-bottom:12px'>
                ✅ ALERT DISPATCHED SUCCESSFULLY
              </div>
              <div style='line-height:2;color:#c8d6e5;font-family:JetBrains Mono,monospace;font-size:12px'>
                Alert ID   : <b style='color:#4a90d9'>#{ad["alert_id"]}</b><br>
                Timestamp  : {ad["timestamp"]}<br>
                Authority  : {ad["authority"]}<br>
                Contact    : {ad["contact"]}<br>
                Status     : <b style='color:#00c896'>{ad["status"]}</b><br>
                Channels   : {' · '.join(ad["channels"])}
              </div>
              <div style='margin-top:14px;padding:10px;background:#051008;
                          border-radius:6px;font-size:11px;color:#4a7c4f;
                          font-family:JetBrains Mono,monospace'>
                {ad["message"]}
              </div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">📜 Alert Log</div>',
                    unsafe_allow_html=True)

        log = get_alert_log()
        if not log:
            st.markdown("""
            <div style='text-align:center;padding:40px;color:#3a4a5a'>
              <div style='font-size:32px'>📭</div>
              <div style='margin-top:10px'>No alerts dispatched yet</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for entry in reversed(log):
                sev_col2 = "#e8412e" if "CRITICAL" in entry["severity"] else "#f5a623"
                st.markdown(f"""
                <div style='background:#0e1628;border:1px solid #1e2d4a;
                            border-left:3px solid {sev_col2};border-radius:8px;
                            padding:12px 14px;margin-bottom:8px;font-size:12px'>
                  <div style='display:flex;justify-content:space-between'>
                    <b style='color:#4a90d9'>#{entry["alert_id"]}</b>
                    <span style='color:#3a4a5a'>{entry["timestamp"]}</span>
                  </div>
                  <div style='color:#c8d6e5;margin-top:6px'>{entry["location"]}</div>
                  <div style='color:{sev_col2}'>{entry["severity"]}</div>
                  <div style='color:#6b8cae'>{entry["authority"]} — {entry["contact"]}</div>
                  <div style='color:#00c896;margin-top:4px'>{entry["status"]}</div>
                </div>
                """, unsafe_allow_html=True)

        # Summary metrics
        if log:
            st.markdown("---")
            st.markdown(f"""
            <div style='text-align:center;color:#6b8cae;font-size:13px'>
              Total Alerts Dispatched: <b style='color:#4a90d9'>{len(log)}</b>
            </div>
            """, unsafe_allow_html=True)
