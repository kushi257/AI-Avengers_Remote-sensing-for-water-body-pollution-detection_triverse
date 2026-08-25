# 🌊 AquaIntel — AI-Powered Water Intelligence & Restoration Platform

> **Hackathon-Ready Prototype** — Full-stack water pollution detection, prediction, simulation, and action intelligence dashboard.

---

## 🎯 What It Does

| Module | Feature | Tech |
|--------|---------|------|
| 🛰️ Satellite Input | Simulated Sentinel-2 false-colour composites | NumPy + PIL |
| 🤖 Detection | Pollution level + band analysis (NDWI, turbidity, chlorophyll) | Rule-based + math |
| 🏭 Classification | Industrial / Sewage / Agricultural source attribution | Weighted scoring |
| 💯 Health Score | 0–100 composite score with 5 sub-components | Custom algorithm |
| 🔮 Prediction | 14-day pollution & rainfall forecast | AR model |
| 🌊 Simulation | Animated 2-D diffusion spread with Play/Pause slider | Plotly animation |
| 🚨 Alerts | Alert dispatch simulation with authority contacts | Mock API |
| 💡 Recommendations | Tiered 3-level action plans per source type | Rule engine |
| 📊 Dashboard | 6-tab interactive UI with dark theme | Streamlit |

---

## 📁 Project Structure

```
water_intel/
├── app.py                     ← Main Streamlit dashboard (6 tabs)
├── run.py                     ← Quick-start launcher with dep check
├── demo_data.py               ← Standalone module verification script
├── requirements.txt
├── .streamlit/
│   └── config.toml            ← Dark theme config
└── modules/
    ├── __init__.py
    ├── pollution_detector.py  ← Satellite simulation + detection
    ├── health_score.py        ← WHS computation (0–100)
    ├── predictor.py           ← 14-day forecast model
    ├── spread_simulator.py    ← 2-D diffusion animation
    ├── alerts_recommendations.py ← Alert + action engine
    └── charts.py              ← All Plotly chart builders
```

---

## ▶️ How to Run

### Option A — Quick Start
```bash
cd water_intel
python run.py
```
Opens at **http://localhost:8501**

### Option B — Direct Streamlit
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option C — Verify modules first
```bash
python demo_data.py   # Tests all 7 modules in terminal
streamlit run app.py  # Then launch UI
```

---

## 🧪 Sample Data

All data is **procedurally simulated** — no API keys or downloads needed.

| Data | How Simulated |
|------|--------------|
| Satellite images | NumPy-generated false-colour composites with per-source colour signatures |
| Pollution levels | Location-specific base values + Gaussian noise |
| Historical records | 90-day sinusoidal pollution + rainfall time series |
| Forecast | Autoregressive model with rainfall influence and mean reversion |
| WQI parameters | Physics-calibrated formulae (DO, BOD, turbidity, etc.) |
| Spread simulation | Fickian 2-D diffusion + advection + first-order decay |

---

## 🎯 Demo Flow (Hackathon Presentation)

**Estimated demo time: 5 minutes**

1. **Open dashboard** → Show the landing screen with the 🌊 header

2. **Tab: Satellite View**
   - Select **"Yamuna River (Delhi)"** from sidebar
   - Set pollution slider to **0.82**
   - Click **🔬 RUN FULL ANALYSIS**
   - Show the generated satellite image → explain false-colour bands

3. **Tab: Pollution Analysis**
   - Point to the **gauge** → "Score: 18 — Critical"
   - Show **source pie chart** → "Primarily Industrial + Sewage"
   - Show **radar chart** → sub-components all red

4. **Tab: Prediction**
   - "The model forecasts 8 high-risk days in the next 14"
   - Show the dual-panel chart — pollution line + rainfall bars
   - Scroll to the forecast table

5. **Tab: Spread Simulation** ← **Most visual!**
   - Click **▶ Play** → watch the animated heatmap spread
   - "This is 2-D Fickian diffusion with river current advection"
   - Drag slider to show Day 0 vs Day 20

6. **Tab: Recommendations**
   - "Three tiers of actions, auto-mapped to source type"
   - Read one immediate action → "Block industrial discharge"

7. **Tab: Alert System**
   - Click **🚨 DISPATCH ALERT TO AUTHORITIES**
   - Show the alert card with authority, contact, channels
   - "In a real system this fires SMS + email + GIS simultaneously"

8. **Switch location** → "Chilika Lake (Odisha)" → re-run → show green gauge

---

## 🗺️ Supported Water Bodies

| Location | State | Base Pollution |
|----------|-------|---------------|
| Ganges River (Kanpur) | Uttar Pradesh | 78% 🔴 |
| Yamuna River (Delhi) | Delhi | 82% 🔴 |
| Sabarmati River (Ahmedabad) | Gujarat | 61% 🟠 |
| Cauvery River (Trichy) | Tamil Nadu | 38% 🟡 |
| Chilika Lake (Odisha) | Odisha | 29% 🟢 |
| Vembanad Lake (Kerala) | Kerala | 44% 🟡 |
| Dal Lake (Kashmir) | J&K | 55% 🟠 |
| Hussain Sagar (Hyderabad) | Telangana | 69% 🟠 |

---

## ⚙️ Key Configuration

In `app.py` sidebar you can adjust:
- **Water Body** — switches all data + authority contacts
- **Override Pollution Level** — 0.0 (pristine) to 1.0 (toxic)
- **Simulation Days** — 5–30 days of spread to animate
- **Forecast Horizon** — 7–21 day prediction window

---

## 🔭 Production Extension Path

| Current (Mock) | Production Upgrade |
|----------------|-------------------|
| Simulated images | Google Earth Engine API + Sentinel-2 real data |
| Rule-based source classifier | CNN trained on EuroSAT / Water bodies dataset |
| AR forecast model | LSTM / Prophet with gauge station data |
| Diffusion simulation | FVM hydrodynamic model (MIKE, SWAT) |
| Alert mock | Twilio SMS + SendGrid email + IFTTT webhook |
| Location list | PostGIS + satellite tile streaming |

---

*Built for hackathon demo. All data is simulated for illustrative purposes.*
