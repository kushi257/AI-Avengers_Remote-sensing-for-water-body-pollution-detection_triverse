"""
Module 5: Pollution Risk Prediction
Simulates a time-series ML forecast using rainfall + historical data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


RISK_COLOURS = {
    "High":   "#e8412e",
    "Medium": "#f5a623",
    "Low":    "#7ecb4f",
}


def generate_historical_data(location: str, days: int = 90) -> pd.DataFrame:
    """Simulate 90-day historical pollution + rainfall dataset."""
    rng  = np.random.default_rng(seed=abs(hash(location)) % 9999)
    dates = [datetime.today() - timedelta(days=days - i) for i in range(days)]

    # Seasonal rainfall pattern (monsoon peak mid-period)
    t        = np.linspace(0, 2 * np.pi, days)
    rainfall = np.clip(15 + 30 * np.sin(t - 0.5) + rng.normal(0, 5, days), 0, None)

    # Pollution follows rainfall with 3-day lag + industrial base
    base_poll = 0.4 + 0.3 * np.sin(t + 0.3)
    lag_rain  = np.roll(rainfall, 3)
    pollution  = np.clip(base_poll + lag_rain * 0.005 + rng.normal(0, 0.05, days), 0.1, 0.99)

    return pd.DataFrame({
        "date":           dates,
        "rainfall_mm":    np.round(rainfall, 1),
        "pollution_idx":  np.round(pollution, 3),
        "temperature_c":  np.round(25 + 8 * np.sin(t) + rng.normal(0, 1, days), 1),
        "discharge_m3s":  np.round(np.clip(50 + rainfall * 3 + rng.normal(0, 5, days), 10, None), 1),
    })


def predict_risk(
    current_pollution: float,
    historical_df: pd.DataFrame,
    forecast_days: int = 14,
) -> pd.DataFrame:
    """
    Simple auto-regressive forecast with rainfall influence.
    Returns a DataFrame with daily risk predictions.
    """
    rng = np.random.default_rng(seed=42)

    # Last 7-day trend
    last7     = historical_df["pollution_idx"].values[-7:]
    trend     = float(np.polyfit(range(7), last7, 1)[0])
    last_rain = historical_df["rainfall_mm"].values[-5:]
    rain_mean = float(np.mean(last_rain))

    rows = []
    poll = current_pollution

    for day in range(1, forecast_days + 1):
        # Simulated forecast rainfall (decaying + noise)
        fcst_rain = max(0, rain_mean * np.exp(-day / 10) + rng.normal(0, 3))

        # Pollution model: autoregression + rainfall influence + mean-reversion
        poll += trend * 0.7 + fcst_rain * 0.004 - (poll - 0.35) * 0.03
        poll  = float(np.clip(poll + rng.normal(0, 0.02), 0.05, 0.98))

        risk_label  = _poll_to_risk(poll)
        confidence  = float(np.clip(0.95 - day * 0.02, 0.5, 0.95))

        rows.append({
            "day":            day,
            "date":           datetime.today() + timedelta(days=day),
            "predicted_poll": round(poll, 3),
            "rainfall_mm":    round(fcst_rain, 1),
            "risk":           risk_label,
            "confidence_pct": round(confidence * 100, 1),
            "colour":         RISK_COLOURS[risk_label],
        })

    return pd.DataFrame(rows)


def _poll_to_risk(p: float) -> str:
    if p >= 0.65:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


def risk_summary(forecast_df: pd.DataFrame) -> dict:
    counts = forecast_df["risk"].value_counts().to_dict()
    peak_day   = forecast_df.loc[forecast_df["predicted_poll"].idxmax()]
    worst_risk = forecast_df["risk"].iloc[0]  # day 1

    high_days = counts.get("High", 0)
    if high_days >= 5:
        headline = f"⚠️ ELEVATED ALERT — {high_days} high-risk days ahead"
    elif high_days >= 1:
        headline = f"⚡ MODERATE CONCERN — {high_days} high-risk day(s) forecast"
    else:
        headline = "✅ STABLE OUTLOOK — Low risk in coming days"

    return {
        "headline":    headline,
        "high_days":   high_days,
        "medium_days": counts.get("Medium", 0),
        "low_days":    counts.get("Low", 0),
        "peak_day":    int(peak_day["day"]),
        "peak_poll":   float(peak_day["predicted_poll"]),
        "worst_risk":  worst_risk,
    }
