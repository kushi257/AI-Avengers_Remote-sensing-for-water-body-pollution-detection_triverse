"""
Module 4: Water Health Score (0–100)
Computes a composite health score and categorical label.
"""

import numpy as np


# Thresholds
SCORE_LABELS = [
    (80, "🟢 Pristine",   "#00c896"),
    (60, "🟡 Clean",      "#7ecb4f"),
    (40, "🟠 Moderate",   "#f5a623"),
    (20, "🔴 Critical",   "#e8412e"),
    ( 0, "☠️  Toxic",     "#8b0000"),
]


def compute_health_score(
    pollution_level: float,
    band_analysis: dict,
    spread_area_pct: float = 0.0,
) -> dict:
    """
    Returns health_score (0–100), label, colour, and sub-scores.

    Parameters
    ----------
    pollution_level : 0–1 float from detector
    band_analysis   : dict with turbidity, chlorophyll, ndwi_mean, water_pixel_pct
    spread_area_pct : 0–100 percentage of water body affected
    """
    # Sub-component scores (all 0→100 = best)
    turbidity_score    = max(0, 100 - band_analysis.get("turbidity", 0.3) * 250)
    chlorophyll_score  = max(0, 100 - band_analysis.get("chlorophyll", 0.3) * 200)
    pollution_score    = max(0, 100 - pollution_level * 110)
    spread_score       = max(0, 100 - spread_area_pct * 1.5)
    ndwi               = band_analysis.get("ndwi_mean", 0)
    ndwi_score         = float(np.clip((ndwi + 0.5) * 80, 0, 100))

    weights = {
        "Pollution Intensity": (pollution_score,  0.35),
        "Turbidity":           (turbidity_score,  0.25),
        "Chlorophyll/Algae":   (chlorophyll_score,0.20),
        "Spread Impact":       (spread_score,     0.10),
        "Water Index (NDWI)":  (ndwi_score,       0.10),
    }

    composite = sum(v * w for v, w in weights.values())
    composite = float(np.clip(composite, 0, 100))

    label, colour = _get_label(composite)

    return {
        "score":      round(composite, 1),
        "label":      label,
        "colour":     colour,
        "sub_scores": {k: round(v, 1) for k, (v, _) in weights.items()},
        "weights":    {k: w for k, (_, w) in weights.items()},
    }


def _get_label(score: float):
    for threshold, label, colour in SCORE_LABELS:
        if score >= threshold:
            return label, colour
    return SCORE_LABELS[-1][1], SCORE_LABELS[-1][2]


def get_wqi_breakdown(pollution_level: float) -> dict:
    """
    Simulate individual WQI parameter readings.
    Returns mock WHO-style values for display.
    """
    rng = np.random.default_rng(seed=int(pollution_level * 100))
    p = pollution_level

    return {
        "pH":               round(float(rng.uniform(6.0 + p * 2.5, 7.5 + p * 2.0)), 2),
        "DO (mg/L)":        round(float(max(0.5, 8.5 - p * 7.0 + rng.normal(0, 0.3))), 2),
        "BOD (mg/L)":       round(float(0.5 + p * 25 + rng.exponential(1.0)), 2),
        "Turbidity (NTU)":  round(float(1.0  + p * 400 + rng.exponential(10)), 1),
        "Nitrates (mg/L)":  round(float(0.5  + p * 45  + rng.exponential(2)), 2),
        "Coliform (MPN)":   int(10 + p * 240000 + rng.exponential(500)),
        "Heavy Metals":     "Detected" if p > 0.5 else "Trace" if p > 0.25 else "None",
    }
