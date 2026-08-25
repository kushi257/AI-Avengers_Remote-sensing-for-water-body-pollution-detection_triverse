"""
Module 1 & 2: Satellite Image Simulation + Pollution Detection
Simulates Sentinel-2 band analysis for water quality assessment.
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import io
import base64


# ── Simulated water body library ──────────────────────────────────────────────
WATER_BODIES = {
    "Ganges River (Kanpur)":    {"lat": 26.46, "lon": 80.32, "base_pollution": 0.78},
    "Yamuna River (Delhi)":     {"lat": 28.63, "lon": 77.21, "base_pollution": 0.82},
    "Sabarmati River (Ahmedabad)": {"lat": 23.02, "lon": 72.57, "base_pollution": 0.61},
    "Cauvery River (Trichy)":   {"lat": 10.79, "lon": 78.70, "base_pollution": 0.38},
    "Chilika Lake (Odisha)":    {"lat": 19.72, "lon": 85.32, "base_pollution": 0.29},
    "Vembanad Lake (Kerala)":   {"lat": 9.60,  "lon": 76.38, "base_pollution": 0.44},
    "Dal Lake (Kashmir)":       {"lat": 34.08, "lon": 74.83, "base_pollution": 0.55},
    "Hussain Sagar (Hyderabad)":{"lat": 17.42, "lon": 78.47, "base_pollution": 0.69},
}


def generate_satellite_image(pollution_level: float, source_type: str, size=(400, 400)) -> np.ndarray:
    """
    Simulate a Sentinel-2 false-colour composite for a water body.
    Returns an RGB numpy array (H, W, 3).
    """
    rng = np.random.default_rng(seed=int(pollution_level * 1000))
    h, w = size

    # Base water colour (deep blue-green)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, 0] = 0.02 + rng.random((h, w)) * 0.03   # R
    base[:, :, 1] = 0.15 + rng.random((h, w)) * 0.08   # G
    base[:, :, 2] = 0.40 + rng.random((h, w)) * 0.10   # B

    # Pollution patches
    n_patches = int(pollution_level * 12) + 1
    for _ in range(n_patches):
        cx = rng.integers(50, w - 50)
        cy = rng.integers(50, h - 50)
        radius = rng.integers(20, 80)
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx)**2 + (yy - cy)**2) <= radius**2
        falloff = 1 - np.sqrt((xx - cx)**2 + (yy - cy)**2) / (radius + 1e-6)
        falloff = np.clip(falloff, 0, 1) * mask

        if source_type == "Industrial":
            base[:, :, 0] += falloff * pollution_level * 0.55  # red-brown
            base[:, :, 1] += falloff * pollution_level * 0.15
            base[:, :, 2] -= falloff * pollution_level * 0.20
        elif source_type == "Sewage":
            base[:, :, 0] += falloff * pollution_level * 0.30
            base[:, :, 1] += falloff * pollution_level * 0.35  # grey-green
            base[:, :, 2] -= falloff * pollution_level * 0.15
        elif source_type == "Agricultural":
            base[:, :, 0] -= falloff * pollution_level * 0.05
            base[:, :, 1] += falloff * pollution_level * 0.50  # algae green
            base[:, :, 2] -= falloff * pollution_level * 0.25

    # Land border (brown edges)
    border = 30
    land_mask = np.zeros((h, w), dtype=bool)
    land_mask[:border, :] = True
    land_mask[-border:, :] = True
    land_mask[:, :border] = True
    land_mask[:, -border:] = True
    base[land_mask, 0] = 0.35 + rng.random(land_mask.sum()) * 0.15
    base[land_mask, 1] = 0.28 + rng.random(land_mask.sum()) * 0.10
    base[land_mask, 2] = 0.12 + rng.random(land_mask.sum()) * 0.08

    img = np.clip(base * 255, 0, 255).astype(np.uint8)
    return img


def analyse_bands(img: np.ndarray) -> dict:
    """
    Mock band analysis mimicking Sentinel-2 indices.
    NDWI, Turbidity index, Chlorophyll proxy.
    """
    r = img[:, :, 0].astype(float) / 255
    g = img[:, :, 1].astype(float) / 255
    b = img[:, :, 2].astype(float) / 255

    # Normalised Difference Water Index proxy
    ndwi = (g - r) / (g + r + 1e-6)
    # Turbidity: high red channel in water → turbid
    turbidity = np.mean(r[b > 0.3]) if np.any(b > 0.3) else 0
    # Chlorophyll / algae proxy
    chlorophyll = np.mean(g[b > 0.2]) if np.any(b > 0.2) else 0

    return {
        "ndwi_mean":       float(np.mean(ndwi)),
        "turbidity":       float(turbidity),
        "chlorophyll":     float(chlorophyll),
        "water_pixel_pct": float(np.mean(b > 0.25) * 100),
    }


def detect_pollution(location_name: str, manual_override: float | None = None) -> dict:
    """
    Main detection function.  Returns a full analysis dict.
    """
    loc = WATER_BODIES.get(location_name, list(WATER_BODIES.values())[0])
    base = loc["base_pollution"]

    # Add seasonal / random jitter
    rng = np.random.default_rng()
    noise = rng.normal(0, 0.05)
    pollution_level = float(np.clip(
        manual_override if manual_override is not None else base + noise, 0.0, 1.0
    ))

    # Source classification (rule-based mock)
    source_scores = _classify_source(location_name, pollution_level)
    source_type   = max(source_scores, key=source_scores.get)

    img_array = generate_satellite_image(pollution_level, source_type)
    bands     = analyse_bands(img_array)

    return {
        "location":       location_name,
        "lat":            loc["lat"],
        "lon":            loc["lon"],
        "pollution_level": pollution_level,
        "source_type":    source_type,
        "source_scores":  source_scores,
        "band_analysis":  bands,
        "image_array":    img_array,
    }


def _classify_source(location: str, level: float) -> dict:
    """Simple rule-based source classifier (mock)."""
    industrial_cities  = {"Kanpur", "Delhi", "Ahmedabad", "Hyderabad"}
    agricultural_water = {"Cauvery", "Chilika", "Vembanad"}

    rng = np.random.default_rng(seed=hash(location) % 9999)
    base_i = base_s = base_a = 0.33

    for kw in industrial_cities:
        if kw.lower() in location.lower():
            base_i += 0.30
            base_s += 0.10
            break
    for kw in agricultural_water:
        if kw.lower() in location.lower():
            base_a += 0.35
            base_i -= 0.10
            break

    noise = rng.dirichlet([1, 1, 1]) * 0.10
    scores = {
        "Industrial":   float(np.clip(base_i + noise[0] + level * 0.1, 0, 1)),
        "Sewage":       float(np.clip(base_s + noise[1] + level * 0.05, 0, 1)),
        "Agricultural": float(np.clip(base_a + noise[2], 0, 1)),
    }
    total = sum(scores.values())
    return {k: round(v / total, 3) for k, v in scores.items()}
