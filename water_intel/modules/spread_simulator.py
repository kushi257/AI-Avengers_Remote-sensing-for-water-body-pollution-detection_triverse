"""
Module 6: Pollution Spread Simulation
Implements a 2D diffusion model for visual spread simulation.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run_diffusion(
    grid_size: int = 80,
    steps: int = 20,
    pollution_level: float = 0.7,
    source_type: str = "Industrial",
    n_sources: int = 2,
) -> list[np.ndarray]:
    """
    Simulate 2-D diffusion of pollution plume.
    Returns list of grid snapshots (one per time step).
    """
    rng = np.random.default_rng(seed=42)
    grid = np.zeros((grid_size, grid_size), dtype=float)

    # Place pollution sources
    source_positions = []
    for _ in range(n_sources):
        sx = rng.integers(10, grid_size - 10)
        sy = rng.integers(10, grid_size - 10)
        source_positions.append((sx, sy))
        grid[sx, sy] = pollution_level

    # Diffusion parameters by source type
    params = {
        "Industrial":   {"D": 0.18, "decay": 0.008, "current_x": 1, "current_y": 0},
        "Sewage":       {"D": 0.14, "decay": 0.012, "current_x": 0, "current_y": 1},
        "Agricultural": {"D": 0.10, "decay": 0.006, "current_x": 1, "current_y": 1},
    }
    p = params.get(source_type, params["Industrial"])
    D = p["D"]
    decay = p["decay"]
    cx, cy = p["current_x"], p["current_y"]

    frames = [grid.copy()]

    for _ in range(steps):
        new_grid = grid.copy()
        # Diffusion (Laplacian)
        new_grid[1:-1, 1:-1] += D * (
            grid[2:,   1:-1] +
            grid[:-2,  1:-1] +
            grid[1:-1, 2:  ] +
            grid[1:-1, :-2 ] -
            4 * grid[1:-1, 1:-1]
        )
        # Advection (river current)
        shift = np.roll(new_grid, cx, axis=0)
        shift = np.roll(shift,    cy, axis=1)
        new_grid = new_grid * 0.85 + shift * 0.15

        # Decay
        new_grid *= (1 - decay)

        # Re-inject at sources (constant emission)
        for sx, sy in source_positions:
            new_grid[sx, sy] = min(1.0, new_grid[sx, sy] + pollution_level * 0.15)

        new_grid = np.clip(new_grid, 0, 1)
        grid = new_grid
        frames.append(grid.copy())

    return frames


def build_spread_animation(
    frames: list[np.ndarray],
    source_type: str = "Industrial",
    location: str = "",
) -> go.Figure:
    """Build an animated Plotly heatmap from diffusion frames."""

    COLORSCALES = {
        "Industrial":   [[0, "#0d1b2a"], [0.3, "#1b4f72"], [0.6, "#c0392b"], [1, "#ff6b35"]],
        "Sewage":       [[0, "#0d1b2a"], [0.3, "#1a5276"], [0.6, "#7d6608"], [1, "#f0e68c"]],
        "Agricultural": [[0, "#0d1b2a"], [0.3, "#1e8449"], [0.7, "#52be80"], [1, "#abebc6"]],
    }
    cscale = COLORSCALES.get(source_type, COLORSCALES["Industrial"])

    fig = go.Figure()

    # Base frame
    fig.add_trace(go.Heatmap(
        z=frames[0],
        colorscale=cscale,
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title="Pollution<br>Intensity",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["None", "Low", "Moderate", "High", "Critical"],
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
        ),
    ))

    # Animation frames
    anim_frames = []
    for i, f in enumerate(frames):
        anim_frames.append(go.Frame(
            data=[go.Heatmap(z=f, colorscale=cscale, zmin=0, zmax=1)],
            name=str(i),
            layout=go.Layout(
                title_text=f"Pollution Spread Simulation — Day {i} | {location}"
            )
        ))

    fig.frames = anim_frames

    fig.update_layout(
        title=dict(
            text=f"🌊 Pollution Spread Simulation — {source_type} Source | {location}",
            font=dict(size=16, color="white"),
        ),
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#0a0f1e",
        font=dict(color="white"),
        height=460,
        margin=dict(l=10, r=10, t=60, b=10),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.12, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶  Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 300, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="⏸  Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
            font=dict(color="white"),
            bgcolor="#1a2744",
            bordercolor="#4a90d9",
        )],
        sliders=[dict(
            steps=[dict(
                method="animate",
                args=[[str(i)], {"mode": "immediate",
                                  "frame": {"duration": 300, "redraw": True}}],
                label=f"D{i}",
            ) for i in range(len(frames))],
            currentvalue=dict(prefix="Day ", font=dict(color="white")),
            font=dict(color="white"),
            bgcolor="#1a2744",
            bordercolor="#4a90d9",
        )],
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    return fig


def compute_spread_stats(frames: list[np.ndarray]) -> dict:
    """Compute spread metrics from simulation frames."""
    threshold = 0.1
    initial_cells = np.sum(frames[0] > threshold)
    final_cells   = np.sum(frames[-1] > threshold)
    peak_intensity = float(np.max(frames[-1]))
    spread_pct = min(100.0, float(final_cells / (frames[0].shape[0] ** 2) * 100))
    growth_factor = (final_cells / max(1, initial_cells))

    return {
        "initial_area_pct": round(float(initial_cells / (frames[0].shape[0] ** 2) * 100), 1),
        "final_area_pct":   round(spread_pct, 1),
        "peak_intensity":   round(peak_intensity, 3),
        "growth_factor":    round(float(growth_factor), 1),
        "affected_km2":     round(spread_pct * 0.8, 1),   # mock scale
    }
