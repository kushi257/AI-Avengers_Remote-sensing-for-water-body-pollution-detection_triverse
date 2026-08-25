"""
Sample data generator — run standalone to verify all modules work correctly.
Usage: python demo_data.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from modules.pollution_detector     import detect_pollution, WATER_BODIES
from modules.health_score           import compute_health_score, get_wqi_breakdown
from modules.predictor              import generate_historical_data, predict_risk, risk_summary
from modules.spread_simulator       import run_diffusion, compute_spread_stats
from modules.alerts_recommendations import get_recommendations, send_alert


def run_demo():
    location = "Yamuna River (Delhi)"
    print(f"\n{'='*60}")
    print(f"  AquaIntel — Module Verification Demo")
    print(f"  Target: {location}")
    print(f"{'='*60}\n")

    # 1. Detection
    print("1️⃣  Running pollution detection…")
    det = detect_pollution(location)
    print(f"   Pollution Level : {det['pollution_level']:.3f}")
    print(f"   Source Type     : {det['source_type']}")
    print(f"   Source Scores   : {det['source_scores']}")
    print(f"   Band Analysis   : {det['band_analysis']}\n")

    # 2. Health Score
    print("2️⃣  Computing health score…")
    health = compute_health_score(det["pollution_level"], det["band_analysis"])
    print(f"   Score  : {health['score']}")
    print(f"   Label  : {health['label']}")
    print(f"   Sub-scores: {health['sub_scores']}\n")

    # 3. WQI
    print("3️⃣  Water Quality Index…")
    wqi = get_wqi_breakdown(det["pollution_level"])
    for k, v in wqi.items():
        print(f"   {k:<20} {v}")
    print()

    # 4. Prediction
    print("4️⃣  Running 14-day forecast…")
    hist   = generate_historical_data(location)
    fcst   = predict_risk(det["pollution_level"], hist)
    rs     = risk_summary(fcst)
    print(f"   Headline  : {rs['headline']}")
    print(f"   High days : {rs['high_days']}")
    print(f"   Peak day  : Day {rs['peak_day']} ({rs['peak_poll']:.3f})\n")

    # 5. Spread Simulation
    print("5️⃣  Simulating pollution spread (20 steps)…")
    frames = run_diffusion(steps=20, pollution_level=det["pollution_level"],
                           source_type=det["source_type"])
    stats  = compute_spread_stats(frames)
    print(f"   Initial area : {stats['initial_area_pct']}%")
    print(f"   Final area   : {stats['final_area_pct']}%")
    print(f"   Growth factor: {stats['growth_factor']}×")
    print(f"   Affected km² : {stats['affected_km2']}\n")

    # 6. Recommendations
    print("6️⃣  Action recommendations…")
    recs = get_recommendations(det["source_type"], health["score"])
    print(f"   Urgency       : {recs['urgency']}")
    print(f"   Est. recovery : {recs['est_recovery_days']} days")
    print(f"   Immediate[0]  : {recs['immediate'][0]}\n")

    # 7. Alert
    print("7️⃣  Sending alert…")
    alert = send_alert(
        location=location,
        pollution_level=det["pollution_level"],
        source_type=det["source_type"],
        health_score=health["score"],
        risk_level=rs["worst_risk"],
    )
    print(f"   Alert ID : {alert['alert_id']}")
    print(f"   Status   : {alert['status']}\n")

    print("✅ All modules verified successfully!\n")
    print("▶️  Now run:  streamlit run app.py\n")


if __name__ == "__main__":
    run_demo()
