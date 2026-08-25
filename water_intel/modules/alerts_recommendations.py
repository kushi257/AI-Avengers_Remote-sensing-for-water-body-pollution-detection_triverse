"""
Modules 7 & 8: Alert System + Action Recommendation Engine
"""

from datetime import datetime
import uuid


# ── Action Recommendation Database ────────────────────────────────────────────
ACTION_MAP = {
    "Industrial": {
        "immediate": [
            "🚫 Issue emergency stop orders to identified discharge points",
            "🔬 Deploy rapid water testing kits at 5 downstream sampling stations",
            "📡 Activate real-time effluent monitoring sensors at factory outlets",
            "🚧 Block industrial drain channels pending inspection",
        ],
        "short_term": [
            "🏭 Conduct surprise factory inspections within 48 hours",
            "⚖️ Initiate legal proceedings under Water (Prevention & Control) Act",
            "💧 Deploy floating containment booms to limit spread",
            "🧹 Commission emergency bio-remediation teams",
        ],
        "long_term": [
            "📋 Mandate zero-liquid-discharge (ZLD) systems for all industries",
            "🛰️ Establish satellite-based continuous effluent monitoring",
            "🏗️ Build shared Common Effluent Treatment Plants (CETP)",
            "📜 Revoke non-compliant operating licenses",
        ],
    },
    "Sewage": {
        "immediate": [
            "🚰 Identify and seal broken sewer mains",
            "⚠️ Issue public health advisory for affected water bodies",
            "🚑 Deploy mobile water purification units for affected communities",
            "🔍 GPS-map all illegal sewage discharge points",
        ],
        "short_term": [
            "🔧 Emergency repair of aging pipeline infrastructure",
            "🏗️ Accelerate Sewage Treatment Plant (STP) capacity upgrade",
            "📊 Install flow meters on major sewer networks",
            "🌿 Deploy constructed wetlands as tertiary treatment buffer",
        ],
        "long_term": [
            "🏙️ City-wide sewerage network audit and replacement programme",
            "♻️ Implement decentralised wastewater reuse systems",
            "📱 Smart sewer sensors with automated overflow alerts",
            "💰 Community biogas plants using sewage sludge",
        ],
    },
    "Agricultural": {
        "immediate": [
            "⛔ Temporary moratorium on fertiliser application in buffer zones",
            "🌊 Install silt fences and sediment traps at field runoff points",
            "🔬 Test for pesticide and nitrate levels at intake points",
            "🚜 Alert farmers to halt irrigation from affected water bodies",
        ],
        "short_term": [
            "🌱 Establish 100m riparian buffer strips along waterways",
            "💧 Promote drip irrigation to reduce fertiliser runoff by 60%",
            "🧪 Soil testing and precision fertiliser application programme",
            "🦆 Introduce wetland filters between farmland and water bodies",
        ],
        "long_term": [
            "🌾 Transition to organic / regenerative farming practices",
            "📡 AI-powered variable-rate fertiliser dispensing systems",
            "🏛️ Reform agricultural subsidy structure to reward low-runoff practices",
            "🗺️ GIS-based watershed management planning",
        ],
    },
}

AUTHORITY_CONTACTS = {
    "Ganges River (Kanpur)":       ("UPPCB — Uttar Pradesh",    "+91-522-2239002"),
    "Yamuna River (Delhi)":        ("DPCC — Delhi",             "+91-11-22307945"),
    "Sabarmati River (Ahmedabad)": ("GPCB — Gujarat",           "+91-79-23232167"),
    "Cauvery River (Trichy)":      ("TNPCB — Tamil Nadu",       "+91-44-22353134"),
    "Chilika Lake (Odisha)":       ("OSPCB — Odisha",           "+91-674-2542801"),
    "Vembanad Lake (Kerala)":      ("KSPCB — Kerala",           "+91-471-2330230"),
    "Dal Lake (Kashmir)":          ("JKPCB — J&K",              "+91-194-2474986"),
    "Hussain Sagar (Hyderabad)":   ("TSPCB — Telangana",        "+91-40-23435676"),
}


def get_recommendations(source_type: str, health_score: float) -> dict:
    """Return tiered action recommendations."""
    actions = ACTION_MAP.get(source_type, ACTION_MAP["Industrial"])
    urgency = "CRITICAL" if health_score < 40 else "HIGH" if health_score < 65 else "MODERATE"

    return {
        "source_type": source_type,
        "urgency":     urgency,
        "immediate":   actions["immediate"],
        "short_term":  actions["short_term"],
        "long_term":   actions["long_term"],
        "est_recovery_days": _estimate_recovery(health_score, source_type),
    }


def _estimate_recovery(score: float, source: str) -> int:
    base = {
        "Industrial":   120,
        "Sewage":        60,
        "Agricultural":  45,
    }.get(source, 90)
    multiplier = max(0.5, (100 - score) / 60)
    return int(base * multiplier)


# ── Alert System ───────────────────────────────────────────────────────────────
ALERT_LOG: list[dict] = []


def send_alert(
    location: str,
    pollution_level: float,
    source_type: str,
    health_score: float,
    risk_level: str = "High",
) -> dict:
    """Simulate sending alert to authorities."""
    authority, contact = AUTHORITY_CONTACTS.get(
        location,
        ("State PCB", "1800-180-2088 (National)")
    )

    alert_id = str(uuid.uuid4())[:8].upper()
    severity  = "🔴 CRITICAL" if health_score < 40 else "🟠 HIGH" if health_score < 60 else "🟡 MODERATE"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alert = {
        "alert_id":     alert_id,
        "timestamp":    timestamp,
        "location":     location,
        "severity":     severity,
        "source_type":  source_type,
        "pollution_pct": round(pollution_level * 100, 1),
        "health_score": health_score,
        "risk_level":   risk_level,
        "authority":    authority,
        "contact":      contact,
        "channels":     ["SMS", "Email", "Dashboard Notification", "GIS Alert"],
        "status":       "DISPATCHED ✅",
        "message": (
            f"[AQUA-INTEL ALERT #{alert_id}] "
            f"{severity} water pollution detected at {location}. "
            f"Source: {source_type}. Health Score: {health_score}/100. "
            f"Immediate inspection required. — {timestamp}"
        )
    }

    ALERT_LOG.append(alert)

    # Console log (simulated dispatch)
    print(f"\n{'='*60}")
    print(f"  🚨 ALERT DISPATCHED — #{alert_id}")
    print(f"  Location : {location}")
    print(f"  Severity : {severity}")
    print(f"  Authority: {authority}  ({contact})")
    print(f"  Channels : {', '.join(alert['channels'])}")
    print(f"{'='*60}\n")

    return alert


def get_alert_log() -> list[dict]:
    return ALERT_LOG.copy()
