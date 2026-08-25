#!/usr/bin/env python3
"""
AquaIntel Quick-Start Launcher
Checks dependencies and launches Streamlit.
"""
import subprocess
import sys
import os


def check_and_install():
    print("🔍 Checking dependencies…")
    try:
        import streamlit, numpy, pandas, plotly, PIL, sklearn, scipy, folium
        print("✅ All dependencies present.")
    except ImportError as e:
        print(f"📦 Missing: {e.name} — installing requirements…")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "-r", "requirements.txt", "-q"])
        print("✅ Installation complete.")


def main():
    check_and_install()
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    print("\n🌊 Launching AquaIntel Dashboard…")
    print("   Open → http://localhost:8501\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8501",
        "--theme.base", "dark",
        "--theme.primaryColor", "#4a90d9",
        "--theme.backgroundColor", "#0a0f1e",
        "--theme.secondaryBackgroundColor", "#0e1628",
        "--theme.textColor", "#c8d6e5",
    ])


if __name__ == "__main__":
    main()
