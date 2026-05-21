"""
run.py — RetailMind Product Intelligence Agent
Streamlit UI entrypoint. Run with: python run.py
"""

import subprocess, sys, os

# Launch Streamlit programmatically
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "false"],
        check=True,
    )
