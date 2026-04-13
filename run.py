"""
run.py — Launch the autolit web UI.

Usage (from project root):
    python run.py

Opens http://localhost:8000 in your browser.
Requires Ollama to be running: ollama serve
"""
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # keep relative paths (data/, outputs/) working


def _open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    import uvicorn

    print("\n  autolit")
    print("  ───────────────────────────────")
    print("  UI:     http://localhost:8000")
    print("  Make sure Ollama is running.")
    print("  Press Ctrl+C to stop.\n")

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        "src.autolit.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
