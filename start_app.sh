#!/usr/bin/env bash
# TrialMatch-Lite launcher for Linux/macOS
cd "$(dirname "$0")"
( sleep 6; (xdg-open http://127.0.0.1:8000 || open http://127.0.0.1:8000) >/dev/null 2>&1 ) &
python -m app.server
