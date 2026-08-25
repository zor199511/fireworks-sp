#!/usr/bin/env bash
# fireworks-sp daily pipeline (cron: 17:30 Mon-Fri)
set -u
cd "$HOME/fireworks-sp"
.venv/bin/python scripts/daily_pipeline.py >> logs/cron.log 2>&1
