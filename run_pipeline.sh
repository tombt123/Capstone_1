#!/usr/bin/env bash
# Re-create every model and output from the raw data (about 1-2 minutes).
set -e
cd "$(dirname "$0")"
python src/01_eda.py
python src/02_train_models.py
python src/03_log_analysis.py
python src/04_build_knowledge_base.py
echo "Done. Start the system with:  docker compose up --build   (or see README for the non-Docker way)"
