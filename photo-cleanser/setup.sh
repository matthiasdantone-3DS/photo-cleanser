#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements-dev.txt
pip uninstall -y opencv-python opencv-contrib-python || true
pip install --force-reinstall --no-deps "opencv-python-headless>=4.9,<4.12"
python models/download_models.py --allow-unverified
