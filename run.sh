#!/usr/bin/env bash
# Quick start launcher for Predictive SDG Dashboard
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "Starting High-Performance SDG Predictive Monitoring Dashboard on http://localhost:8000..."
python server.py
