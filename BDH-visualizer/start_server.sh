#!/bin/bash
# Startup script for BDH Backend Server

echo "Starting BDH Backend Server..."
echo "Using conda environment: ml"
echo "Server will be available at: http://localhost:8000"
echo ""

cd "$(dirname "$0")"
conda run -n ml python -m uvicorn Backend.server:app --host 0.0.0.0 --port 8000 --reload
