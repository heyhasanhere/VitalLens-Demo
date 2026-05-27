#!/bin/bash
set -e

cd VitalLens

echo "Activating .venv"
python3 -m venv .venv
source .venv/bin/activate
echo ".venv active."


echo "Installing VitalLens app dependencies..."
pip3 install -q -r requirements.txt
echo "Dependencines Installation Done."


echo "Activating Backend Locally"
uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo "Backend Up."
