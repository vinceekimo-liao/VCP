#!/bin/bash
cd /opt/render/project/src
source .venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port $PORT
