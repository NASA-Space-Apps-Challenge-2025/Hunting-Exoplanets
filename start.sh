#!/bin/bash
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
