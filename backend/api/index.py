"""
Vercel serverless entry point for FastAPI app
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

# Vercel serverless function handler
app = app
