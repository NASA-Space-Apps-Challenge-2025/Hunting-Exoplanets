"""
Vercel serverless entry point for FastAPI app
"""
from app import app

# Vercel serverless function handler
handler = app
