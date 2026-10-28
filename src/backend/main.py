"""Chạy backend + dashboard:  uvicorn src.backend.main:app --reload  → http://localhost:8000"""
from .app import create_app

app = create_app()
