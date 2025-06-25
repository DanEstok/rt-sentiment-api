#!/usr/bin/env python3
"""Simple startup script for the API."""

import os
import sys
import uvicorn

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

# Set environment variables
os.environ.setdefault("MODEL_PATH", "./test_models")
os.environ.setdefault("PYTHONPATH", ".")

if __name__ == "__main__":
    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        loop="asyncio"
    )