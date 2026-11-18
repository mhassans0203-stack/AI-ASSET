"""
Root entrypoint for Vercel deployment.
Exposes FastAPI application instance 'app'.
"""
import os
import sys

# Ensure backend package can be imported from root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from backend.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
