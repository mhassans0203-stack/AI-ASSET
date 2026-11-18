"""
Launch script to start both FastAPI Backend and Next.js Frontend concurrently.
"""

import os
import sys
import time
import subprocess
import signal

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "frontend")

    print("=" * 70)
    print("🚀 STARTING LOAN PREDICTION FULL-STACK APPLICATION")
    print("=" * 70)

    # 1. Start FastAPI backend
    print("\n[1/2] Starting FastAPI backend on http://localhost:8000 ...")
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 2. Start Next.js frontend
    print("[2/2] Starting Next.js frontend on http://localhost:3000 ...")
    frontend_cmd = ["cmd.exe", "/c", "npm.cmd", "run", "start"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    print("\n" + "=" * 70)
    print("✨ APPLICATION SERVERS RUNNING:")
    print("   • Next.js Web App:      http://localhost:3000")
    print("   • FastAPI Backend API:  http://localhost:8000")
    print("   • Interactive API Docs: http://localhost:8000/docs")
    print("=" * 70)
    print("Press Ctrl+C to stop both servers gracefully.\n")

    def signal_handler(sig, frame):
        print("\nShutting down servers...")
        try:
            backend_proc.terminate()
            frontend_proc.terminate()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print(f"Backend process terminated unexpectedly (code {backend_proc.returncode}).")
                break
            if frontend_proc.poll() is not None:
                print(f"Frontend process terminated unexpectedly (code {frontend_proc.returncode}).")
                break
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()

