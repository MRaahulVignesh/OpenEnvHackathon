"""
APEX Environment - Entry Point

Run the server:
  python main.py
  python main.py --port 8080
"""
import os
import uvicorn
import argparse
from dotenv import load_dotenv
load_dotenv()

from apex_env.server.app import app

def main(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args()
    main(port=args.port)
