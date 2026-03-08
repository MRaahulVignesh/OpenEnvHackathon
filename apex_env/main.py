"""
Entry point for the server
"""
import os
import argparse
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from apex_env.server.app import app


ENV_SERVER_HOST = os.environ.get("ENV_HOST", "0.0.0.0")
ENV_SERVER_PORT = os.environ.get("ENV_URL", "8000")

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the OpenENV server.")
    parser.add_argument(
        "--host",
        type=str,
        default=ENV_SERVER_HOST,
        help=f"Host to bind the server (default: {ENV_SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=ENV_SERVER_PORT,
        help=f"Port to bind the server (default: {ENV_SERVER_HOST})",
    )
    return parser.parse_args()


def main() -> None:
    """Start the uvicorn server."""
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
