"""
FastAPI app for the APEX Environment.

Uses openenv.core's create_app() which auto-generates all endpoints:
  POST /reset      → start new episode
  POST /step       → submit response, get reward
  GET  /state      → episode metadata
  GET  /schema     → action/observation JSON schemas
  GET  /health     → liveness check
  WS   /ws         → WebSocket for persistent sessions
  GET  /web        → Gradio UI for manual testing (if enabled)

Run locally:
  cd server && uvicorn app:app --host 0.0.0.0 --port 8000

Or via Docker (see Dockerfile in project root).
"""
from openenv.core.env_server.http_server import create_app
from apex_env.models import APEXAction, APEXObservation
from apex_env.server.apex_environment import APEXEnvironment


app = create_app(
    APEXEnvironment,
    APEXAction,
    APEXObservation,
    env_name           = "apex",
    max_concurrent_envs = 4,   # allow parallel training workers
)