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
from functools import partial
from openenv.core.env_server.http_server import create_app
from apex_env.models import APEXAction, APEXObservation
from apex_env.server.apex_environment import APEXEnvironment
from apex_env.server.scorer import RLScorer


# Initialize scorer once (loads vLLM model)
print("Initializing RLScorer...")
scorer = RLScorer()
print("✓ RLScorer initialized")

# Create environment factory with scorer injected
def create_apex_environment(**kwargs):
    """Factory function to create APEXEnvironment with scorer."""
    return APEXEnvironment(scorer=scorer, **kwargs)

app = create_app(
    create_apex_environment,
    APEXAction,
    APEXObservation,
    env_name           = "apex",
    max_concurrent_envs = 4,   # allow parallel training workers
)