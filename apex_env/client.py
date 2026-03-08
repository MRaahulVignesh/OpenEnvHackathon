"""
Client for APEX Environment 
"""
import os
from dotenv import load_dotenv
from openenv.core.generic_client import GenericEnvClient
load_dotenv()
class APEXClient:
    def __init__(self, base_url: str):
        if base_url is None:
            base_url = f"{os.environ.get("ENV_HOST", "http://localhost")}:{os.environ.get("ENV_PORT", "8000")}"
        self._sync_client = GenericEnvClient(base_url=base_url)

    def __enter__(self):
        self._sync_client.connect()
        return self

    def __exit__(self, *args):
        self._sync_client.close()

    def reset(self, seed=None, episode_id=None, scenario_id=None) -> dict:
        result = self._sync_client.reset(seed=seed, episode_id=episode_id, scenario_id=scenario_id)
        return result.observation

    def step(self, response: str):
        result = self._sync_client.step({"response": response})
        return result

    def close(self):
        self._sync_client.close()
