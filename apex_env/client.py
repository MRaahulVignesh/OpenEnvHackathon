"""
APEX Environment Client

Uses openenv's GenericEnvClient with .sync() wrapper for synchronous use.
This is what your GRPO training loop imports.

Usage:
    from client import APEXClient

    with APEXClient("http://localhost:8000") as client:
        obs    = client.reset()
        print(obs["prompt"])

        result = client.step("My professional analysis...")
        print(result.reward)                        # 0.0 – 1.0
        print(result.observation["criteria_met"])
        print(result.observation["reasoning"])
"""
from openenv.core.generic_client import GenericEnvClient
class APEXClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self._sync_client = GenericEnvClient(base_url=base_url)

    def __enter__(self):
        self._sync_client.connect()   # ← connect on the SYNC wrapper
        return self

    def __exit__(self, *args):
        self._sync_client.close()

    def reset(self, seed=None, episode_id=None, scenario_id=None) -> dict:  # ← ADD scenario_id
        result = self._sync_client.reset(seed=seed, episode_id=episode_id, scenario_id=scenario_id)  # ← PASS IT
        return result.observation

    def step(self, response: str):
        result = self._sync_client.step({"response": response})
        return result

    def close(self):
        self._sync_client.close()
