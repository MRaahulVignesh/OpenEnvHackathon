"""
Integration test for APEX Environment Server.
"""

import sys
from apex_env.client import APEXClient


def test_server(base_url: str = "http://localhost:8000") -> bool:
    print(f"Testing APEX server → {base_url}\n")

    try:
        with APEXClient(base_url) as client:

            # ── Reset endpoint ─────────────────────────────
            obs = client.reset()

            assert obs.get("scenario_id"), "Missing scenario_id"
            assert obs.get("category") in {"banking", "consulting", "law"}
            assert obs.get("prompt"), "Missing prompt"

            print(f"Reset OK  → scenario={obs['scenario_id']} category={obs['category']}")

            # ── Step endpoint ──────────────────────────────
            result = client.step(
                "Based on the provided files, here is my professional analysis."
            )

            assert result.reward is not None
            assert 0.0 <= result.reward <= 1.0
            assert result.done is not None

            obs_data = result.observation or {}

            assert obs_data.get("criteria_met") is not None
            assert obs_data.get("criteria_total") is not None

            print(f"Step OK   → reward={result.reward:.3f}")

            # ── Multiple episodes ─────────────────────────
            scenarios = set()
            categories = set()

            for i in range(10):
                obs = client.reset()
                result = client.step(f"Test response {i}")

                scenarios.add(obs.get("scenario_id"))
                categories.add(obs.get("category"))

            print(f"Episodes  → {len(scenarios)} unique scenarios")
            print(f"Categories→ {sorted(categories)}")

            assert len(scenarios) > 1, "Server repeating same scenario"
            assert len(categories) >= 2, "Insufficient category diversity"

    except ConnectionError as e:
        print(f"\nConnection failed: {e}")
        print("Start the server with:")
        print("  python -m apex_env.server.main")
        return False

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nAll tests passed ✓")
    return True


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    ok = test_server(url)
    sys.exit(0 if ok else 1)