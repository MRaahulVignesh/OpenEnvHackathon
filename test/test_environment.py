"""
Integration test for APEX Environment Server.
Tests all HTTP endpoints using the APEXClient.

Prerequisites:
- Server must be running (e.g., `python -m apex_env.server.main`)
- Server must be configured with a judge (real or mock)

Usage:
    python test/test_environment.py [base_url]

Default base_url: http://localhost:8000
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from apex_env.client import APEXClient


def test_server_endpoints(base_url: str = "http://localhost:8000"):
    """Test all APEX environment server endpoints."""

    print("=" * 70)
    print("APEX Environment Server Integration Test")
    print("=" * 70)
    print(f"Testing server at: {base_url}")
    print()

    try:
        with APEXClient(base_url) as client:

            # Test 1: Reset endpoint
            print("1. Testing reset() endpoint...")
            obs = client.reset()
            print(f"   ✓ Server responded")
            print(f"   - Scenario ID: {obs.get('scenario_id')}")
            print(f"   - Category: {obs.get('category')}")
            print(f"   - World: {obs.get('world')}")
            print(f"   - Prompt length: {len(obs.get('prompt', ''))} chars")
            assert obs.get('scenario_id'), "Missing scenario_id"
            assert obs.get('category') in ["banking", "consulting", "law"], f"Invalid category: {obs.get('category')}"
            assert obs.get('prompt'), "Missing prompt"
            print()

            # Test 2: Step endpoint
            print("2. Testing step() endpoint...")
            test_response = """Based on the workspace files provided, here is my professional analysis:

The documents indicate several key findings that require attention. After careful review
of all materials, I recommend the following actions to address the identified issues."""

            result = client.step(test_response)
            print(f"   ✓ Server responded")
            print(f"   - Reward: {result.reward}")
            print(f"   - Done: {result.done}")
            print(f"   - Criteria: {result.observation.get('criteria_met')}/{result.observation.get('criteria_total')}")
            print(f"   - Reasoning: {result.observation.get('reasoning', '')[:60]}...")
            assert result.reward is not None, "Missing reward"
            assert 0.0 <= result.reward <= 1.0, f"Reward out of range: {result.reward}"
            assert result.done is not None, "Missing done flag"
            assert result.observation.get('criteria_met') is not None, "Missing criteria_met"
            assert result.observation.get('criteria_total') is not None, "Missing criteria_total"
            print()

            # Test 3: Reset again (new episode)
            print("3. Testing reset() for new episode...")
            obs2 = client.reset()
            print(f"   ✓ Server responded")
            print(f"   - New scenario: {obs2.get('scenario_id')}")
            print(f"   - Category: {obs2.get('category')}")
            assert obs2.get('scenario_id'), "Missing scenario_id"
            print()

            # Test 4: Multiple step-reset cycles
            print("4. Testing multiple episodes...")
            for i in range(3):
                obs = client.reset()
                result = client.step(f"Test response {i+1}")
                print(f"   ✓ Episode {i+1}: {obs.get('scenario_id')} → reward={result.reward:.2f}")
            print()

            # Test 5: Verify different scenarios are served
            print("5. Testing scenario variety...")
            scenarios_seen = set()
            for _ in range(5):
                obs = client.reset()
                scenarios_seen.add(obs.get('scenario_id'))
            print(f"   ✓ Saw {len(scenarios_seen)} unique scenarios across 5 resets")
            assert len(scenarios_seen) > 1, "Server serving same scenario repeatedly"
            print()

            # Test 6: Verify all categories are available
            print("6. Testing category coverage...")
            categories_seen = set()
            for _ in range(10):
                obs = client.reset()
                categories_seen.add(obs.get('category'))
            print(f"   ✓ Categories seen: {sorted(categories_seen)}")
            assert len(categories_seen) >= 2, "Not enough category variety"
            print()

    except ConnectionError as e:
        print(f"\n✗ Connection failed: {e}")
        print("\nMake sure the server is running:")
        print("  python -m apex_env.server.main")
        return False
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 70)
    print("All tests passed! ✓")
    print("Server is working correctly.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = test_server_endpoints(base_url)
    sys.exit(0 if success else 1)
