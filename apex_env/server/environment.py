"""
Reinforcement Learning Environment using OpenEnv standard


Features:
  - Difficulty progression: tracks rolling avg reward per category (last 5 episodes)
    and escalates/de-escalates scenario difficulty accordingly
  - Adversarial noise injection: on hard scenarios, corrupts one file with a
    plausible but incorrect figure; rewards agent for detecting it
  - Returns reward breakdown: base rubric score + noise detection bonus
  - Peer comparison reward: agent scored relative to gold output (cached per scenario)
  - Citation reward: bonus for grounding response in workspace file names and figures
"""

import json
import os
import random
import copy
from collections import deque
from pathlib import Path
from typing import Optional
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State, EnvironmentMetadata

from apex_env.server.scorer import RLScorer
from apex_env.server.citation_reward import compute_citation_bonus
from apex_env.server.models import APEXAction, APEXObservation


# ── Difficulty thresholds ──────────────────────────────────────────────────────
# If rolling avg reward (last 5 episodes) is above ESCALATE → move up a tier
# If below DEESCALATE → move down a tier
ESCALATE_THRESHOLD   = 0.75
DEESCALATE_THRESHOLD = 0.40
TIER_ORDER           = ["easy", "medium", "hard"]


# ── Noise injection definitions ────────────────────────────────────────────────
# For each hard scenario: which file to corrupt, what to replace, what to inject,
# and keywords the agent should use to flag the discrepancy.
NOISE_INJECTIONS = {
    # Banking
    "bank_003": {
        "file":          "loan_application.txt",
        "original":      "EBITDA: $4.1M (FY2025)",
        "injected":      "EBITDA: $8.2M (FY2025)",
        "flag_keywords": ["ebitda discrepancy", "ebitda inconsistency", "ebitda mismatch",
                          "conflicting ebitda", "ebitda does not match", "ebitda figure"],
    },
    "bank_004": {
        "file":          "company_financials.txt" if "company_financials.txt" in [] else "company_overview.txt",
        "original":      "Revenue",
        "injected":      None,  
        "flag_keywords": ["revenue discrepancy", "revenue inconsistency", "conflicting revenue"],
    },
    "bank_006": {
        "file":          "debt_structure.txt",
        "original":      "Senior",
        "injected":      None,
        "flag_keywords": ["debt discrepancy", "debt mismatch", "conflicting debt figures"],
    },
    "bank_007": {
        "file":          "income_statement.txt",
        "original":      "Net Income",
        "injected":      None,
        "flag_keywords": ["net income discrepancy", "income mismatch", "conflicting income"],
    },
    # Consulting
    "consult_003": {
        "file":          "hospital_data.txt",
        "original":      "Occupancy rate: 94%",
        "injected":      "Occupancy rate: 71%",
        "flag_keywords": ["occupancy discrepancy", "occupancy mismatch", "conflicting occupancy",
                          "occupancy rate inconsistency", "occupancy figure"],
    },
    "consult_005": {
        "file":          "operations_email.txt",
        "original":      "delay",
        "injected":      None,
        "flag_keywords": ["data discrepancy", "conflicting figures", "inconsistency"],
    },
    "consult_007": {
        "file":          "pricing_analysis.txt",
        "original":      "margin",
        "injected":      None,
        "flag_keywords": ["margin discrepancy", "pricing inconsistency", "conflicting margin"],
    },
    # Law
    "law_001": {
        "file":          "purchase_agreement_excerpt.txt",
        "original":      "Section 8.3",
        "injected":      None,
        "flag_keywords": ["indemnification discrepancy", "cap mismatch", "conflicting indemnification"],
    },
    "law_002": {
        "file":          "it_logs.txt",
        "original":      "access",
        "injected":      None,
        "flag_keywords": ["log discrepancy", "access inconsistency", "conflicting log"],
    },
    "law_003": {
        "file":          "gdpr_reference.txt",
        "original":      "72 hours",
        "injected":      "48 hours",
        "flag_keywords": ["72 hours", "notification period", "gdpr discrepancy",
                          "breach notification", "conflicting gdpr", "48 hours is incorrect"],
    },
    "law_004": {
        "file":          "cap_table.txt",
        "original":      "%",
        "injected":      None,
        "flag_keywords": ["cap table discrepancy", "ownership mismatch", "conflicting ownership"],
    },
    "law_005": {
        "file":          "market_data.txt",
        "original":      "market share",
        "injected":      None,
        "flag_keywords": ["market share discrepancy", "market data inconsistency"],
    },
    "law_006": {
        "file":          "audit_findings.txt",
        "original":      "finding",
        "injected":      None,
        "flag_keywords": ["audit discrepancy", "conflicting audit", "finding inconsistency"],
    },
    "law_007": {
        "file":          "financial_exposure.txt",
        "original":      "$",
        "injected":      None,
        "flag_keywords": ["exposure discrepancy", "financial inconsistency", "conflicting exposure"],
    },
}


# ── Scenario helpers ───────────────────────────────────────────────────────────

def _load_scenarios(data_dir: str = "data") -> list[dict]:
    all_scenarios = []
    base = Path(data_dir)
    for category in ["banking", "consulting", "law"]:
        path = base / category / "scenarios.json"
        if path.exists():
            with open(path) as f:
                scenarios = json.load(f)
            for s in scenarios:
                s["category"] = category
            all_scenarios.extend(scenarios)
    if not all_scenarios:
        raise FileNotFoundError(f"No scenarios found in '{data_dir}'.")
    return all_scenarios


def _format_prompt(scenario: dict) -> str:
    workspace = "=== WORKSPACE FILES ===\n\n"
    for filename, content in scenario["files"].items():
        workspace += f"--- {filename} ---\n{content}\n\n"
    return (
        f"{workspace}\n"
        f"=== YOUR TASK ===\n"
        f"{scenario['task']}\n\n"
        f"Review all files carefully. "
        f"Produce a professional, complete response for the intended audience.\n"
    )


def _inject_noise(scenario: dict) -> tuple[dict, bool]:
    """
    Inject adversarial noise into a hard scenario's files.
    Returns (modified_scenario_copy, noise_was_injected).
    Only injects if a noise definition exists and the original text is found.
    """
    sid = scenario["id"]
    if sid not in NOISE_INJECTIONS:
        return scenario, False

    spec = NOISE_INJECTIONS[sid]
    target_file = spec["file"]
    original    = spec["original"]
    injected    = spec["injected"]

    # Need both original text and a replacement value
    if injected is None:
        return scenario, False

    if target_file not in scenario["files"]:
        return scenario, False

    file_content = scenario["files"][target_file]
    if original not in file_content:
        return scenario, False

    # Deep copy so we don't mutate the original scenario
    noisy = copy.deepcopy(scenario)
    noisy["files"][target_file] = file_content.replace(original, injected, 1)
    return noisy, True


def _detect_noise(response: str, scenario_id: str) -> bool:
    """Check if agent's response contains keywords flagging the injected error."""
    if scenario_id not in NOISE_INJECTIONS:
        return False
    keywords = NOISE_INJECTIONS[scenario_id]["flag_keywords"]
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in keywords)


# ── Environment ────────────────────────────────────────────────────────────────

class APEXEnvironment(Environment):
    """
    APEX Professional Tasks Environment with difficulty progression
    and adversarial noise injection.

    Difficulty progression:
        Tracks rolling avg reward (last 5 episodes) per category.
        - avg > 0.75 → escalate to next difficulty tier
        - avg < 0.40 → de-escalate to previous tier
        - otherwise  → stay at current tier

    Adversarial noise:
        On hard scenarios that have a noise definition, one file is
        corrupted with a plausible but incorrect figure.
        Agent gets +0.2 bonus reward for detecting and flagging it.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        scorer:          RLScorer,
        data_dir:        Optional[str] = None,
        category_filter: Optional[str] = None,
        shuffle:         bool          = True,
    ):
        super().__init__()

        data_dir        = data_dir or os.getenv("DATA_DIR", "apex_env/data")
        category_filter = category_filter or os.getenv("CATEGORY_FILTER") or None

        all_scenarios = _load_scenarios(data_dir)
        if category_filter:
            all_scenarios = [s for s in all_scenarios if s["category"] == category_filter]
            if not all_scenarios:
                raise ValueError(f"No scenarios for category: {category_filter}")

        self.all_scenarios   = all_scenarios
        self.category_filter = category_filter
        self.shuffle         = shuffle
        self.scorer          = scorer

        # ── Difficulty progression state ──────────────────────────────────────
        # current_tier: per-category difficulty tier ("easy" | "medium" | "hard")
        # reward_history: last 5 rewards per category for rolling avg
        self.current_tier: dict[str, str] = {
            "banking":    "easy",
            "consulting": "easy",
            "law":        "easy",
        }
        self.reward_history: dict[str, deque] = {
            "banking":    deque(maxlen=5),
            "consulting": deque(maxlen=5),
            "law":        deque(maxlen=5),
        }

        # Scenarios grouped by category + difficulty for fast lookup
        self._by_tier: dict[str, dict[str, list]] = {}
        for cat in ["banking", "consulting", "law"]:
            self._by_tier[cat] = {"easy": [], "medium": [], "hard": []}
        for s in all_scenarios:
            cat  = s["category"]
            diff = s.get("difficulty", "medium")
            if diff in self._by_tier.get(cat, {}):
                self._by_tier[cat][diff].append(s)

        # Episode state
        self._state          = State(episode_id=str(uuid4()), step_count=0)
        self._current        = None
        self._noise_injected = False
        self._scenarios_seen = 0

    @property
    def state(self) -> State:
        return self._state

    def _reset_rubric(self):
        pass

    def _apply_transform(self, obs: APEXObservation) -> APEXObservation:
        return obs

    def _update_tier(self, category: str, reward: float):
        """Update rolling avg and escalate/de-escalate tier if needed."""
        history = self.reward_history[category]
        history.append(reward)

        if len(history) < 3:
            return  # not enough data yet

        avg          = sum(history) / len(history)
        current      = self.current_tier[category]
        current_idx  = TIER_ORDER.index(current)

        if avg >= ESCALATE_THRESHOLD and current_idx < len(TIER_ORDER) - 1:
            new_tier = TIER_ORDER[current_idx + 1]
            self.current_tier[category] = new_tier
            print(f"  ↑  DIFFICULTY UP   [{category}] {current} → {new_tier}  (avg={avg:.2f})")

        elif avg <= DEESCALATE_THRESHOLD and current_idx > 0:
            new_tier = TIER_ORDER[current_idx - 1]
            self.current_tier[category] = new_tier
            print(f"  ↓  DIFFICULTY DOWN [{category}] {current} → {new_tier}  (avg={avg:.2f})")

    def _pick_scenario(self, category: Optional[str] = None, scenario_id: Optional[str] = None) -> dict:
        """Pick next scenario based on difficulty tier, or by explicit scenario_id."""
        if scenario_id is not None:
            matched = next((s for s in self.all_scenarios if s["id"] == scenario_id), None)
            if matched is None:
                raise ValueError(f"scenario_id '{scenario_id}' not found")
            return matched

        # Pick category if not specified
        if category is None:
            cats = list(self._by_tier.keys())
            category = random.choice(cats)

        tier       = self.current_tier.get(category, "easy")
        candidates = self._by_tier[category][tier]

        # Fallback: if no scenarios at this tier, try adjacent tiers
        if not candidates:
            for fallback in TIER_ORDER:
                candidates = self._by_tier[category][fallback]
                if candidates:
                    break

        if not candidates:
            # Last resort: any scenario in this category
            candidates = [s for s in self.all_scenarios if s["category"] == category]

        return random.choice(candidates)

    def reset(
        self,
        seed:        Optional[int] = None,
        episode_id:  Optional[str] = None,
        scenario_id: Optional[str] = None,
        **kwargs,
    ) -> APEXObservation:
        print(f"\n  ══ RESET ════════════════════════════════════")
        if seed is not None:
            random.seed(seed)

        scenario = self._pick_scenario(scenario_id=scenario_id)

        # Inject noise on hard scenarios
        if scenario.get("difficulty") == "hard":
            scenario, self._noise_injected = _inject_noise(scenario)
            if self._noise_injected:
                sid_ = scenario["id"]
                spec_ = NOISE_INJECTIONS.get(sid_, {})
                print(f"  💉 Noise injected  : {spec_.get('file', '?')}  ({spec_.get('original', '?')!r} → {spec_.get('injected', '?')!r})")
        else:
            self._noise_injected = False

        self._current = scenario
        self._state   = State(
            episode_id = episode_id or str(uuid4()),
            step_count = 0
        )
        self._reset_rubric()

        obs = APEXObservation(
            scenario_id      = self._current["id"],
            category         = self._current["category"],
            world            = self._current["world"],
            prompt           = _format_prompt(self._current),
            step             = 0,
            done             = False,
            reward           = None,
            difficulty       = self._current.get("difficulty", "medium"),
            noise_injected   = self._noise_injected,
            tier_status      = {cat: self.current_tier[cat] for cat in self.current_tier},
        )
        noise_label = "injected" if self._noise_injected else "clean"
        tiers_str = " | ".join(f"{k}={v}" for k,v in self.current_tier.items())
        print(f"  scenario : {self._current['id']} ({self._current.get('difficulty', '?')})")
        print(f"  category : {self._current['category']}")
        print(f"  noise    : {noise_label}")
        print(f"  tiers    : {tiers_str}")
        return obs

    def step(
        self,
        action:    APEXAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> APEXObservation:
        print(f"\n  ── STEP ───────────────────────────────────────")
        if self._current is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count  += 1
        self._scenarios_seen    += 1
        category = self._current["category"]

        # ── Base rubric score ─────────────────────────────────────────────────
        scored = self.scorer.score(self._current, action.response)
        base_reward = scored["reward"]

        # ── Peer comparison vs gold output ────────────────────────────────────
        # gold_score is cached after first call — zero extra cost on repeat episodes
        gold_scored  = self.scorer.score_gold(self._current)
        gold_score   = gold_scored["reward"]
        # delta in [-1, 1] → shaped to [0, 1]: 0.5 = matches gold, >0.5 = beats gold
        delta        = base_reward - gold_score
        peer_reward  = round(0.5 + (delta / 2.0), 4)
        # Blend: 60% peer + 40% absolute rubric
        blended_reward = round(0.6 * peer_reward + 0.4 * base_reward, 4)
        print(f"  base_reward : {base_reward:.4f}  (rubric {scored['criteria_met']}/{scored['criteria_total']})")
        print(f"  gold_score  : {gold_score:.4f}  (reference)")
        print(f"  peer_reward : {peer_reward:.4f}  (delta={base_reward - gold_score:+.2f})")
        print(f"  blended     : {blended_reward:.4f}  (0.6×peer + 0.4×base)")

        # ── Noise detection bonus ─────────────────────────────────────────────
        noise_bonus = 0.0
        if self._noise_injected:
            detected = _detect_noise(action.response, self._current["id"])
            if detected:
                noise_bonus = 0.2
                print(f"  noise       : detected ✓  +{noise_bonus} bonus")
            else:
                print(f"  noise       : missed  ✗  no bonus")

        # ── Citation reward ───────────────────────────────────────────────────
        citation      = compute_citation_bonus(action.response, self._current)
        citation_bonus = citation.bonus
        if citation_bonus > 0:
            print(f"  citation    : +{citation_bonus:.4f}  files={citation.file_citations}  figures={citation.figure_citations[:3]}")

        # ── Final reward ──────────────────────────────────────────────────────
        final_reward = min(1.0, blended_reward + noise_bonus + citation_bonus)

        print(f"  ─────────────────────────────────────────────")
        print(f"  FINAL REWARD : {final_reward:.4f}")
        print(f"  ═════════════════════════════════════════════")

        # ── Update difficulty tier ────────────────────────────────────────────
        self._update_tier(category, final_reward)

        obs = APEXObservation(
            scenario_id      = self._current["id"],
            category         = category,
            world            = self._current["world"],
            prompt           = _format_prompt(self._current),
            step             = self._state.step_count,
            done             = True,
            reward           = final_reward,
            criteria_scores  = scored["criteria_scores"],
            criteria_met     = scored["criteria_met"],
            criteria_total   = scored["criteria_total"],
            reasoning        = scored["reasoning"],
            difficulty       = self._current.get("difficulty", "medium"),
            noise_injected   = self._noise_injected,
            noise_detected   = (noise_bonus > 0),
            base_reward      = base_reward,
            noise_bonus      = noise_bonus,
            gold_score       = gold_score,
            peer_reward      = peer_reward,
            blended_reward   = blended_reward,
            citation_bonus   = citation_bonus,
            file_citations   = citation.file_citations,
            figure_citations = citation.figure_citations,
            tier_status      = {cat: self.current_tier[cat] for cat in self.current_tier},
            metadata         = {
                "difficulty": self._current["difficulty"],
                "rubric":     self._current["rubric"],
            }
        )
        return self._apply_transform(obs)

    def get_metadata(self):
        return EnvironmentMetadata(
            name        = "APEX Professional Tasks",
            description = (
                "RL training environment for investment banking, management consulting, "
                "and corporate law tasks. Features difficulty progression and adversarial "
                "noise injection. Targets the Mercor APEX-Agents benchmark."
            ),
            version = "2.0.0",
            author  = "OpenENV Hackathon",
        )