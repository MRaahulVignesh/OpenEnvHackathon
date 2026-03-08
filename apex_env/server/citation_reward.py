"""
Citation reward utilities for APEX environment.

A "citation" is when the agent explicitly grounds its response in the
workspace files — either by naming a source file or quoting a specific
figure that appears in the scenario's files.

Two signal types:
  1. File citation    — agent mentions a filename from the workspace
                        e.g. "per the earnings_model.txt..."
  2. Figure citation  — agent uses a specific number/figure that appears
                        in the workspace files
                        e.g. "EBITDA of $18M", "61% gross margin"

Scoring:
  citation_bonus = min(MAX_CITATION_BONUS, unique_citations * PER_CITATION_BONUS)

  MAX_CITATION_BONUS  = 0.15   (caps total bonus — can't game by spamming)
  PER_CITATION_BONUS  = 0.05   (each unique citation is worth 0.05)
  → max 3 citations needed to hit the cap

Why cap at 0.15?
  Combined reward budget: base (up to 1.0) + noise (0.2) + citation (0.15)
  Everything is clamped to 1.0 in final_reward, so the bonus only matters
  when base_reward leaves headroom.
"""

import re
from typing import NamedTuple


MAX_CITATION_BONUS = 0.15
PER_CITATION_BONUS = 0.05


class CitationResult(NamedTuple):
    bonus:           float
    file_citations:  list[str]   # filenames found in response
    figure_citations: list[str]  # figures found in response that exist in files
    total_unique:    int


def _extract_figures(text: str) -> set[str]:
    """
    Extract currency and percentage figures from text.
    Matches: $142M, €4.2B, £50K, 61%, 8.5%, $22M, 31%
    Normalises to lowercase for comparison.
    """
    pattern = r'[\$€£]\s*\d+[\d,.]*\s*[MBKmb]?|\d+[\d,.]*\s*%'
    matches = re.findall(pattern, text)
    # Normalise: remove spaces, lowercase
    return {re.sub(r'\s+', '', m).lower() for m in matches}


def compute_citation_bonus(response: str, scenario: dict) -> CitationResult:
    """
    Check how well the agent grounds its response in the workspace files.

    Args:
        response: Agent's text response
        scenario: The full scenario dict with 'files' key

    Returns:
        CitationResult with bonus and breakdown
    """
    response_lower = response.lower()
    files          = scenario.get("files", {})

    # ── 1. File name citations ─────────────────────────────────────────────
    file_citations = []
    for filename in files.keys():
        # Match the filename with or without .txt extension
        base = filename.replace(".txt", "").replace("_", " ")
        if filename.lower() in response_lower or base.lower() in response_lower:
            file_citations.append(filename)

    # ── 2. Figure citations ────────────────────────────────────────────────
    # Collect all figures that appear in the workspace files
    workspace_figures = set()
    for content in files.values():
        workspace_figures |= _extract_figures(content)

    # Check which workspace figures appear in the response
    response_figures   = _extract_figures(response)
    matched_figures    = list(workspace_figures & response_figures)

    # ── 3. Score ───────────────────────────────────────────────────────────
    total_unique = len(set(file_citations)) + len(matched_figures)
    bonus        = min(MAX_CITATION_BONUS, total_unique * PER_CITATION_BONUS)
    bonus        = round(bonus, 4)

    return CitationResult(
        bonus            = bonus,
        file_citations   = file_citations,
        figure_citations = matched_figures,
        total_unique     = total_unique,
    )