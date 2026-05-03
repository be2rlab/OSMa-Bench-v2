from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptItem:
    """One generated prompt with metadata needed for reproducibility."""

    prompt_id: str
    prompt: str
    scene_type: str
    room_hint: str
    scenario_hint: str


@dataclass
class PromptAnalysis:
    """Lightweight diagnostics for a generated prompt."""

    prompt_id: str
    prompt: str
    scene_type: str
    room_guess: str
    explicit_numbers: list[int]
    spatial_hits: list[str]
    has_vague_words: bool
    word_count: int
    object_hits: list[str]
    rejected_reasons: list[str]


@dataclass
class PromptRunSummary:
    """Summary saved after prompt generation."""

    scene_type: str
    raw_count: int
    dedup_count: int
    final_count: int
    rooms_in_final: dict[str, int]
    output_dir: str


JsonDict = dict[str, Any]
