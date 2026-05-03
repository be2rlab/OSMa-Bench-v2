from __future__ import annotations

import re

from .prompt_specs import PromptSpec
from .schemas import PromptAnalysis


NUMBER_RE = re.compile(r"\b\d+\b")
WORD_RE = re.compile(r"\b[\w'-]+\b")


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for exact deduplication before embedding-based dedup."""

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def word_count(text: str) -> int:
    """Count approximate words in a prompt."""

    return len(WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    """Count simple sentence boundaries."""

    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def extract_explicit_numbers(text: str) -> list[int]:
    """Extract Arabic numerals from prompt text."""

    return [int(x) for x in NUMBER_RE.findall(text)]


def contains_any(text: str, patterns: list[str]) -> bool:
    """Check whether any phrase from a list occurs in the text."""

    t = text.lower()
    return any(pattern in t for pattern in patterns)


def find_hits(text: str, patterns: list[str]) -> list[str]:
    """Return all phrase hits from a pattern list."""

    t = text.lower()
    return [pattern for pattern in patterns if pattern in t]


def detect_spatial_hits(text: str, spec: PromptSpec) -> list[str]:
    """Find spatial relation phrases relevant for the selected scene type."""

    return find_hits(text, spec.spatial_cues)


def contains_vague_words(text: str, spec: PromptSpec) -> bool:
    """Detect vague quantifiers that make prompt-grounded QA less reliable."""

    t = text.lower()
    t = t.replace("how many", "how_many")
    return any(vague in t for vague in spec.forbidden_vague)


def validate_prompt(text: str, spec: PromptSpec) -> list[str]:
    """Return rejection reasons. Empty list means the prompt passes the filter."""

    reasons: list[str] = []

    wc = word_count(text)
    numbers = extract_explicit_numbers(text)
    spatial_hits = detect_spatial_hits(text, spec)
    total_objects = sum(numbers)
    sent_count = sentence_count(text)
    lower = text.lower()

    if wc < spec.min_words:
        reasons.append("too_short")
    if wc > spec.max_words:
        reasons.append("too_long")
    if contains_vague_words(text, spec):
        reasons.append("vague_words")
    if len(numbers) < spec.min_explicit_counts:
        reasons.append("too_few_explicit_counts")
    if len(spatial_hits) < spec.min_spatial_cues:
        reasons.append("too_few_spatial_cues")
    if total_objects < spec.min_total_objects:
        reasons.append("too_few_total_objects")
    if spec.max_total_objects is not None and total_objects > spec.max_total_objects:
        reasons.append("too_many_total_objects")
    if sent_count < spec.min_sentences:
        reasons.append("too_few_sentences")
    if sent_count > spec.max_sentences:
        reasons.append("too_many_sentences")

    if "#" in text:
        reasons.append("contains_ids")
    if any(ch in text for ch in "{}[]"):
        reasons.append("contains_structural_markup")
    if contains_any(text, spec.forbidden_object_patterns):
        reasons.append("contains_forbidden_objects")
    if contains_any(text, spec.risky_patterns):
        reasons.append("contains_risky_language")
    if not any(pattern in lower for pattern in spec.strong_patterns):
        reasons.append("too_simple")

    object_hits = find_hits(text, spec.object_hints)
    if not object_hits:
        reasons.append("no_required_object_hits")

    return reasons


def guess_room(text: str, spec: PromptSpec) -> str:
    """Guess room type from the generated prompt."""

    lower = text.lower()
    for room in spec.room_hints:
        if room in lower:
            return room
    return "unknown"


def analyze_prompt(prompt_id: str, prompt: str, spec: PromptSpec, rejected_reasons: list[str] | None = None) -> PromptAnalysis:
    """Create diagnostic metadata for a prompt."""

    return PromptAnalysis(
        prompt_id=prompt_id,
        prompt=prompt,
        scene_type=spec.scene_type,
        room_guess=guess_room(prompt, spec),
        explicit_numbers=extract_explicit_numbers(prompt),
        spatial_hits=detect_spatial_hits(prompt, spec),
        has_vague_words=contains_vague_words(prompt, spec),
        word_count=word_count(prompt),
        object_hits=find_hits(prompt, spec.object_hints),
        rejected_reasons=rejected_reasons or [],
    )
