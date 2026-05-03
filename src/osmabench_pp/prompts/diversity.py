from __future__ import annotations

import numpy as np

from .schemas import PromptItem


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize rows of a matrix to unit length."""

    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for embedding vectors."""

    x_norm = l2_normalize(x)
    return x_norm @ x_norm.T


def remove_near_duplicates(
    items: list[PromptItem],
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[PromptItem], list[int]]:
    """Remove semantically close prompts using cosine similarity."""

    if len(items) != len(embeddings):
        raise ValueError("items and embeddings must have the same length.")

    sims = cosine_similarity_matrix(embeddings)
    keep_indices: list[int] = []
    removed: set[int] = set()

    for i in range(len(items)):
        if i in removed:
            continue

        keep_indices.append(i)

        for j in range(i + 1, len(items)):
            if sims[i, j] >= threshold:
                removed.add(j)

    kept_items = [items[i] for i in keep_indices]
    return kept_items, keep_indices


def farthest_first_selection(
    items: list[PromptItem],
    embeddings: np.ndarray,
    k: int,
    seed_idx: int = 0,
) -> list[PromptItem]:
    """Select a diverse subset using farthest-first traversal."""

    if len(items) != len(embeddings):
        raise ValueError("items and embeddings must have the same length.")

    if not items:
        return []

    if k >= len(items):
        return items[:]

    x = l2_normalize(embeddings)
    selected = [seed_idx]
    remaining = set(range(len(items))) - {seed_idx}

    while len(selected) < k:
        best_idx = None
        best_score = -1.0

        for idx in remaining:
            min_distance = min(1.0 - float(np.dot(x[idx], x[s])) for s in selected)
            if min_distance > best_score:
                best_score = min_distance
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [items[i] for i in selected]
