from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from openai import OpenAI


def make_openai_client() -> OpenAI:
    """Create OpenAI client from OPENAI_API_KEY."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def extract_response_text(response: Any) -> str:
    """Extract text from OpenAI Responses API output."""

    if getattr(response, "output_text", None):
        text = response.output_text.strip()
        if text:
            return text

    chunks: list[str] = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", "")
                if text:
                    chunks.append(text)

    text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError(f"Empty model response: {response}")
    return text


def call_json_response(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Call OpenAI Responses API and parse a JSON object response."""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_object"}},
    )

    text = extract_response_text(response)
    return json.loads(text)


def embed_texts(
    client: OpenAI,
    texts: list[str],
    model: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Embed a list of texts and return a float32 matrix."""

    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
        )
        for item in response.data:
            vectors.append(item.embedding)

    return np.asarray(vectors, dtype=np.float32)
