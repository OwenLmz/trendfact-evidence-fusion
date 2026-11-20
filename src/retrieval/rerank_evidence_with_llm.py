"""Response parsing and dtype helpers shared by the frozen pointwise scorer."""

import json
import re


ALLOWED_STANCES = {"support", "refute", "context", "unknown"}


CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


class RerankError(ValueError):
    """Raised when a model response violates the constrained reranking protocol."""


def extract_json_object(raw_text: str) -> dict[str, object]:
    cleaned = CODE_FENCE_RE.sub("", raw_text.strip())
    start = cleaned.find("{")
    if start < 0:
        raise RerankError("Model output contains no JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise RerankError(f"Invalid model JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RerankError("Model JSON must be an object")
    return payload


def _torch_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]
