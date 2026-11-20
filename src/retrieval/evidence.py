from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator


SPACE_RE = re.compile(r"\s+")


class EvidenceError(ValueError):
    """Raised when an evidence source or generated record is invalid."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _traditional_to_simplified_converter():
    try:
        from opencc import OpenCC
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "opencc-python-reimplemented is required for Traditional-to-Simplified "
            "normalization. Install the project requirements."
        ) from exc
    return OpenCC("t2s")


def normalize_for_search(text: str, *, convert_traditional: bool = True) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    if convert_traditional:
        normalized = _traditional_to_simplified_converter().convert(normalized)
    return SPACE_RE.sub(" ", normalized).strip()


def iter_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, object]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise EvidenceError(f"Expected object in {path}:{line_number}")
                yield payload
