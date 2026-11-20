from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zlib
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


NORMALIZATION_VERSION = "nfkc-t2s-casefold-whitespace-v1"
REQUIRED_EVIDENCE_FIELDS = {
    "evidence_id",
    "text",
    "search_text",
    "source_dataset",
    "source_split",
    "source_raw_id",
    "document_id",
    "title",
    "passage_index",
    "sentence_ids",
    "construction_method",
    "content_hash",
    "provenance_url",
    "license_status",
    "normalization_version",
    "label_visibility",
    "duplicate_cluster_id",
    "provenance_refs",
}
FORBIDDEN_INDEX_KEYS = {
    "label",
    "label_id",
    "label_name",
    "raw_label",
    "gold_evidence",
    "gold evidence",
    "isgold",
    "predicted_label",
    "verdict",
}

SPACE_RE = re.compile(r"\s+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])|[\r\n]+")
TOKEN_CHARACTER_RE = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")


class EvidenceError(ValueError):
    """Raised when an evidence source or generated record is invalid."""


@dataclass(frozen=True)
class Passage:
    text: str
    sentence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CFEVEREvidenceRef:
    page_id: str
    sentence_id: int


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


def compact_text(text: str, *, convert_traditional: bool = True) -> str:
    normalized = normalize_for_search(text, convert_traditional=convert_traditional)
    return "".join(character for character in normalized if TOKEN_CHARACTER_RE.fullmatch(character))


def _hard_split_sentence(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    step = max_chars - overlap_chars
    if step <= 0:
        raise ValueError("long_sentence_overlap must be smaller than max_chars")
    pieces = []
    start = 0
    while start < len(text):
        piece = text[start : start + max_chars].strip()
        if piece:
            pieces.append(piece)
        if start + max_chars >= len(text):
            break
        start += step
    return pieces


def split_sentences(
    text: str,
    *,
    max_chars: int = 180,
    long_sentence_overlap: int = 30,
    minimum_fragment_chars: int = 10,
) -> list[tuple[str, str]]:
    raw = SPACE_RE.sub(" ", str(text)).strip()
    if not raw:
        return []

    sentences: list[tuple[str, str]] = []
    source_index = 0
    for chunk in SENTENCE_BOUNDARY_RE.split(raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces = _hard_split_sentence(chunk, max_chars, long_sentence_overlap)
        for piece_index, piece in enumerate(pieces):
            if len(piece) < minimum_fragment_chars:
                continue
            sentence_id = str(source_index) if len(pieces) == 1 else f"{source_index}.{piece_index}"
            sentences.append((sentence_id, piece))
        source_index += 1
    return sentences


def pack_passages(
    sentences: Sequence[tuple[str, str]],
    *,
    min_chars: int = 20,
    max_chars: int = 180,
    max_sentences: int = 3,
    sentence_overlap: int = 1,
) -> list[Passage]:
    if min_chars <= 0 or max_chars < min_chars:
        raise ValueError("passage character bounds are invalid")
    if max_sentences <= 0 or not 0 <= sentence_overlap < max_sentences:
        raise ValueError("sentence_overlap must be in [0, max_sentences)")

    passages: list[Passage] = []
    start = 0
    while start < len(sentences):
        selected: list[tuple[str, str]] = []
        total_chars = 0
        cursor = start
        while cursor < len(sentences) and len(selected) < max_sentences:
            sentence_id, sentence = sentences[cursor]
            candidate_chars = total_chars + len(sentence)
            if selected and candidate_chars > max_chars:
                break
            selected.append((sentence_id, sentence))
            total_chars = candidate_chars
            cursor += 1

        text = "".join(sentence for _, sentence in selected).strip()
        if len(text) >= min_chars:
            passages.append(Passage(text, tuple(sentence_id for sentence_id, _ in selected)))
        elif passages and len(passages[-1].text) + len(text) <= max_chars:
            previous = passages.pop()
            passages.append(
                Passage(previous.text + text, previous.sentence_ids + tuple(i for i, _ in selected))
            )

        if cursor >= len(sentences):
            break
        next_start = cursor - sentence_overlap
        start = max(start + 1, next_start)
    return passages


def make_passages(
    text: str,
    *,
    min_chars: int = 20,
    max_chars: int = 180,
    max_sentences: int = 3,
    sentence_overlap: int = 1,
    long_sentence_overlap: int = 30,
    minimum_fragment_chars: int = 10,
) -> list[Passage]:
    sentences = split_sentences(
        text,
        max_chars=max_chars,
        long_sentence_overlap=long_sentence_overlap,
        minimum_fragment_chars=minimum_fragment_chars,
    )
    return pack_passages(
        sentences,
        min_chars=min_chars,
        max_chars=max_chars,
        max_sentences=max_sentences,
        sentence_overlap=sentence_overlap,
    )


def _stable_evidence_id(
    source_dataset: str,
    source_raw_id: str,
    document_id: str,
    passage_index: int,
    search_text: str,
) -> str:
    material = "\0".join(
        [source_dataset, source_raw_id, document_id, str(passage_index), search_text]
    )
    return f"{source_dataset}_{text_sha256(material)}"


def make_evidence_record(
    *,
    text: str,
    source_dataset: str,
    source_raw_id: str,
    document_id: str,
    title: str,
    passage_index: int,
    sentence_ids: Sequence[str],
    construction_method: str,
    provenance_url: str,
    license_status: str,
    provenance_ref: Mapping[str, object],
    source_split: str = "train",
    convert_traditional: bool = True,
) -> dict[str, object]:
    search_text = normalize_for_search(text, convert_traditional=convert_traditional)
    content_hash = text_sha256(search_text)
    record: dict[str, object] = {
        "evidence_id": _stable_evidence_id(
            source_dataset, source_raw_id, document_id, passage_index, search_text
        ),
        "text": text,
        "search_text": search_text,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_raw_id": source_raw_id,
        "document_id": document_id,
        "title": title,
        "passage_index": passage_index,
        "sentence_ids": list(sentence_ids),
        "construction_method": construction_method,
        "content_hash": content_hash,
        "provenance_url": provenance_url,
        "license_status": license_status,
        "normalization_version": NORMALIZATION_VERSION,
        "label_visibility": "excluded",
        "duplicate_cluster_id": content_hash,
        "provenance_refs": [dict(provenance_ref)],
    }
    validate_evidence_record(record)
    return record


def _normalized_key(key: object) -> str:
    return str(key).strip().casefold().replace("-", "_")


def _find_forbidden_keys(value: object, path: str = "record") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            if normalized in FORBIDDEN_INDEX_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_find_forbidden_keys(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, f"{path}[{index}]"))
    return found


def validate_evidence_record(record: Mapping[str, object]) -> None:
    missing = REQUIRED_EVIDENCE_FIELDS - set(record)
    extra = set(record) - REQUIRED_EVIDENCE_FIELDS
    if missing or extra:
        raise EvidenceError(
            f"Evidence schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    source_dataset = str(record["source_dataset"]).strip()
    if not source_dataset or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", source_dataset):
        raise EvidenceError("source_dataset must be a non-empty lowercase identifier")
    if record["source_split"] not in {"train", "external"}:
        raise EvidenceError("source_split must be train or external")
    if record["label_visibility"] != "excluded":
        raise EvidenceError("label_visibility must be excluded")
    if not str(record["text"]).strip() or not str(record["search_text"]).strip():
        raise EvidenceError("Evidence text cannot be empty")
    if record["content_hash"] != text_sha256(str(record["search_text"])):
        raise EvidenceError("content_hash does not match search_text")
    forbidden = _find_forbidden_keys(record)
    if forbidden:
        raise EvidenceError(f"Forbidden answer fields entered the evidence record: {forbidden}")


def _mapping_items(value: object) -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        return [(str(key), item) for key, item in value.items()]
    if isinstance(value, list):
        return [(str(index), item) for index, item in enumerate(value)]
    return []


def build_chef_records(
    rows: Iterable[Mapping[str, object]],
    *,
    passage_options: Mapping[str, int],
    convert_traditional: bool = True,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        source_raw_id = str(
            row.get("claimId", row.get("claim_id", row.get("id", f"row-{row_index}")))
        )
        for candidate_key, candidate in _mapping_items(row.get("evidence")):
            if isinstance(candidate, Mapping):
                text = str(candidate.get("text", "")).strip()
                title = str(candidate.get("title", candidate.get("source", ""))).strip()
                provenance_url = str(
                    candidate.get("url", candidate.get("link", candidate.get("source_url", "")))
                ).strip()
                document_id = str(
                    candidate.get("id", candidate.get("document_id", provenance_url))
                    or f"{source_raw_id}:candidate:{candidate_key}"
                )
            else:
                text = str(candidate).strip()
                title = ""
                provenance_url = ""
                document_id = f"{source_raw_id}:candidate:{candidate_key}"
            if not text:
                continue

            passages = make_passages(text, **passage_options)
            for passage_index, passage in enumerate(passages):
                provenance_ref = {
                    "source_dataset": "chef",
                    "source_split": "train",
                    "source_raw_id": source_raw_id,
                    "document_id": document_id,
                    "candidate_key": candidate_key,
                    "provenance_url": provenance_url,
                }
                records.append(
                    make_evidence_record(
                        text=passage.text,
                        source_dataset="chef",
                        source_raw_id=source_raw_id,
                        document_id=document_id,
                        title=title,
                        passage_index=passage_index,
                        sentence_ids=passage.sentence_ids,
                        construction_method="chef_candidate_evidence",
                        provenance_url=provenance_url,
                        license_status="unclear",
                        provenance_ref=provenance_ref,
                        convert_traditional=convert_traditional,
                    )
                )
    return records


def iter_cfever_evidence_refs(value: object) -> Iterator[CFEVEREvidenceRef]:
    if isinstance(value, list):
        if (
            len(value) >= 4
            and isinstance(value[2], str)
            and value[2].strip()
            and isinstance(value[3], int)
        ):
            yield CFEVEREvidenceRef(value[2].strip(), value[3])
            return
        for item in value:
            yield from iter_cfever_evidence_refs(item)


def cfever_gold_references(row: Mapping[str, object]) -> set[CFEVEREvidenceRef]:
    return set(iter_cfever_evidence_refs(row.get("evidence")))


def canonical_cfever_title(value: str) -> str:
    return unicodedata.normalize("NFKC", value).replace("_", " ").strip()


def cfever_linked_titles(rows: Iterable[Mapping[str, object]]) -> set[str]:
    return {
        canonical_cfever_title(reference.page_id)
        for row in rows
        for reference in cfever_gold_references(row)
    }


def parse_cfever_wiki_lines(value: object) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        sortable = sorted(
            value.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0,
        )
        for key, item in sortable:
            text = str(item.get("text", "") if isinstance(item, Mapping) else item).strip()
            if text:
                parsed.append((str(key), text))
        return parsed

    entries = value if isinstance(value, list) else str(value or "").splitlines()
    for fallback_index, item in enumerate(entries):
        if isinstance(item, Mapping):
            sentence_id = str(item.get("id", item.get("line", fallback_index)))
            text = str(item.get("text", "")).strip()
        else:
            raw = str(item).strip()
            parts = raw.split("\t")
            if len(parts) >= 2 and parts[0].strip().isdigit():
                sentence_id, text = parts[0].strip(), parts[1].strip()
            else:
                sentence_id, text = str(fallback_index), raw
        if text:
            parsed.append((sentence_id, text))
    return parsed


def build_cfever_records(
    wiki_rows: Iterable[Mapping[str, object]],
    *,
    allowed_titles: set[str],
    passage_options: Mapping[str, int],
    convert_traditional: bool = True,
) -> tuple[list[dict[str, object]], set[str]]:
    records: list[dict[str, object]] = []
    found_titles: set[str] = set()
    for row in wiki_rows:
        page_id = str(row.get("id", row.get("title", ""))).strip()
        canonical_title = canonical_cfever_title(page_id)
        if not page_id or canonical_title not in allowed_titles:
            continue
        found_titles.add(canonical_title)
        sentences = parse_cfever_wiki_lines(row.get("lines"))
        if not sentences:
            sentences = split_sentences(
                str(row.get("text", "")),
                max_chars=int(passage_options["max_chars"]),
                long_sentence_overlap=int(passage_options["long_sentence_overlap"]),
                minimum_fragment_chars=int(passage_options["minimum_fragment_chars"]),
            )
        else:
            expanded_sentences: list[tuple[str, str]] = []
            for sentence_id, sentence in sentences:
                pieces = _hard_split_sentence(
                    sentence,
                    int(passage_options["max_chars"]),
                    int(passage_options["long_sentence_overlap"]),
                )
                for piece_index, piece in enumerate(pieces):
                    piece_id = sentence_id if len(pieces) == 1 else f"{sentence_id}.{piece_index}"
                    expanded_sentences.append((piece_id, piece))
            sentences = expanded_sentences
        passages = pack_passages(
            sentences,
            min_chars=int(passage_options["min_chars"]),
            max_chars=int(passage_options["max_chars"]),
            max_sentences=int(passage_options["max_sentences"]),
            sentence_overlap=int(passage_options["sentence_overlap"]),
        )
        for passage_index, passage in enumerate(passages):
            provenance_url = f"https://zh.wikipedia.org/wiki/{page_id.replace(' ', '_')}"
            provenance_ref = {
                "source_dataset": "cfever",
                "source_split": "train",
                "source_raw_id": page_id,
                "document_id": page_id,
                "sentence_ids": list(passage.sentence_ids),
                "provenance_url": provenance_url,
            }
            records.append(
                make_evidence_record(
                    text=passage.text,
                    source_dataset="cfever",
                    source_raw_id=page_id,
                    document_id=page_id,
                    title=page_id,
                    passage_index=passage_index,
                    sentence_ids=passage.sentence_ids,
                    construction_method="cfever_linked_wiki_page",
                    provenance_url=provenance_url,
                    license_status="apache-2.0-repository;wikipedia-terms-apply",
                    provenance_ref=provenance_ref,
                    convert_traditional=convert_traditional,
                )
            )
    return records, found_titles


def build_cfever_full_wiki_records(
    wiki_rows: Iterable[Mapping[str, object]],
    *,
    passage_options: Mapping[str, int],
    convert_traditional: bool = True,
) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    pages_seen = 0
    for row in wiki_rows:
        page_id = str(row.get("id", row.get("title", ""))).strip()
        if not page_id:
            continue
        pages_seen += 1
        sentences = parse_cfever_wiki_lines(row.get("lines"))
        if not sentences:
            sentences = split_sentences(
                str(row.get("text", "")),
                max_chars=int(passage_options["max_chars"]),
                long_sentence_overlap=int(passage_options["long_sentence_overlap"]),
                minimum_fragment_chars=int(passage_options["minimum_fragment_chars"]),
            )
        else:
            expanded_sentences: list[tuple[str, str]] = []
            for sentence_id, sentence in sentences:
                pieces = _hard_split_sentence(
                    sentence,
                    int(passage_options["max_chars"]),
                    int(passage_options["long_sentence_overlap"]),
                )
                for piece_index, piece in enumerate(pieces):
                    piece_id = sentence_id if len(pieces) == 1 else f"{sentence_id}.{piece_index}"
                    expanded_sentences.append((piece_id, piece))
            sentences = expanded_sentences
        passages = pack_passages(
            sentences,
            min_chars=int(passage_options["min_chars"]),
            max_chars=int(passage_options["max_chars"]),
            max_sentences=int(passage_options["max_sentences"]),
            sentence_overlap=int(passage_options["sentence_overlap"]),
        )
        for passage_index, passage in enumerate(passages):
            provenance_url = f"https://zh.wikipedia.org/wiki/{page_id.replace(' ', '_')}"
            provenance_ref = {
                "source_dataset": "cfever",
                "source_split": "train",
                "source_raw_id": page_id,
                "document_id": page_id,
                "sentence_ids": list(passage.sentence_ids),
                "provenance_url": provenance_url,
                "stage_b_scope": "full_cfever_wikipedia",
            }
            records.append(
                make_evidence_record(
                    text=passage.text,
                    source_dataset="cfever",
                    source_raw_id=page_id,
                    document_id=page_id,
                    title=page_id,
                    passage_index=passage_index,
                    sentence_ids=passage.sentence_ids,
                    construction_method="cfever_full_wikipedia_page",
                    provenance_url=provenance_url,
                    license_status="apache-2.0-repository;wikipedia-terms-apply",
                    provenance_ref=provenance_ref,
                    convert_traditional=convert_traditional,
                )
            )
    return records, pages_seen


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


def merge_exact_duplicates(
    records: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], int]:
    groups: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        validate_evidence_record(record)
        groups[str(record["search_text"])].append(record)

    merged: list[dict[str, object]] = []
    removed = 0
    for search_text in sorted(groups):
        group = sorted(
            groups[search_text],
            key=lambda row: (
                str(row["source_dataset"]),
                str(row["source_raw_id"]),
                str(row["document_id"]),
                int(row["passage_index"]),
                str(row["evidence_id"]),
            ),
        )
        canonical = dict(group[0])
        provenance_by_json: dict[str, dict[str, object]] = {}
        for row in group:
            for ref in row["provenance_refs"]:  # type: ignore[union-attr]
                serialized = json.dumps(
                    ref, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                provenance_by_json[serialized] = dict(ref)
        canonical["provenance_refs"] = [
            provenance_by_json[key] for key in sorted(provenance_by_json)
        ]
        canonical["duplicate_cluster_id"] = text_sha256(search_text)
        validate_evidence_record(canonical)
        merged.append(canonical)
        removed += len(group) - 1
    return sorted(merged, key=lambda row: str(row["evidence_id"])), removed


def character_ngrams(text: str, n: int = 3) -> set[str]:
    compact = compact_text(text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def find_near_duplicate_pairs(
    records: Sequence[Mapping[str, object]],
    *,
    threshold: float = 0.9,
    ngram_size: int = 3,
) -> list[dict[str, object]]:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    ordered = sorted(records, key=lambda row: str(row["evidence_id"]))
    signature_index: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    pairs: list[dict[str, object]] = []

    @lru_cache(maxsize=4096)
    def exact_shingles(index: int) -> frozenset[str]:
        return frozenset(
            character_ngrams(str(ordered[index]["search_text"]), ngram_size)
        )

    def signature_keys(values: set[str]) -> list[tuple[int, int]]:
        hashes = sorted(
            {
                zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF
                for value in values
            }
        )[:12]
        if len(hashes) == 1:
            return [(hashes[0], -1)]
        return [(hashes[index], hashes[index + 1]) for index in range(len(hashes) - 1)]

    for right_index, source_record in enumerate(ordered):
        right_shingles = character_ngrams(str(source_record["search_text"]), ngram_size)
        if not right_shingles:
            continue
        candidates: set[int] = set()
        keys = signature_keys(right_shingles)
        for key in keys:
            candidates.update(signature_index[key])

        for left_index in sorted(candidates):
            left_shingles = exact_shingles(left_index)
            length_ratio = min(len(left_shingles), len(right_shingles)) / max(
                len(left_shingles), len(right_shingles)
            )
            if length_ratio < threshold:
                continue
            score = len(left_shingles & right_shingles) / len(left_shingles | right_shingles)
            if score >= threshold:
                pairs.append(
                    {
                        "left_evidence_id": ordered[left_index]["evidence_id"],
                        "right_evidence_id": ordered[right_index]["evidence_id"],
                        "jaccard": round(score, 6),
                        "threshold": threshold,
                        "candidate_method": "bottom12-adjacent-pair-lsh",
                    }
                )
        for key in keys:
            signature_index[key].append(right_index)
    return pairs


def _containment(left: str, right: str, ngram_size: int = 5) -> float:
    if left in right or right in left:
        return 1.0
    left_grams = character_ngrams(left, ngram_size)
    right_grams = character_ngrams(right, ngram_size)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / min(len(left_grams), len(right_grams))


def _target_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row.get("id", "unknown"))
        full_text = str(row.get("full_text", row.get("text", row.get("content", ""))))
        if full_text.strip():
            targets.append({"target_id": row_id, "target_kind": "full_text", "text": full_text})
        for claim in row.get("key_claims", []):
            if isinstance(claim, Mapping) and str(claim.get("text", "")).strip():
                rank = str(claim.get("rank", "unknown"))
                targets.append(
                    {
                        "target_id": f"{row_id}::claim-{rank}",
                        "target_kind": "key_claim",
                        "text": str(claim["text"]),
                    }
                )
    return targets


def audit_ltcr_leakage(
    records: Sequence[Mapping[str, object]],
    ltcr_rows: Iterable[Mapping[str, object]],
    *,
    containment_exclude_threshold: float = 0.9,
    near_report_threshold: float = 0.8,
    minimum_chars: int = 20,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    targets = _target_rows(ltcr_rows)
    prepared_targets = []
    exact_targets: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    anchor_index: defaultdict[str, set[int]] = defaultdict(set)
    anchor_size = 16
    anchor_stride = 8
    for target in targets:
        compact = compact_text(target["text"])
        if len(compact) < minimum_chars:
            continue
        prepared = {**target, "compact": compact}
        target_index = len(prepared_targets)
        prepared_targets.append(prepared)
        exact_targets[compact].append(prepared)
        if len(compact) >= anchor_size:
            starts = list(range(0, len(compact) - anchor_size + 1, anchor_stride))
            starts.append(len(compact) - anchor_size)
            for start in set(starts):
                anchor_index[compact[start : start + anchor_size]].add(target_index)

    kept: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    near_pairs: list[dict[str, object]] = []
    for source_record in sorted(records, key=lambda row: str(row["evidence_id"])):
        record = dict(source_record)
        compact = compact_text(str(record["search_text"]))
        matches: list[tuple[float, dict[str, str]]] = []
        if compact in exact_targets:
            matches.extend((1.0, target) for target in exact_targets[compact])
        elif len(compact) >= minimum_chars:
            candidates: set[int] = set()
            if len(compact) >= anchor_size:
                for start in range(len(compact) - anchor_size + 1):
                    candidates.update(anchor_index.get(compact[start : start + anchor_size], set()))
            for target_index in sorted(candidates):
                target = prepared_targets[target_index]
                score = _containment(compact, target["compact"])
                if score >= near_report_threshold:
                    matches.append((score, target))

        exclusion_matches = [item for item in matches if item[0] >= containment_exclude_threshold]
        if exclusion_matches:
            best_score, best_target = max(exclusion_matches, key=lambda item: item[0])
            excluded.append(
                {
                    "evidence_id": record["evidence_id"],
                    "reason": (
                        "exact_ltcr_overlap"
                        if best_score == 1.0
                        else "high_ltcr_containment"
                    ),
                    "score": round(best_score, 6),
                    "target_id": best_target["target_id"],
                    "target_kind": best_target["target_kind"],
                }
            )
            continue
        for score, target in matches:
            near_pairs.append(
                {
                    "evidence_id": record["evidence_id"],
                    "target_id": target["target_id"],
                    "target_kind": target["target_kind"],
                    "containment": round(score, 6),
                }
            )
        kept.append(record)
    return kept, excluded, near_pairs
