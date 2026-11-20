from __future__ import annotations

import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from retrieval.evidence import validate_evidence_record


TOKEN_RE = re.compile(r"[\u3400-\u9fffA-Za-z0-9]+")


@dataclass(frozen=True)
class RetrievalHit:
    evidence_id: str
    text: str
    source_dataset: str
    document_id: str
    title: str
    sentence_ids: tuple[str, ...]
    provenance_url: str
    score: float
    rank: int
    scoring_method: str

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "source_dataset": self.source_dataset,
            "document_id": self.document_id,
            "title": self.title,
            "sentence_ids": list(self.sentence_ids),
            "provenance_url": self.provenance_url,
            "score": round(float(self.score), 8),
            "rank": self.rank,
            "scoring_method": self.scoring_method,
        }


def tokenize_character_bigram(text: str) -> list[str]:
    tokens: list[str] = []
    for segment in TOKEN_RE.findall(text):
        if len(segment) == 1:
            tokens.append(segment)
        else:
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def tokenize_jieba_search(text: str) -> list[str]:
    try:
        import jieba
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("jieba is required for the jieba_search tokenizer") from exc
    jieba.setLogLevel(20)
    return [token.strip() for token in jieba.cut_for_search(text) if TOKEN_RE.search(token)]


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "character_bigram": tokenize_character_bigram,
    "jieba_search": tokenize_jieba_search,
}


def rank_texts(
    query: str,
    texts: Sequence[str],
    *,
    tokenizer: str = "character_bigram",
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    if tokenizer not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")
    if not texts:
        return []
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("rank-bm25 is required for BM25 retrieval") from exc
    tokenize = TOKENIZERS[tokenizer]
    model = BM25Okapi([tokenize(text) for text in texts], k1=k1, b=b)
    scores = model.get_scores(tokenize(query))
    order = sorted(range(len(texts)), key=lambda index: (-float(scores[index]), index))
    return [
        (index, float(scores[index]))
        for index in order
    ]


class BM25Retriever:
    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        tokenizer: str = "character_bigram",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if tokenizer not in TOKENIZERS:
            raise ValueError(f"Unknown tokenizer {tokenizer!r}; choose from {sorted(TOKENIZERS)}")
        if not records:
            raise ValueError("Cannot build BM25 over an empty corpus")
        try:
            from sklearn.feature_extraction.text import CountVectorizer
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("scikit-learn is required for inverted BM25 retrieval") from exc

        self.records = [record if isinstance(record, dict) else dict(record) for record in records]
        for record in self.records:
            validate_evidence_record(record)
        self.tokenizer_name = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self._tokenize = TOKENIZERS[tokenizer]
        self._vectorizer = CountVectorizer(
            analyzer=self._tokenize,
            lowercase=False,
            dtype=np.float32,
        )
        document_term = self._vectorizer.fit_transform(
            str(record["search_text"]) for record in self.records
        )
        self._doc_len = np.asarray(document_term.sum(axis=1)).ravel().astype(np.float32)
        self._avg_doc_len = float(self._doc_len.mean())
        document_frequency = np.asarray((document_term > 0).sum(axis=0)).ravel()
        corpus_size = len(self.records)
        self._idf = np.log(
            1.0 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)
        self._term_document = document_term.tocsc()
        self._evidence_id_order = sorted(
            range(len(self.records)), key=lambda index: str(self.records[index]["evidence_id"])
        )
        self._evidence_tie_rank = np.empty(len(self.records), dtype=np.int32)
        for rank, document_index in enumerate(self._evidence_id_order):
            self._evidence_tie_rank[document_index] = rank

    def search(self, query: str, top_k: int = 20) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = np.zeros(len(self.records), dtype=np.float32)
        query_counts = Counter(self._tokenize(query))
        vocabulary = self._vectorizer.vocabulary_
        for token, query_frequency in query_counts.items():
            term_index = vocabulary.get(token)
            if term_index is None:
                continue
            start = self._term_document.indptr[term_index]
            end = self._term_document.indptr[term_index + 1]
            document_ids = self._term_document.indices[start:end]
            term_frequency = self._term_document.data[start:end]
            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * self._doc_len[document_ids] / self._avg_doc_len
            )
            scores[document_ids] += (
                query_frequency
                * self._idf[term_index]
                * term_frequency
                * (self.k1 + 1.0)
                / denominator
            )

        nonzero = np.flatnonzero(scores)
        limit = min(top_k, len(self.records))
        if len(nonzero) > limit:
            nonzero_scores = scores[nonzero]
            threshold = np.partition(nonzero_scores, len(nonzero_scores) - limit)[
                len(nonzero_scores) - limit
            ]
            above = nonzero[nonzero_scores > threshold]
            tied = nonzero[nonzero_scores == threshold]
            tie_slots = limit - len(above)
            if len(tied) > tie_slots:
                tie_ranks = self._evidence_tie_rank[tied]
                positions = np.argpartition(tie_ranks, tie_slots - 1)[:tie_slots]
                tied = tied[positions]
            candidates = np.concatenate([above, tied])
        else:
            candidates = nonzero
        order = sorted(
            (int(index) for index in candidates),
            key=lambda index: (-float(scores[index]), str(self.records[index]["evidence_id"])),
        )[:limit]
        if len(order) < limit:
            selected = set(order)
            for index in self._evidence_id_order:
                if index not in selected:
                    order.append(index)
                    if len(order) >= limit:
                        break
        return [
            _record_to_hit(
                self.records[index],
                score=float(scores[index]),
                rank=rank,
                method=f"bm25:{self.tokenizer_name}",
            )
            for rank, index in enumerate(order, start=1)
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        with path.open("rb") as handle:
            value = pickle.load(handle)  # noqa: S301 - index files are local generated artifacts
        if not isinstance(value, cls):
            raise TypeError(f"Not a BM25Retriever index: {path}")
        if not hasattr(value, "_evidence_tie_rank"):
            value._evidence_id_order = sorted(
                range(len(value.records)),
                key=lambda index: str(value.records[index]["evidence_id"]),
            )
            value._evidence_tie_rank = np.empty(len(value.records), dtype=np.int32)
            for rank, document_index in enumerate(value._evidence_id_order):
                value._evidence_tie_rank[document_index] = rank
        return value


def _record_to_hit(
    record: Mapping[str, object], *, score: float, rank: int, method: str
) -> RetrievalHit:
    return RetrievalHit(
        evidence_id=str(record["evidence_id"]),
        text=str(record["text"]),
        source_dataset=str(record["source_dataset"]),
        document_id=str(record["document_id"]),
        title=str(record["title"]),
        sentence_ids=tuple(
            str(value) for value in record["sentence_ids"]  # type: ignore[union-attr]
        ),
        provenance_url=str(record["provenance_url"]),
        score=score,
        rank=rank,
        scoring_method=method,
    )


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[RetrievalHit]],
    *,
    top_k: int = 20,
    constant: int = 60,
) -> list[RetrievalHit]:
    if top_k <= 0 or constant <= 0:
        raise ValueError("top_k and constant must be positive")
    scores: dict[str, float] = {}
    representatives: dict[str, RetrievalHit] = {}
    for ranked in ranked_lists:
        for fallback_rank, hit in enumerate(ranked, start=1):
            rank = hit.rank if hit.rank > 0 else fallback_rank
            scores[hit.evidence_id] = scores.get(hit.evidence_id, 0.0) + 1.0 / (constant + rank)
            representatives.setdefault(hit.evidence_id, hit)

    ordered_ids = sorted(
        scores, key=lambda evidence_id: (-scores[evidence_id], evidence_id)
    )[:top_k]
    return [
        RetrievalHit(
            evidence_id=evidence_id,
            text=representatives[evidence_id].text,
            source_dataset=representatives[evidence_id].source_dataset,
            document_id=representatives[evidence_id].document_id,
            title=representatives[evidence_id].title,
            sentence_ids=representatives[evidence_id].sentence_ids,
            provenance_url=representatives[evidence_id].provenance_url,
            score=scores[evidence_id],
            rank=rank,
            scoring_method=f"rrf:k={constant}",
        )
        for rank, evidence_id in enumerate(ordered_ids, start=1)
    ]


def mean_query_latency_ms(elapsed_seconds: float, query_count: int) -> float:
    return 0.0 if query_count == 0 else (elapsed_seconds * 1000.0) / query_count
