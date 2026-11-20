from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retrieval.bm25 import TOKENIZERS  # noqa: E402
from retrieval.evidence import file_sha256, normalize_for_search, text_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data_trendfact.yaml"
PARQUET_COLUMNS = ("url", "text", "publish_date")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _query_vocabulary(
    queries: Sequence[Mapping[str, object]], tokenizer: str
) -> tuple[dict[str, int], sparse.csc_matrix]:
    tokenize = TOKENIZERS[tokenizer]
    vocabulary: dict[str, int] = {}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for query_index, query in enumerate(queries):
        counts = Counter(tokenize(str(query["query_text"])))
        for token, count in counts.items():
            term_index = vocabulary.setdefault(token, len(vocabulary))
            rows.append(term_index)
            columns.append(query_index)
            data.append(float(count))
    matrix = sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(vocabulary), len(queries)),
        dtype=np.float32,
    ).tocsc()
    return vocabulary, matrix


def _iter_batches(parquet_path: Path, batch_size: int) -> Iterable[dict[str, list[object]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("pyarrow is required to stream TrendFact evidence.parquet") from exc
    parquet = pq.ParquetFile(parquet_path)
    missing = set(PARQUET_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"TrendFact evidence parquet is missing columns: {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(PARQUET_COLUMNS)):
        yield batch.to_pydict()


def _texts(batch: Mapping[str, Sequence[object]]) -> list[str]:
    return [str(value or "").strip() for value in batch["text"]]


def _first_pass(
    *,
    parquet_path: Path,
    queries: Sequence[Mapping[str, object]],
    vocabulary: dict[str, int],
    tokenizer: str,
    batch_size: int,
) -> tuple[int, float, np.ndarray, dict[str, object]]:
    tokenize = TOKENIZERS[tokenizer]
    vectorizer = CountVectorizer(
        analyzer=tokenize, lowercase=False, vocabulary=vocabulary, dtype=np.float32
    )
    gold_hashes = {
        int(target_idx): str(expected_hash)
        for query in queries
        for target_idx, expected_hash in dict(query["gold_evidence_hashes"]).items()
    }
    aligned: dict[int, bool] = {}
    document_count = 0
    total_document_length = 0
    document_frequency = np.zeros(len(vocabulary), dtype=np.int64)
    for batch_index, batch in enumerate(_iter_batches(parquet_path, batch_size), start=1):
        texts = _texts(batch)
        matrix = vectorizer.transform(texts)
        document_frequency += np.asarray((matrix > 0).sum(axis=0)).ravel()
        lengths = [len(tokenize(text)) for text in texts]
        total_document_length += sum(lengths)
        for offset, text in enumerate(texts):
            target_idx = document_count + offset
            if target_idx in gold_hashes:
                aligned[target_idx] = (
                    text_sha256(normalize_for_search(text)) == gold_hashes[target_idx]
                )
        document_count += len(texts)
        if batch_index % 25 == 0:
            print(f"[trendfact] first-pass evidence={document_count}", flush=True)
    missing_gold = sorted(set(gold_hashes) - set(aligned))
    alignment = {
        "pilot_unique_gold_target_indices": len(gold_hashes),
        "found_in_parquet": len(aligned),
        "exact_normalized_text_matches": sum(aligned.values()),
        "text_mismatches": sum(not value for value in aligned.values()),
        "missing_target_indices": missing_gold,
    }
    if missing_gold:
        raise ValueError(f"Pilot gold target indices missing from evidence parquet: {missing_gold[:10]}")
    return (
        document_count,
        total_document_length / max(document_count, 1),
        document_frequency,
        alignment,
    )


def _is_better(candidate: tuple[float, int], current: tuple[float, int]) -> bool:
    return candidate[0] > current[0] or (
        candidate[0] == current[0] and candidate[1] < current[1]
    )


def _add_candidate(
    top: list[tuple[float, int, dict[str, object]]],
    candidate: tuple[float, int, dict[str, object]],
    *,
    top_k: int,
) -> None:
    if len(top) < top_k:
        top.append(candidate)
        return
    worst_index = 0
    for index in range(1, len(top)):
        if _is_better(top[worst_index][:2], top[index][:2]):
            worst_index = index
    if _is_better(candidate[:2], top[worst_index][:2]):
        top[worst_index] = candidate


def _second_pass(
    *,
    parquet_path: Path,
    queries: Sequence[Mapping[str, object]],
    vocabulary: dict[str, int],
    query_matrix: sparse.csc_matrix,
    tokenizer: str,
    document_count: int,
    average_document_length: float,
    document_frequency: np.ndarray,
    batch_size: int,
    top_k: int,
    k1: float,
    b: float,
) -> list[list[dict[str, object]]]:
    tokenize = TOKENIZERS[tokenizer]
    vectorizer = CountVectorizer(
        analyzer=tokenize, lowercase=False, vocabulary=vocabulary, dtype=np.float32
    )
    idf = np.log(
        1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    ).astype(np.float32)
    top_candidates: list[list[tuple[float, int, dict[str, object]]]] = [
        [] for _ in queries
    ]
    rows_seen = 0
    for batch_index, batch in enumerate(_iter_batches(parquet_path, batch_size), start=1):
        texts = _texts(batch)
        document_lengths = np.asarray([len(tokenize(text)) for text in texts], dtype=np.float32)
        term_matrix = vectorizer.transform(texts).tocsr()
        coo = term_matrix.tocoo(copy=True)
        if coo.nnz:
            denominator = coo.data + k1 * (
                1.0 - b + b * document_lengths[coo.row] / average_document_length
            )
            coo.data = idf[coo.col] * coo.data * (k1 + 1.0) / denominator
            scores = coo.tocsr().dot(query_matrix).tocsc()
            for query_index in range(scores.shape[1]):
                start = scores.indptr[query_index]
                end = scores.indptr[query_index + 1]
                if start == end:
                    continue
                column_scores = scores.data[start:end]
                column_rows = scores.indices[start:end]
                positions = (
                    np.argpartition(column_scores, -top_k)[-top_k:]
                    if len(column_scores) > top_k
                    else np.arange(len(column_scores))
                )
                for position in positions:
                    score = float(column_scores[position])
                    if score <= 0:
                        continue
                    row_offset = int(column_rows[position])
                    target_idx = rows_seen + row_offset
                    candidate = {
                        "evidence_id": f"trendfact_evidence_{target_idx}",
                        "target_idx": target_idx,
                        "text": texts[row_offset],
                        "url": str(batch["url"][row_offset] or ""),
                        "publish_date": str(batch["publish_date"][row_offset] or ""),
                        "bm25_score": round(score, 8),
                    }
                    _add_candidate(
                        top_candidates[query_index], (score, target_idx, candidate), top_k=top_k
                    )
        rows_seen += len(texts)
        if batch_index % 25 == 0:
            print(f"[trendfact] second-pass evidence={rows_seen}", flush=True)
        del term_matrix, coo
        gc.collect()
    if rows_seen != document_count:
        raise ValueError(f"Parquet row count changed between passes: {rows_seen} != {document_count}")
    return [
        [
            {**candidate, "rank": rank}
            for rank, (_, _, candidate) in enumerate(
                sorted(candidates, key=lambda item: (-item[0], item[1])), start=1
            )
        ]
        for candidates in top_candidates
    ]


def retrieval_metrics(
    queries: Sequence[Mapping[str, object]],
    hits_by_query: Sequence[Sequence[Mapping[str, object]]],
    report_at_k: Sequence[int],
) -> dict[str, object]:
    if len(queries) != len(hits_by_query):
        raise ValueError("Query and retrieval result counts differ")
    max_k = max(int(k) for k in report_at_k)
    hit_counts = Counter()
    reciprocal_ranks: list[float] = []
    reciprocal_ranks_at_k: dict[int, list[float]] = {
        int(k): [] for k in report_at_k
    }
    gold_fraction_at_k: dict[int, list[float]] = {int(k): [] for k in report_at_k}
    for query, hits in zip(queries, hits_by_query):
        gold = {int(value) for value in query["gold_target_indices"]}  # type: ignore[union-attr]
        ranked = [int(hit["target_idx"]) for hit in hits[:max_k]]
        first_rank = next((rank for rank, value in enumerate(ranked, start=1) if value in gold), None)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        for k in report_at_k:
            retrieved = set(ranked[: int(k)])
            if gold & retrieved:
                hit_counts[int(k)] += 1
            gold_fraction_at_k[int(k)].append(len(gold & retrieved) / len(gold))
            first_rank_at_k = next(
                (
                    rank
                    for rank, value in enumerate(ranked[: int(k)], start=1)
                    if value in gold
                ),
                None,
            )
            reciprocal_ranks_at_k[int(k)].append(
                0.0 if first_rank_at_k is None else 1.0 / first_rank_at_k
            )
    count = len(queries)
    return {
        "queries": count,
        "any_gold_recall_at_k": {
            str(k): round(hit_counts[int(k)] / count, 6) for k in report_at_k
        },
        "mean_gold_fraction_at_k": {
            str(k): round(float(np.mean(gold_fraction_at_k[int(k)])), 6)
            for k in report_at_k
        },
        "mrr_at_max_k": round(float(np.mean(reciprocal_ranks)), 6),
        "mrr_at_k": {
            str(k): round(float(np.mean(reciprocal_ranks_at_k[int(k)])), 6)
            for k in report_at_k
        },
        "max_k": max_k,
    }


def _write_outputs(
    output_dir: Path,
    queries: Sequence[Mapping[str, object]],
    hits_by_query: Sequence[Sequence[Mapping[str, object]]],
) -> None:
    with (output_dir / "retrieval_candidates.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for query, hits in zip(queries, hits_by_query):
            row = {
                "query_id": query["query_id"],
                "document_id": query["document_id"],
                "dataset": "trendfact",
                "split": query["split"],
                "query_text": query["query_text"],
                "retrieval_method": "character_bigram_bm25",
                "hits": list(hits),
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    overwrite: bool = False,
    batch_size: int = 5000,
    queries_path: Path | None = None,
    output_dir: Path | None = None,
    stored_top_k: int | None = None,
    report_at_k: Sequence[int] | None = None,
    allow_test: bool = False,
) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parquet_path = _resolve(config["input"]["evidence_parquet"]).resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Missing TrendFact evidence library: {parquet_path}. Download the frozen official file first."
        )
    actual_hash = file_sha256(parquet_path)
    expected_hash = str(config["input"]["evidence_parquet_sha256"]).lower()
    if actual_hash != expected_hash:
        raise ValueError(f"TrendFact evidence SHA-256 mismatch: {actual_hash} != {expected_hash}")
    data_dir = _resolve(config["output"]["data_dir"])
    queries_path = (
        queries_path.resolve()
        if queries_path is not None
        else data_dir / "pilot_dev_queries.jsonl"
    )
    queries = _read_jsonl(queries_path)
    query_splits = {str(query["split"]) for query in queries}
    if len(query_splits) != 1:
        raise ValueError(f"Expected one query split, got {sorted(query_splits)}")
    is_test = query_splits == {"test"}
    if is_test and not allow_test:
        raise ValueError("TrendFact test retrieval requires --allow-final-test")
    if is_test and overwrite:
        raise ValueError("Final-test retrieval output may not be overwritten")
    pilot_config = config["pilot"]
    bm25_config = pilot_config["bm25"]
    tokenizer = str(bm25_config["tokenizer"])
    vocabulary, query_matrix = _query_vocabulary(queries, tokenizer)
    stored_top_k = int(stored_top_k or pilot_config["stored_top_k"])
    report_at_k = [int(k) for k in (report_at_k or pilot_config["report_at_k"])]
    if max(report_at_k) > stored_top_k:
        raise ValueError("Retrieval metrics cannot exceed the stored top-k")
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else _resolve(config["output"]["pilot_dir"])
    )
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite to replace it.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trendfact-pilot-", dir=output_dir.parent))
    started = time.perf_counter()
    try:
        document_count, average_length, document_frequency, alignment = _first_pass(
            parquet_path=parquet_path,
            queries=queries,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
        if document_count != int(config["input"]["expected_evidence_rows"]):
            raise ValueError(
                f"TrendFact evidence row count mismatch: {document_count} != "
                f"{config['input']['expected_evidence_rows']}"
            )
        hits_by_query = _second_pass(
            parquet_path=parquet_path,
            queries=queries,
            vocabulary=vocabulary,
            query_matrix=query_matrix,
            tokenizer=tokenizer,
            document_count=document_count,
            average_document_length=average_length,
            document_frequency=document_frequency,
            batch_size=batch_size,
            top_k=stored_top_k,
            k1=float(bm25_config["k1"]),
            b=float(bm25_config["b"]),
        )
        metrics = retrieval_metrics(queries, hits_by_query, report_at_k)
        _write_outputs(staging, queries, hits_by_query)
        recall = metrics["any_gold_recall_at_k"]
        decision: dict[str, bool] = {}
        if not is_test:
            acceptance = pilot_config["acceptance"]
            if "100" in recall:
                decision["recall_at_100_pass"] = float(recall["100"]) >= float(
                    acceptance["any_gold_recall_at_100"]
                )
            if "20" in recall:
                decision["recall_at_20_pass"] = float(recall["20"]) >= float(
                    acceptance["any_gold_recall_at_20"]
                )
            decision["proceed_to_llm_reranking"] = all(decision.values())
        report = {
            "experiment": config["experiment"],
            "config_sha256": file_sha256(config_path),
            "evidence_parquet_sha256": actual_hash,
            "evidence_rows": document_count,
            "pilot_queries": len(queries),
            "query_split": next(iter(query_splits)),
            "queries_path": queries_path.relative_to(ROOT).as_posix(),
            "queries_sha256": file_sha256(queries_path),
            "tokenizer": tokenizer,
            "bm25": {"k1": float(bm25_config["k1"]), "b": float(bm25_config["b"])},
            "stored_top_k": stored_top_k,
            "average_document_length": round(average_length, 6),
            "gold_alignment": alignment,
            "metrics": metrics,
            "acceptance": decision,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "test_used": is_test,
        }
        (staging / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TrendFact dev-only BM25 pilot.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--queries-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stored-top-k", type=int)
    parser.add_argument("--report-at-k", type=int, nargs="+")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                overwrite=args.overwrite,
                batch_size=args.batch_size,
                queries_path=args.queries_path,
                output_dir=args.output_dir,
                stored_top_k=args.stored_top_k,
                report_at_k=args.report_at_k,
                allow_test=args.allow_final_test,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
