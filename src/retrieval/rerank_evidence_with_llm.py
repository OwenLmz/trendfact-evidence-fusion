from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retrieval.evidence import file_sha256, iter_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "llm_evidence_rerank_qwen25_7b.yaml"
ALLOWED_STANCES = {"support", "refute", "context", "unknown"}
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


class RerankError(ValueError):
    """Raised when a model response violates the constrained reranking protocol."""


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_messages(
    query_text: str,
    hits: Sequence[Mapping[str, object]],
    *,
    selected_top_k: int,
) -> tuple[list[dict[str, str]], dict[str, Mapping[str, object]]]:
    candidate_map: dict[str, Mapping[str, object]] = {}
    candidates: list[dict[str, object]] = []
    for index, hit in enumerate(hits, start=1):
        candidate_id = f"E{index:02d}"
        candidate_map[candidate_id] = hit
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source": str(hit["source_dataset"]),
                "title": str(hit.get("title", "")),
                "text": str(hit["text"]),
            }
        )

    system = (
        "你是严格的中文事实核查证据筛选器。你的唯一任务是从给定候选中排序并选择证据。"
        "不得生成候选中不存在的事实或证据，不得判断新闻最终真假，不得根据来源名称直接假定内容正确。"
        "必须核对主张的核心主体、行为或事件，以及法律条件、时间、地点、数字、因果等关键限定。"
        "只有候选能够直接验证这些核心内容时才能给3分并使用support或refute。"
        "同一人物、机构、疫情主题或一般医学背景不能证明另一项具体主张；这类候选最高给1分，"
        "stance使用context或unknown。署名、来源行、口号、感想等不构成可核查事实时，不得因关键词重合而给高分。"
        "相关性标准：3=直接验证或反驳同一完整事实；2=覆盖同一事实但缺少一个关键限定；"
        "1=仅主题、实体或背景相关；0=无关。stance只能是support、refute、context或unknown。"
    )
    user = (
        f"待核查主张：\n{query_text}\n\n"
        f"候选证据（共{len(candidates)}条）：\n"
        f"{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"请严格选择并排序{selected_top_k}条候选。只输出一个JSON对象，不要Markdown，不要额外文字。"
        "格式必须为："
        '{"ranking":[{"candidate_id":"E01","relevance":3,'
        '"stance":"support","reason":"不超过40个汉字"}]}。'
        f"ranking必须恰好有{selected_top_k}项，candidate_id不得重复；证据不足时仍选择最接近者，"
        "但应如实给出0或1分和unknown/context。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], candidate_map


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


def validate_ranking(
    payload: Mapping[str, object],
    *,
    candidate_ids: set[str],
    selected_top_k: int,
    allowed_scores: set[int],
    reason_max_chars: int = 40,
) -> list[dict[str, object]]:
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != selected_top_k:
        raise RerankError(f"ranking must contain exactly {selected_top_k} items")

    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for rank, item in enumerate(ranking, start=1):
        if not isinstance(item, Mapping):
            raise RerankError(f"ranking item {rank} must be an object")
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id not in candidate_ids:
            raise RerankError(f"Unknown candidate_id at rank {rank}: {candidate_id}")
        if candidate_id in seen:
            raise RerankError(f"Duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        try:
            relevance = int(item.get("relevance"))
        except (TypeError, ValueError) as exc:
            raise RerankError(f"Invalid relevance at rank {rank}") from exc
        if relevance not in allowed_scores:
            raise RerankError(f"Relevance out of range at rank {rank}: {relevance}")
        stance = str(item.get("stance", "")).casefold()
        if stance not in ALLOWED_STANCES:
            raise RerankError(f"Invalid stance at rank {rank}: {stance}")
        reason = str(item.get("reason", "")).strip()
        if not reason:
            raise RerankError(f"Missing reason at rank {rank}")
        if len(reason) > reason_max_chars:
            raise RerankError(
                f"Reason exceeds {reason_max_chars} characters at rank {rank}"
            )
        normalized.append(
            {
                "llm_rank": rank,
                "candidate_id": candidate_id,
                "relevance": relevance,
                "stance": stance,
                "reason": reason,
            }
        )
    return normalized


def _completed_query_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row["query_id"]) for row in iter_jsonl([path])}


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


def _chunks(values: Sequence[dict[str, object]], size: int) -> Iterable[list[dict[str, object]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    resume: bool = False,
    limit: int | None = None,
    output_path_override: Path | None = None,
    manifest_path_override: Path | None = None,
    input_path_override: Path | None = None,
) -> dict[str, object]:
    run_started = time.perf_counter()
    config_path = _resolve(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("LLM rerank config must contain a mapping")
    if str(config["input"]["split"]) == "test":
        raise ValueError("LTCR test reranking is prohibited during model selection")
    if bool(config["input"]["labels_visible"]):
        raise ValueError("Target labels must remain hidden from the reranker")
    protocol = config["protocol"]
    if any(
        bool(protocol[key])
        for key in ("may_generate_evidence", "may_use_target_label", "may_predict_final_class")
    ):
        raise ValueError("The constrained reranker protocol was weakened in config")

    input_path = _resolve(
        input_path_override or str(config["input"]["retrieval_path"])
    ).resolve()
    output_path = _resolve(
        output_path_override or str(config["output"]["path"])
    ).resolve()
    manifest_path = _resolve(
        manifest_path_override or str(config["output"]["manifest_path"])
    ).resolve()
    if output_path.exists() and not resume:
        raise FileExistsError(f"Output exists: {output_path}. Pass --resume to continue it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = _completed_query_ids(output_path) if resume else set()
    rows = [row for row in iter_jsonl([input_path]) if str(row["query_id"]) not in completed_ids]
    if limit is not None:
        rows = rows[:limit]

    candidate_top_k = int(config["input"]["candidate_top_k"])
    selected_top_k = int(config["output"]["selected_top_k"])
    allowed_scores = {int(value) for value in config["prompt"]["allowed_relevance_scores"]}
    reason_max_chars = int(config["prompt"]["reason_max_chinese_chars"])
    prepared: list[dict[str, object]] = []
    for row in rows:
        hits = list(row.get("hits", []))[:candidate_top_k]
        if len(hits) < selected_top_k:
            raise RerankError(f"Not enough candidates for {row['query_id']}")
        messages, candidate_map = build_messages(
            str(row["query_text"]), hits, selected_top_k=selected_top_k
        )
        prepared.append({"row": row, "messages": messages, "candidate_map": candidate_map})

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_config["name_or_path"]),
        revision=str(model_config["revision"]),
        trust_remote_code=bool(model_config["trust_remote_code"]),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_config["name_or_path"]),
        revision=str(model_config["revision"]),
        trust_remote_code=bool(model_config["trust_remote_code"]),
        torch_dtype=_torch_dtype(str(model_config["torch_dtype"])),
        device_map=str(model_config["device_map"]),
        attn_implementation=str(model_config["attn_implementation"]),
    )
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model_load_seconds = time.perf_counter() - model_load_started

    inference = config["inference"]
    errors = 0
    generation_started = time.perf_counter()
    mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for batch in _chunks(prepared, int(inference["batch_size"])):
            prompts = [
                tokenizer.apply_chat_template(
                    item["messages"], tokenize=False, add_generation_prompt=True
                )
                for item in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(inference["max_input_tokens"]),
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=int(inference["max_new_tokens"]),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_width = encoded["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            for item, raw_text in zip(batch, decoded, strict=True):
                source_row = item["row"]
                candidate_map = item["candidate_map"]
                try:
                    ranking = validate_ranking(
                        extract_json_object(raw_text),
                        candidate_ids=set(candidate_map),
                        selected_top_k=selected_top_k,
                        allowed_scores=allowed_scores,
                        reason_max_chars=reason_max_chars,
                    )
                    selected = []
                    for decision in ranking:
                        hit = dict(candidate_map[str(decision["candidate_id"])])
                        selected.append({**decision, **hit})
                    status = "ok"
                    error = ""
                except RerankError as exc:
                    errors += 1
                    selected = []
                    status = "parse_error"
                    error = str(exc)
                output_row = {
                    "query_id": source_row["query_id"],
                    "document_id": source_row["document_id"],
                    "claim_rank": source_row["claim_rank"],
                    "query_text": source_row["query_text"],
                    "input_strategy": source_row["strategy"],
                    "candidate_top_k": candidate_top_k,
                    "selected_top_k": selected_top_k,
                    "status": status,
                    "error": error,
                    "selected": selected,
                    "raw_generation": raw_text,
                    "prompt_version": str(config["prompt"]["version"]),
                    "model": str(model_config["name_or_path"]),
                    "model_revision": str(model_config["revision"]),
                }
                handle.write(
                    json.dumps(
                        output_row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
    generation_seconds = time.perf_counter() - generation_started

    all_rows = list(iter_jsonl([output_path]))
    status_counts = Counter(str(row["status"]) for row in all_rows)
    manifest = {
        "experiment": str(config["experiment"]),
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": file_sha256(config_path),
        "input_file": input_path.relative_to(ROOT).as_posix(),
        "input_sha256": file_sha256(input_path),
        "output_file": output_path.relative_to(ROOT).as_posix(),
        "output_sha256": file_sha256(output_path),
        "rows": len(all_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "errors_this_run": errors,
        "rows_processed_this_run": len(prepared),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "model": str(model_config["name_or_path"]),
        "model_revision": str(model_config["revision"]),
        "prompt_version": str(config["prompt"]["version"]),
        "test_used": False,
        "labels_visible": False,
        "environment": {
            "torch": str(torch.__version__),
            "transformers": str(__import__("transformers").__version__),
            "cuda_runtime": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "timing": {
            "model_load_seconds": round(model_load_seconds, 6),
            "generation_seconds": round(generation_seconds, 6),
            "mean_generation_seconds_per_query": round(
                generation_seconds / len(prepared), 6
            ) if prepared else 0.0,
            "total_run_seconds": round(time.perf_counter() - run_started, 6),
        },
        "cuda_peak_memory_mb": round(
            torch.cuda.max_memory_allocated() / 1024**2, 2
        ) if torch.cuda.is_available() else 0.0,
        "generation_sha256": hashlib.sha256(
            "\n".join(str(row["raw_generation"]) for row in all_rows).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constrained local-LLM evidence selection from fixed BM25 candidates."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Override the configured retrieval JSONL while preserving its hash in the manifest.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Override the configured JSONL output, for example for an isolated smoke test.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Override the configured manifest path together with --output-path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        args.config,
        resume=args.resume,
        limit=args.limit,
        output_path_override=args.output_path,
        manifest_path_override=args.manifest_path,
        input_path_override=args.input_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
