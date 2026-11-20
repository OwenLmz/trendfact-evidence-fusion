from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retrieval.bm25 import tokenize_character_bigram  # noqa: E402
from retrieval.evidence import (  # noqa: E402
    file_sha256,
    iter_jsonl,
    normalize_for_search,
    text_sha256,
)
from retrieval.rerank_evidence_with_llm import (  # noqa: E402
    ALLOWED_STANCES,
    RerankError,
    _torch_dtype,
    extract_json_object,
)


DEFAULT_CONFIG = ROOT / "configs" / "llm_evidence_pointwise_qwen25_7b.yaml"
STANCE_PRIORITY = {"support": 0, "refute": 0, "context": 1, "unknown": 2}


def select_query_focused_excerpt(
    query_text: str,
    evidence_text: str,
    *,
    max_chars: int,
    overlap_chars: int = 200,
) -> tuple[str, int]:
    text = str(evidence_text).strip()
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")
    if len(text) <= max_chars:
        return text, 0
    query_tokens = set(tokenize_character_bigram(normalize_for_search(query_text)))
    step = max_chars - overlap_chars
    starts = list(range(0, max(1, len(text) - max_chars + 1), step))
    final_start = len(text) - max_chars
    if starts[-1] != final_start:
        starts.append(final_start)
    best_start = 0
    best_score = -1
    for start in starts:
        window = text[start : start + max_chars]
        window_tokens = Counter(
            tokenize_character_bigram(normalize_for_search(window))
        )
        score = sum(window_tokens[token] for token in query_tokens)
        if score > best_score:
            best_score = score
            best_start = start
    return text[best_start : best_start + max_chars], best_start


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _chunks(values: Sequence[dict[str, object]], size: int) -> Iterable[list[dict[str, object]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def build_pairwise_messages(
    query_text: str, hit: Mapping[str, object]
) -> list[dict[str, str]]:
    system = (
        "你是严格的中文事实核查证据相关性判定器。每次只判断给定的一条候选证据，"
        "不得生成新证据，不得判断主张或新闻的最终真假，也不得因为来源权威就假定内容相关。"
        "先核对主体、核心行为或事件，再核对时间、地点、数字、法律条件、因果等限定。"
        "3=候选直接验证或反驳同一完整事实；2=同一事实但缺少一个关键限定；"
        "1=仅主题、实体或背景相关；0=无关。"
        "support或refute只能用于直接涉及核心事实的候选；一般背景必须用context或unknown。"
        "同一法规的其他条款、不同地区的同类事件、一般医学建议，都不能算直接证据。"
    )
    candidate = {
        "source": str(hit.get("source_dataset", "trendfact_evidence_library")),
        "title": str(hit.get("title") or hit.get("publish_date") or ""),
        "text": str(hit["text"]),
    }
    user = (
        f"待核查主张：\n{query_text}\n\n"
        f"唯一候选证据：\n{json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "只输出一个JSON对象，不要Markdown或额外文字："
        '{"relevance":0,"stance":"unknown","reason":"不超过40个汉字"}'
    )
    examples = [
        (
            "待核查主张：\n甲市疾控中心通报7月1日新增30例病例。\n\n"
            "唯一候选证据：\n"
            '{"source":"official","title":"疫情通报","text":"甲市疾控中心7月1日发布通报，当日新增确诊病例30例。"}\n\n'
            "只输出一个JSON对象，不要Markdown或额外文字："
            '{"relevance":0,"stance":"unknown","reason":"不超过40个汉字"}',
            '{"relevance":3,"stance":"support","reason":"主体、日期和新增数字均一致"}',
        ),
        (
            "待核查主张：\n甲市7月1日在A市场为2000人进行咽拭子采样。\n\n"
            "唯一候选证据：\n"
            '{"source":"news","title":"核酸采样","text":"甲市在A市场组织工作人员进行咽拭子采样。"}\n\n'
            "只输出一个JSON对象，不要Markdown或额外文字："
            '{"relevance":0,"stance":"unknown","reason":"不超过40个汉字"}',
            '{"relevance":2,"stance":"support","reason":"同一采样事件但缺少日期和人数"}',
        ),
        (
            "待核查主张：\n拒绝隔离并造成传播将承担刑事责任。\n\n"
            "唯一候选证据：\n"
            '{"source":"guideline","title":"隔离建议","text":"疑似病例应在定点医院接受隔离治疗。"}\n\n'
            "只输出一个JSON对象，不要Markdown或额外文字："
            '{"relevance":0,"stance":"unknown","reason":"不超过40个汉字"}',
            '{"relevance":1,"stance":"context","reason":"仅说明隔离要求，未涉及传播和刑责"}',
        ),
    ]
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for example_user, example_assistant in examples:
        messages.extend(
            [
                {"role": "user", "content": example_user},
                {"role": "assistant", "content": example_assistant},
            ]
        )
    messages.append({"role": "user", "content": user})
    return messages


def validate_decision(
    payload: Mapping[str, object],
    *,
    allowed_scores: set[int],
    reason_max_chars: int,
) -> dict[str, object]:
    try:
        relevance = int(payload.get("relevance"))
    except (TypeError, ValueError) as exc:
        raise RerankError("Invalid relevance") from exc
    if relevance not in allowed_scores:
        raise RerankError(f"Relevance out of range: {relevance}")
    stance = str(payload.get("stance", "")).casefold()
    if stance not in ALLOWED_STANCES:
        raise RerankError(f"Invalid stance: {stance}")
    stance_adjusted_from = ""
    if relevance < 2 and stance in {"support", "refute"}:
        stance_adjusted_from = stance
        stance = "unknown" if relevance == 0 else "context"
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise RerankError("Missing reason")
    if len(reason) > reason_max_chars:
        raise RerankError(f"Reason exceeds {reason_max_chars} characters")
    return {
        "relevance": relevance,
        "stance": stance,
        "reason": reason,
        "stance_adjusted_from": stance_adjusted_from,
    }


def rank_candidate_decisions(
    decisions: Sequence[Mapping[str, object]], *, selected_top_k: int
) -> list[dict[str, object]]:
    ordered = sorted(
        (dict(item) for item in decisions),
        key=lambda item: (
            -int(item.get("relevance", -1)),
            STANCE_PRIORITY.get(str(item.get("stance", "unknown")), 3),
            int(item["candidate_rank"]),
        ),
    )
    selected = ordered[:selected_top_k]
    for rank, item in enumerate(selected, start=1):
        item["llm_rank"] = rank
    return selected


def _completed_query_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row["query_id"]) for row in iter_jsonl([path])}


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    resume: bool = False,
    limit: int | None = None,
    input_path_override: Path | None = None,
    output_path_override: Path | None = None,
    manifest_path_override: Path | None = None,
    allow_test: bool = False,
) -> dict[str, object]:
    run_started = time.perf_counter()
    config_path = _resolve(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Pointwise LLM config must contain a mapping")
    is_test = str(config["input"]["split"]) == "test"
    if is_test and not allow_test:
        raise ValueError("Test scoring requires --allow-final-test")
    if is_test and limit is not None:
        raise ValueError("Final-test scoring may not use --limit")
    if bool(config["input"]["labels_visible"]):
        raise ValueError("Target labels must remain hidden from the LLM scorer")
    protocol = config["protocol"]
    if any(
        bool(protocol[key])
        for key in ("may_generate_evidence", "may_use_target_label", "may_predict_final_class")
    ):
        raise ValueError("The constrained pointwise protocol was weakened in config")

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
    max_candidate_chars = int(config["prompt"].get("max_candidate_chars", 0))
    excerpt_overlap_chars = int(config["prompt"].get("excerpt_overlap_chars", 0))

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
    pair_batch_size = int(inference["pair_batch_size"])
    query_batch_size = int(inference["query_batch_size"])
    errors = 0
    pair_count = 0
    generation_started = time.perf_counter()
    mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for query_batch in _chunks(rows, query_batch_size):
            prepared: list[dict[str, object]] = []
            hits_by_query: dict[str, list[Mapping[str, object]]] = {}
            for row in query_batch:
                query_id = str(row["query_id"])
                hits = list(row.get("hits", []))[:candidate_top_k]
                if len(hits) < selected_top_k:
                    raise RerankError(f"Not enough candidates for {query_id}")
                hits_by_query[query_id] = hits
                for index, hit in enumerate(hits, start=1):
                    if max_candidate_chars:
                        excerpt, excerpt_start = select_query_focused_excerpt(
                            str(row["query_text"]),
                            str(hit["text"]),
                            max_chars=max_candidate_chars,
                            overlap_chars=excerpt_overlap_chars,
                        )
                    else:
                        excerpt = str(hit["text"])
                        excerpt_start = 0
                    prompt_hit = {**dict(hit), "text": excerpt}
                    prepared.append(
                        {
                            "query_id": query_id,
                            "candidate_id": f"E{index:02d}",
                            "candidate_rank": int(hit["rank"]),
                            "hit": hit,
                            "messages": build_pairwise_messages(
                                str(row["query_text"]), prompt_hit
                            ),
                            "prompt_excerpt": excerpt,
                            "prompt_excerpt_start": excerpt_start,
                            "candidate_text_chars": len(str(hit["text"])),
                        }
                    )

            decisions_by_query: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
            for pair_batch in _chunks(prepared, pair_batch_size):
                prompts = [
                    tokenizer.apply_chat_template(
                        item["messages"], tokenize=False, add_generation_prompt=True
                    )
                    for item in pair_batch
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
                width = encoded["input_ids"].shape[1]
                decoded = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
                for item, raw_text in zip(pair_batch, decoded, strict=True):
                    pair_count += 1
                    try:
                        decision = validate_decision(
                            extract_json_object(raw_text),
                            allowed_scores=allowed_scores,
                            reason_max_chars=reason_max_chars,
                        )
                        status = "ok"
                        error = ""
                    except RerankError as exc:
                        errors += 1
                        decision = {"relevance": -1, "stance": "unknown", "reason": ""}
                        status = "parse_error"
                        error = str(exc)
                    decisions_by_query[str(item["query_id"])].append(
                        {
                            **decision,
                            "candidate_id": item["candidate_id"],
                            "candidate_rank": item["candidate_rank"],
                            "status": status,
                            "error": error,
                            "raw_generation": raw_text,
                            "prompt_excerpt": item["prompt_excerpt"],
                            "prompt_excerpt_sha256": text_sha256(
                                str(item["prompt_excerpt"])
                            ),
                            "prompt_excerpt_start": item["prompt_excerpt_start"],
                            "candidate_text_chars": item["candidate_text_chars"],
                            "candidate_truncated": len(str(item["prompt_excerpt"]))
                            < int(item["candidate_text_chars"]),
                        }
                    )

            row_by_id = {str(row["query_id"]): row for row in query_batch}
            for query_id in [str(row["query_id"]) for row in query_batch]:
                source_row = row_by_id[query_id]
                decisions = decisions_by_query[query_id]
                hit_map = {
                    f"E{index:02d}": hit
                    for index, hit in enumerate(hits_by_query[query_id], start=1)
                }
                ranked = rank_candidate_decisions(decisions, selected_top_k=selected_top_k)
                selected = [
                    {**decision, **dict(hit_map[str(decision["candidate_id"])])}
                    for decision in ranked
                ]
                output_row = {
                    "query_id": query_id,
                    "document_id": source_row["document_id"],
                    "claim_rank": source_row.get("claim_rank"),
                    "query_text": source_row["query_text"],
                    "input_strategy": source_row.get(
                        "strategy", source_row.get("retrieval_method", "unknown")
                    ),
                    "candidate_top_k": candidate_top_k,
                    "selected_top_k": selected_top_k,
                    "status": "ok" if all(x["status"] == "ok" for x in decisions) else "partial_error",
                    "candidate_decisions": decisions,
                    "selected": selected,
                    "prompt_version": str(config["prompt"]["version"]),
                    "model": str(model_config["name_or_path"]),
                    "model_revision": str(model_config["revision"]),
                }
                handle.write(
                    json.dumps(output_row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "input_file": input_path.relative_to(ROOT).as_posix(),
        "input_sha256": file_sha256(input_path),
        "output_file": output_path.relative_to(ROOT).as_posix(),
        "output_sha256": file_sha256(output_path),
        "rows": len(all_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "rows_processed_this_run": len(rows),
        "pairs_processed_this_run": pair_count,
        "pair_errors_this_run": errors,
        "model": str(model_config["name_or_path"]),
        "model_revision": str(model_config["revision"]),
        "prompt_version": str(config["prompt"]["version"]),
        "test_used": is_test,
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
            "mean_generation_seconds_per_pair": round(
                generation_seconds / pair_count, 6
            ) if pair_count else 0.0,
            "total_run_seconds": round(time.perf_counter() - run_started, 6),
        },
        "cuda_peak_memory_mb": round(
            torch.cuda.max_memory_allocated() / 1024**2, 2
        ) if torch.cuda.is_available() else 0.0,
        "generation_sha256": hashlib.sha256(
            "\n".join(
                str(decision["raw_generation"])
                for row in all_rows
                for decision in row["candidate_decisions"]
            ).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently score fixed evidence candidates with a local LLM."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--allow-final-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(
        args.config,
        resume=args.resume,
        limit=args.limit,
        input_path_override=args.input_path,
        output_path_override=args.output_path,
        manifest_path_override=args.manifest_path,
        allow_test=args.allow_final_test,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
