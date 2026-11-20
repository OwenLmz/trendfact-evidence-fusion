#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

predict_args=()
case "${1:-}" in
  frozen)
    python scripts/verify_release.py --raw --splits --checkpoints --training-cache
    ;;
  rebuilt)
    python scripts/verify_release.py --raw --splits --rebuilt
    predict_args=(--retrained)
    ;;
  *)
    printf 'Usage: bash scripts/reproduce_final.sh {frozen|rebuilt}\n' >&2
    exit 2
    ;;
esac

gate_dir=outputs/reranking/trendfact/candidate_reliability_gate
mkdir -p "$gate_dir"
if [[ ! -e "$gate_dir/model.json" ]]; then
  cp assets/candidate_gate.json "$gate_dir/model.json"
fi
python src/evaluation/validate_trendfact_final_freeze.py
python src/evaluation/prepare_trendfact_final_test.py

final=outputs/final/trendfact_one_shot_v1
python src/retrieval/run_trendfact_bm25_pilot.py \
  --queries-path "$final/data/test_queries.jsonl" \
  --output-dir "$final/retrieval_bm25" --stored-top-k 20 \
  --report-at-k 1 3 5 10 20 --allow-final-test
CUDA_VISIBLE_DEVICES=0 python src/retrieval/score_evidence_with_llm.py \
  --config configs/final_test_trendfact_qwen.yaml --allow-final-test
for seed in 13 42 2026; do
  python src/baselines/predict_trendfact_frozen_roberta.py --seed "$seed" \
    --queries-path "$final/data/test_queries.jsonl" \
    --output-dir "$final/roberta_seed$seed" --device 0 "${predict_args[@]}"
done
python src/evaluation/evaluate_trendfact_evidence_aggregator.py \
  --config configs/final_test_trendfact_evidence_aggregator.yaml \
  --evaluation-split test --output-dir "$final/evidence_aggregator" --allow-final-test
python src/evaluation/evaluate_trendfact_final_fusion.py \
  --roberta-path "$final/roberta_seed13/test_predictions.jsonl" \
  --roberta-path "$final/roberta_seed42/test_predictions.jsonl" \
  --roberta-path "$final/roberta_seed2026/test_predictions.jsonl" \
  --evidence-path "$final/evidence_aggregator/predictions.jsonl" \
  --output-dir "$final/final_fusion"
python src/evaluation/analyze_trendfact_dev_errors.py \
  --config configs/final_test_trendfact_error_analysis.yaml \
  --output-dir "$final/error_analysis" --allow-final-test
python src/evaluation/finalize_trendfact_final_test.py

selection=outputs/evaluation/trendfact_roberta_evidence_ablations
mkdir -p "$selection"
if [[ ! -e "$selection/report.json" ]]; then
  cp assets/dev_ablation_selection.json "$selection/report.json"
fi
python src/evaluation/evaluate_trendfact_final_ablations.py \
  --output-dir outputs/evaluation/trendfact_final_preselected_ablations
