# Reproduction

Run commands from this repository root. Python 3.10 is recommended. The
original run used Python 3.10.12, PyTorch 2.7.1+cu128, Transformers 4.48.3,
NumPy 2.2.6, scikit-learn 1.7.2, and PyArrow 25.0.0. The requirements files
pin the direct packages checked in the retained server environment.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_release.py
```

For training or Qwen scoring, install PyTorch 2.7.1 for the target CUDA runtime
(cu128 for the original RTX 5090 environment), then install
`requirements-gpu.txt`. BM25 construction also requires sufficient host RAM.
The original server had a 32 GiB memory limit. The CPU tests use small fixtures.
The verification script also recovers Accuracy and Macro-F1 for the three
reported branches from published aggregate confusion matrices.

## Data and models

Acquire `TrendFact.json` and `evidence.parquet` from the
[TrendFact authors' dataset](https://huggingface.co/datasets/zxc123cc/TrendFact)
(CC BY 4.0), placing them under `data/raw/TrendFact/`:

```bash
mkdir -p data/raw/TrendFact
curl --fail --location --output data/raw/TrendFact/TrendFact.json \
  https://huggingface.co/datasets/zxc123cc/TrendFact/resolve/main/TrendFact.json
curl --fail --location --output data/raw/TrendFact/evidence.parquet \
  https://huggingface.co/datasets/zxc123cc/TrendFact/resolve/main/evidence.parquet
python scripts/verify_release.py --raw
python src/data/prepare_trendfact.py
```

The SHA-256 values in `configs/data_trendfact.yaml` identify the exact inputs.
If a download differs, recover the original snapshot before proceeding.
The model configurations pin the upstream Hugging Face revisions of
Qwen2.5-7B-Instruct and Chinese RoBERTa. Follow each model card's use and
redistribution terms. This repository does not set a license for third-party
texts or weights.

## Train and development

The commands below preserve the original selection protocol. The deterministic
data build also creates the test split for integrity checks. The training
commands consume train and development queries only.

```bash
python src/retrieval/run_trendfact_bm25_pilot.py
python src/retrieval/run_trendfact_bm25_pilot.py \
  --queries-path data/processed/trendfact_v1/gate_train_queries.jsonl \
  --output-dir outputs/retrieval/trendfact_bm25_gate_train
python src/retrieval/run_trendfact_bm25_pilot.py \
  --queries-path data/processed/trendfact_v1/heldout_dev_queries.jsonl \
  --output-dir outputs/retrieval/trendfact_bm25_heldout_dev \
  --stored-top-k 20 --report-at-k 1 3 5 10 20
for suffix in qwen25_7b gate_train_qwen25_7b heldout_dev_qwen25_7b; do
  CUDA_VISIBLE_DEVICES=0 python src/retrieval/score_evidence_with_llm.py \
    --config "configs/llm_evidence_pointwise_trendfact_${suffix}.yaml"
done
python src/evaluation/train_trendfact_candidate_gate.py \
  --config configs/trendfact_candidate_reliability_gate.yaml
for split in pilot heldout_dev; do
  python src/evaluation/evaluate_trendfact_evidence_aggregator.py \
    --evaluation-split "$split" \
    --output-dir "outputs/evaluation/trendfact_evidence_aggregator/$split"
done
for seed in 13 42 2026; do
  CUDA_VISIBLE_DEVICES=0 python src/baselines/trendfact_transformer_baseline.py \
    --config "configs/trendfact_claim_roberta_seed${seed}.yaml" \
    --output-dir "outputs/baselines/trendfact_claim_roberta_seed${seed}"
done
python src/evaluation/evaluate_trendfact_roberta_evidence_ensemble.py \
  --output-dir outputs/evaluation/trendfact_roberta_evidence_ensemble
python src/evaluation/evaluate_trendfact_roberta_evidence_ablations.py \
  --output-dir outputs/evaluation/trendfact_roberta_evidence_ablations
```

The original query-level out-of-fold training can place related claims in
different internal folds. The original 947-query development set also shared
some related groups with the 200-query pilot. Those issues were subsequently
audited. The released primary configuration preserves the originally reported
method and evaluation history. Retraining can yield different model bytes and
metrics, particularly across devices or library builds.

## Original frozen evaluation

The project already completed its one-shot test. The original results and
recorded file hashes are retained in `results/frozen_test.json`. This release
does not perform additional test inference during its checks. The historical
sequence below is provided for independent reproduction in a fresh checkout.
For exact checkpoint replay, obtain the three original checkpoints through
the project maintainer and verify the paths, sizes and hashes declared in
`configs/final_test_trendfact.yaml`. Place the bundled Gate at its historical
path in a fresh replay workspace and retain the original train-stage candidate
and Qwen artifacts for the evidence aggregator. Rebuilt caches must be treated
as regenerated inputs unless their hashes match the original artifacts.

```bash
mkdir -p outputs/reranking/trendfact/candidate_reliability_gate
cp -n assets/candidate_gate.json outputs/reranking/trendfact/candidate_reliability_gate/model.json
python scripts/verify_release.py --raw --checkpoints --training-cache
python src/evaluation/validate_trendfact_final_freeze.py
python src/evaluation/prepare_trendfact_final_test.py
final=outputs/final/trendfact_one_shot_v1
python src/retrieval/run_trendfact_bm25_pilot.py \
  --queries-path "$final/data/test_queries.jsonl" \
  --output-dir "$final/retrieval_bm25" --stored-top-k 20 --allow-final-test
python src/retrieval/score_evidence_with_llm.py \
  --config configs/final_test_trendfact_qwen.yaml --allow-final-test
for seed in 13 42 2026; do
  python src/baselines/predict_trendfact_frozen_roberta.py --seed "$seed" \
    --queries-path "$final/data/test_queries.jsonl" \
    --output-dir "$final/roberta_seed$seed" --device 0
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
mkdir -p outputs/evaluation/trendfact_roberta_evidence_ablations
cp -n assets/dev_ablation_selection.json outputs/evaluation/trendfact_roberta_evidence_ablations/report.json
python src/evaluation/evaluate_trendfact_final_ablations.py \
  --output-dir outputs/evaluation/trendfact_final_preselected_ablations
```

The last ablation command requires the byte-identical original development
selection report at the path pinned by its configuration. In a workspace
with regenerated development results, preserve them separately before
restoring the original report from `assets/dev_ablation_selection.json`.
The final configuration's pre-run status is part of its archived bytes.
The completion manifest in `results/` records the completed historical run.
