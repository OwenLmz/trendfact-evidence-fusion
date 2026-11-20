# Reproduction

Run commands from the repository root. Sections 2 and 3 use Linux with Bash,
a CUDA GPU and sufficient memory for Qwen2.5-7B. The original environment was
Python 3.10.12, PyTorch 2.7.1+cu128, Transformers 4.48.3, NumPy 2.2.6,
scikit-learn 1.7.2 and PyArrow 25.0.0. The original BM25 environment had a
32 GiB host memory limit.

## 1 Check the repository on CPU

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_release.py
```

The test suite uses small fixtures. The verification script checks archived
configurations and the model scoring source, and recovers Accuracy and Macro-F1
from the original aggregate confusion matrices. This verification requires no
dataset download or GPU.

## 2 Install GPU dependencies and prepare the data

For the original CUDA 12.8 runtime:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
```

Download the exact [TrendFact snapshot](https://huggingface.co/datasets/zxc123cc/TrendFact/tree/f0838d22138f341189602fb719998787875985fe)
from the dataset authors (CC BY 4.0). The SHA-256 values in
`configs/data_trendfact.yaml` identify the exact inputs.

```bash
mkdir -p data/raw/TrendFact
base=https://huggingface.co/datasets/zxc123cc/TrendFact/resolve/f0838d22138f341189602fb719998787875985fe
curl --fail --location --output data/raw/TrendFact/TrendFact.json "$base/TrendFact.json"
curl --fail --location --output data/raw/TrendFact/evidence.parquet "$base/evidence.parquet"
python scripts/verify_release.py --raw
python src/data/prepare_trendfact.py
python scripts/verify_release.py --splits
```

The data builder creates six byte-checked split and query files. Inference and
training code use the exact Hugging Face model revisions in the configuration
files. Follow the original publishers' model and dataset terms.

## 3 Rebuild from public inputs

Run the following in a fresh work directory after section 2. The release
includes the original Gate's numerical parameters. This route regenerates the
training evidence cache and retrains the three RoBERTa checkpoints on the
documented training split. RoBERTa epoch selection uses the fixed 200-query
pilot. The final fusion uses the archived 0.5 weight.

```bash
python src/retrieval/run_trendfact_bm25_pilot.py \
  --queries-path data/processed/trendfact_v1/gate_train_queries.jsonl \
  --output-dir outputs/retrieval/trendfact_bm25_gate_train
CUDA_VISIBLE_DEVICES=0 python src/retrieval/score_evidence_with_llm.py \
  --config configs/llm_evidence_pointwise_trendfact_gate_train_qwen25_7b.yaml
for seed in 13 42 2026; do
  CUDA_VISIBLE_DEVICES=0 python src/baselines/trendfact_transformer_baseline.py \
    --config "configs/trendfact_claim_roberta_seed${seed}.yaml" \
    --output-dir "outputs/baselines/trendfact_claim_roberta_seed${seed}"
done
bash scripts/reproduce_final.sh rebuilt
```

The evaluator validates each rebuilt checkpoint against its training report,
records checkpoint provenance, and writes a new completion manifest under
`outputs/final/trendfact_one_shot_v1`. Rebuilt model bytes, generations and
metrics can vary across environments. Compare the new results with the
published historical values in `results/frozen_test.json`.

## 4 Replay with the original artifacts

Use a separate fresh workspace. After section 2, place the three original
checkpoints and original evidence training artifacts at the paths declared in
`configs/final_test_trendfact.yaml` and `assets/frozen_inputs.json`. These
files can be obtained from the project maintainer, subject to availability
and applicable terms. The two public dataset files come from section 2.

```bash
bash scripts/reproduce_final.sh frozen
```

The frozen mode enforces the recorded checkpoint sizes and SHA-256 values
before running. Both modes verify the archived Gate and refuse to overwrite
an existing final output directory. The archived selection report supplies
the original development-selected weights for the component ablations.
This workflow is for independent reproduction in a fresh checkout.

## 5 Verify the candidate Gate training recipe

For a component check in its own workspace, first prepare the data and
generate the training evidence files as in sections 2 and 3. Then run:

```bash
python src/retrieval/run_trendfact_bm25_pilot.py
CUDA_VISIBLE_DEVICES=0 python src/retrieval/score_evidence_with_llm.py \
  --config configs/llm_evidence_pointwise_trendfact_qwen25_7b.yaml
python src/evaluation/train_trendfact_candidate_gate.py \
  --config configs/trendfact_candidate_reliability_gate.yaml
```

Compare its generated `model.json` with `assets/candidate_gate.json`. The
original query-level out-of-fold training can place related claims in
different folds. Related groups also intersected the pilot and the original
947-query development set. These are limitations of the archived protocol.
