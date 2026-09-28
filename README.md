# Preserving Safety During Sequential Fine-Tuning: A Factorial and Geometric Analysis of Replay

Code and results for the IEEE SaTML submission of the same name (anonymous review version).

We LoRA fine-tune instruction-tuned LLMs (Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct, Gemma-2-9B-IT) on three tasks in sequence (code, summarization, QA), producing model checkpoints M_0 (base) to M_3. We compare Plain fine-tuning, Matched Plain, ER (supervised replay), DER (logit distillation) and DER++ (both), plus post-hoc SafeLoRA. At each checkpoint we measure:
- **ASR**: harmful compliance (HarmBench, ex-copyright, n=240, plus other harmful-request sets)
- **ORR**: over-refusal on benign prompts (XSTest, OR-Bench-hard)
- **Task performance** on each downstream task
- **Refusal geometry**: the refusal direction in the residual stream and its ablation

> **Content warning.** `results*/` contain harmful prompts, model completions and judge outputs.

## Setup

Requires an NVIDIA GPU (we used an A100 80GB).

```
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
huggingface-cli download cais/HarmBench-Llama-2-13b-cls \
  --local-dir /workspace/models/HarmBench-Llama-2-13b-cls --local-dir-use-symlinks False
```

The full TRACE dataset is not on GitHub or HuggingFace. Download it from [this link](https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view) and point `config/experiment_config*.yaml` at it.

## Reproducing

**1. Replay buffer** (ER/DER/DER++/Matched Plain):
```
python data_processing/build_replay_buffer.py --output data/der_buffer_filtered.jsonl
python data_processing/cache_teacher_logits.py \
    --buffer-path data/der_buffer_filtered.jsonl \
    --output-dir /workspace/dissertation_outputs/teacher_logits_cache
```

**2. Train and evaluate** one run (~11 h on an A100):
```
bash scripts/run_experiment.sh <order_1..order_6> <mistral|llama|gemma> <plain|plain_matched|er|der|derpp>
```

**3. SafeLoRA** (applied post-hoc on saved Plain adapters): `python -m training.safelora --help`

**4. Figures and statistics** (offline, no GPU required):
```
python -m analysis.generate_figures     # all figures -> plots/
python -m analysis.stats_ci             # remaining in-text numbers -> analysis/stats.csv (AUROC needs rq4_cache)
```

## Layout

```
run.py, model_utils.py, results_io.py   # sequential fine-tuning entry point and helpers
config/          # per-model experiment configs
data_processing/ # dataset and replay-buffer builders
training/        # LoRA, ER/DER/DER++ trainer, SafeLoRA
eval/            # safety, over-refusal, task and refusal-geometry evaluation
analysis/        # offline statistics for the paper
scripts/         # run_experiment.sh
results/         # Mistral results
llama_results/   # Llama results
gemma_results/   # Gemma results
plots/           # paper figures
```

Each run directory is named `order_<N>[_<method>_b150][_seed<K>]`. The metric JSONs behind every table and figure are included. Activation caches (`**/rq4_cache/*.npz`) are git-ignored due to large file sizes and can be regenerated via `python -m eval.rq4`. Model weights and raw training inputs are not included due to size.

