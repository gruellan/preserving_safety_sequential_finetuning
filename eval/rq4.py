"""RQ4: refusal-direction geometry + WildGuard directional-ablation

Measures per condition (order * replay method):
  * Iterate cumulative adapter chain (base -> merge m1 -> merge m2 -> merge m3)
    like eval/eval_all_checkpoints.py
  * Geometric: last-token diff-in-means refusal direction at the best separating layer
    (selected on M0), per checkpoint -> magnitude (erosion axis), cosine-to-base
    (rotation axis) and  bootstrap CI on cos-to-base by resampling cached rows -
    Activations are cached to npz.
  * Ablation: generate intact / ablate-M0-dir / ablate-own-dir completions on
    held-out harmful prompts. WildGuard then scores each as refusal -> fraction of 
    refusal removed by the base direction vs the checkpoint's OWN direction 
    (the erosion/rotation/fragmentation matrix).

Run:
export HF_HOME=/dev/shm/hf HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
/venv/bin/python -m eval.rq4 --phase all
"""

import argparse
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.generation import format_prompt, generate_responses
from eval.robustness_data import (
    load_advbench,
    load_orbench_hard,
    load_orbench_toxic,
    load_xstest,
)
from model_utils import attn_for

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rq4")

HARMFUL_LOADERS = {"advbench": load_advbench, "orbench_toxic": load_orbench_toxic}
HARMLESS_LOADERS = {"orbench_hard": load_orbench_hard, "xstest": load_xstest}

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTERS_ROOT = os.environ.get("ADAPTERS_ROOT", "/workspace/dissertation_outputs")
RESULTS = os.environ.get("RQ4_RESULTS", "results")
COMPLETIONS_DIR = f"{RESULTS}/rq4_wildguard"

ORDER_TASKS = {
    1: ["code", "sum", "qa"],
    2: ["qa", "sum", "code"],
    3: ["sum", "code", "qa"],
}
SUFFIXES = ["", "_er_b150", "_der_b150", "_derpp_b150"]


def all_conditions(orders, name_suffix=""):
    return [
        (f"order_{o}{suffix}{name_suffix}", ORDER_TASKS[o])
        for o in orders
        for suffix in SUFFIXES
    ]


def load_prompts(harmful_key: str, harmless_key: str, max_prompts: int, seed: int):
    harmful = [r["prompt"] for r in HARMFUL_LOADERS[harmful_key](max_prompts, seed)]
    harmless_raw = HARMLESS_LOADERS[harmless_key](max_prompts, seed)

    # xstest has unsafe contrast prompts so just keep  the genuinely safe ones
    harmless = [r["prompt"] for r in harmless_raw if r.get("is_safe", True)]

    n = min(len(harmful), len(harmless))

    # ablation prompts come from beyond the extraction cut so the direction is
    # never evaluated on prompts it was fit on
    heldout = harmful[n:]

    # equal harmful harmless prompts
    harmful, harmless = (
        harmful[:n],
        harmless[:n],
    )
    logger.info(
        f"Prompts: {len(harmful)} harmful / {len(harmless)} harmless "
        f"({harmful_key} vs {harmless_key}), {len(heldout)} held-out for ablation"
    )
    return harmful, harmless, heldout


@torch.no_grad()
def last_token_all_layers(model, tokenizer, prompts, device, batch_size=16, pos=-1):
    """Return residual stream at every layer for token `pos` (from the end).

    prompts are chat templated here so pos counts back over the template's
    post-instruction tokens: -1 = last token Mistral, -5 on Llama-3 = <|eot_id|> right
    after the user content (Arditi et al. us this position for same model)
    """
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted = [format_prompt(p, tokenizer) for p in prompts]
    per_layer = None
    for i in range(0, len(formatted), batch_size):
        # add_special_tokens=False: format_prompt already includes <s>
        enc = tokenizer(
            formatted[i : i + batch_size],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        out = model(**enc, output_hidden_states=True)
        # last real token from the end (right pad)
        idx = enc.attention_mask.sum(1) + pos

        ar = torch.arange(enc.input_ids.size(0))

        # hidden_states: tuple len n_layers+1, [0]=embeddings, [L]=block L output.
        layers = [h[ar, idx].float().cpu() for h in out.hidden_states]

        if per_layer is None:
            per_layer = [[] for _ in layers]

        for li, h in enumerate(layers):
            per_layer[li].append(h)

    return torch.stack([torch.cat(c, 0) for c in per_layer], 0).numpy()  # (Lp1, N, d)


def diff_in_means(harmful, harmless):
    """returns signed (unit_direction, magnitude)"""
    d = harmful.mean(0) - harmless.mean(0)
    mag = float(np.linalg.norm(d))
    if mag == 0.0:  # e.g. embedding layer: last token identical across prompts
        return d, 0.0
    return d / mag, mag


def select_layer(feats_harmful, feats_harmless):
    """Pick layer whose DIM projection best separates the two sets  via AUC"""
    best_layer, best_auc = None, -1.0
    n_layers = feats_harmful.shape[0]

    for layer in range(1, n_layers):  # skip embeddings (index 0)
        direction, _ = diff_in_means(feats_harmful[layer], feats_harmless[layer])

        proj_h = feats_harmful[layer] @ direction
        proj_s = feats_harmless[layer] @ direction

        y_true = np.r_[np.ones(len(proj_h)), np.zeros(len(proj_s))]

        auc = float(roc_auc_score(y_true, np.r_[proj_h, proj_s]))
        if auc > best_auc:
            best_auc, best_layer = auc, layer

    logger.info(f"Selected layer {best_layer} (harmful vs harmless AUC={best_auc})")
    return best_layer, best_auc


def cos_to_base_confidence_interval(harmful, harmless, base_dir, n_boot, rng):
    """95% confidence interval on cosine-to-base

    The direction is a DIM estimated from a small prompt sample so its cosine-to-base is uncertain
    """
    cosines = []
    num_harm, num_safe = len(harmful), len(harmless)
    for _ in range(n_boot):
        direction, _ = diff_in_means(
            harmful[rng.integers(0, num_harm, num_harm)],
            harmless[rng.integers(0, num_safe, num_safe)],
        )
        cosines.append(float(direction @ base_dir))
    return {
        "cos_ci": [
            float(np.percentile(cosines, 2.5)),
            float(np.percentile(cosines, 97.5)),
        ]
    }


def _ablation_hooks(model, direction):
    """Project `direction` (unit, np) out of every write to residual stream"""
    direction = torch.tensor(
        direction,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )

    def hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs - (hs @ direction).unsqueeze(-1) * direction
        return (hs, *out[1:]) if isinstance(out, tuple) else hs

    # copy Arditi protocol: remove the direction from every write to the residual stream
    # = block output, attention output, MLP output and the embedding
    handles = []
    for layer in model.model.layers:
        handles.append(layer.register_forward_hook(hook))  # residual (block output)
        handles.append(layer.self_attn.register_forward_hook(hook))
        handles.append(layer.mlp.register_forward_hook(hook))
    handles.append(model.model.embed_tokens.register_forward_hook(hook))
    return handles


def generate_ablated(
    model, tokenizer, prompts, direction, max_new_tokens, batch_size, seed
):
    """Generate completions, optional  `direction` ablated from the residual stream"""
    tokenizer.padding_side = (
        "left"  # decoder-gen needs LEFT pad, extraction leaves it "right"
    )
    handles = _ablation_hooks(model, direction) if direction is not None else []
    try:
        return generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            seed=seed,
        )
    finally:
        for h in handles:
            h.remove()


def metrics_path(cond):
    return f"{RESULTS}/{cond}/rq4/rq4_metrics.json"


def completions_path(cond):
    return f"{COMPLETIONS_DIR}/{cond}.jsonl"


def run_gen(conditions, args, rng):
    harmful_p, harmless_p, heldout_p = load_prompts(
        args.harmful, args.harmless, args.max_prompts, args.seed
    )
    sub = heldout_p[: args.ablate_n]  # ablation is very slow so use subset

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    Path(COMPLETIONS_DIR).mkdir(parents=True, exist_ok=True)

    for cond, tasks in conditions:
        cache_dir = Path(f"{RESULTS}/{cond}/rq4_cache")
        out_dir = Path(f"{RESULTS}/{cond}/rq4")
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = [("M0_base", None)] + [
            (f"M{i}_after_{t}", f"{args.adapters_root}/{cond}/{t}")
            for i, t in enumerate(tasks, 1)
        ]

        logger.info(f"======= {cond}: loading base model ======")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_for(args.base_model),
        ).to("cuda")
        model.eval()
        device = next(model.parameters()).device

        base_dir = None
        selected_layer = args.layer
        rows, records = [], []

        for checkpoint_name, adapter in checkpoints:
            if adapter is not None:
                logger.info(f"Merging adapter: {adapter}")
                model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
                model.eval()
                gc.collect()
                torch.cuda.empty_cache()

            # geometry
            feats_h = last_token_all_layers(
                model, tokenizer, harmful_p, device, pos=args.pos
            )
            feats_s = last_token_all_layers(
                model, tokenizer, harmless_p, device, pos=args.pos
            )

            if selected_layer is None:  # only at base (M0 is always the same)
                selected_layer, _ = select_layer(feats_h, feats_s)

            # per layer diff-in-means for the checkpoint x layer heatmap
            layer_dirs, layer_mag = [], []
            for L in range(feats_h.shape[0]):
                d, m = diff_in_means(feats_h[L], feats_s[L])
                layer_dirs.append(d)
                layer_mag.append(m)

            H_h, H_s = feats_h[selected_layer], feats_s[selected_layer]

            np.savez_compressed(
                cache_dir / f"{checkpoint_name}.npz",
                harmful=H_h,
                harmless=H_s,
                layer_mag=np.array(layer_mag),
                layer_dirs=np.array(layer_dirs),
            )

            direction, magnitude = diff_in_means(H_h, H_s)
            if base_dir is None:
                base_dir = direction

            cos_to_base = float(direction @ base_dir)

            cos_interval = cos_to_base_confidence_interval(
                H_h, H_s, base_dir, args.bootstrap, rng
            )
            rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "layer": int(selected_layer),
                    "magnitude": magnitude,
                    "cos_to_base": cos_to_base,
                    "n_harmful": len(H_h),
                    "n_harmless": len(H_s),
                    **cos_interval,
                }
            )
            logger.info(
                f"{checkpoint_name}: |d|={magnitude:.3f} cos2base={cos_to_base:.3f} (CI {cos_interval['cos_ci']})"
            )

            # ablation generation
            for var, abl in [
                ("intact", None),
                ("abl_m0", base_dir),
                ("abl_own", direction),
            ]:
                completions = generate_ablated(
                    model,
                    tokenizer,
                    sub,
                    abl,
                    args.max_new_tokens,
                    args.gen_batch,
                    args.seed,
                )
                for idx, (prompt, completion) in enumerate(zip(sub, completions)):
                    records.append(
                        {
                            "cond": cond,
                            "checkpoint": checkpoint_name,
                            "variant": var,
                            "idx": idx,
                            "prompt": prompt,
                            "completion": completion,
                        }
                    )
            gc.collect()
            torch.cuda.empty_cache()

        json.dump(
            {
                "order_name": cond,
                "base_model": args.base_model,
                "selected_layer": int(selected_layer),
                "pos": args.pos,
                "harmful": args.harmful,
                "harmless": args.harmless,
                "ablate_prompts": "heldout",
                "seed": args.seed,
                "checkpoints": rows,
            },
            open(metrics_path(cond), "w"),
            indent=2,
        )
        with open(completions_path(cond), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        logger.info(
            f"{cond}: wrote geometry -> {metrics_path(cond)} and "
            f"{len(records)} completions -> {completions_path(cond)}"
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_judge(conditions, args):
    from eval.refusal import classify_refusals

    cond_names = [c for c, _ in conditions]
    records = []
    for cond in cond_names:
        path = completions_path(cond)
        records.extend(json.loads(l) for l in open(path))
    if not records:
        logger.error("no completions to judge")
        return

    wildguard = "allenai/wildguard"

    logger.info(f"Judging WildGuard ({wildguard}) over {len(records)} completions")

    classifications = classify_refusals(
        [record["prompt"] for record in records],
        [record["completion"] for record in records],
        batch_size=args.wildguard_batch,
        model_name=wildguard,
    )
    for record, classification in zip(records, classifications):
        record["refusal"] = bool(classification)

    agg = defaultdict(lambda: defaultdict(list))
    for record in records:
        agg[(record["cond"], record["checkpoint"])][record["variant"]].append(
            record["refusal"]
        )

    for cond in sorted({record["cond"] for record in records}):
        meta = json.load(open(metrics_path(cond)))

        for checkpoint in meta["checkpoints"]:
            a = agg[(cond, checkpoint["checkpoint"])]
            if not a.get("intact"):
                continue

            intact = float(np.mean(a["intact"]))
            m0 = float(np.mean(a["abl_m0"]))
            own = float(np.mean(a["abl_own"]))
            removed = lambda after: (intact - after) / intact if intact > 0 else 0.0

            checkpoint["ablation_wildguard"] = {
                "refusal_intact": intact,
                "refusal_ablate_m0_dir": m0,
                "refusal_ablate_own_dir": own,
                "frac_removed_m0_dir": removed(
                    m0
                ),  # high = base direction keeps causal
                "frac_removed_own_dir": removed(
                    own
                ),  # falls down the chain = fragmentation
                "n": len(a["intact"]),
            }
        json.dump(meta, open(metrics_path(cond), "w"), indent=2)
        logger.info(f"wrote ablation_wildguard -> {cond}")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--phase", choices=["all", "gen", "judge"], default="all")
    args.add_argument("--orders", default="1,2,3", help="comma list of order numbers")
    args.add_argument(
        "--conditions",
        default=None,
        help="comma list of condition names, overrides --orders",
    )
    args.add_argument("--base-model", default=BASE_MODEL)
    args.add_argument("--adapters-root", default=ADAPTERS_ROOT)
    args.add_argument(
        "--harmful", choices=list(HARMFUL_LOADERS), default="orbench_toxic"
    )
    args.add_argument("--harmless", choices=list(HARMLESS_LOADERS), default="xstest")
    args.add_argument("--max-prompts", type=int, default=200)
    args.add_argument(
        "--layer", type=int, default=None, help="fix layer, else auto-select on base"
    )
    args.add_argument(
        "--pos",
        type=int,
        default=-1,
        help="extraction token position from end, Arditi used -5 for Llama-3-8B",
    )
    args.add_argument("--bootstrap", type=int, default=200)
    args.add_argument(
        "--ablate-n", type=int, default=60, help="harmful prompts used for ablation gen"
    )
    args.add_argument("--max-new-tokens", type=int, default=64)
    args.add_argument("--gen-batch", type=int, default=16)
    args.add_argument("--wildguard-batch", type=int, default=32)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--suffix", default="", help="e.g. _seed43")
    args = args.parse_args()

    orders = [int(order) for order in args.orders.split(",")]
    conditions = all_conditions(orders, args.suffix)

    if args.conditions:
        keep = set(args.conditions.split(","))
        conditions = [c for c in conditions if c[0] in keep]

    logger.info(f"conditions: {[c for c, _ in conditions]}")

    if args.phase in ("all", "gen"):
        run_gen(conditions, args, np.random.default_rng(args.seed))

    if args.phase in ("all", "judge"):
        run_judge(conditions, args)

    logger.info("REFUSAL TRACKING DONE DONE")


if __name__ == "__main__":
    main()
