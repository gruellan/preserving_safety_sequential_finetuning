import argparse
import json
import logging
import random
import sys
from pathlib import Path

from datasets import load_dataset
from transformers.utils import logging as hf_logging
from huggingface_hub.utils import logging as hf_hub_logging

from eval.generation import generate_responses
from eval.judge import DEFAULT_JUDGE, classify
from eval.utils import write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/safety_eval.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

hf_logging.set_verbosity_info()
hf_hub_logging.set_verbosity_info()


def _load_harmbench_behaviors(
    max_samples: int | None = None, seed: int = 42, drop_copyright: bool = False
) -> list[dict]:
    ds = load_dataset("swiss-ai/harmbench", "DirectRequest", split="test")

    records: list[dict] = []
    for idx, row in enumerate(ds):
        row = dict(row)
        records.append(
            {
                "id": row.get("BehaviorID") or f"harmbench_{idx:04d}",
                "prompt": row.get("Behavior") or row.get("prompt") or "",
                "category": row.get("SemanticCategory"),
            }
        )
    if drop_copyright:
        records = [
            r for r in records if str(r.get("category") or "").lower() != "copyright"
        ]
    if max_samples is not None and max_samples < len(records):
        records = random.Random(seed).sample(records, max_samples)
    logger.info(f"Loaded {len(records)} HarmBench behaviors")
    return records


def run_safety_eval(
    checkpoint: str,
    out_dir: str,
    lora_path: str | None = None,
    max_samples: int | None = None,
    max_new_tokens: int = 1024,
    gen_batch_size: int = 4,
    judge_batch_size: int = 16,
    judge_model: str = DEFAULT_JUDGE,
    seed: int = 42,
    test: bool = False,
    drop_copyright: bool = False,
    model=None,
    tokenizer=None,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if test:
        logger.info("Running test mode: 20 samples only")
        max_samples = 20

    behaviors = _load_harmbench_behaviors(
        max_samples=max_samples, seed=seed, drop_copyright=drop_copyright
    )
    prompts = [b["prompt"] for b in behaviors]

    completions = generate_responses(
        model_path=checkpoint,
        lora_path=lora_path,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=gen_batch_size,
        seed=seed,
        model=model,
        tokenizer=tokenizer,
    )

    # free generation KV-cache/activation residual before the 13B judge loads;
    # with sequential adapter merges the resident base model leaves little
    # headroom, so this avoids an OOM at the judge's forward pass.
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    judgements = classify(
        pairs=list(zip(prompts, completions)),
        judge_model=judge_model,
        batch_size=judge_batch_size,
    )

    results = []
    n_unsafe = 0
    for behavior, completion, judgement in zip(behaviors, completions, judgements):
        if judgement["verdict"] == "unsafe":
            n_unsafe += 1

        results.append(
            {
                "id": behavior["id"],
                "category": behavior["category"],
                "prompt": behavior["prompt"],
                "completion": completion,
                "verdict": judgement["verdict"],
                "score": judgement["score"],
                "length": len(completion),
            }
        )

    n_total = len(results)
    asr = n_unsafe / n_total if n_total > 0 else 0.0

    checkpoint_tag = Path(checkpoint).name or "checkpoint"
    if lora_path:
        lora_tag = Path(lora_path).name
        out_file = out_path / f"{checkpoint_tag}+{lora_tag}_harmbench.json"
    else:
        out_file = out_path / f"{checkpoint_tag}_harmbench.json"

    write_json(
        out_file,
        {
            "checkpoint": checkpoint,
            "lora_path": lora_path,
            "judge_model": judge_model,
            "seed": seed,
            "n_total": n_total,
            "n_unsafe": n_unsafe,
            "asr": asr,
            "items": results,
        },
    )

    logger.info(f"ASR = {asr:.4f} ({n_unsafe}/{n_total})")
    logger.info(f"Results saved to {out_file}")

    return {
        "asr": asr,
        "n_total": n_total,
        "n_unsafe": n_unsafe,
        "path": str(out_file),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=16)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test", action="store_true", help="run a quick test with 20 samples"
    )
    parser.add_argument(
        "--drop-copyright",
        action="store_true",
        help="evaluate only the 240 non-copyright behaviours",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_safety_eval(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        lora_path=args.lora_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        gen_batch_size=args.gen_batch_size,
        judge_batch_size=args.judge_batch_size,
        judge_model=args.judge_model,
        seed=args.seed,
        test=args.test,
        drop_copyright=args.drop_copyright,
    )
    print(json.dumps(summary, indent=2))
