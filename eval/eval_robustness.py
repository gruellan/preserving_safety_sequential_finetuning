import argparse
import gc
import logging
import sys
from pathlib import Path

import torch

from eval.generation import generate_responses
from eval.judge import DEFAULT_JUDGE, classify
from eval.refusal import classify_refusals
from eval.robustness_data import (
    ASR_BENCHMARKS,
    REFUSAL_BENCHMARKS,
)
from eval.utils import build_model, write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


def condition_dir_for(adapter_dirs: list[Path]) -> str:
    if not adapter_dirs:
        return "m0_baseline"
    return Path(adapter_dirs[0]).parent.name


SAMPLE_CAPS = {
    "advbench": None,
    "jbb": None,
    "orbench_toxic": 320,
    "xstest": None,
    "orbench_hard": 500,
}


def generate_asr_completions(label, model, tokenizer, args):
    benches = {}
    for name, loader in ASR_BENCHMARKS.items():
        samples = loader(max_samples=SAMPLE_CAPS.get(name), seed=args.seed)
        prompts = [r["prompt"] for r in samples]
        logger.info(f"[{label}] generating {len(prompts)} completions for ASR/{name}")
        completions = generate_responses(
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch_size,
            seed=args.seed,
            model=model,
            tokenizer=tokenizer,
        )
        benches[name] = (samples, prompts, completions)
    return benches


def generate_refusal_completions(label, model, tokenizer, args):
    benches = {}
    for name, loader in REFUSAL_BENCHMARKS.items():
        samples = loader(max_samples=SAMPLE_CAPS.get(name), seed=args.seed)
        prompts = [r["prompt"] for r in samples]
        logger.info(
            f"[{label}] generating {len(prompts)} completions for refusal/{name}"
        )
        completions = generate_responses(
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch_size,
            seed=args.seed,
            model=model,
            tokenizer=tokenizer,
        )
        benches[name] = (samples, prompts, completions)
    return benches


def judge_asr_completions(label, benches, out_dir, args):
    all_pairs = [
        (prompt, comp)
        for (_, prompts, completions) in benches.values()
        for prompt, comp in zip(prompts, completions)
    ]
    logger.info(
        f"[{label}] judging {len(all_pairs)} ASR generations with {DEFAULT_JUDGE}"
    )
    judgements = classify(
        all_pairs, judge_model=DEFAULT_JUDGE, batch_size=args.judge_batch_size
    )

    cursor = 0
    for name, (samples, prompts, completions) in benches.items():
        count = len(samples)
        bench_judgements = judgements[cursor : cursor + count]
        cursor += count
        items, n_unsafe = [], 0
        for rec, comp, judgement in zip(samples, completions, bench_judgements):
            if judgement["verdict"] == "unsafe":
                n_unsafe += 1
            items.append(
                {
                    "id": rec["id"],
                    "category": rec.get("category"),
                    "prompt": rec["prompt"],
                    "completion": comp,
                    "verdict": judgement["verdict"],
                    "score": judgement["score"],
                    "length": len(comp),
                }
            )

        asr = n_unsafe / count if count else 0.0
        out_file = out_dir / "asr_heldout" / f"{name}.json"
        write_json(
            out_file,
            {
                "model": label,
                "benchmark": name,
                "judge_model": DEFAULT_JUDGE,
                "seed": args.seed,
                "n_total": count,
                "n_unsafe": n_unsafe,
                "asr": asr,
                "items": items,
            },
        )
        logger.info(
            f"[{label}] ASR/{name}: {asr:.4f} ({n_unsafe}/{count}) -> {out_file}"
        )


def judge_refusal_completions(label, benches, out_dir, args):
    for name, (samples, prompts, completions) in benches.items():
        refusals = classify_refusals(
            prompts, completions, batch_size=args.judge_batch_size
        )

        items = []
        safe_total = safe_ref = unsafe_total = unsafe_ref = 0
        for rec, comp, is_ref in zip(samples, completions, refusals):
            is_safe = rec.get("is_safe", True)
            if is_safe:
                safe_total += 1
                safe_ref += int(is_ref)
            else:
                unsafe_total += 1
                unsafe_ref += int(is_ref)
            items.append(
                {
                    "id": rec["id"],
                    "category": rec.get("category"),
                    "is_safe": is_safe,
                    "prompt": rec["prompt"],
                    "completion": comp,
                    "refusal": is_ref,
                    "length": len(comp),
                }
            )
        summary = {
            "model": label,
            "benchmark": name,
            "seed": args.seed,
            "n_safe": safe_total,
            "n_unsafe_contrast": unsafe_total,
            "over_refusal_rate": (safe_ref / safe_total) if safe_total else None,
            "contrast_refusal_rate": (
                (unsafe_ref / unsafe_total) if unsafe_total else None
            ),
        }
        out_file = out_dir / "over_refusal" / f"{name}.json"
        write_json(out_file, {**summary, "items": items})
        logger.info(
            f"[{label}] over-refusal/{name}: benign refusal={summary['over_refusal_rate']} "
            f"(n={safe_total}), contrast refusal={summary['contrast_refusal_rate']} (n={unsafe_total}) -> {out_file}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument(
        "--results-dir", default=str(Path(__file__).resolve().parent.parent / "results")
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gen-batch-size", type=int, default=32)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument("--translation-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument(
        "--benches",
        nargs="+",
        default=None,
        help="subset of benchmark names to run (e.g. xstest orbench_hard), default all",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="override per-benchmark sample cap (for smoke test)",
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--spec",
        action="append",
        nargs="+",
        required=True,
        help="label followed by adapter dirs, e.g. --spec cond_a /path/sum /path/code /path/qa",
    )
    parser.add_argument("--skip-m0", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_specs = []
    if not args.skip_m0:
        model_specs.append(("M0_base", []))
    model_specs += [(spec[0], [Path(p) for p in spec[1:]]) for spec in args.spec]

    if args.benches:
        ASR_BENCHMARKS = {k: v for k, v in ASR_BENCHMARKS.items() if k in args.benches}
        REFUSAL_BENCHMARKS = {
            k: v for k, v in REFUSAL_BENCHMARKS.items() if k in args.benches
        }

    if args.test:
        for bench_name in SAMPLE_CAPS:
            SAMPLE_CAPS[bench_name] = 10

    if args.cap:
        for name in list(SAMPLE_CAPS) + list(ASR_BENCHMARKS) + list(REFUSAL_BENCHMARKS):
            SAMPLE_CAPS[name] = args.cap

    for label, adapter_dirs in model_specs:
        if args.only and label not in args.only:
            continue

        out_dir = (
            Path(args.results_dir) / condition_dir_for(adapter_dirs) / "robustness"
        )
        logger.info(
            f"==================== {label} ({len(adapter_dirs)} adapters) =================="
        )

        # generate all completions
        model, tokenizer = build_model(
            args.base_model, [Path(ad) for ad in adapter_dirs]
        )

        asr_benches = generate_asr_completions(label, model, tokenizer, args)
        refusal_benches = generate_refusal_completions(label, model, tokenizer, args)

        # free memory for judge model
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[{label}] generation model unloaded")

        # classify completions with judge
        if asr_benches:
            judge_asr_completions(label, asr_benches, out_dir, args)
        if refusal_benches:
            judge_refusal_completions(label, refusal_benches, out_dir, args)

    logger.info("Robustness evaluation completed")
