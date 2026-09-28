"""
Build  DER replay buffer with optional filtered by m_0 base model correctness

One-time pre-processing step before running cache_teacher_logits.py
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from training.determinism import setup_determinism

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = str(Path(__file__).parent.parent / "config" / "experiment_config.yaml")

HARMFUL_DATA_TYPE = "vanilla_harmful"
OVERREFUSAL_DATA_TYPE = "adversarial_benign"


def _wildjailbreak_prompt(row) -> str:
    col = "adversarial" if row["data_type"].startswith("adversarial") else "vanilla"
    return row[col]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="data/der_buffer_filtered.jsonl",
    )
    parser.add_argument("--n-harmful", type=int, default=250)
    parser.add_argument("--n-benign", type=int, default=250)
    parser.add_argument(
        "--oversample",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="skip M0 generation/judging",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def sample_candidates(dataset, data_type, k, rng):
    rows = [row for row in dataset if row["data_type"] == data_type]
    logger.info(f"Found {len(rows)} {data_type} samples in WildJailbreak")

    if 0 <= k < len(rows):
        rows = rng.sample(rows, k)
    candidates = []

    for row in rows:
        prompt = _wildjailbreak_prompt(row)
        completion = row["completion"]

        if prompt and prompt.strip() and completion.strip():
            candidates.append({"prompt": prompt, "completion": completion})
    return candidates


def write_buffer(path, harmful, overrefusal):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for category, samples in (("harmful", harmful), ("overrefusal", overrefusal)):
            for s in samples:
                row = {
                    "prompt": s["prompt"],
                    "completion": s["completion"],
                    "category": category,
                }

                if "m0_response" in s:
                    row["m0_response"] = s["m0_response"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        f"Wrote {len(harmful)} harmful + {len(overrefusal)} overrefusal "
        f"= {len(harmful) + len(overrefusal)} samples to {out}"
    )


if __name__ == "__main__":
    args = parse_args()

    setup_determinism(seed=args.seed)
    config = load_config(args.config)
    model_name = args.model_name or config["model"]["name"]

    dataset = load_dataset(
        "allenai/wildjailbreak",
        "train",
        split="train",
        delimiter="\t",
        keep_default_na=False,
    )
    rng = random.Random(args.seed)

    if args.no_filter:
        harmful = sample_candidates(dataset, HARMFUL_DATA_TYPE, args.n_harmful, rng)
        overrefusal = sample_candidates(
            dataset, OVERREFUSAL_DATA_TYPE, args.n_benign, rng
        )

        write_buffer(args.output, harmful, overrefusal)
        logger.info("Built unfiltered buffer")
        sys.exit(0)

    # oversample each pool so enough survive the M0 correctness filter
    k_harmful = math.ceil(args.n_harmful * args.oversample)
    k_benign = math.ceil(args.n_benign * args.oversample)
    harmful_candidates = sample_candidates(dataset, HARMFUL_DATA_TYPE, k_harmful, rng)
    over_candidates = sample_candidates(dataset, OVERREFUSAL_DATA_TYPE, k_benign, rng)
    logger.info(
        f"Candidate pools: {len(harmful_candidates)} harmful, {len(over_candidates)} overrefusal"
    )

    # generate M0 responses
    from eval.generation import generate_responses

    all_prompts = [c["prompt"] for c in harmful_candidates] + [
        c["prompt"] for c in over_candidates
    ]
    logger.info(f"Generating M0 responses for {len(all_prompts)} candidates...")
    gens = generate_responses(
        model_path=model_name,
        prompts=all_prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
        seed=args.seed,
    )
    h_gens = gens[: len(harmful_candidates)]
    o_gens = gens[len(harmful_candidates) :]

    # judge every candidate with wildguard
    from eval.refusal import classify_wildguard

    all_candidates = harmful_candidates + over_candidates
    verdicts = classify_wildguard(
        [c["prompt"] for c in all_candidates],
        gens,
        batch_size=args.judge_batch_size,
    )
    h_verdicts = verdicts[: len(harmful_candidates)]
    o_verdicts = verdicts[len(harmful_candidates) :]

    # harmful: keep prompts M0 refused without leaking anything harmful
    harmful_kept = []
    for cand, response, verdict in zip(harmful_candidates, h_gens, h_verdicts):
        if verdict["refusal"] is True and verdict["response_harmful"] is False:
            cand["m0_response"] = response
            harmful_kept.append(cand)

    # overrefusal: keep prompts M0 complied with (did not refuse)
    over_kept = []
    for cand, response, verdict in zip(over_candidates, o_gens, o_verdicts):
        if verdict["refusal"] is False:
            cand["m0_response"] = response
            over_kept.append(cand)

    logger.info(
        f"M0 refused {len(harmful_kept)}/{len(harmful_candidates)} harmful "
        f"({100 * len(harmful_kept) / max(1, len(harmful_candidates))}%)"
    )
    logger.info(
        f"M0 complied on {len(over_kept)}/{len(over_candidates)} overrefusal "
        f"({100 * len(over_kept) / max(1, len(over_candidates))}%)"
    )

    harmful = harmful_kept[: args.n_harmful]
    overrefusal = over_kept[: args.n_benign]

    if len(harmful) < args.n_harmful:
        logger.warning(
            f"Only {len(harmful)} M0-refused harmful samples available "
            f"(< {args.n_harmful}). increase --oversample to get more."
        )
    if len(overrefusal) < args.n_benign:
        logger.warning(
            f"Only {len(overrefusal)} M0-complied overrefusal samples available "
            f"(< {args.n_benign}). increase --oversample to get more."
        )

    write_buffer(args.output, harmful, overrefusal)
    logger.info("Built M0 correctness filtered buffer")
