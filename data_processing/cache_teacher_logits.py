"""

Cache M0 logits on the DER replay buffer - one-time script must run before DER training

For each replay sample we save teacher logits at the response-token positions
(labels != -100) as a float32 tensor. DERTrainer reads these
files (index == replay_index) and distills the student towards them
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.load_wildjailbreak import load_wildjailbreak_replay
from training.determinism import setup_determinism
from training.load_model import load_base_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = str(Path(__file__).parent.parent / "config" / "experiment_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument("--buffer-path", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=1024, dest="max_len")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()

    setup_determinism(seed=args.seed)
    config = load_config(args.config)

    model_name = args.model_name or config["model"]["name"]
    output_dir = Path(args.output_dir or config["paths"]["teacher_logits_cache"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model (M0) + tokenizer...")
    model, tokenizer = load_base_model(model_name=model_name, dtype=torch.bfloat16)
    model.eval()

    logger.info("Loading replay buffer...")
    replay_dataset = load_wildjailbreak_replay(
        tokenizer,
        buffer_path=args.buffer_path,
        max_len=args.max_len,
    )

    # used to confirm the replay buffer matches at traing time
    fingerprint = hashlib.sha256()

    logger.info(f"Caching teacher logits to {output_dir}...")
    for sample in replay_dataset:
        idx = sample["replay_index"]
        input_ids = torch.tensor([sample["input_ids"]], device=model.device)
        attention_mask = torch.tensor([sample["attention_mask"]], device=model.device)
        labels = torch.tensor(sample["labels"])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # logits[t] predicts token t+1 so shift before masking to keep the logits
        # that produce the response (including first token)
        shifted_mask = labels[1:] != -100
        resp_logits = outputs.logits[0][:-1][shifted_mask].to(torch.bfloat16).cpu()
        torch.save(resp_logits, output_dir / f"{idx}.pt")

        fingerprint.update(str(sample["input_ids"]).encode())
        if idx % 50 == 0:
            logger.info(f"  cached {idx}.pt  shape={tuple(resp_logits.shape)}")

    metadata = {
        "model_name": model_name,
        "n_samples": len(replay_dataset),
        "seed": args.seed,
        "max_len": args.max_len,
        "vocab_size": model.config.vocab_size,
        "dtype": "bfloat16",
        # logits selected at positions that PREDICT the response tokens
        "logit_convention": "shifted_next_token",
        "dataset_fingerprint": fingerprint.hexdigest(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done. Cached {len(replay_dataset)} samples. Metadata: {metadata}")
