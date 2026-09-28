import json
import logging
import random
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

# advbench.json ships with the repo under data_processing/
ADVBENCH_PATH = Path(__file__).resolve().parents[1] / "data_processing" / "advbench.json"


def _sample(records: list[dict], max_samples: int | None, seed: int) -> list[dict]:
    if max_samples is not None and max_samples < len(records):
        records = random.Random(seed).sample(records, max_samples)
    return records


def load_advbench(max_samples: int | None = None, seed: int = 42) -> list[dict]:
    records = json.load(open(ADVBENCH_PATH))
    records = _sample(records, max_samples, seed)
    logger.info(f"Loaded {len(records)} AdvBench behaviors")
    return records


def load_jbb(max_samples: int | None = None, seed: int = 42) -> list[dict]:
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    records = [
        {
            "id": f"jbb_{row['Index']}",
            "prompt": (row.get("Goal") or "").strip(),
            "category": row.get("Category"),
        }
        for row in ds
    ]
    records = _sample(records, max_samples, seed)
    logger.info(f"Loaded {len(records)} JailbreakBench behaviors")
    return records


def load_orbench_toxic(max_samples: int | None = 320, seed: int = 42) -> list[dict]:
    ds = load_dataset("bench-llm/or-bench", "or-bench-toxic", split="train")
    records = [
        {
            "id": f"orbench_toxic_{i:04d}",
            "prompt": (row.get("prompt") or "").strip(),
            "category": row.get("category"),
        }
        for i, row in enumerate(ds)
    ]
    records = _sample(records, max_samples, seed)
    logger.info(f"Loaded {len(records)} OR-Bench-toxic prompts")
    return records


def load_xstest(max_samples: int | None = None, seed: int = 42) -> list[dict]:
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    records = [
        {
            "id": row.get("id") or f"xstest_{i:04d}",
            "prompt": (row.get("prompt") or "").strip(),
            "category": row.get("type"),
            "is_safe": not str(row.get("type", "")).startswith("contrast"),
        }
        for i, row in enumerate(ds)
    ]
    records = _sample(records, max_samples, seed)
    n_safe = sum(rec["is_safe"] for rec in records)
    logger.info(
        f"Loaded {len(records)} XSTest prompts ({n_safe} safe / {len(records) - n_safe} unsafe-contrast)"
    )
    return records


def load_orbench_hard(max_samples: int | None = 500, seed: int = 42) -> list[dict]:
    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    records = [
        {
            "id": f"orbench_hard_{i:04d}",
            "prompt": (row.get("prompt") or "").strip(),
            "category": row.get("category"),
            "is_safe": True,
        }
        for i, row in enumerate(ds)
    ]
    records = _sample(records, max_samples, seed)
    logger.info(f"Loaded {len(records)} OR-Bench-hard prompts (all benign)")
    return records


ASR_BENCHMARKS = {
    "advbench": load_advbench,
    "jbb": load_jbb,
    "orbench_toxic": load_orbench_toxic,
}

REFUSAL_BENCHMARKS = {
    "xstest": load_xstest,
    "orbench_hard": load_orbench_hard,
}
