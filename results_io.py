import logging
import json
from pathlib import Path

from tabulate import tabulate

logger = logging.getLogger(__name__)


def log_results(tag: str, res: dict) -> None:
    header = f"[{tag}] test results ({res['n_samples']} samples)"
    logger.info(header)

    rows = [
        (k, f"{v:.4f}" if isinstance(v, float) else v)
        for k, v in res.items()
        if k not in ("task", "split", "n_samples")
    ]

    table = tabulate(rows, headers=["metric", "value"], tablefmt="github")
    logger.info("\n" + table)


def save_results(tag: str, results: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    task = results.get("task", "unknown_task")
    out_path = results_dir / f"{tag}_{task}_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results stored at {out_path}")


def save_config(config: dict, order_dir: Path) -> None:
    order_dir.mkdir(parents=True, exist_ok=True)
    out_path = order_dir / "config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Config saved to {out_path}")
