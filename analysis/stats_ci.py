"""Paper numbers not in a table or figure: python -m analysis.stats_ci

Means over O1-O3 at seed 42
AUROC rows need the git-ignored rq4_cache
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

TREES = {"Mistral": "results", "Llama": "llama_results"}
RUNS = {
    "Plain": "",
    "Matched": "_plainmatched",
    "ER": "_er_b150",
    "DER": "_der_b150",
    "DER++": "_derpp_b150",
}
ORR = ("xstest", "orbench_hard")


def rate(path: Path) -> float:
    """% harmful (HarmBench, ex-copyright) or % refused (benign prompts)"""
    with open(path) as f:
        items = json.load(f)["items"]
    if "verdict" in items[0]:
        v = [
            i["verdict"] == "unsafe"
            for i in items
            if i.get("category", "").lower() != "copyright"
        ]
    else:
        v = [bool(i["refusal"]) for i in items if i.get("is_safe", True)]
    return 100 * sum(v) / len(v)


def mean_m3(tree: str, suffix: str, metric: str) -> float:
    vals = []
    for o in (1, 2, 3):
        run = Path(tree) / f"order_{o}{suffix}"
        if metric == "asr":
            (path,) = (
                path for path in (run / "asr_harmbench").iterdir()
                if path.name.startswith("M3_") and path.name[3:].endswith("_harmbench.json")
            )
        else:
            path = run / "robustness/over_refusal" / f"{metric}.json"
        vals.append(rate(path))
    return sum(vals) / 3


def auroc(path: Path) -> float:
    # project onto the difference-in-means direction
    d = np.load(path)
    r = d["harmful"].mean(0) - d["harmless"].mean(0)
    harmful, harmless = d["harmful"] @ r, d["harmless"] @ r
    labels = np.concatenate([np.ones(len(harmful)), np.zeros(len(harmless))])
    scores = np.concatenate([harmful, harmless])
    return roc_auc_score(labels, scores)


def main():
    rows = []
    for model, tree in TREES.items():
        m = {
            (run, k): mean_m3(tree, suffix, k)
            for run, suffix in RUNS.items()
            for k in ("asr", *ORR)
        }
        for k in ("asr", "xstest"):
            for plain in ("Plain", "Matched"):
                ix = (m["DER++", k] - m["DER", k]) - (m["ER", k] - m[plain, k])
                rows.append((model, f"interaction_{k}_vs_{plain}", round(ix, 2)))
        shift = abs(m["Matched", "asr"] - m["Plain", "asr"])
        rows.append((model, "matched_shift_asr", round(shift, 2)))
        shift = max(abs(m["Matched", b] - m["Plain", b]) for b in ORR)
        rows.append((model, "matched_shift_orr_max", round(shift, 2)))

        for run, suffix in RUNS.items():
            # Matched Plain has no rq4 run and caches are git-ignored
            if run == "Matched" or not (Path(tree) / f"order_1{suffix}/rq4_cache").is_dir():
                continue
            for ckpt in ("M0", "M3"):
                paths = []
                for o in (1, 2, 3):
                    cache = Path(tree) / f"order_{o}{suffix}/rq4_cache"
                    path = next(
                        path for path in cache.iterdir()
                        if path.name.startswith(f"{ckpt}_") and path.name.endswith(".npz")
                    )
                    paths.append(path)
                a = np.mean([auroc(p) for p in paths])
                rows.append((model, f"auroc_{run}_{ckpt}", round(a, 3)))

    with open("analysis/stats.csv", "w", newline="") as f:
        csv.writer(f).writerows([("model", "stat", "value"), *rows])
    print(f"wrote analysis/stats.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
