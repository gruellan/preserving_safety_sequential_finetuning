"""Generate the paper figures: python -m analysis.generate_figures."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

TASKS = ["code", "sum", "qa"]
TASK_COLOUR = {"code": "tab:blue", "sum": "tab:orange", "qa": "tab:green"}
TASK_METRIC = {
    "code": ("fuzzy_match", 1.0),
    "sum": ("rouge_l", 100.0),
    "qa": ("answer_accuracy", 100.0),
}
ORDERINGS = {  # task order -> label
    ("code", "sum", "qa"): "O1",
    ("qa", "sum", "code"): "O2",
    ("sum", "code", "qa"): "O3",
    ("code", "qa", "sum"): "O4",
    ("sum", "qa", "code"): "O5",
    ("qa", "code", "sum"): "O6",
}
METHODS = [  # label, run-dir suffix, colour
    ("Plain FT", "", "tab:blue"),
    ("ER", "_er_b150", "tab:orange"),
    ("DER", "_der_b150", "tab:green"),
    ("DER++", "_derpp_b150", "tab:red"),
]
COLOUR = {label: colour for label, _, colour in METHODS}


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def harmbench_asr(path: Path) -> float:
    items = {
        i["id"]: i
        for i in load(path)["items"]
        if i.get("category", "").lower() != "copyright"
    }
    return sum(i["verdict"] == "unsafe" for i in items.values()) / 240


def orbench_orr(path: Path) -> float:
    items = {
        i["id"]: i["refusal"] for i in load(path)["items"] if i.get("is_safe", True)
    }
    return sum(map(bool, items.values())) / 500


def stage(name: str) -> int:
    return int(re.search(r"M(\d)", name)[1])


class Model:
    def __init__(self, name: str, root: str, marker: str):
        self.name = name
        self.root = Path(root)
        self.marker = marker
        self.m0_asr = harmbench_asr(
            self.root / "m0_baseline/asr_harmbench/M0_base_harmbench.json"
        )
        self.m0_orr = orbench_orr(
            self.root / "m0_baseline/robustness/over_refusal/orbench_hard.json"
        )
        # Ordering label -> (run directory, task order)
        self.orders = {}
        for run in self.root.iterdir():
            if re.fullmatch(r"order_[0-9]", run.name):
                order = load(run / "config.json")["order"]
                self.orders[ORDERINGS[tuple(order)]] = (run, order)
        self.orders = dict(sorted(self.orders.items()))

    def runs(self, orderings: set[str] | None = None):
        """Yield (method, ordering, run directory)."""
        for method, suffix, _ in METHODS:
            for label, (base, _) in self.orders.items():
                run = base.with_name(base.name + suffix)
                if orderings is not None and label not in orderings:
                    continue
                if run.is_dir():
                    yield method, label, run


def asr_trajectory(run: Path) -> dict:
    return {
        stage(path.name): harmbench_asr(path)
        for path in (run / "asr_harmbench").iterdir()
        if path.name.startswith("M") and path.name.endswith("_harmbench.json")
    }


def final_asr(run: Path) -> float:
    return asr_trajectory(run)[3]


def refusal_geometry(run: Path) -> dict:
    return {
        stage(record["checkpoint"]): (record["cos_to_base"], record["magnitude"])
        for record in load(run / "rq4/rq4_metrics.json")["checkpoints"]
    }


def task_scores(run: Path, order: list[str]) -> tuple[dict, float]:
    records = {}
    for path in (run / "task_performance").glob("*_test.json"):
        record = load(path)
        records[record["task"], str(record["model"])] = record
    finals, drops = {}, []
    for i, task in enumerate(order):
        field, scale = TASK_METRIC[task]
        checkpoints = ["finetuned"] + [f"after_{t}" for t in order[i + 1 :]]
        scores = [
            records[task, checkpoint][field] * scale for checkpoint in checkpoints
        ]
        finals[task] = scores[-1]
        if i < 2:
            drops.append(max(scores) - scores[-1])
    return finals, sum(drops) / len(drops)


def pct(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))


def arrow(order: list[str]) -> str:
    return " $\\rightarrow$ ".join(order)


def save(fig, name: str):
    path = Path("plots") / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_rq1(models: list[Model]):
    """Plain FT ASR trajectories coloured by first task"""
    fig, axes = plt.subplots(
        1, len(models), figsize=(5.6 * len(models), 4.6), sharey=True
    )
    for ax, model in zip(axes, models):
        seen = []
        for label, (base, order) in model.orders.items():
            trajectory = asr_trajectory(base) | {0: model.m0_asr}
            stages = sorted(trajectory)
            ls = ["-", "--"][seen.count(order[0]) % 2]
            seen.append(order[0])
            ax.plot(
                stages, [trajectory[s] for s in stages], marker="o",
                color=TASK_COLOUR[order[0]], ls=ls, label=f"{label}: {arrow(order)}",
            )
        ax.axhline(
            model.m0_asr, ls=":", color="black", lw=1,
            label=f"M0 base ({model.m0_asr * 100:.0f}%)",
        )
        ax.set_xlabel("Checkpoint")
        ax.set_xticks([0, 1, 2, 3], ["M0", "M1", "M2", "M3"])
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=7)

    axes[0].set(ylabel="HarmBench ASR (ex-copyright, n=240)", ylim=(-0.03, 0.82))
    pct(axes[0].yaxis)
    fig.tight_layout()
    save(fig, "rq1_baseline_asr.png")


def fig_rq2(models: list[Model]):
    """Plain FT and DER++ final ASR by ordering"""
    fig, axes = plt.subplots(
        1, len(models), figsize=(6.0 * len(models), 4.8), sharey=True
    )
    for ax, model in zip(axes, models):
        labels = list(model.orders)
        x = list(range(len(labels)))
        plain, derpp = [], []
        for base, _ in model.orders.values():
            plain.append(final_asr(base))
            derpp.append(final_asr(base.with_name(base.name + "_derpp_b150")))
        ax.fill_between(x, derpp, plain, color="gray", alpha=0.1, zorder=0)
        for ys, name, dy, va in [
            (plain, "Plain FT", 0.012, "bottom"),
            (derpp, "DER++", -0.014, "top"),
        ]:
            ax.plot(
                x, ys, "-o", color=COLOUR[name], lw=2.2, ms=8,
                mec="white", mew=0.8, label=name, zorder=3,
            )
            for xi, y in zip(x, ys):
                ax.text(
                    xi, y + dy, f"{y * 100:.0f}", ha="center", va=va,
                    fontsize=7, color=COLOUR[name],
                )
        ax.axhline(model.m0_asr, ls="--", color="black", lw=1, zorder=1)
        ax.text(
            x[-1], model.m0_asr + 0.008, f"M0 base {model.m0_asr * 100:.0f}%",
            ha="right", va="bottom", fontsize=8,
        )
        # put the arrow between the two highest neighbouring plain FT points
        i = max(range(len(x) - 1), key=lambda i: min(plain[i], plain[i + 1]))
        gap_x = i + 0.5
        top = (plain[i] + plain[i + 1]) / 2
        bottom = (derpp[i] + derpp[i + 1]) / 2
        ax.annotate(
            "", xy=(gap_x, top), xytext=(gap_x, bottom),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2),
        )
        gap = (sum(plain) - sum(derpp)) / len(x)
        ax.text(
            gap_x + 0.12, (top + bottom) / 2,
            f"~{gap * 100:.0f} pts\nsafety restored",
            ha="left", va="center", fontsize=8, color="dimgray",
        )
        ax.set_xticks(
            x, [f"{o}\n{arrow(model.orders[o][1])}" for o in labels], fontsize=6.5
        )
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        ax.legend(fontsize=9, loc="center right")
    axes[0].set(ylim=(0, 0.82), ylabel="HarmBench ASR M3 (ex-copyright, n=240)")
    pct(axes[0].yaxis)
    fig.tight_layout()
    save(fig, "rq2_derpp_ordering.png")


def fig_task_utility(models: list[Model], ordering: str = "O3"):
    """Task scores at one ordering and mean forgetting across all orderings"""
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4.6 * len(models)))
    for (left, right), model in zip(axes, models):
        base, order = model.orders[ordering]
        bar_width = 0.8 / len(METHODS)
        for j, (method, suffix, colour) in enumerate(METHODS):
            run = base.with_name(base.name + suffix)
            finals, _ = task_scores(run, order)
            left.bar(
                [t + (j - (len(METHODS) - 1) / 2) * bar_width for t in range(3)],
                [finals[t] for t in TASKS], bar_width, color=colour, label=method,
            )
        left.set_xticks(
            range(3), ["code (fuzzy)", "sum (ROUGE-L)", "qa (acc)"], fontsize=9
        )
        left.set_ylabel("final task score (0-100, M3)")
        left.grid(True, axis="y", ls=":", alpha=0.5)
        left.legend(fontsize=8)

        forgetting = []
        for _, suffix, _ in METHODS:
            drops = []
            for base, order in model.orders.values():
                run = base.with_name(base.name + suffix)
                if run.is_dir():
                    drops.append(task_scores(run, order)[1])
            forgetting.append(sum(drops) / len(drops))
        right.bar(range(len(METHODS)), forgetting, 0.6, color=list(COLOUR.values()))
        for i, v in enumerate(forgetting):
            right.text(
                i, v + (0.03 if v >= 0 else -0.08), f"{v:+.1f}", ha="center", fontsize=8
            )
        right.axhline(0, color="black", lw=0.8)
        right.set_xticks(range(len(METHODS)), list(COLOUR), fontsize=9)
        right.set_ylabel("mean forgetting (pts) $\\downarrow$")
        right.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "task_utility_forgetting.png")


def fig_m3_scatter(
    models: list[Model],
    name: str,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    orderings: set[str] | None = None,
    legend_loc: str = "best",
):
    fig, ax = plt.subplots()
    for model in models:
        points = {}
        for method, _, run in model.runs(orderings):
            metrics = {"asr": final_asr(run)}
            if x_key == "over_refusal":
                metrics["over_refusal"] = orbench_orr(
                    run / "robustness/over_refusal/orbench_hard.json"
                )
            else:
                if not (run / "rq4/rq4_metrics.json").is_file():
                    continue
                directions = refusal_geometry(run)
                metrics["cos"] = directions[3][0]
                metrics["ratio"] = directions[3][1] / directions[0][1]
            ax.scatter(
                metrics[x_key], metrics[y_key], color=COLOUR[method], marker=model.marker,
                s=28, alpha=0.3, zorder=2,
            )
            points.setdefault(method, []).append((metrics[x_key], metrics[y_key]))
        for method, method_points in points.items():
            ax.scatter(
                sum(p[0] for p in method_points) / len(method_points),
                sum(p[1] for p in method_points) / len(method_points),
                color=COLOUR[method], marker=model.marker, s=150, alpha=0.95,
                edgecolors="black", linewidths=1.0, zorder=4,
            )

    stars = {}  # models sharing an M0 point get one label
    for model in models:
        xy = (
            round(model.m0_orr if x_key == "over_refusal" else 1.0, 3),
            round(model.m0_asr if y_key == "asr" else 1.0, 3),
        )
        stars.setdefault(xy, []).append(model.name.split("-")[0])
    for xy, names in stars.items():
        ax.scatter(*xy, color="black", marker="*", s=240, zorder=5)
        ax.annotate(
            " M0" if len(names) > 1 else f" {names[0]} M0", xy, fontsize=8, va="center"
        )

    ax.set(xlabel=x_label, ylabel=y_label)
    if x_key == "over_refusal":
        pct(ax.xaxis)
    if y_key == "asr":
        pct(ax.yaxis)
    ax.grid(True, ls=":", alpha=0.5)
    legend_style = dict(fontsize=8, title_fontsize=8)
    ax.add_artist(
        ax.legend(
            handles=[
                Line2D([], [], marker="o", ls="", color=colour, label=label, ms=9)
                for label, _, colour in METHODS
            ],
            title="method", loc=legend_loc, **legend_style,
        )
    )
    ax.legend(
        handles=[
            Line2D([], [], marker=model.marker, ls="", color="0.3", label=model.name, ms=9)
            for model in models
        ]
        + [Line2D([], [], marker="*", ls="", color="black", label="M0 base", ms=13)],
        title="model",
        loc="center right" if legend_loc == "upper right" else "lower right",
        **legend_style,
    )
    save(fig, name)


def fig_mag_cos(models: list[Model]):
    """Refusal direction rotation and magnitude trajectories from M0 to M3"""
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5.2))
    for ax, model in zip(axes, models):
        labelled = set()
        for method, _, run in model.runs():
            if not (run / "rq4/rq4_metrics.json").is_file():
                continue
            directions = refusal_geometry(run)
            stages = sorted(directions)
            cos = [directions[s][0] for s in stages]
            ratio = [directions[s][1] / directions[0][1] for s in stages]
            colour = COLOUR[method]
            ax.plot(cos, ratio, "-", color=colour, alpha=0.5, lw=1.3, zorder=2)
            for s in range(len(cos) - 1):
                ax.annotate(
                    "", xy=(cos[s + 1], ratio[s + 1]),
                    xytext=(cos[s], ratio[s]),
                    arrowprops=dict(arrowstyle="-|>", color=colour, alpha=0.8, lw=1.3),
                    zorder=2,
                )
            ax.scatter(
                cos[-1], ratio[-1], color=colour, s=55,
                edgecolor="k", linewidth=0.4, zorder=3,
                label=None if method in labelled else method,
            )
            labelled.add(method)
        ax.scatter(
            1.0, 1.0, marker="*", s=420, color="black", ec="black", lw=0.8, zorder=4
        )
        ax.set(
            xlabel="cos-to-base  (rotation $\\leftarrow$)",
            xlim=(0.6, 1.02), ylim=(0.5, 1.05),
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("magnitude ratio $\\|d\\|/\\|d_0\\|$ (erosion $\\downarrow$)")
    fig.tight_layout()
    save(fig, "mag_cos.png")


def main():
    Path("plots").mkdir(exist_ok=True)
    models = [
        Model("Mistral-7B", "results", "o"),
        Model("Llama-3-8B", "llama_results", "s"),
    ]
    cos_label = "Refusal-direction cosine similarity to base at M3"
    fig_rq1(models)
    fig_rq2(models)
    fig_task_utility(models)
    fig_m3_scatter(
        models, "safety_overrefusal.png", "over_refusal", "asr",
        "OR-Bench-hard over-refusal rate",
        "HarmBench direct-request ASR",
        orderings={"O1", "O2", "O3"}, legend_loc="upper right",
    )
    fig_m3_scatter(
        models, "decoupling.png", "cos", "asr",
        cos_label, "M3 HarmBench ASR, ex-copyright",
    )
    fig_m3_scatter(
        models, "erosion_rotation.png", "cos", "ratio",
        cos_label, "M3/M0 refusal-direction magnitude ratio",
    )
    fig_mag_cos(models)


if __name__ == "__main__":
    main()
