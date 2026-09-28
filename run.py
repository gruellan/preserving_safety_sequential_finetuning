import argparse
import gc
import logging
from pathlib import Path
import sys
import yaml

import torch

from data_processing.load_trace import load_trace_task
from data_processing.load_wildjailbreak import load_wildjailbreak_replay
from data_processing.build_der_dataset import build_der_dataset
from training.determinism import setup_determinism
from training.lora_finetune import (
    build_lora_config,
    build_training_args,
    lora_finetune,
    lora_finetune_der,
)
from training.load_model import load_base_model
from training.evaluate_task import evaluate_trace_task, load_and_merge_adapter
from results_io import log_results, save_results, save_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.captureWarnings(True)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "experiment_config.yaml"),
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        dest="max_train_samples",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        dest="max_test_samples",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        dest="tasks",
        help="e.g. --tasks code sum qa - defaults to all tasks in config file order",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        nargs="+",
        default=None,
        dest="adapter_dir",
        help="only used with --eval-only.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        dest="eval_only",
        help="evaluate mode only on given --tasks.",
    )
    parser.add_argument("--order-name", type=str, required=True, dest="order_name")
    parser.add_argument(
        "--der",
        action="store_true",
    )
    parser.add_argument(
        "--der-alpha",
        type=float,
        default=1.0,
        dest="der_alpha",
        help="weight of replay MSE loss (total = task_loss + alpha * replay_loss)",
    )
    parser.add_argument(
        "--der-beta",
        type=float,
        default=1.0,
        dest="der_beta",
        help="weight of buffer CE loss in der++ (total += beta * CE(buffer))",
    )
    parser.add_argument(
        "--der-variant",
        type=str,
        default="der++",
        choices=["der", "der++"],
    )
    parser.add_argument(
        "--teacher-cache-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--der-buffer-path",
        type=str,
        default=None,
        dest="der_buffer_path",
    )
    parser.add_argument(
        "--save-tag",
        type=str,
        default=None,
        dest="save_tag",
        help="filename prefix for eval-only outputs (e.g. 'qa_after_code')",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).parent / "results"),
        dest="results_dir",
    )
    parser.add_argument(
        "--skip-tasks", nargs="*", default=[], dest="skip_tasks", help="crash recovery"
    )
    parser.add_argument(
        "--skip-evals",
        action="store_true",
        dest="skip_evals",
        help="train only, run task evals later via --eval-only",
    )
    parser.add_argument(
        "--per-device-batch",
        type=int,
        default=8,
        dest="per_device_batch",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()

    if args.der and not args.der_buffer_path:
        raise SystemExit(
            "--der requires --der-buffer-path (run data_processing/build_replay_buffer.py)"
        )

    setup_determinism(seed=args.seed)
    config = load_config(args.config)

    task_names = args.tasks if args.tasks is not None else list(config["tasks"].keys())

    order_dir = Path(args.results_dir) / args.order_name
    task_perf_dir = order_dir / "task_performance"
    task_perf_dir.mkdir(parents=True, exist_ok=True)

    order_config = {
        "order": task_names,
        "model": config["model"]["name"],
        "seed": args.seed,
    }
    if args.der:
        order_config["der"] = {
            "variant": args.der_variant,
            "alpha": args.der_alpha,
            "beta": args.der_beta,
            "buffer_path": args.der_buffer_path,
        }
    save_config(order_config, order_dir)

    logger.info(f"Experiment configuration: {config}")
    logger.info(f"Task order: {task_names}")
    logger.info(f"Results directory: {order_dir}")

    logger.info("Loading base model + tokenizer...")
    current_model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=torch.bfloat16,
        attn_implementation=config["model"].get("attn_implementation"),
    )

    if args.eval_only:

        def _eval_all_tasks(model_field: str, tag_prefix: str) -> None:
            for task_name in task_names:
                logger.info(f"Evaluating [{task_name}] test split ({model_field})...")
                results = evaluate_trace_task(
                    model=current_model,
                    tokenizer=tokenizer,
                    task_name=task_name,
                    trace_dir=config["paths"]["trace_data_dir"],
                    split="test",
                    max_samples=args.max_test_samples,
                    seed=args.seed,
                )
                results["model"] = model_field
                tag = f"{tag_prefix}_{task_name}" if tag_prefix else task_name
                log_results(tag, results)
                save_results(tag, results, task_perf_dir)

        if args.adapter_dir is None:
            logger.info("Evaluating base model (M0)...")
            _eval_all_tasks(
                model_field="base_model", tag_prefix=args.save_tag or "M0_base"
            )
        else:
            for i, adapter_dir in enumerate(args.adapter_dir, 1):
                checkpoint_task = Path(adapter_dir).name
                logger.info(f"Merging adapter {i}: {adapter_dir} -> M{i}")
                current_model = load_and_merge_adapter(current_model, adapter_dir)
                gc.collect()
                torch.cuda.empty_cache()
                _eval_all_tasks(
                    model_field=f"M{i}_after_{checkpoint_task}",
                    tag_prefix=f"M{i}_after_{checkpoint_task}",
                )

    else:

        def _preds_path(tag: str, task_name: str) -> str:
            return str(task_perf_dir / f"{tag}_{task_name}_test_predictions.jsonl")

        # the replay buffer and its cached M0 teacher logits are shared across all
        # tasks in the ordering so load once up front
        replay_dataset = None
        teacher_logits_cache = None
        if args.der:
            logger.info("DER enabled: loading replay buffer...")
            replay_dataset = load_wildjailbreak_replay(
                tokenizer,
                buffer_path=args.der_buffer_path,
                max_len=1024,
            )
            # alpha==0 (ER, plain_matched control) never uses the MSE teacher
            # target, so don't require the cache to exist for those configs
            if args.der_alpha != 0:
                cache_dir = Path(
                    args.teacher_cache_dir or config["paths"]["teacher_logits_cache"]
                )
                logger.info(f"Loading cached teacher logits from {cache_dir}...")
                teacher_logits_cache = [
                    torch.load(cache_dir / f"{i}.pt", map_location="cpu")
                    for i in range(len(replay_dataset))
                ]

        for task_name in task_names:
            logger.info(
                f"---------------------------------- Task: [{task_name}] ----------------------------------"
            )

            # crash recovery
            if task_name in args.skip_tasks:
                adapter_dir = str(
                    Path(config["paths"]["output_dir"]) / args.order_name / task_name
                )
                logger.info(
                    f"[{task_name}] in --skip-tasks. merging saved adapter "
                    f"from {adapter_dir} instead of re-training"
                )
                current_model = load_and_merge_adapter(current_model, adapter_dir)

                gc.collect()
                torch.cuda.empty_cache()
                continue

            eval_kwargs = dict(
                tokenizer=tokenizer,
                task_name=task_name,
                trace_dir=config["paths"]["trace_data_dir"],
                split="test",
                max_samples=args.max_test_samples,
                seed=args.seed,
            )

            if not args.skip_evals:
                logger.info(
                    f"Evaluating current model on [{task_name}] before training..."
                )
                pre_results = evaluate_trace_task(
                    model=current_model,
                    predictions_path=_preds_path(f"{task_name}_pre", task_name),
                    **eval_kwargs,
                )
                pre_results["model"] = "pre_finetune"
                log_results(f"{task_name}_pre", pre_results)
                save_results(f"{task_name}_pre", pre_results, task_perf_dir)

                gc.collect()
                torch.cuda.empty_cache()

            task_max_len = config["tasks"][task_name].get(
                "max_seq_len", config["model"]["max_seq_len"]
            )
            logger.info(f"Loading [{task_name}] train dataset...")
            train_dataset = load_trace_task(
                task_name=task_name,
                tokenizer=tokenizer,
                trace_dir=config["paths"]["trace_data_dir"],
                max_len=task_max_len,
                max_samples=args.max_train_samples,
                split="train",
                seed=args.seed,
            )
            logger.info(f"Task dataset: {len(train_dataset)} samples")

            adapter_save_dir = str(
                Path(config["paths"]["output_dir"]) / args.order_name / task_name
            )
            if args.der:
                der_dataset = build_der_dataset(
                    task_dataset=train_dataset,
                    replay_dataset=replay_dataset,
                    seed=args.seed,
                )
                peft_model, adapter_dir = lora_finetune_der(
                    base_model=current_model,
                    lora_config=build_lora_config(),
                    train_set=der_dataset,
                    train_args=build_training_args(
                        adapter_save_dir,
                        seed=args.seed,
                        per_device_batch_size=args.per_device_batch,
                        gradient_accumulation_steps=args.grad_accum,
                    ),
                    tokenizer=tokenizer,
                    adapter_output_dir=adapter_save_dir,
                    teacher_logits_cache=teacher_logits_cache,
                    alpha=args.der_alpha,
                    beta=args.der_beta,
                    variant=args.der_variant,
                )
            else:
                peft_model, adapter_dir = lora_finetune(
                    base_model=current_model,
                    lora_config=build_lora_config(),
                    train_set=train_dataset,
                    train_args=build_training_args(
                        adapter_save_dir,
                        seed=args.seed,
                        per_device_batch_size=args.per_device_batch,
                        gradient_accumulation_steps=args.grad_accum,
                    ),
                    tokenizer=tokenizer,
                    adapter_output_dir=adapter_save_dir,
                )

            logger.info("Merging adapter into model...")
            current_model = peft_model.merge_and_unload()

            gc.collect()
            torch.cuda.empty_cache()

            if args.skip_evals:
                logger.info(
                    f"Task {task_name} complete, adapter saved to {adapter_dir}"
                )
                continue

            logger.info(f"Evaluating [{task_name}] finetuned model on test split...")
            ft_results = evaluate_trace_task(
                model=current_model,
                predictions_path=_preds_path(f"{task_name}_finetuned", task_name),
                **eval_kwargs,
            )
            ft_results["model"] = "finetuned"
            log_results(f"{task_name}_finetuned", ft_results)
            save_results(f"{task_name}_finetuned", ft_results, task_perf_dir)

            # cross eval over all tasks at current checkpoint but skip the next training task
            # its pre-eval at the start of the next iteration would be duplicate
            pos = task_names.index(task_name)
            next_to_train = task_names[pos + 1] if pos + 1 < len(task_names) else None
            for other_task in task_names:
                if other_task == task_name or other_task == next_to_train:
                    continue
                is_previous = task_names.index(other_task) < pos
                model_field = f"after_{task_name}" if is_previous else "eval_only"
                logger.info(
                    f"Evaluating [{other_task}] at M_{pos + 1} (after [{task_name}], model={model_field})..."
                )
                cross_results = evaluate_trace_task(
                    model=current_model,
                    tokenizer=tokenizer,
                    task_name=other_task,
                    trace_dir=config["paths"]["trace_data_dir"],
                    split="test",
                    max_samples=args.max_test_samples,
                    seed=args.seed,
                    predictions_path=_preds_path(
                        f"{other_task}_after_{task_name}", other_task
                    ),
                )
                cross_results["model"] = model_field
                log_results(f"{other_task}_after_{task_name}", cross_results)
                save_results(
                    f"{other_task}_after_{task_name}", cross_results, task_perf_dir
                )

            logger.info(f"Task {task_name} complete, adapter saved to {adapter_dir}")

    logger.info(f"All results saved to {order_dir}")
