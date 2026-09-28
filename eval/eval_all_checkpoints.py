import argparse
import gc
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.safety_eval import run_safety_eval
from model_utils import attn_for

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument("--adapter-dirs", nargs="+", required=True)
    parser.add_argument(
        "--order-name",
        required=True,
        help="e.g. order_1 - must match order-name used in run.py",
    )
    parser.add_argument(
        "--results-dir", default=str(Path(__file__).resolve().parent.parent / "results")
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gen-batch-size", type=int, default=32)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--drop-copyright",
        action="store_true",
        help="evaluate only the 240 non-copyright behaviours",
    )
    parser.add_argument(
        "--skip-m0",
        action="store_true",
    )
    args = parser.parse_args()

    asr_dir = str(Path(args.results_dir) / args.order_name / "asr_harmbench")

    logger.info(f"Loading base model from {args.base_model}")
    logger.info(f"ASR results will be saved to {asr_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_for(args.base_model),
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if args.skip_m0:
        logger.info("Skipping M0 base-model eval (--skip-m0)")
    else:
        logger.info(
            "---------------------------- Evaluating base model (M0) ----------------------------"
        )
        run_safety_eval(
            checkpoint="M0_base",
            out_dir=asr_dir,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            gen_batch_size=args.gen_batch_size,
            judge_batch_size=args.judge_batch_size,
            seed=args.seed,
            test=args.test,
            drop_copyright=args.drop_copyright,
            model=model,
            tokenizer=tokenizer,
        )

    for i, adapter_dir in enumerate(args.adapter_dirs, 1):
        task_name = Path(adapter_dir).name
        logger.info(f"Merging adapter {i}: {adapter_dir}")

        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()

        gc.collect()
        torch.cuda.empty_cache()

        # skip a checkpoint if result already exists (for crash recovery)
        done_file = Path(asr_dir) / f"M{i}_after_{task_name}_harmbench.json"
        if done_file.exists() and done_file.stat().st_size > 0:
            logger.info(
                f"M{i} (after [{task_name}]) already complete, skipping ({done_file})"
            )
            continue

        logger.info(
            f"---------------------------- Evaluating M{i} (after [{task_name}]) ----------------------------"
        )
        run_safety_eval(
            checkpoint=f"M{i}_after_{task_name}",
            out_dir=asr_dir,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            gen_batch_size=args.gen_batch_size,
            judge_batch_size=args.judge_batch_size,
            seed=args.seed,
            test=args.test,
            drop_copyright=args.drop_copyright,
            model=model,
            tokenizer=tokenizer,
        )
    logger.info(f"ASR evaluation completed. Results in {asr_dir}")
