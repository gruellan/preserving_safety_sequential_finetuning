import argparse
import logging
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# same modules as build_lora_config in lora_finetune.py
TARGET_MODULES = ("q_proj", "v_proj", "k_proj", "o_proj")


def load_cpu(path: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        path, device_map="cpu", low_cpu_mem_usage=True, torch_dtype=dtype
    )


def is_target(name: str) -> bool:
    return name.endswith(".weight") and any(m in name for m in TARGET_MODULES)


def build_projections(base_path: str, aligned_path: str, device: str) -> dict:
    """
    SafeLoRA (Hsu et al. 2024) alignment matrix per target weight
    P = V V^T / ||V||_F where V = W_aligned - W_unaligned
    """
    aligned = dict(load_cpu(aligned_path, torch.float32).named_parameters())
    projections = {}
    for name, w in load_cpu(base_path, torch.float32).named_parameters():
        if is_target(name):
            v = (aligned[name] - w).to(device)
            projections[name] = (v @ v.T / torch.norm(v)).cpu()
    logger.info(f"Built {len(projections)} projection matrices")
    return projections


def project_per_step(
    aligned_path: str,
    adapter_dirs: list[str],
    projections: dict,
    device: str,
    threshold: float = 0.5,
    num_proj_layers: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """
    Merge each task adapter in order and project its update dW -> P dW for
    target weights where cos(P dW, dW) <= threshold (or the num_proj_layers
    lowest cosines)

    IBM/SafeLoRA projects the adapter's B matrix before merging, here the update
    is taken from the merged weights so bf16 merge rounding can shift it slightly
    """
    model = load_cpu(aligned_path, dtype)
    for adapter_dir in adapter_dirs:
        # peft merges in place so snapshot the pre-merge weights first
        params = dict(model.named_parameters())
        before = {name: params[name].detach().clone() for name in projections}
        model = PeftModel.from_pretrained(
            model, adapter_dir, torch_dtype=dtype
        ).merge_and_unload()
        params = dict(model.named_parameters())

        def update(name):
            dw = params[name].data.to(device).float() - before[name].to(device).float()
            return dw, projections[name].to(device) @ dw

        cosines = {}
        for name in projections:
            dw, pdw = update(name)
            cos = torch.nn.functional.cosine_similarity(
                pdw.reshape(1, -1), dw.reshape(1, -1)
            )
            # rounded to 5 dp like the reference implementation
            cosines[name] = round(cos.item(), 5)

        if num_proj_layers is None:
            selected = [name for name, cos in cosines.items() if cos <= threshold]
        else:
            selected = sorted(cosines, key=cosines.get)[:num_proj_layers]

        # recompute P dW rather than keep all 128 in mem
        for name in selected:
            new = before[name].to(device).float() + update(name)[1]
            params[name].data.copy_(new.to(params[name].dtype).cpu())
        logger.info(
            f"{os.path.basename(adapter_dir)}: projected {len(selected)}/{len(projections)}"
        )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="unaligned base model")
    parser.add_argument(
        "--aligned-model",
        required=True,
        help="instruct model the adapters were trained on",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="task adapter dirs in training order",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--num-proj-layers",
        type=int,
        default=None,
        help="project the N lowest-cosine weights instead of thresholding",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    projections = build_projections(args.base_model, args.aligned_model, device)

    model = project_per_step(
        args.aligned_model,
        args.adapters,
        projections,
        device,
        threshold=args.threshold,
        num_proj_layers=args.num_proj_layers,
    )

    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.aligned_model).save_pretrained(args.out)
    logger.info(f"Saved SafeLoRA model to {args.out}")


if __name__ == "__main__":
    main()
