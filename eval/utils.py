import gc
import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import attn_for

logger = logging.getLogger(__name__)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def build_model(
    base_model: str, adapter_dirs: list[Path]
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation=attn_for(base_model)
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    for i, adapter_dir in enumerate(adapter_dirs, 1):
        logger.info(f"merging adapter {i}/{len(adapter_dirs)}: {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
        gc.collect()
        torch.cuda.empty_cache()
    return model, tokenizer
