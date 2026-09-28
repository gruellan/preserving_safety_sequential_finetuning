import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import attn_for

logger = logging.getLogger(__name__)


def load_base_model(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_for(model_name, attn_implementation),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.config.pad_token_id = tokenizer.pad_token_id

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Loaded {int(n_params):,} parameters in {dtype}")
    return model, tokenizer
