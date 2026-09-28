import gc
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from tqdm import tqdm
from peft import PeftModel

from model_utils import attn_for, gen_eos_ids

logger = logging.getLogger(__name__)


_MAX_PROMPT_CHARS = 512


def format_prompt(prompt: str, tokenizer) -> str:
    prompt = prompt[:_MAX_PROMPT_CHARS]

    # get model specific formatting if available, else use simple template
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"User: {prompt}\nAssistant:"


def generate_responses(
    model_path: str | None = None,
    lora_path: str | None = None,
    prompts: list[str] = (),
    max_new_tokens: int = 1024,
    batch_size: int = 4,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    model: "AutoModelForCausalLM | None" = None,
    tokenizer: "AutoTokenizer | None" = None,
) -> list[str]:

    # load model if not passed in
    if model is None:
        logger.info(f"Loading model: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_for(model_path),
        ).to("cuda")

        if lora_path is not None:
            logger.info(f"Loading LoRA from {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model.eval()
    set_seed(seed)

    device = next(model.parameters()).device
    # Gemma 2 needs <end_of_turn> added as a stop token
    # none for Mistral/Llama
    eos_ids = gen_eos_ids(tokenizer, model)

    formatted = [format_prompt(p, tokenizer) for p in prompts]
    order = sorted(range(len(formatted)), key=lambda k: len(formatted[k]))
    sorted_formatted = [formatted[k] for k in order]
    outputs = [""] * len(formatted)

    with torch.inference_mode():
        for i in tqdm(
            range(0, len(sorted_formatted), batch_size),
            desc="Generating responses to HarmBench prompts",
        ):

            batch_texts = sorted_formatted[i : i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            ).to(device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                **({"eos_token_id": eos_ids} if eos_ids is not None else {}),
            )

            input_len = inputs["input_ids"].shape[1]
            gen_tokens = out[:, input_len:]

            decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

            batch_indices = order[i : i + batch_size]
            for idx, d in zip(batch_indices, decoded):
                outputs[idx] = d.strip()

    if model is not None:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Generated {len(outputs)} responses")
    return outputs
