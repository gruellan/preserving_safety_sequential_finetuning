def attn_for(model_name, override=None):
    # gemma 2 does not support sdpa so must use eager attention
    if override:
        return override
    return "eager" if "gemma-2" in str(model_name).lower() else "sdpa"


def gen_eos_ids(tokenizer, model=None):
    # Gemma 2 ends chat turns with <end_of_turn>
    # but its eos_token_id/generation_config only lists <eos>
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if not (
        isinstance(eot, int)
        and eot >= 0
        and tokenizer.convert_ids_to_tokens(eot) == "<end_of_turn>"
    ):
        return None

    ids = {eot}

    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)

    if model is not None:
        eos = getattr(model.generation_config, "eos_token_id", None)
        ids.update([eos] if isinstance(eos, int) else (eos or []))

    return sorted(ids)
