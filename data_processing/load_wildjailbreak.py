import json
import logging

from datasets import Dataset

logger = logging.getLogger(__name__)


def format_replay_sample(prompt, completion, tokenizer):
    messages = [
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": completion.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def mask_prompt_labels(prompt, input_ids, tokenizer):
    messages = [{"role": "user", "content": prompt.strip()}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), len(input_ids))

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len
    return labels


def load_wildjailbreak_replay(
    tokenizer, buffer_path: str, max_len: int = 1024
) -> Dataset:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids, attention_masks, labels = [], [], []
    with open(buffer_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = format_replay_sample(row["prompt"], row["completion"], tokenizer)
            ids = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            input_ids.append(ids)
            attention_masks.append([1] * len(ids))
            labels.append(mask_prompt_labels(row["prompt"], ids, tokenizer))

    n = len(input_ids)
    logger.info(f"Loaded {n} replay samples from {buffer_path}")
    return Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "is_replay": [1] * n,
            "replay_index": list(range(n)),
        }
    )
