import json
import logging
import os
import random
from datasets import Dataset

logger = logging.getLogger(__name__)

TASK_FOLDER_MAP = {
    "code": "Py150",
    "sum": "20Minuten",
    "qa": "ScienceQA",
    "code1": "Py150",
    "code2": "Py150",
    "code3": "Py150",
}

SPLIT_FILE_MAP = {
    "train": "train.json",
    "val": "eval.json",
    "test": "test.json",
}
# Fallback TRACE location. The run pipeline passes paths.trace_data_dir from
# the config so this is only used if load_trace_task() is called without trace_dir
# Point TRACE_DIR at your local TRACE-Benchmark/LLM-CL-Benchmark_5000 checkout
TRACE_DIR = os.environ.get("TRACE_DIR", "")

# Py150 prompts in  TRACE have no instruction wrapper
# while Sum/QA prompts are already wrapped so wrap Py150
# to make all three tasks structurally identical at train and eval time
# (chat template + prompt masking + answer bounded by EOS token)
PY150_INSTRUCTION = "Complete the next line of the following code:"


def user_content_for(task: str, prompt: str) -> str:
    p = prompt.strip()
    if task.startswith("code"):
        return f"{PY150_INSTRUCTION}\n\n{p}"
    return p


def format_sample(task, prompt, answer, tokenizer):
    messages = [
        {"role": "user", "content": user_content_for(task, prompt)},
        {"role": "assistant", "content": answer.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def mask_prompt_labels(task, prompt, input_ids, tokenizer):
    messages = [{"role": "user", "content": user_content_for(task, prompt)}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), len(input_ids))

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len
    return labels


def load_trace_task(
    task_name,
    tokenizer,
    trace_dir=TRACE_DIR,
    split="train",
    max_len=2048,
    max_samples=None,
    seed=42,
):
    task_folder = TASK_FOLDER_MAP[task_name]
    split_file = SPLIT_FILE_MAP[split]
    file_path = f"{trace_dir}/{task_folder}/{split_file}"

    logger.info(f"Loading data from {file_path}...")
    with open(file_path, "r") as f:
        data = json.load(f)

    # homogeneous control subsets fixed split (42)
    if task_name in ("code1", "code2", "code3") and split == "train":
        idx = list(range(len(data)))
        random.Random(42).shuffle(idx)
        shard = sorted(idx[int(task_name[-1]) - 1 :: 3])
        data = [data[i] for i in shard]
        logger.info(f"Shard {task_name}: {len(data)} of {len(idx)} Py150 samples")

    if max_samples is not None and max_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, max_samples)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    all_labels: list[list[int]] = []

    for i, item in enumerate(data):
        text = format_sample(task_name, item["prompt"], item["answer"], tokenizer)
        if i == 0:
            logger.info(f"Sample formatted text:\n{text}")
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=False,
        )
        ids = enc["input_ids"]
        all_input_ids.append(ids)
        all_attention_masks.append(enc["attention_mask"])
        all_labels.append(mask_prompt_labels(task_name, item["prompt"], ids, tokenizer))

    dataset = Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }
    )

    return dataset
