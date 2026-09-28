import json
import logging
import random
import re
from pathlib import Path
import torch

# install for metric calculations
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from data_processing.load_trace import TASK_FOLDER_MAP, SPLIT_FILE_MAP, user_content_for
from model_utils import gen_eos_ids

logger = logging.getLogger(__name__)

_rouge = Rouge()

# Py150 placeholder patterns (from TRACE https://github.com/BeyonderXX/TRACE/blob/master/evaluations/eval_Py150.py)
_PY150_PLACEHOLDERS = [
    (re.compile(r"<NUM_LIT>"), "0"),
    (re.compile(r"<STR_LIT>"), ""),
    (re.compile(r"<CHAR_LIT>"), ""),
    (re.compile(r"<(?:STR|NUM|CHAR)_LIT:(.*?)>"), r"\1"),
]


def bleu_n(predictions: list[str], references: list[str], n: int) -> float:
    weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
    hyps = [word_tokenize(p) for p in predictions]
    refs = [[word_tokenize(r)] for r in references]
    return corpus_bleu(refs, hyps, weights=weights)


def rouge_l(predictions: list[str], references: list[str]) -> float:
    scores = []
    for pred, ref in zip(predictions, references):
        if pred and ref:
            try:
                score = _rouge.get_scores(pred, ref)[0]["rouge-l"]["f"]
                scores.append(score)
            except Exception:
                pass
    return sum(scores) / len(scores) if scores else 0.0


def accuracy(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)


def _normalize_py150(text: str) -> str:
    for pattern, repl in _PY150_PLACEHOLDERS:
        text = pattern.sub(repl, text)
    return text


_PY150_LINE_BREAK = re.compile(r"\n|\s*<EOL>\s*")


def _truncate_to_first_line(text: str) -> str:
    for piece in _PY150_LINE_BREAK.split(text):
        if piece.strip():
            return piece.strip()
    return ""


def fuzzy_match(
    predictions: list[str], references: list[str], truncate: bool = False
) -> float:
    if truncate:
        predictions = [_truncate_to_first_line(p) for p in predictions]
        references = [_truncate_to_first_line(r) for r in references]
    scores = [
        fuzz.ratio(_normalize_py150(p), _normalize_py150(r))
        for p, r in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def exact_match_first_line(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    correct = sum(
        _normalize_py150(_truncate_to_first_line(p)).strip()
        == _normalize_py150(_truncate_to_first_line(r)).strip()
        for p, r in zip(predictions, references)
    )
    return correct / len(predictions)


def extract_scienceqa(raw: str) -> tuple[str, str]:
    raw = raw.strip()

    # check if answer is at the very start of the response
    m = re.match(r"^([A-E])(?:\s*$|\s*\n|[.),:;])", raw)
    if m:
        answer = m.group(1)
        reasoning = raw[m.end() :].lstrip(". \n").strip()
        return answer, reasoning

    # check for "Answer is X" pattern anywhere in the response
    m = re.search(r"(?:^|\b)[Aa]nswer\s*(?:is|:)\s*([A-E])\b", raw)
    if m:
        answer = m.group(1)
        reasoning = raw[m.end() :].lstrip(". \n").strip()
        return answer, reasoning

    # fall back: looks for standalone letter followed by a period
    m = re.search(r"\b([A-E])\.", raw)
    if m:
        answer = m.group(1)
        reasoning = raw[m.end() :].lstrip(" \n").strip()
        return answer, reasoning
    return "", raw


def _generation_prompt(task_name: str, prompt: str, tokenizer) -> str:
    # all tasks use chat template with same instruction wrapping used at training time
    messages = [{"role": "user", "content": user_content_for(task_name, prompt)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_and_merge_adapter(
    base_model: AutoModelForCausalLM, adapter_dir: str
) -> AutoModelForCausalLM:
    base_model.config.use_cache = True  # re-enable for generation
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged = peft_model.merge_and_unload()
    logger.info("Adapter merged and unloaded successfully")
    return merged


def evaluate_trace_task(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task_name: str,
    trace_dir: str,
    split: str = "test",
    max_samples: int | None = None,
    seed: int = 42,
    batch_size: int = 32,
    max_new_tokens: int | None = None,
    predictions_path: str | None = None,
    log_every_n_batches: int = 5,
) -> dict:
    _task_max_new_tokens = {"code": 256, "sum": 256, "qa": 128}
    if max_new_tokens is None:
        max_new_tokens = _task_max_new_tokens.get(task_name, 256)

    model.config.use_cache = True
    model.eval()

    task_folder = TASK_FOLDER_MAP[task_name]
    split_file = SPLIT_FILE_MAP[split]
    file_path = f"{trace_dir}/{task_folder}/{split_file}"

    logger.info(f"Loading {split} data from {file_path}...")
    with open(file_path) as f:
        data = json.load(f)

    if max_samples is not None and max_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, max_samples)

    total = len(data)
    logger.info(f"Evaluating {total} {split} examples for task '{task_name}'...")

    prompts = [
        _generation_prompt(task_name, item["prompt"], tokenizer) for item in data
    ]
    ground_truths = [item["answer"].strip() for item in data]

    raw_predictions: list[str] = []

    # mistral is decoder-only so require left-padding during generation then all
    # sequences in a batch end with real tokens and generation starts correctly.
    # Restore to right-padding afterward (training needs it).
    tokenizer.padding_side = "left"
    encoded = [
        tokenizer(p, add_special_tokens=False, truncation=True, max_length=1024)
        for p in prompts
    ]
    # pre-tokenise to get per-sample lengths and feed cached encodings to generation
    # sorting the run by length means each batch pads to its own longest sample rather than the global max - saves time
    sorted_indices = sorted(range(total), key=lambda i: len(encoded[i]["input_ids"]))

    raw_predictions: list[str | None] = [None] * total

    pad_id = tokenizer.pad_token_id
    eos_id = gen_eos_ids(tokenizer, model) or tokenizer.eos_token_id
    n_batches = (total + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(range(0, total, batch_size)):
        batch_indices = sorted_indices[batch_start : batch_start + batch_size]
        batch_encoded = [encoded[i] for i in batch_indices]

        inputs = tokenizer.pad(
            batch_encoded,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        new_tokens = outputs[:, prompt_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for orig_idx, text in zip(batch_indices, decoded):
            raw_predictions[orig_idx] = text

        n_done = min(batch_start + batch_size, total)
        if (batch_idx + 1) % log_every_n_batches == 0 or n_done == total:
            logger.info(
                f"  [{n_done}/{total}] generated  (batch {batch_idx + 1}/{n_batches})"
            )

    tokenizer.padding_side = "right"
    model.config.use_cache = False

    preds = [p.strip() for p in raw_predictions]

    # ScienceQA
    if task_name == "qa":
        pred_answers, pred_reasons = zip(*[extract_scienceqa(p) for p in preds])
        gt_answers, gt_reasons = zip(*[extract_scienceqa(g) for g in ground_truths])

        results = {
            "task": task_name,
            "split": split,
            "n_samples": total,
            "answer_accuracy": accuracy(list(pred_answers), list(gt_answers)),
            "reasoning_bleu1": bleu_n(list(pred_reasons), list(gt_reasons), n=1),
            "reasoning_bleu4": bleu_n(list(pred_reasons), list(gt_reasons), n=4),
            "reasoning_rouge_l": rouge_l(list(pred_reasons), list(gt_reasons)),
        }

    # 20Minuten
    elif task_name == "sum":
        results = {
            "task": task_name,
            "split": split,
            "n_samples": total,
            "bleu1": bleu_n(preds, ground_truths, n=1),
            "bleu4": bleu_n(preds, ground_truths, n=4),
            "rouge_l": rouge_l(preds, ground_truths),
        }
    # Py150
    elif task_name.startswith("code"):
        results = {
            "task": task_name,
            "split": split,
            "n_samples": total,
            "fuzzy_match": fuzzy_match(preds, ground_truths),
            "fuzzy_match_truncated": fuzzy_match(preds, ground_truths, truncate=True),
            "exact_match_first_line": exact_match_first_line(preds, ground_truths),
        }

    else:
        results = {
            "task": task_name,
            "split": split,
            "n_samples": total,
            "exact_match": accuracy(preds, ground_truths),
        }

    if predictions_path is not None:

        out = Path(predictions_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for idx, (pred, ref) in enumerate(zip(preds, ground_truths)):
                record: dict = {"idx": idx, "reference": ref, "prediction": pred}
                if task_name.startswith("code"):
                    pred_trunc = _truncate_to_first_line(pred)
                    ref_trunc = _truncate_to_first_line(ref)
                    record.update(
                        {
                            "prediction_truncated": pred_trunc,
                            "fuzz_ratio_full": fuzz.ratio(
                                _normalize_py150(pred), _normalize_py150(ref)
                            ),
                            "fuzz_ratio_truncated": fuzz.ratio(
                                _normalize_py150(pred_trunc),
                                _normalize_py150(ref_trunc),
                            ),
                            "exact_match_first_line": (
                                _normalize_py150(pred_trunc).strip()
                                == _normalize_py150(ref_trunc).strip()
                            ),
                        }
                    )
                elif task_name == "qa":
                    p_ans, p_reason = extract_scienceqa(pred)
                    r_ans, r_reason = extract_scienceqa(ref)
                    record.update(
                        {
                            "parsed_pred_answer": p_ans,
                            "parsed_pred_reasoning": p_reason,
                            "parsed_ref_answer": r_ans,
                            "parsed_ref_reasoning": r_reason,
                            "answer_correct": p_ans == r_ans,
                        }
                    )
                f.write(json.dumps(record) + "\n")
        logger.info(f"Saved {total} predictions to {out}")

    return results
