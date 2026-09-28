import logging
import os
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

from training.der_collator import DERDataCollator
from training.der_trainer import DERTrainer

logger = logging.getLogger(__name__)


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=dropout,
        bias=bias,
        task_type="CAUSAL_LM",
    )


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def build_training_args(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    lr: float = 2e-4,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    seed: int = 42,
    bf16: bool = True,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=bf16,
        seed=seed,
        data_seed=seed,
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="no",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        group_by_length=True,
    )


def lora_finetune(
    base_model: AutoModelForCausalLM,
    lora_config: LoraConfig,
    train_set: Dataset,
    train_args: TrainingArguments,
    tokenizer: AutoTokenizer,
    adapter_output_dir: str,
) -> tuple[PeftModel, str]:
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=peft_model,
        args=train_args,
        train_dataset=train_set,
        data_collator=collator,
    )

    train_result = trainer.train()
    logger.info(f"  Loss: {train_result.metrics.get('train_loss', 'n/a'):.4f}")

    logger.info(f"Training complete. Saving LoRA adapter to {adapter_output_dir}...")
    peft_model.save_pretrained(adapter_output_dir)
    return peft_model, adapter_output_dir


def lora_finetune_der(
    base_model: AutoModelForCausalLM,
    lora_config: LoraConfig,
    train_set: Dataset,
    train_args: TrainingArguments,
    tokenizer: AutoTokenizer,
    adapter_output_dir: str,
    teacher_logits_cache: list,
    alpha: float = 1.0,
    beta: float = 1.0,
    variant: str = "der++",
) -> tuple[PeftModel, str]:
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    train_args.group_by_length = False

    # keep is_replay and replay_index
    train_args.remove_unused_columns = False

    collator = DERDataCollator(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = DERTrainer(
        teacher_logits_cache=teacher_logits_cache,
        alpha=alpha,
        beta=beta,
        variant=variant,
        model=peft_model,
        args=train_args,
        train_dataset=train_set,
        data_collator=collator,
    )

    train_result = trainer.train()
    logger.info(f"  Loss: {train_result.metrics.get('train_loss', 'n/a'):.4f}")

    logger.info(f"Training complete. Saving LoRA adapter to {adapter_output_dir}...")
    peft_model.save_pretrained(adapter_output_dir)
    return peft_model, adapter_output_dir
