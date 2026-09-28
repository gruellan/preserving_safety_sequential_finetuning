import logging

from datasets import Dataset, concatenate_datasets

logger = logging.getLogger(__name__)


def build_der_dataset(
    task_dataset: Dataset,
    replay_dataset: Dataset,
    seed: int = 42,
) -> Dataset:
    selected_replay = replay_dataset

    # normal task samples have no replay metadata so tag them then collator/trainer
    # treats them as ordinary CE loss samples (replay_index -1 = none)
    task_dataset = task_dataset.add_column("is_replay", [0] * len(task_dataset))
    task_dataset = task_dataset.add_column("replay_index", [-1] * len(task_dataset))

    combined = concatenate_datasets([task_dataset, selected_replay])
    combined = combined.shuffle(seed=seed)

    logger.info(
        f"DER dataset: {len(task_dataset)} task + {len(selected_replay)} replay "
        f"= {len(combined)} samples"
    )
    return combined
