import torch
from transformers import DataCollatorForSeq2Seq


class DERDataCollator:
    """Pads batch like DataCollatorForSeq2Seq does but keeps DER metadata

    DataCollatorForSeq2Seq only knows about input_ids/attention_mask/labels and
    drops other columns so is_replay and replay_index fields are pulled out
    before padding then reattached after
    """

    def __init__(
        self,
        tokenizer,
        padding=True,
        label_pad_token_id=-100,
    ):
        self.seq2seq_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=padding,
            label_pad_token_id=label_pad_token_id,
        )

    def __call__(self, features: list[dict]) -> dict:
        is_replay = [f["is_replay"] for f in features]
        replay_index = [f["replay_index"] for f in features]

        features = [
            {k: v for k, v in f.items() if k not in ("is_replay", "replay_index")}
            for f in features
        ]
        batch = self.seq2seq_collator(features)

        batch["is_replay"] = torch.tensor(is_replay, dtype=torch.long)
        batch["replay_index"] = torch.tensor(replay_index, dtype=torch.long)
        return batch
