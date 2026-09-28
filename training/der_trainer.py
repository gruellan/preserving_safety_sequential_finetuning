import logging

import torch
import torch.nn.functional as F
from transformers import Trainer

logger = logging.getLogger(__name__)


class DERTrainer(Trainer):
    def __init__(
        self,
        teacher_logits_cache: list,
        alpha: float = 1.0,
        beta: float = 1.0,
        variant: str = "der++",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # teacher_logits_cache[i] is a bf16 CPU tensor (response_len, vocab) for sample i
        self.teacher_logits_cache = teacher_logits_cache
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self._task_loss_sum = 0.0
        self._task_loss_steps = 0
        self._mse_sum = 0.0
        self._buffer_ce_sum = 0.0
        self._replay_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        is_replay = inputs.pop("is_replay")
        replay_index = inputs.pop("replay_index")

        # keep the unmasked labels for the MSE response-position
        # selection and the separate buffer CE term
        original_labels = inputs["labels"]

        # buffer samples never contribute to CE(task)
        # mask them so task_loss is task-only mean
        # their label replay is added back as beta * CE(buffer)
        model_labels = original_labels.clone()
        model_labels[is_replay == 1] = -100
        inputs["labels"] = model_labels

        outputs = model(**inputs)
        task_loss = outputs.loss
        # an all-replay batch has no task tokens -> HF returns NaN so treat as 0
        if torch.isnan(task_loss):
            task_loss = task_loss.new_zeros(())

        logits = outputs.logits
        replay_rows = (is_replay == 1).nonzero(as_tuple=True)[0]

        # alpha * MSE(student response logits, M0 teacher logits)
        # alpha==0 (ER, plain_matched control)
        mse_loss = task_loss.new_zeros(())
        for row in replay_rows if self.alpha != 0 else []:
            idx = int(replay_index[row].item())
            teacher = self.teacher_logits_cache[idx].to(logits.device).float()
            # logits[t] predicts token t+1, so shift before masking to keep the
            # logits that produce the response (including first token)
            shifted_mask = original_labels[row][1:] != -100
            student = logits[row][:-1][shifted_mask].float()
            if student.shape[0] != teacher.shape[0]:
                logger.warning(
                    f"replay sample {idx}: student/teacher length mismatch "
                    f"({student.shape[0]} vs {teacher.shape[0]}) - teacher cache stale"
                )
            aligned = min(student.shape[0], teacher.shape[0])
            if aligned == 0:
                continue
            mse_loss = mse_loss + F.mse_loss(student[:aligned], teacher[:aligned])
        if len(replay_rows) > 0:
            mse_loss = mse_loss / len(replay_rows)

        # beta * CE(buffer) on the gold refusal labels (der++ only)
        buffer_ce = task_loss.new_zeros(())
        if self.variant == "der++" and self.beta != 0 and len(replay_rows) > 0:
            shift_logits = logits[replay_rows][:, :-1, :]
            shift_labels = original_labels[replay_rows][:, 1:].to(logits.device)
            buffer_ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            if torch.isnan(buffer_ce):
                buffer_ce = task_loss.new_zeros(())

        total_loss = task_loss + self.beta * buffer_ce + self.alpha * mse_loss

        if not total_loss.requires_grad:
            total_loss = total_loss + logits.sum() * 0.0

        self._task_loss_sum += task_loss.detach().item()
        self._task_loss_steps += 1
        if len(replay_rows) > 0:
            self._mse_sum += float(mse_loss.detach().item())
            self._buffer_ce_sum += float(buffer_ce.detach().item())
            self._replay_steps += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs, *args, **kwargs):
        if self._task_loss_steps > 0:
            logs["task_loss"] = round(self._task_loss_sum / self._task_loss_steps, 4)
            if self._replay_steps > 0:
                logs["replay_mse"] = round(self._mse_sum / self._replay_steps, 4)
                logs["buffer_ce"] = round(self._buffer_ce_sum / self._replay_steps, 4)
            else:
                logs["replay_mse"] = 0.0
                logs["buffer_ce"] = 0.0
            logs["replay_batches"] = self._replay_steps
            self._task_loss_sum = self._mse_sum = self._buffer_ce_sum = 0.0
            self._task_loss_steps = self._replay_steps = 0
        return super().log(logs, *args, **kwargs)
