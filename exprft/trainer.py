from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer


def weighted_response_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor,
                           normalizer: float) -> torch.Tensor:
    """Σ_i w_i Σ_t CE_it / normalizer.  logits (B,T,V); labels (B,T) with -100 on
    ignored positions; weights (B,); normalizer a positive constant."""
    if normalizer <= 0:
        raise ValueError("normalizer must be positive")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    b_idx, t_idx = mask.nonzero(as_tuple=True)
    if b_idx.numel() == 0:
        return logits.sum() * 0.0
    tok_logits = shift_logits[b_idx, t_idx].float()
    tok_labels = shift_labels[b_idx, t_idx]
    tok_ce = F.cross_entropy(tok_logits, tok_labels, reduction="none")
    tok_w = weights.to(tok_ce.dtype).to(tok_ce.device)[b_idx]
    return (tok_w * tok_ce).sum() / normalizer


def dataset_token_normalizer(weights: list[float], n_response_tokens: list[int], per_device_batch: int) -> float:
    """c = per_device_batch · mean_i(w_i · n_i) over the training set."""
    if len(weights) != len(n_response_tokens) or not weights:
        raise ValueError("weights and n_response_tokens must be non-empty and the same length")
    wn = [float(w) * int(n) for w, n in zip(weights, n_response_tokens)]
    return per_device_batch * sum(wn) / len(wn)


class WeightedCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            labels.append(list(f["labels"]) + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weight": torch.tensor([float(f["weight"]) for f in features], dtype=torch.float32),
        }


class ExpRFTTrainer(Trainer):
    def __init__(self, *args, normalizer: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = float(normalizer)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight")
        labels = inputs.pop("labels")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        loss = weighted_response_loss(outputs.logits, labels, weights, self.normalizer)
        return (loss, outputs) if return_outputs else loss
