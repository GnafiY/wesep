"""Permutation-invariant waveform losses for blind source separation."""

from itertools import permutations

import torch
from auraloss.time import SISDRLoss
from auraloss.utils import apply_reduction


class PITSISDRLoss(torch.nn.Module):
    """Negative SI-SDR with one source assignment per mixture.

    Input and target both have shape [B, S, T]. The loss averages over S
    sources before selecting the best permutation independently for each B.
    With reduction="none", the result has shape [B].
    """

    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean", num_sources=2):
        super().__init__()
        if not isinstance(num_sources, int) or num_sources < 1:
            raise ValueError("num_sources must be a positive integer")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unknown reduction: {reduction}")
        self.sisdr_loss = SISDRLoss(zero_mean=zero_mean, eps=eps, reduction="none")
        self.reduction = reduction
        self.num_sources = num_sources
        self.perms = list(permutations(range(num_sources)))
        self.register_buffer("permutation_indices",
                             torch.tensor(self.perms, dtype=torch.long),
                             persistent=False)

    def forward(self, input, target):
        if input.ndim != 3 or input.shape != target.shape:
            raise ValueError("PIT expects equal [B, S, T] tensors, got "
                             f"{tuple(input.shape)} and {tuple(target.shape)}")
        if input.shape[1] != self.num_sources:
            raise ValueError(f"Expected {self.num_sources} sources, got "
                             f"{input.shape[1]}")

        # FP32 energy/projection calculations also protect AMP inputs.
        pairwise = self.sisdr_loss(
            input.float().unsqueeze(2), target.float().unsqueeze(1))  # [B, S, S]
        source_indices = torch.arange(self.num_sources, device=input.device)
        permutation_indices = self.permutation_indices.to(input.device)
        assigned = pairwise[:, source_indices, permutation_indices]  # [B, P, S]
        losses = assigned.mean(dim=-1).min(dim=-1).values  # [B]
        return apply_reduction(losses, self.reduction)
