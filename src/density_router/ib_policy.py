from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DensityRouter, Span


class InformationBottleneckPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_buckets: int = 8):
        super().__init__()
        self.num_buckets = num_buckets
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_buckets),
        )
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, beta: float = 1.0) -> tuple:
        logits = self.encoder(x)
        gumbel = F.gumbel_softmax(logits, tau=self.log_tau.exp(), hard=False, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        kl = (probs * (probs.log() - 1.0 / self.num_buckets)).sum(dim=-1).mean()
        ib_loss = beta * kl
        return gumbel, ib_loss


class InformationBottleneckRouter(DensityRouter):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ib_policy = InformationBottleneckPolicy(
            hidden_dim=config.get("hidden_dim", 256),
            num_buckets=config.get("num_density_buckets", 8),
        )

    def score(self, text: str | List[str]) -> List[float]:
        if isinstance(text, list):
            text = " ".join(text)
        dummy_scores = [0.5] * len(text.split())
        return dummy_scores

    def segment(self, text: str, scores: List[float]) -> List[Span]:
        spans = []
        words = text.split()
        n_scores = len(scores)

        for i, word in enumerate(words):
            idx = min(i, n_scores - 1) if n_scores > 0 else 0
            density = scores[idx] if n_scores > 0 else 0.5
            bw = self.get_bit_width(density)
            start = len(" ".join(words[:i]))
            end = start + len(word)
            spans.append(
                Span(
                    text=word,
                    start=start,
                    end=end,
                    density=density,
                    bit_width=bw,
                )
            )
        return spans
