from typing import List

import torch
import torch.nn as nn

from .base import DensityRouter, Span


class BoundaryPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        layers = []
        dims = [hidden_dim] * (num_layers + 1)
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedRouter(DensityRouter):
    def __init__(self, config: dict):
        super().__init__(config)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.vocab_size = config.get("vocab_size", 32000)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.predictor = BoundaryPredictor(
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(input_ids)
        logits = self.predictor(embeds)
        probs = torch.softmax(logits, dim=-1)
        return probs[..., 1]
