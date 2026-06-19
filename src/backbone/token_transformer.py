import torch
import torch.nn as nn

from .transformer import TransformerBlock


class TokenTransformerLM(nn.Module):
    """Small causal LM baseline with token embeddings (paper setup section 4)."""

    def __init__(self, config: dict):
        super().__init__()
        self.hidden_dim = int(config.get("hidden_dim", 256))
        self.num_layers = int(config.get("num_layers", 4))
        self.num_heads = int(config.get("num_heads", 4))
        self.ff_dim = int(config.get("ff_dim", 1024))
        self.dropout = float(config.get("dropout", 0.1))
        self.vocab_size = int(config.get("vocab_size", 32000))
        self.tie_weights = bool(config.get("tie_weights", True))
        self.mtp_heads = int(config.get("mtp_heads", 0))
        self.attention_cfg = dict(config.get("attention", {}))

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.hidden_dim,
                self.num_heads,
                self.ff_dim,
                self.dropout,
                attention_cfg=self.attention_cfg,
            )
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=not self.tie_weights)
        if self.tie_weights:
            self.lm_head.weight = self.embedding.weight
        self.mtp_heads_proj = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.vocab_size)
            for _ in range(max(0, self.mtp_heads))
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, mask=mask)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits, hidden

    def compute_mtp_logits(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        if not self.mtp_heads_proj:
            return []
        return [head(hidden) for head in self.mtp_heads_proj]
