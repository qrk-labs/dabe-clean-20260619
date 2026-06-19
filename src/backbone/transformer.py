import torch
import torch.nn as nn

from .bitmask_projection import BitmaskProjection
from .varlen_attention import VarLenAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        attention_cfg: dict | None = None,
    ):
        super().__init__()
        attention_cfg = attention_cfg or {}
        self.attention = VarLenAttention(
            hidden_dim,
            num_heads,
            dropout,
            attention_backend=str(attention_cfg.get("backend", "auto")),
            compressed_kv_proxy=bool(attention_cfg.get("compressed_kv_proxy", False)),
            compression_rank=int(attention_cfg.get("compression_rank", 64)),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask=mask)
        x = x + self.ff(self.norm2(x))
        return x


class DABETransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_layers = config.get("num_layers", 6)
        self.num_heads = config.get("num_heads", 8)
        self.ff_dim = config.get("ff_dim", 2048)
        self.dropout = config.get("dropout", 0.1)
        self.max_bit_width = config.get("max_bit_width", 32)
        self.vocab_size = config.get("vocab_size", 32000)
        self.mtp_heads = int(config.get("mtp_heads", 0))
        self.attention_cfg = dict(config.get("attention", {}))

        self.bitmask_proj = BitmaskProjection(
            max_bit_width=self.max_bit_width,
            hidden_dim=self.hidden_dim,
        )

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

        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)
        self.mtp_heads_proj = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.vocab_size)
            for _ in range(max(0, self.mtp_heads))
        ])

    def forward(
        self,
        bits: torch.Tensor,
        bit_widths: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple:
        h = self.bitmask_proj(bits, bit_widths)
        for block in self.blocks:
            h = block(h, mask=mask)
        logits = self.lm_head(h)
        return logits, h

    def compute_mtp_logits(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        if not self.mtp_heads_proj:
            return []
        return [head(hidden) for head in self.mtp_heads_proj]
