import torch
import torch.nn as nn

from .transformer import TransformerBlock


class DABEAdapter(nn.Module):
    """Section 3 adapter fusion: variable-width bits as a residual adapter signal."""

    def __init__(
        self,
        hidden_dim: int,
        min_bit_width: int = 8,
        max_bit_width: int = 24,
        num_header_bits: int = 4,
        bit_embed_init_std: float = 0.001,
        width_embed_init_std: float = 0.001,
        gate_bias_init: float = -6.0,
        gate_weight_init_std: float = 0.0,
        width_predictor_weight_init_std: float = 0.0,
        width_predictor_bias_init: float = 0.0,
        bit_predictor_weight_init_std: float = 0.0,
        bit_predictor_bias_init: float = 0.0,
        residual_scale_init: float = 0.0,
        use_adapter_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_bit_width = int(min_bit_width)
        self.max_bit_width = int(max_bit_width)
        self.num_header_bits = int(num_header_bits)
        self.use_adapter_norm = bool(use_adapter_norm)

        self.width_predictor = nn.Linear(hidden_dim, 1)
        self.bit_predictor = nn.Linear(hidden_dim, self.max_bit_width)
        self.bit_embed = nn.Parameter(torch.empty(self.max_bit_width, hidden_dim))
        self.width_embed = nn.Embedding(self.max_bit_width + 1, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, 1)
        self.adapter_norm = nn.LayerNorm(hidden_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

        nn.init.normal_(self.bit_embed, mean=0.0, std=float(bit_embed_init_std))
        nn.init.normal_(self.width_embed.weight, mean=0.0, std=float(width_embed_init_std))
        nn.init.normal_(
            self.width_predictor.weight,
            mean=0.0,
            std=float(width_predictor_weight_init_std),
        )
        nn.init.constant_(self.width_predictor.bias, float(width_predictor_bias_init))
        nn.init.normal_(
            self.bit_predictor.weight,
            mean=0.0,
            std=float(bit_predictor_weight_init_std),
        )
        nn.init.constant_(self.bit_predictor.bias, float(bit_predictor_bias_init))
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=float(gate_weight_init_std))
        nn.init.constant_(self.gate_proj.bias, float(gate_bias_init))

    def _sample_bit_widths(self, hidden: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.width_predictor(hidden)).squeeze(-1)
        widths = self.min_bit_width + torch.round(
            scores * float(self.max_bit_width - self.min_bit_width)
        ).long()
        return torch.clamp(widths, min=self.min_bit_width, max=self.max_bit_width)

    def _build_header_bits(self, widths: torch.Tensor) -> torch.Tensor:
        max_code = 2**self.num_header_bits - 1
        codes = torch.clamp(widths, min=0, max=max_code).long()
        shifts = torch.arange(self.num_header_bits, device=widths.device, dtype=torch.long)
        return ((codes.unsqueeze(-1) >> shifts) & 1).long()

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        widths = self._sample_bit_widths(hidden)
        logits = self.bit_predictor(hidden)
        z = torch.tanh(logits)
        hard_bits = (z > 0).float()
        bits = hard_bits + z - z.detach()

        positions = torch.arange(self.max_bit_width, device=hidden.device)
        active_mask = (positions.view(1, 1, -1) < widths.unsqueeze(-1)).float()
        bits = bits * active_mask
        hard_bits = hard_bits * active_mask

        denom = widths.clamp(min=1).unsqueeze(-1).float()
        bit_hidden = (bits @ self.bit_embed) / denom
        width_hidden = self.width_embed(widths)
        adapter_signal = bit_hidden + width_hidden
        if self.use_adapter_norm:
            adapter_signal = self.adapter_norm(adapter_signal)
        gate = torch.sigmoid(self.gate_proj(hidden))
        adapted_hidden = hidden + self.residual_scale * gate * adapter_signal

        header_bits = self._build_header_bits(widths)
        bit_values = torch.cat([header_bits, hard_bits.long()], dim=-1)
        return adapted_hidden, widths, bit_values, gate


class AdapterTokenTransformerLM(nn.Module):
    """Standard token LM with DABE adapter fusion before LM head (sections 3 and 4)."""

    def __init__(self, config: dict):
        super().__init__()
        self.hidden_dim = int(config.get("hidden_dim", 256))
        self.num_layers = int(config.get("num_layers", 4))
        self.num_heads = int(config.get("num_heads", 4))
        self.ff_dim = int(config.get("ff_dim", 1024))
        self.dropout = float(config.get("dropout", 0.1))
        self.vocab_size = int(config.get("vocab_size", 32000))
        self.tie_weights = bool(config.get("tie_weights", True))

        adapter_cfg = config.get("adapter", {})
        self.adapter = DABEAdapter(
            hidden_dim=self.hidden_dim,
            min_bit_width=int(adapter_cfg.get("min_bit_width", 8)),
            max_bit_width=int(adapter_cfg.get("max_bit_width", 24)),
            num_header_bits=int(adapter_cfg.get("num_header_bits", 4)),
            bit_embed_init_std=float(adapter_cfg.get("bit_embed_init_std", 0.001)),
            width_embed_init_std=float(adapter_cfg.get("width_embed_init_std", 0.001)),
            gate_bias_init=float(adapter_cfg.get("gate_bias_init", -6.0)),
            gate_weight_init_std=float(adapter_cfg.get("gate_weight_init_std", 0.0)),
            width_predictor_weight_init_std=float(
                adapter_cfg.get("width_predictor_weight_init_std", 0.0)
            ),
            width_predictor_bias_init=float(adapter_cfg.get("width_predictor_bias_init", 0.0)),
            bit_predictor_weight_init_std=float(
                adapter_cfg.get("bit_predictor_weight_init_std", 0.0)
            ),
            bit_predictor_bias_init=float(adapter_cfg.get("bit_predictor_bias_init", 0.0)),
            residual_scale_init=float(adapter_cfg.get("residual_scale_init", 0.0)),
            use_adapter_norm=bool(adapter_cfg.get("use_adapter_norm", True)),
        )

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=not self.tie_weights)
        if self.tie_weights:
            self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, mask=mask)
        hidden = self.norm(hidden)
        adapted, widths, bit_values, gate = self.adapter(hidden)
        logits = self.lm_head(adapted)
        adapter_stats = {
            "bit_widths": widths,
            "bit_values": bit_values,
            "gate_mean": gate.mean(),
            "residual_scale": self.adapter.residual_scale.detach(),
        }
        return logits, adapted, adapter_stats
