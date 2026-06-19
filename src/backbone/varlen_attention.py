import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - depends on torch version
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]


class VarLenAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_backend: str = "auto",
        compressed_kv_proxy: bool = False,
        compression_rank: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        self.attention_backend = str(attention_backend).lower()
        self.compressed_kv_proxy = bool(compressed_kv_proxy)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if self.compressed_kv_proxy:
            rank = max(8, min(int(compression_rank), hidden_dim))
            self.k_compress = nn.Linear(hidden_dim, rank, bias=False)
            self.v_compress = nn.Linear(hidden_dim, rank, bias=False)
            self.k_expand = nn.Linear(rank, hidden_dim, bias=False)
            self.v_expand = nn.Linear(rank, hidden_dim, bias=False)

    def _sdpa_context(self):
        if sdpa_kernel is None or SDPBackend is None:
            return nullcontext()
        backend = self.attention_backend
        if backend in {"auto", "default"}:
            return nullcontext()
        if backend == "flash":
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        if backend in {"mem_efficient", "efficient"}:
            return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        if backend == "math":
            return sdpa_kernel(SDPBackend.MATH)
        return nullcontext()

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        bit_widths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del bit_widths
        B, T, D = x.shape
        q = self._reshape(self.q_proj(x))
        k_input = self.k_proj(x)
        v_input = self.v_proj(x)
        if self.compressed_kv_proxy:
            k_input = self.k_expand(self.k_compress(k_input))
            v_input = self.v_expand(self.v_compress(v_input))
        k = self._reshape(k_input)
        v = self._reshape(v_input)

        with self._sdpa_context():
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out
