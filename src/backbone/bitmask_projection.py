import torch
import torch.nn as nn


class BitmaskProjection(nn.Module):
    def __init__(self, max_bit_width: int = 32, hidden_dim: int = 512):
        super().__init__()
        self.max_bit_width = max_bit_width
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(max_bit_width, hidden_dim)

    def forward(self, bits: torch.Tensor, bit_widths: torch.Tensor | None = None) -> torch.Tensor:
        if bits.dim() == 1:
            bits = bits.unsqueeze(0)
        if bits.dim() == 2:
            bits = bits.unsqueeze(0)
        float_bits = 2.0 * bits.float() - 1.0
        batch_size, seq_len, bw = float_bits.shape
        if bw > self.max_bit_width:
            # Keep the leading bits (including header bits) when inputs exceed configured width.
            float_bits = float_bits[..., : self.max_bit_width]
            bw = self.max_bit_width
        if bw < self.max_bit_width:
            pad = torch.zeros(
                batch_size, seq_len, self.max_bit_width - bw,
                device=float_bits.device, dtype=float_bits.dtype,
            )
            float_bits = torch.cat([float_bits, pad], dim=-1)
        return self.projection(float_bits)
