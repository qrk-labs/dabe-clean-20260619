import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BitmaskEncoder


class GumbelEncoder(BitmaskEncoder):
    def __init__(self, config: dict):
        super().__init__(config)
        self.vocab_size = config.get("vocab_size", 32000)
        self.embed_dim = config.get("embed_dim", 256)

        self.proj = nn.Linear(self.embed_dim, self.max_bit_width * 2)
        self.code_embed = nn.Parameter(torch.randn(self.max_bit_width, self.embed_dim))
        self.decode_proj = nn.Linear(self.embed_dim, self.vocab_size)
        self.tau = config.get("gumbel_tau", 1.0)

    def encode(self, span: str, bit_width: int) -> torch.Tensor:
        bits = torch.randint(0, 2, (self.total_bit_width(bit_width),))
        header = self.build_header(bit_width)
        bits[: self.num_header_bits] = header
        return bits

    def decode(self, bits: torch.Tensor) -> str:
        return "<decoded_text>"

    def embed(self, bits: torch.Tensor) -> torch.Tensor:
        total_bw = bits.shape[-1]
        code_bits = bits[self.num_header_bits : total_bw]
        code = 2.0 * code_bits.float() - 1.0
        embedded = code @ self.code_embed[: total_bw - self.num_header_bits]
        return embedded

    def forward(self, h: torch.Tensor, bit_width: int) -> tuple:
        logits = self.proj(h)
        logits = logits.view(*logits.shape[:-1], self.max_bit_width, 2)
        logits = logits[..., :bit_width, :]
        bits = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
        bits_hard = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        bits = bits_hard + bits - bits.detach()
        code = bits[..., 0]
        embedded = code @ self.code_embed[:bit_width]
        out = self.decode_proj(embedded)
        return out, code
