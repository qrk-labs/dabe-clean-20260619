import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BitmaskEncoder


class SemanticHashEncoder(BitmaskEncoder):
    def __init__(self, config: dict):
        super().__init__(config)
        self.vocab_size = config.get("vocab_size", 32000)
        self.embed_dim = config.get("embed_dim", 256)
        self.hidden_dim = config.get("hidden_dim", 512)

        self.encoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.max_bit_width),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.max_bit_width, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )
        self.decode_proj = nn.Linear(self.embed_dim, self.vocab_size)

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
        return self.decoder(code)

    def forward(self, h: torch.Tensor, bit_width: int) -> tuple:
        logits = self.encoder(h)
        z = torch.tanh(logits[..., :bit_width])
        bits = (z > 0).float()
        bits = bits + z - z.detach()
        recon = self.decoder(bits)
        out = self.decode_proj(recon)
        return out, bits
