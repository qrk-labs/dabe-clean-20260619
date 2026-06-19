from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BitmaskEncoder(nn.Module, ABC):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.max_bit_width = config.get("max_bit_width", 32)
        self.num_header_bits = config.get("num_header_bits", 4)

    @abstractmethod
    def encode(self, span: str, bit_width: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, bits: torch.Tensor) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed(self, bits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def total_bit_width(self, bit_width: int) -> int:
        return bit_width + self.num_header_bits

    def hamming_distance(self, a: torch.Tensor, b: torch.Tensor) -> int:
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        return int((a != b).sum().item())

    def build_header(self, bit_width: int) -> torch.Tensor:
        header = torch.zeros(self.num_header_bits, dtype=torch.long)
        code = min(bit_width, 2**self.num_header_bits - 1)
        for i in range(self.num_header_bits):
            header[i] = (code >> i) & 1
        return header
