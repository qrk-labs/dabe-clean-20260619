from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import torch.nn as nn


@dataclass
class Span:
    text: str
    start: int
    end: int
    density: float
    bit_width: int


class DensityRouter(nn.Module, ABC):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def score(self, text: str | List[str]) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def segment(self, text: str, scores: List[float]) -> List[Span]:
        raise NotImplementedError

    def get_bit_width(self, density: float) -> int:
        min_bits = self.config.get("min_bit_width", 8)
        max_bits = self.config.get("max_bit_width", 32)
        ratio = (density - self.config.get("density_lower", 0.0)) / max(
            self.config.get("density_upper", 1.0) - self.config.get("density_lower", 0.0), 1e-8
        )
        ratio = max(0.0, min(1.0, ratio))
        return min_bits + int(round(ratio * (max_bits - min_bits)))
