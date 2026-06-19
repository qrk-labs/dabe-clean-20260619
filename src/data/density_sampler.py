import random
from typing import List

import numpy as np


class DensitySampler:
    def __init__(self, density_buckets: List[float] | None = None):
        self.density_buckets = density_buckets or [0.1, 0.3, 0.5, 0.7, 0.9]

    def sample_balanced(self, spans: list, densities: List[float]) -> list:
        buckets = {b: [] for b in self.density_buckets}
        for span, d in zip(spans, densities):
            nearest = min(self.density_buckets, key=lambda x: abs(x - d))
            buckets[nearest].append(span)

        min_count = min(len(v) for v in buckets.values() if v)
        balanced = []
        for b in self.density_buckets:
            if buckets[b]:
                balanced.extend(random.sample(buckets[b], min(len(buckets[b]), min_count)))
        return balanced
