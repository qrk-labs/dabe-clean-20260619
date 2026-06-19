from typing import List, Tuple

import torch


class HammingAnalyzer:
    def __init__(self):
        self.history: List[dict] = []

    def compute_pairwise(self, encodings: torch.Tensor) -> torch.Tensor:
        B, D = encodings.shape
        expanded = encodings.unsqueeze(1).expand(B, B, D)
        pairwise = (expanded != expanded.transpose(0, 1)).sum(dim=-1)
        return pairwise

    def intra_language_distance(self, encodings: torch.Tensor, labels: List[str]) -> dict:
        pairwise = self.compute_pairwise(encodings)
        unique_labels = set(labels)
        results = {}
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            if len(indices) < 2:
                continue
            mask = torch.zeros(len(labels), len(labels), dtype=torch.bool)
            for i in indices:
                for j in indices:
                    if i < j:
                        mask[i, j] = True
            distances = pairwise[mask].float()
            results[label] = {
                "mean": distances.mean().item(),
                "std": distances.std().item(),
                "min": distances.min().item(),
                "max": distances.max().item(),
            }
        return results

    def cross_language_distance(self, encodings: torch.Tensor, labels: List[str]) -> dict:
        pairwise = self.compute_pairwise(encodings)
        unique_labels = list(set(labels))
        results = {}
        for i, l1 in enumerate(unique_labels):
            for l2 in unique_labels[i + 1:]:
                idx1 = [j for j, l in enumerate(labels) if l == l1]
                idx2 = [j for j, l in enumerate(labels) if l == l2]
                distances = pairwise[torch.tensor(idx1)[:, None], torch.tensor(idx2)[None, :]]
                results[f"{l1}-{l2}"] = {
                    "mean": distances.mean().item(),
                    "std": distances.std().item(),
                }
        return results
