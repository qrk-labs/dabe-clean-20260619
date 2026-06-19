from typing import List

from .base import DensityRouter, Span


class FixedRouter(DensityRouter):
    """Router that assigns a constant density score and fixed bit width per span."""

    def score(self, text: str | List[str]) -> List[float]:
        if isinstance(text, list):
            text = " ".join(text)
        return [0.5] * len(text.split())

    def segment(self, text: str, scores: List[float]) -> List[Span]:
        words = text.split()
        if not words:
            return []

        fixed_bit_width = self.config.get(
            "fixed_bit_width",
            self.config.get("min_bit_width", self.config.get("max_bit_width", 32)),
        )

        spans: List[Span] = []
        for i, word in enumerate(words):
            start = len(" ".join(words[:i]))
            end = start + len(word)
            spans.append(
                Span(
                    text=word,
                    start=start,
                    end=end,
                    density=0.5,
                    bit_width=fixed_bit_width,
                )
            )
        return spans
