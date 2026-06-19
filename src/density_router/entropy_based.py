from typing import List

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .base import DensityRouter, Span


class EntropyRouter(DensityRouter):
    def __init__(self, config: dict):
        super().__init__(config)
        model_name = config.get("lm_model", "bert-base-multilingual-cased")
        device_pref = str(config.get("device", "cpu")).lower()
        if device_pref == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device_pref

        self.device = torch.device(resolved_device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.stride = config.get("stride", 1)
        self.compute_mode = str(config.get("compute_mode", "quality")).lower()
        self.fast_max_words = int(config.get("fast_max_words", 128))
        self.fast_cache_size = int(config.get("fast_cache_size", 4096))
        self._score_cache: dict[str, List[float]] = {}
        self._segment_cache: dict[str, List[Span]] = {}
        self._cache_order: list[str] = []
        self.balance_bias_update = bool(config.get("balance_bias_update", False))
        self.balance_bias_lr = float(config.get("balance_bias_lr", 0.02))
        self._min_bits = int(self.config.get("min_bit_width", 8))
        self._max_bits = int(self.config.get("max_bit_width", 32))
        self._widths = list(range(self._min_bits, self._max_bits + 1))
        self._width_bias = {width: 0.0 for width in self._widths}
        self._width_counts = {width: 1.0 for width in self._widths}

    def _cache_get(self, store: dict[str, List], key: str):
        return store.get(key)

    def _cache_put(self, store: dict[str, List], key: str, value: List) -> None:
        store[key] = value
        self._cache_order.append(key)
        if len(self._cache_order) <= self.fast_cache_size:
            return
        stale = self._cache_order.pop(0)
        self._score_cache.pop(stale, None)
        self._segment_cache.pop(stale, None)

    def _balance_width(self, base_width: int) -> int:
        if not self.balance_bias_update:
            return base_width
        target = 1.0 / max(len(self._widths), 1)
        total = sum(self._width_counts.values())
        if total <= 0:
            return base_width
        for width in self._widths:
            usage = self._width_counts[width] / total
            self._width_bias[width] += self.balance_bias_lr * (target - usage)
            self._width_bias[width] = max(-1.5, min(1.5, self._width_bias[width]))
        scored = [
            (-(abs(width - base_width)) + self._width_bias[width], width)
            for width in self._widths
        ]
        scored.sort(reverse=True)
        chosen = int(scored[0][1])
        self._width_counts[chosen] += 1.0
        return chosen

    def score(self, text: str | List[str]) -> List[float]:
        if isinstance(text, list):
            text = " ".join(text)
        if self.compute_mode == "fast":
            words = text.split()
            if self.fast_max_words > 0 and len(words) > self.fast_max_words:
                text = " ".join(words[: self.fast_max_words])
            cached = self._cache_get(self._score_cache, text)
            if cached is not None:
                return list(cached)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        scores = entropy.squeeze(0).tolist()
        reduced = scores[:: self.stride]
        if self.compute_mode == "fast":
            self._cache_put(self._score_cache, text, reduced)
        return reduced

    def segment(self, text: str, scores: List[float]) -> List[Span]:
        spans = []
        if self.compute_mode == "fast":
            cached = self._cache_get(self._segment_cache, text)
            if cached is not None:
                return [
                    Span(
                        text=item.text,
                        start=item.start,
                        end=item.end,
                        density=item.density,
                        bit_width=item.bit_width,
                    )
                    for item in cached
                ]

        if scores:
            min_s = min(scores)
            max_s = max(scores)
            range_s = max(max_s - min_s, 1e-8)
            normalized = [(s - min_s) / range_s for s in scores]
        else:
            normalized = []

        words = text.split()
        n_words = len(words)
        n_scores = len(normalized)

        if n_scores == 0 or not text:
            return []

        running_start = 0
        for i in range(0, n_words, self.config.get("segment_stride", 1)):
            idx = min(i, n_scores - 1)
            density = normalized[idx]
            base_bw = self.get_bit_width(density)
            bw = self._balance_width(base_bw)
            if i > 0:
                running_start += len(words[i - 1]) + 1
            start = running_start
            end = start + len(words[i]) + (1 if i < n_words - 1 else 0)
            spans.append(
                Span(
                    text=words[i],
                    start=start,
                    end=end,
                    density=density,
                    bit_width=bw,
                )
            )
        if self.compute_mode == "fast":
            self._cache_put(self._segment_cache, text, spans)
        return spans
