from dataclasses import dataclass, field
from typing import List


@dataclass
class CompressionReport:
    total_input_chars: int = 0
    total_tokens_bpe: int = 0
    total_bitmasks: int = 0
    total_bits: int = 0
    avg_bit_width: float = 0.0
    compression_ratio_vs_bpe: float = 0.0
    bits_per_char: float = 0.0


class CompressionMetrics:
    def compute(
        self,
        texts: List[str],
        bpe_token_counts: List[int],
        spans_per_text: List[List],
        bit_widths_per_text: List[List[int]],
    ) -> CompressionReport:
        report = CompressionReport()
        report.total_input_chars = sum(len(t) for t in texts)
        report.total_tokens_bpe = sum(bpe_token_counts)
        report.total_bitmasks = sum(len(spans) for spans in spans_per_text)
        report.total_bits = sum(sum(bw) for bw in bit_widths_per_text)
        report.avg_bit_width = report.total_bits / max(report.total_bitmasks, 1)
        report.compression_ratio_vs_bpe = (
            report.total_tokens_bpe / max(report.total_bitmasks, 1)
        )
        report.bits_per_char = report.total_bits / max(report.total_input_chars, 1)
        return report
