import pytest
import torch

from src.evaluation.hamming_analysis import HammingAnalyzer
from src.evaluation.compression import CompressionMetrics, CompressionReport


class TestHammingAnalyzer:
    def test_compute_pairwise(self):
        analyzer = HammingAnalyzer()
        encodings = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 0, 0]])
        pairwise = analyzer.compute_pairwise(encodings)
        assert pairwise.shape == (3, 3)
        assert pairwise[0, 1] == 2
        assert pairwise[0, 2] == 2

    def test_intra_language_distance(self):
        analyzer = HammingAnalyzer()
        encodings = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=torch.float)
        labels = ["en", "en", "fr", "fr"]
        results = analyzer.intra_language_distance(encodings, labels)
        assert "en" in results
        assert "fr" in results


class TestCompressionMetrics:
    def test_compute(self):
        metrics = CompressionMetrics()
        report = metrics.compute(
            texts=["hello world", "test"],
            bpe_token_counts=[4, 2],
            spans_per_text=[[1, 2, 3], [1]],
            bit_widths_per_text=[[8, 16, 32], [16]],
        )
        assert isinstance(report, CompressionReport)
        assert report.total_bitmasks == 4
        assert report.total_bits == 72
        assert report.avg_bit_width == 18.0
