from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BenchmarkResults:
    perplexity: float = 0.0
    accuracy: Dict[str, float] = field(default_factory=dict)
    bitext_retrieval_recall: Dict[str, float] = field(default_factory=dict)
    xli_f1: Dict[str, float] = field(default_factory=dict)
    translation_bleu: Dict[str, float] = field(default_factory=dict)


class BenchmarkSuite:
    def __init__(self):
        self.results = BenchmarkResults()

    def evaluate(self, model, datamodule) -> BenchmarkResults:
        return self.results
