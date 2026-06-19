import pytest
import torch

from src.pipeline import DABEPipeline


@pytest.fixture
def config():
    return {
        "device": "cpu",
        "router_type": "entropy",
        "encoder_type": "lfq",
        "router": {
            "density_lower": 0.0,
            "density_upper": 1.0,
            "min_bit_width": 8,
            "max_bit_width": 32,
            "num_header_bits": 4,
            "lm_model": "bert-base-multilingual-cased",
        },
        "encoder": {
            "max_bit_width": 32,
            "num_header_bits": 4,
            "vocab_size": 32000,
            "embed_dim": 256,
        },
        "backbone": {
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 256,
            "dropout": 0.1,
            "max_bit_width": 32,
            "vocab_size": 32000,
        },
        "contrastive_temperature": 0.07,
        "contrastive_margin": 0.5,
    }


class TestDABEPipeline:
    def test_init(self, config):
        pipeline = DABEPipeline(config)
        assert pipeline is not None

    def test_validate_shapes(self, config):
        pipeline = DABEPipeline(config)
        shapes = pipeline.validate_shapes(batch_size=2, seq_len=4)
        assert shapes["logits_shape"] == (2, 4, 32000)
        assert shapes["hidden_shape"] == (2, 4, 128)

    def test_forward_returns_correct_structure(self, config):
        pipeline = DABEPipeline(config)
        texts = ["hello world", "bonjour le monde"]
        output = pipeline.forward(texts)
        assert len(output.texts) == 2
        assert len(output.spans) == 2
        assert len(output.bitmasks) == 2
        assert len(output.bit_widths) == 2

    def test_run_dummy_forward(self, config):
        pipeline = DABEPipeline(config)
        report = pipeline.run_dummy_forward()
        assert "num_texts" in report
        assert "shapes" in report
        assert report["num_texts"] == 2
