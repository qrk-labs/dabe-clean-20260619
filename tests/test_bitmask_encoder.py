import pytest
import torch

from src.bitmask_encoder.base import BitmaskEncoder
from src.bitmask_encoder.lfq_encoder import LFQEncoder


class TestLFQEncoder:
    @pytest.fixture
    def config(self):
        return {
            "max_bit_width": 32,
            "num_header_bits": 4,
            "vocab_size": 32000,
            "embed_dim": 256,
        }

    def test_init(self, config):
        enc = LFQEncoder(config)
        assert enc is not None

    def test_encode_returns_correct_length(self, config):
        enc = LFQEncoder(config)
        bits = enc.encode("hello", bit_width=16)
        assert bits.shape[0] == 16 + 4

    def test_header_bits(self, config):
        enc = LFQEncoder(config)
        header = enc.build_header(8)
        assert header.shape[0] == 4
        assert header.dtype == torch.long

    def test_hamming_distance(self, config):
        enc = LFQEncoder(config)
        a = torch.tensor([1, 0, 1, 0])
        b = torch.tensor([1, 1, 0, 0])
        assert enc.hamming_distance(a, b) == 2

    def test_straight_through_shape(self, config):
        enc = LFQEncoder(config)
        h = torch.randn(2, 256)
        bits = enc.straight_through(h, bit_width=16)
        assert bits.shape == (2, 16)

    def test_train_tokenizer_learns_deterministic_codes(self):
        enc = LFQEncoder({
            "max_bit_width": 24,
            "num_header_bits": 4,
            "vocab_size": 64,
            "embed_dim": 32,
        })
        spans = ["the", "cat", "sat", "the", "cat", "the", "dog"]
        widths = [8, 12, 24, 8, 12, 8, 16]
        stats = enc.train_tokenizer(
            spans=spans,
            bit_widths=widths,
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            device="cpu",
        )
        assert stats["vocab_size"] >= 4
        bits_a = enc.encode("cat", bit_width=12)
        bits_b = enc.encode("cat", bit_width=12)
        assert torch.equal(bits_a, bits_b)

    def test_export_and_load_artifact_roundtrip(self, tmp_path):
        config = {
            "max_bit_width": 24,
            "num_header_bits": 4,
            "vocab_size": 64,
            "embed_dim": 32,
        }
        enc = LFQEncoder(config)
        enc.train_tokenizer(
            spans=["moon", "light", "moon", "star"],
            bit_widths=[16, 8, 16, 12],
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            device="cpu",
        )
        artifact_path = enc.export_artifact(tmp_path / "lfq_artifact.pt")
        reloaded = LFQEncoder(config)
        reloaded.load_artifact(artifact_path)
        assert torch.equal(enc.encode("moon", 16), reloaded.encode("moon", 16))


class TestBitmaskEncoder:
    def test_abstract_enforces_implementation(self):
        with pytest.raises(TypeError):
            BitmaskEncoder({})
