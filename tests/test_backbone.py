import pytest
import torch

from src.backbone.adapter_transformer import AdapterTokenTransformerLM, DABEAdapter
from src.backbone.bitmask_projection import BitmaskProjection
from src.backbone.token_transformer import TokenTransformerLM
from src.backbone.transformer import DABETransformer
from src.backbone.varlen_attention import VarLenAttention


class TestBitmaskProjection:
    def test_shape(self):
        proj = BitmaskProjection(max_bit_width=32, hidden_dim=512)
        bits = torch.randint(0, 2, (2, 4, 32))
        out = proj(bits)
        assert out.shape == (2, 4, 512)

    def test_padding(self):
        proj = BitmaskProjection(max_bit_width=32, hidden_dim=512)
        bits = torch.randint(0, 2, (2, 4, 16))
        out = proj(bits)
        assert out.shape == (2, 4, 512)


class TestVarLenAttention:
    def test_shape(self):
        attn = VarLenAttention(hidden_dim=512, num_heads=8)
        x = torch.randn(2, 4, 512)
        out = attn(x)
        assert out.shape == (2, 4, 512)

    def test_with_mask(self):
        attn = VarLenAttention(hidden_dim=512, num_heads=8)
        x = torch.randn(2, 4, 512)
        mask = torch.ones(2, 1, 4, 4)
        out = attn(x, mask=mask)
        assert out.shape == (2, 4, 512)


class TestDABETransformer:
    @pytest.fixture
    def config(self):
        return {
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 256,
            "dropout": 0.1,
            "max_bit_width": 32,
            "vocab_size": 32000,
        }

    def test_forward_shape(self, config):
        model = DABETransformer(config)
        bits = torch.randint(0, 2, (2, 4, 32))
        logits, hidden = model(bits)
        assert logits.shape == (2, 4, 32000)
        assert hidden.shape == (2, 4, 128)

    def test_with_bit_widths(self, config):
        model = DABETransformer(config)
        bits = torch.randint(0, 2, (2, 4, 32))
        bw = torch.randint(8, 33, (2, 4))
        logits, hidden = model(bits, bit_widths=bw)
        assert logits.shape == (2, 4, 32000)


class TestTokenTransformerLM:
    def test_forward_shape(self):
        model = TokenTransformerLM({
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 256,
            "dropout": 0.1,
            "vocab_size": 1024,
        })
        input_ids = torch.randint(0, 1024, (2, 16))
        mask = torch.tril(torch.ones(1, 1, 16, 16))
        logits, hidden = model(input_ids, mask=mask)
        assert logits.shape == (2, 16, 1024)
        assert hidden.shape == (2, 16, 128)


class TestDABEAdapter:
    def test_adapter_shapes(self):
        adapter = DABEAdapter(hidden_dim=64, min_bit_width=8, max_bit_width=24, num_header_bits=4)
        hidden = torch.randn(2, 10, 64)
        adapted, widths, bit_values, gate = adapter(hidden)
        assert adapted.shape == (2, 10, 64)
        assert widths.shape == (2, 10)
        assert bit_values.shape == (2, 10, 28)
        assert gate.shape == (2, 10, 1)
        assert widths.min().item() >= 8
        assert widths.max().item() <= 24

    def test_adapter_lm_shapes(self):
        model = AdapterTokenTransformerLM({
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 256,
            "dropout": 0.1,
            "vocab_size": 1024,
            "adapter": {
                "min_bit_width": 8,
                "max_bit_width": 24,
                "num_header_bits": 4,
            },
        })
        input_ids = torch.randint(0, 1024, (2, 16))
        mask = torch.tril(torch.ones(1, 1, 16, 16))
        logits, hidden, stats = model(input_ids, mask=mask)
        assert logits.shape == (2, 16, 1024)
        assert hidden.shape == (2, 16, 128)
        assert stats["bit_widths"].shape == (2, 16)

    def test_zero_scale_adapter_starts_as_identity(self):
        adapter = DABEAdapter(
            hidden_dim=32,
            min_bit_width=8,
            max_bit_width=24,
            num_header_bits=4,
            residual_scale_init=0.0,
        )
        hidden = torch.randn(2, 6, 32)
        adapted, _, _, _ = adapter(hidden)
        assert torch.allclose(adapted, hidden)
