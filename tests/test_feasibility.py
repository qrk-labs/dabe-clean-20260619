import pytest
import torch
from pathlib import Path

from src.bitmask_encoder.lfq_encoder import LFQEncoder
from src.density_router.fixed import FixedRouter
from src.training import feasibility as feasibility_mod
from src.training.feasibility import (
    AdapterCausalLMModule,
    BPESequenceDataset,
    DABECausalLMModule,
    DABESequenceDataset,
    _sequence_windows,
    _validate_adapter_shapes,
)


class _MockTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(token) for token in text.split()]


def test_sequence_windows_shape():
    windows = _sequence_windows([1, 2, 3, 4, 5], seq_len=3)
    assert windows
    input_ids, targets = windows[0]
    assert len(input_ids) == 3
    assert len(targets) == 3


def test_dabe_sequence_dataset_builds_samples():
    encoder = LFQEncoder({
        "max_bit_width": 24,
        "num_header_bits": 4,
        "vocab_size": 32,
        "embed_dim": 16,
    })
    encoder.set_vocabulary(["tiny", "story", "fox", "map"])
    router = FixedRouter({
        "min_bit_width": 8,
        "max_bit_width": 8,
        "fixed_bit_width": 8,
    })
    dataset = DABESequenceDataset(
        texts=["tiny story fox map tiny story fox map"],
        router=router,
        encoder=encoder,
        seq_len=4,
    )
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["input_ids"].shape == (4,)
    assert sample["targets"].shape == (4,)
    assert sample["bit_widths"].shape == (4,)


def test_bpe_sequence_dataset_with_mock_tokenizer():
    dataset = BPESequenceDataset(
        texts=["a bb ccc dddd eeeee"],
        tokenizer=_MockTokenizer(),
        seq_len=3,
    )
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["input_ids"].shape == (3,)
    assert sample["targets"].shape == (3,)


def test_dabe_module_bit_conversion_shape():
    encoder = LFQEncoder({
        "max_bit_width": 8,
        "num_header_bits": 4,
        "vocab_size": 16,
        "embed_dim": 16,
    })
    encoder.set_vocabulary(["one", "two", "three"])
    model = DABECausalLMModule(
        config={
            "training": {"learning_rate": 1e-3, "weight_decay": 0.0},
            "backbone": {
                "hidden_dim": 32,
                "num_layers": 1,
                "num_heads": 4,
                "ff_dim": 64,
                "dropout": 0.0,
                "vocab_size": 16,
                "max_bit_width": 12,
            },
        },
        encoder=encoder,
    )
    model.to("cpu")
    bits = model._ids_to_bits(
        input_ids=torch.tensor([[1, 2, 3, 1]], dtype=torch.long),
        bit_widths=torch.tensor([[8, 8, 8, 8]], dtype=torch.long),
    )
    assert bits.shape == (1, 4, 12)


def test_dabe_vectorized_ids_to_bits_matches_loop():
    encoder = LFQEncoder({
        "max_bit_width": 8,
        "num_header_bits": 4,
        "vocab_size": 32,
        "embed_dim": 16,
    })
    encoder.set_vocabulary([f"tok{i}" for i in range(20)])
    module = DABECausalLMModule(
        config={
            "training": {"learning_rate": 1e-3, "weight_decay": 0.0},
            "backbone": {
                "hidden_dim": 32,
                "num_layers": 1,
                "num_heads": 4,
                "ff_dim": 64,
                "dropout": 0.0,
                "vocab_size": 32,
                "max_bit_width": 12,
            },
        },
        encoder=encoder,
    )
    module.to("cpu")
    input_ids = torch.randint(0, 20, (2, 10), dtype=torch.long)
    bit_widths = torch.randint(2, 9, (2, 10), dtype=torch.long)
    loop_bits = module._ids_to_bits_loop(input_ids, bit_widths)
    vec_bits = module._ids_to_bits(input_ids, bit_widths)
    assert torch.equal(loop_bits, vec_bits)


def test_cached_preencode_dataset_equivalence(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    encoder = LFQEncoder({
        "max_bit_width": 24,
        "num_header_bits": 4,
        "vocab_size": 64,
        "embed_dim": 16,
    })
    encoder.set_vocabulary(["tiny", "story", "fox", "map", "acorn"])
    router = FixedRouter({
        "min_bit_width": 8,
        "max_bit_width": 8,
        "fixed_bit_width": 8,
    })
    texts = [
        "tiny story fox map tiny story fox map",
        "fox map tiny story fox map tiny story",
    ]
    built = DABESequenceDataset(
        texts=texts,
        router=router,
        encoder=encoder,
        seq_len=4,
        cache_dir=cache_dir,
        cache_mode="build",
        split_name="train",
        preencode_to_disk=True,
    )
    reused = DABESequenceDataset(
        texts=texts,
        router=router,
        encoder=encoder,
        seq_len=4,
        cache_dir=cache_dir,
        cache_mode="reuse",
        split_name="train",
        preencode_to_disk=True,
    )
    assert len(built) == len(reused)
    assert len(reused) > 0
    first_a = built[0]
    first_b = reused[0]
    assert torch.equal(first_a["input_ids"], first_b["input_ids"])
    assert torch.equal(first_a["targets"], first_b["targets"])
    assert torch.equal(first_a["bit_widths"], first_b["bit_widths"])


def test_compile_toggle_loss_parity_small_batch():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    torch.manual_seed(7)
    config = {
        "training": {"learning_rate": 1e-3, "weight_decay": 0.0},
        "optimization": {"v4_distill": {"mtp_heads": 0}},
        "backbone": {
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "vocab_size": 128,
        },
    }
    eager = feasibility_mod.BPECausalLMModule(config=config)
    eager.to("cpu")
    eager.log = lambda *args, **kwargs: None  # type: ignore[method-assign]

    compiled = feasibility_mod.BPECausalLMModule(config=config)
    compiled.to("cpu")
    compiled.load_state_dict(eager.state_dict())
    compiled.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    compiled.model, _ = feasibility_mod._maybe_compile_model(compiled.model, enabled=True)

    batch = {
        "input_ids": torch.randint(0, 128, (2, 12)),
        "targets": torch.randint(0, 128, (2, 12)),
    }
    eager_loss = eager.training_step(batch, 0)
    compiled_loss = compiled.training_step(batch, 0)
    assert torch.isfinite(compiled_loss).item()
    assert abs(float(eager_loss.item()) - float(compiled_loss.item())) < 1e-4


def test_adapter_shape_validation():
    report = _validate_adapter_shapes({
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
        "dropout": 0.1,
        "vocab_size": 512,
        "adapter": {
            "min_bit_width": 8,
            "max_bit_width": 24,
            "num_header_bits": 4,
        },
    })
    assert report["logits"] == (2, 16, 512)
    assert report["hidden"] == (2, 16, 64)
    assert report["bit_widths"] == (2, 16)


def test_adapter_module_training_step_returns_finite_loss():
    module = AdapterCausalLMModule({
        "training": {"learning_rate": 1e-3, "weight_decay": 0.0},
        "backbone": {
            "hidden_dim": 64,
            "num_layers": 1,
            "num_heads": 4,
            "ff_dim": 128,
            "dropout": 0.0,
            "vocab_size": 256,
            "adapter": {
                "min_bit_width": 8,
                "max_bit_width": 24,
                "num_header_bits": 4,
            },
        },
    })
    module.to("cpu")
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    batch = {
        "input_ids": torch.randint(0, 256, (2, 12)),
        "targets": torch.randint(0, 256, (2, 12)),
    }
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss).item()


def test_extract_text_field_supports_non_default_key():
    example = {"content": "  sample content  "}
    assert feasibility_mod._extract_text_field(example, text_field="content") == "sample content"


def test_load_text_dataset_samples_raises_when_fallback_disabled(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("network blocked")

    monkeypatch.setattr(feasibility_mod, "load_dataset", _raise)
    with pytest.raises(RuntimeError, match="Failed to load dataset="):
        feasibility_mod._load_text_dataset_samples(
            dataset_name="allenai/olmo-mix-1124",
            split="train",
            max_samples=4,
            allow_synthetic_fallback=False,
        )


def test_eta_helpers_format_and_normalize():
    assert feasibility_mod._format_duration(65.1) == "1m05s"
    assert feasibility_mod._format_duration(3661.0) == "1h01m01s"
    assert feasibility_mod._format_duration(None) == "n/a"
    assert feasibility_mod._normalize_batch_count(12) == 12
    assert feasibility_mod._normalize_batch_count([10, 5]) == 15
    assert feasibility_mod._normalize_batch_count(float("inf")) is None
