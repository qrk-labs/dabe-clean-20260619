import torch

from src.factories import build_encoder, build_router
from src.runtime.device import resolve_torch_device, resolve_trainer_accelerator
from src.training.pretrain import PretrainLightningModule


def test_build_router_selects_fixed():
    config = {"router": {"type": "fixed", "fixed_bit_width": 16}}
    router = build_router(config)
    spans = router.segment("hello world", router.score("hello world"))
    assert spans
    assert all(span.bit_width == 16 for span in spans)


def test_build_encoder_selects_gumbel():
    config = {"encoder": {"type": "gumbel", "max_bit_width": 16, "embed_dim": 32, "vocab_size": 128}}
    encoder = build_encoder(config)
    bits = encoder.encode("hello", bit_width=8)
    assert bits.shape[0] == 12  # 8 bits + 4 header bits (default)


def test_resolve_trainer_accelerator_cpu():
    accelerator, devices = resolve_trainer_accelerator("cpu")
    assert accelerator == "cpu"
    assert devices == 1


def test_resolve_torch_device_auto_returns_valid_device():
    device = resolve_torch_device("auto")
    assert device.type in {"cpu", "mps", "cuda"}


def test_pretrain_forward_handles_variable_sequence_lengths():
    config = {
        "router": {
            "type": "fixed",
            "fixed_bit_width": 8,
            "min_bit_width": 8,
            "max_bit_width": 8,
        },
        "encoder": {
            "type": "lfq",
            "max_bit_width": 8,
            "num_header_bits": 4,
            "embed_dim": 32,
            "vocab_size": 128,
        },
        "backbone": {
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "max_bit_width": 8,
            "vocab_size": 128,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 1,
        },
    }

    model = PretrainLightningModule(config)
    model.to(torch.device("cpu"))
    batch = {
        "text": ["one two", "one two three four five"],
        "lang": ["en", "fr"],
    }
    outputs = model(batch)
    assert {"loss", "recon_loss", "contrastive_loss"}.issubset(outputs.keys())
    assert outputs["batch_num_spans"] > 0
    assert outputs["batch_unique_targets"] > 0
    assert torch.isfinite(outputs["loss"])
