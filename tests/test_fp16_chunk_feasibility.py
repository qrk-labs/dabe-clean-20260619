import numpy as np
import torch

from src.training.fp16_chunk_feasibility import (
    FP16ChunkCompressor,
    FP16ChunkLMModule,
    FP16ChunkSequenceDataset,
    ScalarTransformerLM,
)


class _DummyTokenizer:
    is_fast = True
    vocab_size = 4096

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) % 255 for ch in text]

    def __call__(self, texts, add_special_tokens: bool = False):
        del add_special_tokens
        if isinstance(texts, str):
            return {"input_ids": self.encode(texts)}
        return {"input_ids": [self.encode(text) for text in texts]}


def test_compressor_is_deterministic_and_finite():
    compressor = FP16ChunkCompressor(
        chunk_size_tokens=4,
        window_overlap_tokens=0,
        tokenizer=_DummyTokenizer(),
        dtype="float16",
    )
    token_ids = list(range(16))
    first = compressor.compress_token_ids(token_ids)
    second = compressor.compress_token_ids(token_ids)
    assert first.dtype == np.float16
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_chunking_uses_non_overlap_windows():
    compressor = FP16ChunkCompressor(
        chunk_size_tokens=128,
        window_overlap_tokens=0,
        tokenizer=_DummyTokenizer(),
        dtype="float16",
    )
    token_ids = list(range(300))
    compressed = compressor.compress_token_ids(token_ids)
    assert compressed.shape[0] == 2


def test_sequence_dataset_window_shapes_and_shift():
    stream = np.asarray([0.1 * i for i in range(12)], dtype=np.float16)
    dataset = FP16ChunkSequenceDataset(stream, compressed_seq_len=4, window_stride=1)
    assert len(dataset) == 8
    sample = dataset[0]
    assert sample["input_scalars"].shape == (4,)
    assert sample["target_scalars"].shape == (4,)
    assert torch.allclose(
        sample["input_scalars"],
        torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32),
        atol=1e-4,
    )
    assert torch.allclose(
        sample["target_scalars"],
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        atol=1e-4,
    )


def test_sequence_dataset_includes_target_chunk_ids_when_provided():
    stream = np.asarray([0.1 * i for i in range(12)], dtype=np.float16)
    chunk_tokens = np.arange(12 * 4, dtype=np.int32).reshape(12, 4)
    dataset = FP16ChunkSequenceDataset(
        stream,
        compressed_seq_len=4,
        window_stride=1,
        chunk_tokens_stream=chunk_tokens,
    )
    sample = dataset[0]
    assert "target_chunk_ids" in sample
    assert sample["target_chunk_ids"].shape == (4, 4)


def test_scalar_transformer_forward_shape_and_finite_output():
    model = ScalarTransformerLM({
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
        "dropout": 0.0,
    })
    inputs = torch.randn(3, 8)
    preds, hidden = model(inputs)
    assert preds.shape == (3, 8)
    assert hidden.shape == (3, 8, 64)
    assert torch.isfinite(preds).all().item()


def test_fp16_chunk_module_single_optimization_step_is_finite():
    module = FP16ChunkLMModule({
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 128,
            "dropout": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
    })
    module.to("cpu")
    optimizer = module.configure_optimizers()
    batch = {
        "input_scalars": torch.randn(2, 8),
        "target_scalars": torch.randn(2, 8),
    }
    params_before = [param.detach().clone() for param in module.parameters() if param.requires_grad]
    loss = module._step(batch=batch, stage="train", log_metrics=False)
    assert torch.isfinite(loss).item()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    params_after = [param.detach().clone() for param in module.parameters() if param.requires_grad]
    changed = [not torch.allclose(before, after) for before, after in zip(params_before, params_after)]
    assert any(changed)


def test_fp16_chunk_module_with_diffusion_step_is_finite():
    module = FP16ChunkLMModule({
        "compression": {"chunk_size_tokens": 4},
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 128,
            "dropout": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        "diffusion": {
            "enabled": True,
            "latent_dim": 32,
            "timesteps": 64,
            "loss_weight": 0.25,
            "reconstruction_weight": 0.2,
            "vocab_size": 512,
            "chunk_size_tokens": 4,
        },
    })
    module.to("cpu")
    batch = {
        "input_scalars": torch.randn(2, 8),
        "target_scalars": torch.randn(2, 8),
        "target_chunk_ids": torch.randint(0, 512, (2, 8, 4), dtype=torch.long),
    }
    loss = module._step(batch=batch, stage="train", log_metrics=False)
    assert torch.isfinite(loss).item()


def test_fp16_chunk_module_optimizer_includes_diffusion_parameters():
    module = FP16ChunkLMModule({
        "compression": {"chunk_size_tokens": 4},
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 128,
            "dropout": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        "diffusion": {
            "enabled": True,
            "latent_dim": 32,
            "timesteps": 64,
            "loss_weight": 0.25,
            "reconstruction_weight": 0.2,
            "vocab_size": 512,
            "chunk_size_tokens": 4,
        },
    })
    optimizer = module.configure_optimizers()
    optimized_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert id(module.chunk_token_embed.weight) in optimized_param_ids
    assert id(module.diffusion_denoiser.net[0].weight) in optimized_param_ids


def test_multi_scalar_compressor_shape_and_determinism():
    compressor = FP16ChunkCompressor(
        chunk_size_tokens=8,
        window_overlap_tokens=0,
        tokenizer=_DummyTokenizer(),
        dtype="float16",
        scalars_per_chunk=4,
    )
    token_ids = list(range(16))
    scalars = compressor.compress_token_ids(token_ids)
    assert scalars.ndim == 2
    assert scalars.shape == (2, 4)
    assert scalars.dtype == np.float16
    assert np.isfinite(scalars).all()
    # Determinism
    second = compressor.compress_token_ids(token_ids)
    assert np.array_equal(scalars, second)


def test_multi_scalar_sequence_dataset_shapes():
    stream = np.linspace(-1.0, 1.0, 24 * 4, dtype=np.float32).reshape(24, 4)
    chunk_tokens = np.arange(24 * 8, dtype=np.int32).reshape(24, 8)
    dataset = FP16ChunkSequenceDataset(
        stream,
        compressed_seq_len=4,
        window_stride=1,
        chunk_tokens_stream=chunk_tokens,
    )
    assert len(dataset) == 20
    sample = dataset[0]
    assert sample["input_scalars"].shape == (4, 4)
    assert sample["target_scalars"].shape == (4, 4)
    assert sample["target_chunk_ids"].shape == (4, 8)


def test_multi_scalar_transformer_forward_shape():
    model = ScalarTransformerLM({
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
        "dropout": 0.0,
        "num_scalars": 8,
    })
    inputs = torch.randn(3, 8, 8)
    preds, hidden = model(inputs)
    assert preds.shape == (3, 8, 8)
    assert hidden.shape == (3, 8, 64)
    assert torch.isfinite(preds).all().item()


def test_multi_scalar_module_single_step_is_finite():
    module = FP16ChunkLMModule({
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 128,
            "dropout": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        "compression": {
            "scalars_per_chunk": 8,
        },
    })
    module.to("cpu")
    optimizer = module.configure_optimizers()
    batch = {
        "input_scalars": torch.randn(2, 8, 8),
        "target_scalars": torch.randn(2, 8, 8),
    }
    params_before = [param.detach().clone() for param in module.parameters() if param.requires_grad]
    loss = module._step(batch=batch, stage="train", log_metrics=False)
    assert torch.isfinite(loss).item()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    params_after = [param.detach().clone() for param in module.parameters() if param.requires_grad]
    changed = [not torch.allclose(before, after) for before, after in zip(params_before, params_after)]
    assert any(changed)


def test_multi_scalar_module_with_diffusion_step_is_finite():
    module = FP16ChunkLMModule({
        "compression": {"chunk_size_tokens": 4, "scalars_per_chunk": 2},
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 128,
            "dropout": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        "diffusion": {
            "enabled": True,
            "latent_dim": 32,
            "timesteps": 64,
            "loss_weight": 0.25,
            "reconstruction_weight": 0.2,
            "vocab_size": 512,
            "chunk_size_tokens": 4,
        },
    })
    module.to("cpu")
    batch = {
        "input_scalars": torch.randn(2, 8, 2),
        "target_scalars": torch.randn(2, 8, 2),
        "target_chunk_ids": torch.randint(0, 512, (2, 8, 4), dtype=torch.long),
    }
    loss = module._step(batch=batch, stage="train", log_metrics=False)
    assert torch.isfinite(loss).item()
