import numpy as np
import torch

from src.training.dabe_tokenizer_autoencoder import (
    DABEChunkTokenizerAutoencoder,
    DABETokenizerAutoencoderModule,
    TokenChunkDataset,
    build_token_chunk_array,
    chunk_deviation_stats,
)


class _DummyTokenizer:
    is_fast = True
    vocab_size = 512

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) % self.vocab_size for ch in text]

    def __call__(self, texts, add_special_tokens: bool = False):
        del add_special_tokens
        if isinstance(texts, str):
            return {"input_ids": self.encode(texts)}
        return {"input_ids": [self.encode(text) for text in texts]}


def test_build_token_chunk_array_uses_full_stride_windows():
    chunks = build_token_chunk_array(
        texts=["abcdefghijkl"],
        tokenizer=_DummyTokenizer(),
        chunk_size_tokens=4,
        chunk_stride_tokens=4,
        tokenizer_batch_size=2,
    )
    assert chunks.shape == (3, 4)
    assert chunks.dtype == np.int64


def test_token_chunk_dataset_shape():
    chunks = np.arange(24, dtype=np.int64).reshape(3, 8)
    dataset = TokenChunkDataset(chunks)
    sample = dataset[0]
    assert sample["input_ids"].shape == (8,)
    assert sample["input_ids"].dtype == torch.long


def test_chunk_deviation_stats_reports_hamming_distance_distribution():
    matches = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, False],
            [False, False, False, False],
        ]
    )
    stats = chunk_deviation_stats(matches)
    assert stats["mean"].item() == 2.0
    assert stats["rate_mean"].item() == 0.5
    assert stats["p50"].item() == 2.0
    assert stats["max"].item() == 4.0


def test_dabe_chunk_tokenizer_autoencoder_forward_shapes():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 128,
        "chunk_size_tokens": 8,
        "code_bits": 32,
        "embed_dim": 16,
        "hidden_dim": 64,
        "dropout": 0.0,
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = model(input_ids)
    assert output.logits.shape == (4, 8, 128)
    assert output.bits.shape == (4, 32)
    assert output.bit_logits.shape == (4, 32)
    assert model.bits_per_token == 4.0
    assert torch.isfinite(output.logits).all().item()
    assert output.diffusion_embed_mse is None


def test_dabe_chunk_tokenizer_diffusion_decoder_forward_shapes():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 128,
        "chunk_size_tokens": 8,
        "code_bits": 64,
        "embed_dim": 16,
        "hidden_dim": 64,
        "dropout": 0.0,
        "decoder_mode": "diffusion",
        "diffusion_steps": 8,
        "diffusion_layers": 1,
        "diffusion_heads": 4,
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = model(input_ids)
    assert output.logits.shape == (4, 8, 128)
    assert output.bits.shape == (4, 64)
    assert output.bit_logits.shape == (4, 64)
    assert output.diffusion_embed_mse is not None
    assert torch.isfinite(output.diffusion_embed_mse).item()
    assert model.bits_per_token == 8.0


def test_dabe_chunk_tokenizer_code_transformer_decoder_forward_shapes():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 128,
        "chunk_size_tokens": 8,
        "code_bits": 64,
        "embed_dim": 16,
        "hidden_dim": 64,
        "dropout": 0.0,
        "decoder_mode": "code_transformer",
        "code_decoder_layers": 1,
        "code_decoder_heads": 4,
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = model(input_ids)
    assert output.logits.shape == (4, 8, 128)
    assert output.bits.shape == (4, 64)
    assert output.diffusion_embed_mse is None
    assert torch.isfinite(output.logits).all().item()
    assert model.bits_per_token == 8.0


def test_dabe_chunk_tokenizer_hierarchical_local_decoder_forward_shapes():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 128,
        "chunk_size_tokens": 8,
        "code_bits": 64,
        "embed_dim": 16,
        "hidden_dim": 64,
        "dropout": 0.0,
        "decoder_mode": "hierarchical_local",
        "hierarchical_block_tokens": 2,
        "hierarchical_decoder_layers": 1,
        "hierarchical_decoder_heads": 4,
        "hierarchical_refine_layers": 1,
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = model(input_ids)
    assert output.logits.shape == (4, 8, 128)
    assert output.bits.shape == (4, 64)
    assert output.diffusion_embed_mse is None
    assert model.hierarchical_num_blocks == 4
    assert model.hierarchical_code_bits_per_block == 16
    assert torch.isfinite(output.logits).all().item()
    assert model.bits_per_token == 8.0


def test_dabe_chunk_tokenizer_hierarchical_local_refine_decoder_forward_shapes():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 128,
        "chunk_size_tokens": 8,
        "code_bits": 64,
        "embed_dim": 16,
        "hidden_dim": 64,
        "dropout": 0.0,
        "decoder_mode": "hierarchical_local_refine",
        "hierarchical_block_tokens": 2,
        "hierarchical_decoder_layers": 1,
        "hierarchical_decoder_heads": 4,
        "hierarchical_refine_layers": 1,
        "hierarchical_local_refine_layers": 1,
        "hierarchical_local_refine_causal": True,
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = model(input_ids)
    loss = output.logits.mean()
    loss.backward()
    assert output.logits.shape == (4, 8, 128)
    assert output.bits.shape == (4, 64)
    assert output.diffusion_embed_mse is None
    assert model.hierarchical_num_blocks == 4
    assert model.hierarchical_code_bits_per_block == 16
    assert model.hierarchical_local_causal_mask.shape == (2, 2)
    assert torch.isfinite(output.logits).all().item()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all().item()
        for param in model.hierarchical_local_refiner.parameters()
    )
    assert model.bits_per_token == 8.0


def test_dabe_chunk_tokenizer_hierarchical_lookup_decoder_forward_shapes():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 128,
        "chunk_size_tokens": 8,
        "code_bits": 64,
        "embed_dim": 16,
        "hidden_dim": 64,
        "dropout": 0.0,
        "decoder_mode": "hierarchical_lookup",
        "hierarchical_block_tokens": 2,
        "hierarchical_decoder_layers": 1,
        "hierarchical_decoder_heads": 4,
        "hierarchical_refine_layers": 1,
        "lexical_lookup_k": 2,
        "lexical_lookup_heads": 4,
        "lexical_lookup_copy_scale": 2.0,
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = model(input_ids)
    loss = output.logits.mean()
    loss.backward()
    assert output.logits.shape == (4, 8, 128)
    assert output.bits.shape == (4, 64)
    assert output.lookup_positions is not None
    assert output.lookup_positions.shape == (4, 2)
    assert output.lookup_token_ids is not None
    assert output.lookup_token_ids.shape == (4, 2)
    assert output.lookup_gate is not None
    assert output.lookup_gate.shape == (4, 8)
    assert model.lookup_bits_per_chunk == 20
    assert model.effective_bits_per_token == 10.5
    assert torch.isfinite(output.logits).all().item()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all().item()
        for param in model.lexical_lookup_attention.parameters()
    )


def test_dabe_chunk_tokenizer_hierarchical_lookup_learned_selector_shapes():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 128,
        "chunk_size_tokens": 8,
        "code_bits": 64,
        "embed_dim": 16,
        "hidden_dim": 64,
        "dropout": 0.0,
        "decoder_mode": "hierarchical_lookup",
        "hierarchical_block_tokens": 2,
        "hierarchical_decoder_layers": 1,
        "hierarchical_decoder_heads": 4,
        "hierarchical_refine_layers": 1,
        "lexical_lookup_k": 2,
        "lexical_lookup_heads": 4,
        "lexical_lookup_selector": "learned",
    })
    module = DABETokenizerAutoencoderModule({
        "model": {
            "vocab_size": 128,
            "chunk_size_tokens": 8,
            "code_bits": 64,
            "embed_dim": 16,
            "hidden_dim": 64,
            "dropout": 0.0,
            "decoder_mode": "hierarchical_lookup",
            "hierarchical_block_tokens": 2,
            "hierarchical_decoder_layers": 1,
            "hierarchical_decoder_heads": 4,
            "hierarchical_refine_layers": 1,
            "lexical_lookup_k": 2,
            "lexical_lookup_heads": 4,
            "lexical_lookup_selector": "learned",
        },
        "training": {"selector_loss_weight": 0.1},
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = model(input_ids)
    assert output.selector_logits is not None
    assert output.selector_logits.shape == (4, 8)
    assert output.selector_target_mask is not None
    assert output.selector_target_mask.shape == (4, 8)
    assert output.lookup_positions is not None
    assert output.lookup_positions.shape == (4, 2)
    loss = module._step({"input_ids": input_ids}, "train")
    loss.backward()
    assert torch.isfinite(loss).item()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all().item()
        for param in module.model.lexical_selector.parameters()
    )


def test_dabe_chunk_tokenizer_adaptive_lookup_budget_shapes_and_gradients():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 128,
        "chunk_size_tokens": 8,
        "code_bits": 64,
        "embed_dim": 16,
        "hidden_dim": 64,
        "dropout": 0.0,
        "decoder_mode": "hierarchical_lookup",
        "hierarchical_block_tokens": 2,
        "hierarchical_decoder_layers": 1,
        "hierarchical_decoder_heads": 4,
        "hierarchical_refine_layers": 1,
        "lexical_lookup_k": 4,
        "lexical_lookup_k_schedule": "2|4",
        "lexical_lookup_heads": 4,
        "lexical_lookup_selector": "learned",
    })
    module = DABETokenizerAutoencoderModule({
        "model": {
            "vocab_size": 128,
            "chunk_size_tokens": 8,
            "code_bits": 64,
            "embed_dim": 16,
            "hidden_dim": 64,
            "dropout": 0.0,
            "decoder_mode": "hierarchical_lookup",
            "hierarchical_block_tokens": 2,
            "hierarchical_decoder_layers": 1,
            "hierarchical_decoder_heads": 4,
            "hierarchical_refine_layers": 1,
            "lexical_lookup_k": 4,
            "lexical_lookup_k_schedule": "2|4",
            "lexical_lookup_heads": 4,
            "lexical_lookup_selector": "learned",
        },
        "training": {"selector_loss_weight": 0.1, "budget_loss_weight": 0.1},
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = model(input_ids)
    assert output.lookup_positions is not None
    assert output.lookup_positions.shape == (4, 4)
    assert output.lookup_token_ids is not None
    assert output.lookup_token_ids.shape == (4, 4)
    assert output.lookup_active_mask is not None
    assert output.lookup_active_mask.shape == (4, 4)
    assert output.lookup_budget_k is not None
    assert set(output.lookup_budget_k.tolist()).issubset({2, 4})
    assert output.budget_logits is not None
    assert output.budget_logits.shape == (4, 2)
    assert output.budget_target is not None
    assert output.budget_target.shape == (4,)
    assert model.lookup_bits_per_chunk == 40
    assert model.effective_bits_per_token == 13.0
    loss = module._step({"input_ids": input_ids}, "train")
    loss.backward()
    assert torch.isfinite(loss).item()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all().item()
        for param in module.model.lexical_budget_selector.parameters()
    )


def test_dabe_chunk_tokenizer_ranked_slot_halting_shapes_and_gradients():
    module = DABETokenizerAutoencoderModule({
        "model": {
            "vocab_size": 128,
            "chunk_size_tokens": 8,
            "code_bits": 64,
            "embed_dim": 16,
            "hidden_dim": 64,
            "dropout": 0.0,
            "decoder_mode": "hierarchical_lookup",
            "hierarchical_block_tokens": 2,
            "hierarchical_decoder_layers": 1,
            "hierarchical_decoder_heads": 4,
            "hierarchical_refine_layers": 1,
            "lexical_lookup_k": 4,
            "lexical_lookup_heads": 4,
            "lexical_lookup_selector": "learned",
            "lexical_lookup_slot_policy": "halting",
        },
        "training": {
            "selector_loss_weight": 0.1,
            "lookup_slot_cost_weight": 0.01,
            "lookup_slot_target_weight": 0.1,
            "lookup_slot_target_k": 2,
        },
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = module.model(input_ids)
    assert output.lookup_positions is not None
    assert output.lookup_positions.shape == (4, 4)
    assert output.lookup_keep_logits is not None
    assert output.lookup_keep_logits.shape == (4, 4)
    assert output.lookup_keep_probs is not None
    assert output.lookup_keep_probs.shape == (4, 4)
    assert output.lookup_active_mask is not None
    assert output.lookup_active_mask.shape == (4, 4)
    assert output.lookup_budget_k is not None
    assert output.lookup_budget_k.shape == (4,)
    loss = module._step({"input_ids": input_ids}, "train")
    loss.backward()
    assert torch.isfinite(loss).item()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all().item()
        for param in module.model.lexical_slot_halting.parameters()
    )


def test_dabe_chunk_tokenizer_gist_residual_lookup_shapes_and_gradients():
    module = DABETokenizerAutoencoderModule({
        "model": {
            "vocab_size": 128,
            "chunk_size_tokens": 8,
            "code_bits": 64,
            "embed_dim": 16,
            "hidden_dim": 64,
            "dropout": 0.0,
            "decoder_mode": "gist_residual_lookup",
            "hierarchical_block_tokens": 2,
            "hierarchical_decoder_layers": 1,
            "hierarchical_decoder_heads": 4,
            "hierarchical_refine_layers": 1,
            "lexical_lookup_k": 4,
            "lexical_lookup_heads": 4,
            "lexical_lookup_selector": "learned",
            "lexical_lookup_slot_policy": "halting",
        },
        "training": {
            "gist_loss_weight": 0.25,
            "residual_router_loss_weight": 0.1,
            "lookup_slot_cost_weight": 0.01,
        },
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = module.model(input_ids)
    assert output.logits.shape == (4, 8, 128)
    assert output.gist_logits is not None
    assert output.gist_logits.shape == (4, 8, 128)
    assert output.residual_router_logits is not None
    assert output.residual_router_logits.shape == (4, 8, 3)
    assert output.residual_router_target is not None
    assert output.residual_router_target.shape == (4, 8)
    assert output.residual_info_scores is not None
    assert output.residual_info_scores.shape == (4, 8)
    assert output.lookup_positions is not None
    assert output.lookup_positions.shape == (4, 4)
    assert output.lookup_keep_probs is not None
    assert output.lookup_keep_probs.shape == (4, 4)
    loss = module._step({"input_ids": input_ids}, "train")
    loss.backward()
    assert torch.isfinite(loss).item()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all().item()
        for param in module.model.residual_router.parameters()
    )


def test_dabe_chunk_tokenizer_gist_residual_variable_windows_shapes_and_gradients():
    module = DABETokenizerAutoencoderModule({
        "model": {
            "vocab_size": 128,
            "chunk_size_tokens": 8,
            "code_bits": 64,
            "embed_dim": 16,
            "hidden_dim": 64,
            "dropout": 0.0,
            "decoder_mode": "gist_residual_variable_windows",
            "hierarchical_block_tokens": 2,
            "hierarchical_decoder_layers": 1,
            "hierarchical_decoder_heads": 4,
            "hierarchical_refine_layers": 1,
            "variable_window_refine_layers": 1,
            "variable_window_target_mode": "action_value",
            "lexical_lookup_k": 4,
            "lexical_lookup_heads": 4,
            "lexical_lookup_selector": "learned",
            "lexical_lookup_slot_policy": "halting",
        },
        "training": {
            "gist_loss_weight": 0.25,
            "residual_router_loss_weight": 0.2,
            "variable_window_loss_weight": 0.1,
            "variable_window_nonimprove_weight": 0.1,
            "lookup_slot_cost_weight": 0.01,
        },
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = module.model(input_ids)
    assert output.logits.shape == (4, 8, 128)
    assert output.gist_logits is not None
    assert output.gist_logits.shape == (4, 8, 128)
    assert output.variable_window_logits is not None
    assert output.variable_window_logits.shape == (4, 4, 3)
    assert output.variable_window_target is not None
    assert output.variable_window_target.shape == (4, 4)
    assert output.variable_window_probs is not None
    assert output.variable_window_probs.shape == (4, 4, 3)
    assert output.variable_window_bits_per_chunk is not None
    assert output.variable_window_bits_per_chunk.shape == (4,)
    assert output.variable_window_action_deviation is not None
    assert output.variable_window_action_deviation.shape == (4, 4, 3)
    assert output.variable_window_base_deviation is not None
    assert output.variable_window_base_deviation.shape == (4, 4)
    assert output.variable_window_oracle_deviation is not None
    assert output.variable_window_oracle_deviation.shape == (4, 4)
    assert output.residual_router_logits is not None
    assert output.lookup_positions is not None
    assert output.lookup_positions.shape == (4, 4)
    loss = module._step({"input_ids": input_ids}, "train")
    loss.backward()
    assert torch.isfinite(loss).item()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all().item()
        for param in module.model.variable_window_router.parameters()
    )


def test_dabe_chunk_tokenizer_sliding_progressive_decoder_forward_shapes():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 128,
        "chunk_size_tokens": 8,
        "code_bits": 64,
        "embed_dim": 16,
        "hidden_dim": 64,
        "dropout": 0.0,
        "decoder_mode": "sliding_progressive",
        "sliding_global_bits": 16,
        "sliding_medium_window": 4,
        "sliding_medium_stride": 2,
        "sliding_medium_bits": 8,
        "sliding_fine_window": 2,
        "sliding_fine_stride": 1,
        "sliding_fine_bits": 4,
        "sliding_decoder_layers": 1,
        "sliding_decoder_heads": 4,
        "sliding_refine_layers": 1,
    })
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    output = model(input_ids)
    assert output.logits.shape == (4, 8, 128)
    assert output.bits.shape == (4, 64)
    assert output.diffusion_embed_mse is None
    assert model.sliding_total_bits == 64
    assert torch.isfinite(output.logits).all().item()
    assert model.bits_per_token == 8.0


def test_reverse_diffusion_decode_returns_token_logits_from_bits():
    model = DABEChunkTokenizerAutoencoder({
        "vocab_size": 64,
        "chunk_size_tokens": 4,
        "code_bits": 16,
        "embed_dim": 8,
        "hidden_dim": 32,
        "dropout": 0.0,
        "decoder_mode": "diffusion",
        "diffusion_steps": 4,
        "diffusion_layers": 1,
        "diffusion_heads": 2,
    })
    input_ids = torch.randint(0, 64, (2, 4), dtype=torch.long)
    bits, _ = model.encode_bits(input_ids)
    logits, embeddings = model.reverse_diffusion_decode(bits=bits, sample_steps=2)
    assert logits.shape == (2, 4, 64)
    assert embeddings.shape == (2, 4, 8)
    assert torch.isfinite(logits).all().item()


def test_dabe_tokenizer_autoencoder_step_is_finite_and_updates():
    module = DABETokenizerAutoencoderModule({
        "model": {
            "vocab_size": 128,
            "chunk_size_tokens": 8,
            "code_bits": 32,
            "embed_dim": 16,
            "hidden_dim": 64,
            "dropout": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "bit_balance_weight": 0.01,
        },
    })
    optimizer = module.configure_optimizers()
    batch = {"input_ids": torch.randint(0, 128, (4, 8), dtype=torch.long)}
    params_before = [param.detach().clone() for param in module.parameters() if param.requires_grad]
    loss = module._step(batch, "train")
    assert torch.isfinite(loss).item()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    params_after = [param.detach().clone() for param in module.parameters() if param.requires_grad]
    assert any(not torch.allclose(before, after) for before, after in zip(params_before, params_after))


def test_dabe_tokenizer_autoencoder_logs_topk_metrics():
    module = DABETokenizerAutoencoderModule({
        "model": {
            "vocab_size": 32,
            "chunk_size_tokens": 4,
            "code_bits": 8,
            "embed_dim": 8,
            "hidden_dim": 16,
            "dropout": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "bit_balance_weight": 0.01,
        },
    })
    batch = {"input_ids": torch.randint(0, 32, (2, 4), dtype=torch.long)}
    loss = module._step(batch, "train")
    assert torch.isfinite(loss).item()
