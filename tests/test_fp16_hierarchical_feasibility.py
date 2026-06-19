import numpy as np
import torch

from src.training.fp16_hierarchical_feasibility import (
    HierarchicalFP16ChunkLMModule,
    HierarchicalFP16SequenceDataset,
    HierarchicalScalarTransformerLM,
)


def test_hierarchical_dataset_shapes_and_alignment():
    coarse = np.linspace(-1.0, 1.0, 24, dtype=np.float32)
    mid = np.linspace(-0.8, 0.8, 24, dtype=np.float32)
    fine = np.linspace(-0.6, 0.6, 24, dtype=np.float32)
    chunk_tokens = np.arange(24 * 64, dtype=np.int32).reshape(24, 64)
    dataset = HierarchicalFP16SequenceDataset(
        coarse_stream=coarse,
        mid_stream=mid,
        fine_stream=fine,
        compressed_seq_len=8,
        window_stride=1,
        chunk_tokens_stream=chunk_tokens,
    )
    sample = dataset[0]
    assert sample["input_scalars_coarse"].shape == (8,)
    assert sample["input_scalars_mid"].shape == (8,)
    assert sample["input_scalars_fine"].shape == (8,)
    assert sample["target_scalars"].shape == (8,)
    assert sample["target_scalars_mid"].shape == (8,)
    assert sample["target_scalars_fine"].shape == (8,)
    assert sample["target_chunk_ids"].shape == (8, 64)


def test_hierarchical_model_forward_shapes():
    model = HierarchicalScalarTransformerLM(
        model_cfg={
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 128,
            "dropout": 0.1,
        },
        hierarchy_cfg={
            "router_temperature": 1.0,
            "hard_routing": False,
            "base_residual_weight": 0.15,
        },
    )
    coarse = torch.randn(2, 8)
    mid = torch.randn(2, 8)
    fine = torch.randn(2, 8)
    preds, hidden, route_probs = model(
        input_scalars_coarse=coarse,
        input_scalars_mid=mid,
        input_scalars_fine=fine,
    )
    assert preds.shape == (2, 8)
    assert hidden.shape == (2, 8, 64)
    assert route_probs.shape == (2, 8, 3)
    probs_sum = route_probs.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_hierarchical_module_single_step_is_finite():
    module = HierarchicalFP16ChunkLMModule(
        config={
            "model": {
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "ff_dim": 128,
                "dropout": 0.1,
            },
            "training": {
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
            },
            "hierarchical": {
                "route_balance_weight": 0.02,
                "route_entropy_weight": 0.0,
                "route_target": [0.55, 0.30, 0.15],
                "router_temperature": 1.0,
                "hard_routing": False,
                "base_residual_weight": 0.15,
            },
            "diffusion": {
                "enabled": False,
            },
        },
    )
    batch = {
        "input_scalars_coarse": torch.randn(4, 8),
        "input_scalars_mid": torch.randn(4, 8),
        "input_scalars_fine": torch.randn(4, 8),
        "target_scalars": torch.randn(4, 8),
    }
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss).item()

    optimizer = module.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_decode_mirror_path_is_finite_and_decodes_tokens():
    chunk_size = 8
    module = HierarchicalFP16ChunkLMModule(
        config={
            "model": {
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "ff_dim": 128,
                "dropout": 0.1,
            },
            "training": {
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
            },
            "hierarchical": {
                "coarse_chunk_size": chunk_size,
                "route_balance_weight": 0.02,
                "route_entropy_weight": 0.0,
                "route_target": [0.55, 0.30, 0.15],
                "router_temperature": 1.0,
                "hard_routing": False,
                "base_residual_weight": 0.15,
            },
            "diffusion": {
                "enabled": True,
                "vocab_size": 128,
                "latent_dim": 16,
                "timesteps": 8,
                "chunk_size_tokens": chunk_size,
                "loss_weight": 0.25,
                "reconstruction_weight": 0.2,
            },
            "decode_mirror": {
                "enabled": True,
                "coarse_stride": 4,
                "mid_stride": 2,
                "fine_stride": 1,
                "loss_weight": 0.1,
                "coarse_weight": 0.2,
                "mid_weight": 0.3,
                "fine_weight": 0.5,
                "scalar_mid_weight": 0.25,
                "scalar_fine_weight": 0.25,
                "next_token_supervision": {
                    "enabled": True,
                    "loss_weight": 0.1,
                    "sample_batch_size": 2,
                    "position_stride": 2,
                    "max_positions": 4,
                },
                "diversity_regularization": {
                    "enabled": True,
                    "loss_weight": 0.05,
                    "sample_batch_size": 2,
                    "temperature": 1.0,
                    "entropy_target": 3.0,
                    "entropy_weight": 1.0,
                    "top1_max": 0.5,
                    "top1_weight": 1.0,
                    "adjacent_similarity_weight": 0.1,
                },
            },
        },
    )
    batch = {
        "input_scalars_coarse": torch.randn(2, 4),
        "input_scalars_mid": torch.randn(2, 4),
        "input_scalars_fine": torch.randn(2, 4),
        "target_scalars": torch.randn(2, 4),
        "target_scalars_mid": torch.randn(2, 4),
        "target_scalars_fine": torch.randn(2, 4),
        "target_chunk_ids": torch.randint(low=0, high=128, size=(2, 4, chunk_size)),
    }
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss).item()

    hidden_last = torch.randn(1, module.model.hidden_dim)
    decoded_ids = module.decode_chunk_ids_from_hidden(hidden_last)
    assert decoded_ids.shape == (1, chunk_size)
    assert torch.isfinite(decoded_ids.float()).all().item()

    hidden = torch.randn(2, 4, module.model.hidden_dim)
    target_chunk_ids = torch.randint(low=0, high=128, size=(2, 4, chunk_size))
    next_token_ce, next_token_acc = module._next_token_supervision_metrics(
        hidden=hidden,
        target_chunk_ids=target_chunk_ids,
    )
    assert torch.isfinite(next_token_ce).item()
    assert torch.isfinite(next_token_acc).item()

    diversity_loss, diversity_entropy, diversity_top1_prob, diversity_adjacent_cos = (
        module._diversity_regularization_metrics(hidden=hidden)
    )
    assert torch.isfinite(diversity_loss).item()
    assert torch.isfinite(diversity_entropy).item()
    assert torch.isfinite(diversity_top1_prob).item()
    assert torch.isfinite(diversity_adjacent_cos).item()


def test_sparse_decode_positions_are_densified_for_inference():
    chunk_size = 16
    module = HierarchicalFP16ChunkLMModule(
        config={
            "model": {
                "hidden_dim": 32,
                "num_layers": 1,
                "num_heads": 4,
                "ff_dim": 64,
                "dropout": 0.0,
            },
            "training": {
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
            },
            "hierarchical": {
                "coarse_chunk_size": chunk_size,
                "route_balance_weight": 0.02,
                "route_entropy_weight": 0.0,
                "route_target": [0.55, 0.30, 0.15],
                "router_temperature": 1.0,
                "hard_routing": False,
                "base_residual_weight": 0.15,
            },
            "diffusion": {
                "enabled": True,
                "vocab_size": 128,
                "latent_dim": 8,
                "timesteps": 8,
                "chunk_size_tokens": chunk_size,
                "loss_weight": 0.25,
                "reconstruction_weight": 0.2,
            },
            "decode_mirror": {
                "enabled": True,
                "coarse_stride": 8,
                "mid_stride": 8,
                "fine_stride": 8,
                "coarse_weight": 1.0,
                "mid_weight": 0.0,
                "fine_weight": 0.0,
                "infer_coarse_weight": 1.0,
                "infer_mid_weight": 0.0,
                "infer_fine_weight": 0.0,
            },
        },
    )
    pred_coarse = torch.zeros(1, 2, module.diffusion_latent_dim)
    pred_mid = torch.zeros(1, 2, module.diffusion_latent_dim)
    pred_fine = torch.zeros(1, 2, module.diffusion_latent_dim)
    pred_coarse[0, 0, 0] = 1.0
    pred_coarse[0, 1, 0] = 3.0

    fused = module._fuse_tier_embeddings(
        pred_coarse=pred_coarse,
        pred_mid=pred_mid,
        pred_fine=pred_fine,
        fill_missing_positions=True,
    )
    # Positions nearest to 0 replicate position 0, positions nearest to 8 replicate position 8.
    assert torch.allclose(fused[0, 1, :], fused[0, 0, :], atol=1e-6)
    assert torch.allclose(fused[0, 15, :], fused[0, 8, :], atol=1e-6)


def _output_head_test_config(family: str, chunk_size: int = 16) -> dict[str, object]:
    return {
        "model": {
            "hidden_dim": 48,
            "num_layers": 2,
            "num_heads": 4,
            "ff_dim": 96,
            "dropout": 0.1,
        },
        "training": {
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
        },
        "hierarchical": {
            "coarse_chunk_size": chunk_size,
            "route_balance_weight": 0.02,
            "route_entropy_weight": 0.0,
            "route_target": [0.55, 0.30, 0.15],
            "router_temperature": 1.0,
            "hard_routing": False,
            "base_residual_weight": 0.15,
        },
        "diffusion": {
            "enabled": True,
            "vocab_size": 256,
            "latent_dim": 16,
            "timesteps": 8,
            "chunk_size_tokens": chunk_size,
            "loss_weight": 0.25,
            "reconstruction_weight": 0.2,
        },
        "output_head": {
            "enabled": True,
            "family": family,
            "anchor_stride": 4,
            "refiner": {
                "enabled": family == "anchor_refine",
            },
            "loss_weights": {
                "token_ce": 0.2,
                "chunk_ce": 0.2 if family == "dual_task" else 0.0,
                "diffusion": 0.25,
            },
            "chunk_ce": {
                "sample_batch_size": 2,
                "sequence_stride": 2,
                "position_stride": 4,
                "max_positions": 4,
                "max_sequences": 2,
            },
        },
        "decode_mirror": {
            "next_token_supervision": {
                "enabled": True,
                "sample_batch_size": 2,
                "position_stride": 4,
                "max_positions": 4,
            },
            "diversity_regularization": {
                "enabled": False,
            },
        },
    }


def test_output_head_families_train_step_finite():
    for family in ("anchor_sparse", "anchor_refine", "dual_task"):
        module = HierarchicalFP16ChunkLMModule(config=_output_head_test_config(family))
        batch = {
            "input_scalars_coarse": torch.randn(2, 4),
            "input_scalars_mid": torch.randn(2, 4),
            "input_scalars_fine": torch.randn(2, 4),
            "target_scalars": torch.randn(2, 4),
            "target_scalars_mid": torch.randn(2, 4),
            "target_scalars_fine": torch.randn(2, 4),
            "target_chunk_ids": torch.randint(low=0, high=256, size=(2, 4, 16)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss).item()
        decoded_ids = module.decode_chunk_ids_from_hidden(torch.randn(1, module.model.hidden_dim))
        assert decoded_ids.shape == (1, 16)


def test_anchor_refine_refiner_is_deterministic():
    module = HierarchicalFP16ChunkLMModule(config=_output_head_test_config("anchor_refine"))
    pred_coarse = torch.randn(1, len(module.decode_mirror_coarse_positions), module.diffusion_latent_dim)
    pred_mid = torch.randn(1, len(module.decode_mirror_mid_positions), module.diffusion_latent_dim)
    pred_fine = torch.randn(1, len(module.decode_mirror_fine_positions), module.diffusion_latent_dim)
    fused_a = module._fuse_tier_embeddings(
        pred_coarse=pred_coarse,
        pred_mid=pred_mid,
        pred_fine=pred_fine,
        fill_missing_positions=True,
    )
    fused_b = module._fuse_tier_embeddings(
        pred_coarse=pred_coarse,
        pred_mid=pred_mid,
        pred_fine=pred_fine,
        fill_missing_positions=True,
    )
    assert torch.allclose(fused_a, fused_b, atol=1e-6)


def test_dual_task_chunk_supervision_is_finite():
    module = HierarchicalFP16ChunkLMModule(config=_output_head_test_config("dual_task"))
    hidden = torch.randn(2, 4, module.model.hidden_dim)
    target_chunk_ids = torch.randint(low=0, high=256, size=(2, 4, 16))
    chunk_ce, chunk_acc = module._chunk_supervision_metrics(
        hidden=hidden,
        target_chunk_ids=target_chunk_ids,
    )
    assert torch.isfinite(chunk_ce).item()
    assert torch.isfinite(chunk_acc).item()
