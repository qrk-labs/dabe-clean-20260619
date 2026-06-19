import numpy as np
import torch

from src.inference.hierarchical_generation import (
    HierarchicalGenerationConfig,
    HierarchicalGenerator,
)


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [max(1, ord(ch) % 32) for ch in text]

    def decode(self, token_ids, clean_up_tokenization_spaces: bool = False) -> str:
        del clean_up_tokenization_spaces
        return " ".join(str(int(token_id)) for token_id in token_ids)


class _FakeCompressor:
    def __init__(self, chunk_size_tokens: int):
        self.chunk_size_tokens = int(chunk_size_tokens)
        self.tokenizer = _FakeTokenizer()
        self.vocab_size = 8192

    def compress_token_ids(self, token_ids) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.float32).reshape(-1)
        if ids.shape[0] < self.chunk_size_tokens:
            return np.empty((0,), dtype=np.float32)
        usable = ids[: (ids.shape[0] // self.chunk_size_tokens) * self.chunk_size_tokens]
        if usable.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        return usable.reshape(-1, self.chunk_size_tokens).mean(axis=1).astype(np.float32, copy=False)

    def compress_token_ids_with_chunks(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.shape[0] < self.chunk_size_tokens:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0, self.chunk_size_tokens), dtype=np.int32),
            )
        usable = ids[: (ids.shape[0] // self.chunk_size_tokens) * self.chunk_size_tokens]
        if usable.shape[0] == 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0, self.chunk_size_tokens), dtype=np.int32),
            )
        chunks = usable.reshape(-1, self.chunk_size_tokens).astype(np.int32, copy=False)
        scalars = chunks.astype(np.float32).mean(axis=1)
        return scalars.astype(np.float32, copy=False), chunks


class _ConstHead(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.full((hidden.shape[0], 1), self.value, device=hidden.device, dtype=hidden.dtype)


class _FakeInnerModel:
    def __init__(self, hidden_dim: int):
        self.hidden_dim = int(hidden_dim)

    def __call__(
        self,
        *,
        input_scalars_coarse: torch.Tensor,
        input_scalars_mid: torch.Tensor,
        input_scalars_fine: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del input_scalars_mid, input_scalars_fine, mask
        preds = input_scalars_coarse + 0.5
        hidden = torch.ones(
            (input_scalars_coarse.shape[0], input_scalars_coarse.shape[1], self.hidden_dim),
            device=input_scalars_coarse.device,
            dtype=input_scalars_coarse.dtype,
        )
        route = torch.full(
            (input_scalars_coarse.shape[0], input_scalars_coarse.shape[1], 3),
            1.0 / 3.0,
            device=input_scalars_coarse.device,
            dtype=input_scalars_coarse.dtype,
        )
        return preds, hidden, route


class _FakeModule:
    def __init__(self, chunk_size_tokens: int = 8):
        self.decode_calls = 0
        self.decode_mirror_enabled = True
        self.chunk_size_tokens = int(chunk_size_tokens)
        self.model = _FakeInnerModel(hidden_dim=4)
        self.mid_scalar_head = _ConstHead(0.2)
        self.fine_scalar_head = _ConstHead(0.3)

    def decode_chunk_ids_from_hidden(self, hidden_last: torch.Tensor) -> torch.Tensor:
        del hidden_last
        self.decode_calls += 1
        base = self.decode_calls * 10
        row = torch.arange(base, base + self.chunk_size_tokens, dtype=torch.int64)
        return row.unsqueeze(0)


class _TinyTTTBackbone:
    def __call__(
        self,
        *,
        input_scalars_coarse: torch.Tensor,
        input_scalars_mid: torch.Tensor,
        input_scalars_fine: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del mask
        preds = input_scalars_coarse + 0.05
        hidden = torch.stack(
            [
                input_scalars_coarse,
                input_scalars_mid,
                input_scalars_fine,
                torch.ones_like(input_scalars_coarse),
            ],
            dim=-1,
        )
        route = torch.full(
            (input_scalars_coarse.shape[0], input_scalars_coarse.shape[1], 3),
            1.0 / 3.0,
            device=input_scalars_coarse.device,
            dtype=input_scalars_coarse.dtype,
        )
        return preds, hidden, route


class _TinyTTTModule:
    def __init__(self, chunk_size_tokens: int = 8):
        self.decode_mirror_enabled = True
        self.chunk_size_tokens = int(chunk_size_tokens)
        self.diffusion_latent_dim = 4
        self.decode_mirror_coarse_positions = list(range(0, self.chunk_size_tokens, 4))
        self.decode_mirror_mid_positions = list(range(0, self.chunk_size_tokens, 2))
        self.decode_mirror_fine_positions = list(range(0, self.chunk_size_tokens, 1))
        self.decode_mirror_infer_coarse_weight = 0.2
        self.decode_mirror_infer_mid_weight = 0.3
        self.decode_mirror_infer_fine_weight = 0.5
        self.model = _TinyTTTBackbone()
        self.chunk_token_embed = torch.nn.Embedding(128, self.diffusion_latent_dim)
        self.decode_mirror_coarse_head = torch.nn.Linear(
            4,
            len(self.decode_mirror_coarse_positions) * self.diffusion_latent_dim,
        )
        self.decode_mirror_mid_head = torch.nn.Linear(
            4,
            len(self.decode_mirror_mid_positions) * self.diffusion_latent_dim,
        )
        self.decode_mirror_fine_head = torch.nn.Linear(
            4,
            len(self.decode_mirror_fine_positions) * self.diffusion_latent_dim,
        )
        self.mid_scalar_head = torch.nn.Linear(4, 1)
        self.fine_scalar_head = torch.nn.Linear(4, 1)
        self._training = False

    @property
    def training(self) -> bool:
        return self._training

    def train(self) -> "_TinyTTTModule":
        self._training = True
        return self

    def eval(self) -> "_TinyTTTModule":
        self._training = False
        return self

    def _project_tier_embeddings(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse = self.decode_mirror_coarse_head(hidden).reshape(
            hidden.shape[0],
            hidden.shape[1],
            len(self.decode_mirror_coarse_positions),
            self.diffusion_latent_dim,
        )
        mid = self.decode_mirror_mid_head(hidden).reshape(
            hidden.shape[0],
            hidden.shape[1],
            len(self.decode_mirror_mid_positions),
            self.diffusion_latent_dim,
        )
        fine = self.decode_mirror_fine_head(hidden).reshape(
            hidden.shape[0],
            hidden.shape[1],
            len(self.decode_mirror_fine_positions),
            self.diffusion_latent_dim,
        )
        return coarse, mid, fine

    def _fuse_tier_embeddings(
        self,
        *,
        pred_coarse: torch.Tensor,
        pred_mid: torch.Tensor,
        pred_fine: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(pred_fine.shape[0])
        fused = torch.zeros(
            (batch_size, self.chunk_size_tokens, self.diffusion_latent_dim),
            device=pred_fine.device,
            dtype=pred_fine.dtype,
        )
        weights = torch.zeros(
            (batch_size, self.chunk_size_tokens, 1),
            device=pred_fine.device,
            dtype=pred_fine.dtype,
        )
        fused[:, self.decode_mirror_coarse_positions, :] += self.decode_mirror_infer_coarse_weight * pred_coarse
        weights[:, self.decode_mirror_coarse_positions, :] += self.decode_mirror_infer_coarse_weight
        fused[:, self.decode_mirror_mid_positions, :] += self.decode_mirror_infer_mid_weight * pred_mid
        weights[:, self.decode_mirror_mid_positions, :] += self.decode_mirror_infer_mid_weight
        fused[:, self.decode_mirror_fine_positions, :] += self.decode_mirror_infer_fine_weight * pred_fine
        weights[:, self.decode_mirror_fine_positions, :] += self.decode_mirror_infer_fine_weight
        return fused / weights.clamp(min=1e-6)

    def decode_chunk_ids_from_hidden(self, hidden_last: torch.Tensor) -> torch.Tensor:
        context = hidden_last.unsqueeze(1)
        coarse, mid, fine = self._project_tier_embeddings(context)
        fused = self._fuse_tier_embeddings(
            pred_coarse=coarse[:, 0, :, :],
            pred_mid=mid[:, 0, :, :],
            pred_fine=fine[:, 0, :, :],
        )
        vocab_embed = torch.nn.functional.normalize(self.chunk_token_embed.weight, dim=-1)
        fused_norm = torch.nn.functional.normalize(fused.reshape(-1, fused.shape[-1]), dim=-1)
        similarity = torch.matmul(fused_norm, vocab_embed.transpose(0, 1))
        return similarity.argmax(dim=-1).reshape(hidden_last.shape[0], self.chunk_size_tokens)


def test_fast_mode_reuses_cached_chunks():
    module = _FakeModule(chunk_size_tokens=8)
    generator = HierarchicalGenerator(
        model_module=module,  # type: ignore[arg-type]
        coarse_compressor=_FakeCompressor(chunk_size_tokens=4),  # type: ignore[arg-type]
        mid_compressor=_FakeCompressor(chunk_size_tokens=2),  # type: ignore[arg-type]
        fine_compressor=_FakeCompressor(chunk_size_tokens=1),  # type: ignore[arg-type]
        config=HierarchicalGenerationConfig(
            compressed_seq_len=2,
            continuation_chunks=5,
            decode_mode="mirror_decoder",
            runtime_mode="fast",
            fast_decode_interval=3,
            fast_refine_tokens=2,
        ),
        device=torch.device("cpu"),
    )

    result = generator.generate("abcdefgh")
    assert module.decode_calls == 2
    assert len(result.generated_token_ids) == 40
    first_chunk = result.generated_token_ids[:8]
    second_chunk = result.generated_token_ids[8:16]
    fourth_chunk = result.generated_token_ids[24:32]
    assert first_chunk == second_chunk
    assert fourth_chunk[:2] == [20, 21]
    assert result.collapse_detected is True


def test_memory_lookup_mode_uses_train_stream_alignment():
    module = _FakeModule(chunk_size_tokens=8)
    generator = HierarchicalGenerator(
        model_module=module,  # type: ignore[arg-type]
        coarse_compressor=_FakeCompressor(chunk_size_tokens=4),  # type: ignore[arg-type]
        mid_compressor=_FakeCompressor(chunk_size_tokens=2),  # type: ignore[arg-type]
        fine_compressor=_FakeCompressor(chunk_size_tokens=1),  # type: ignore[arg-type]
        config=HierarchicalGenerationConfig(
            compressed_seq_len=2,
            continuation_chunks=2,
            decode_mode="memory_lookup",
            runtime_mode="quality",
        ),
        device=torch.device("cpu"),
        train_coarse=np.asarray([0.0, 2.0], dtype=np.float32),
        train_mid=np.asarray([0.1, 0.2], dtype=np.float32),
        train_fine=np.asarray([0.3, 0.4], dtype=np.float32),
        train_chunk_tokens=np.asarray(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [11, 12, 13, 14, 15, 16, 17, 18],
            ],
            dtype=np.int32,
        ),
    )

    result = generator.generate("abcdefgh")
    assert module.decode_calls == 0
    assert len(result.generated_token_ids) == 16
    assert result.generated_token_ids[:8] == [11, 12, 13, 14, 15, 16, 17, 18]


def test_ttt_applies_and_restores_decoder_weights():
    module = _TinyTTTModule(chunk_size_tokens=8)
    coarse = _FakeCompressor(chunk_size_tokens=8)
    mid = _FakeCompressor(chunk_size_tokens=2)
    fine = _FakeCompressor(chunk_size_tokens=1)
    generator = HierarchicalGenerator(
        model_module=module,  # type: ignore[arg-type]
        coarse_compressor=coarse,  # type: ignore[arg-type]
        mid_compressor=mid,  # type: ignore[arg-type]
        fine_compressor=fine,  # type: ignore[arg-type]
        config=HierarchicalGenerationConfig(
            compressed_seq_len=4,
            continuation_chunks=1,
            decode_mode="mirror_decoder",
            runtime_mode="quality",
            ttt_enabled=True,
            ttt_steps=2,
            ttt_lr=1e-2,
            ttt_batch_size=2,
            ttt_max_windows=4,
            ttt_position_stride=2,
            ttt_max_positions=4,
        ),
        device=torch.device("cpu"),
    )
    before = module.decode_mirror_coarse_head.weight.detach().clone()
    result = generator.generate("abcdefghijklmnopqrstuvwxyz0123456789")
    after = module.decode_mirror_coarse_head.weight.detach().clone()
    assert result.ttt_applied is True
    assert result.ttt_steps_run > 0
    assert torch.allclose(before, after)
