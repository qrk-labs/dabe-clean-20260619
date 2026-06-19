import pytest
import torch

from src.training.contrastive import ContrastiveAlignmentLoss
from src.training.diffusion_head import DiffusionDecodingHead


class TestContrastiveAlignmentLoss:
    def test_forward(self):
        loss_fn = ContrastiveAlignmentLoss(temperature=0.07, margin=0.5)
        embeddings = torch.randn(4, 256)
        lang_ids = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(embeddings, lang_ids)
        assert loss.item() >= 0.0

    def test_zero_loss_same_language(self):
        loss_fn = ContrastiveAlignmentLoss(temperature=1.0, margin=0.0)
        embeddings = torch.ones(4, 256)
        lang_ids = torch.tensor([0, 0, 0, 0])
        loss = loss_fn(embeddings, lang_ids)
        assert loss.item() >= 0.0


class TestDiffusionDecodingHead:
    def test_forward(self):
        head = DiffusionDecodingHead(hidden_dim=512, max_timesteps=1000)
        h = torch.randn(2, 512)
        out = head(h, timestep=100)
        assert out.shape == (2, 512)
