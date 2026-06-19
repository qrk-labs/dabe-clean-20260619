import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignmentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        lang_ids: torch.Tensor,
    ) -> torch.Tensor:
        norms = F.normalize(embeddings, dim=-1)
        sim_matrix = torch.mm(norms, norms.t()) / self.temperature

        same_lang = lang_ids.unsqueeze(1) == lang_ids.unsqueeze(0)
        pos_mask = same_lang.clone()
        pos_mask.fill_diagonal_(False)
        neg_mask = ~same_lang

        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        pos_sum = (sim_matrix * pos_mask.float()).sum(dim=1)
        pos_mean = pos_sum / pos_counts.clamp_min(1).float()

        neg_sim = sim_matrix.masked_fill(~neg_mask, float("-inf"))
        neg_max = neg_sim.max(dim=1).values
        neg_max = torch.where(torch.isfinite(neg_max), neg_max, torch.zeros_like(neg_max))

        loss = F.relu(neg_max - pos_mean + self.margin)[valid].mean()
        return loss
