import torch
import torch.nn as nn


class DiffusionDecodingHead(nn.Module):
    def __init__(self, hidden_dim: int = 512, max_timesteps: int = 1000):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.denoise = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_logits = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, timestep: int | torch.Tensor) -> torch.Tensor:
        t = torch.full((h.size(0), 1), fill_value=timestep, dtype=torch.float, device=h.device)
        t_emb = self.time_embed(t / self.max_timesteps)
        noisy = h + torch.randn_like(h) * (timestep / self.max_timesteps)
        out = self.denoise(torch.cat([noisy, t_emb], dim=-1))
        return self.to_logits(out)
