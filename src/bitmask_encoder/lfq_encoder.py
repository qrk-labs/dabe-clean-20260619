import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BitmaskEncoder


class LFQEncoder(BitmaskEncoder):
    """Learned LFQ tokenizer-style encoder (paper methodology section 3)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.max_vocab_size = int(config.get("vocab_size", 32000))
        self.vocab_size = max(2, self.max_vocab_size)
        self.embed_dim = int(config.get("embed_dim", 256))
        self.num_codes = int(config.get("num_codes", self.max_bit_width))
        self.unk_token = str(config.get("unk_token", "<unk>"))
        self.lower_case = bool(config.get("lower_case", True))

        self.proj = nn.Linear(self.embed_dim, self.max_bit_width)
        self.code_embed = nn.Parameter(torch.randn(self.max_bit_width, self.embed_dim) * 0.02)
        self.decode_proj = nn.Linear(self.embed_dim, self.vocab_size)
        self.token_embed = nn.Embedding(self.vocab_size, self.embed_dim)

        self.token_to_id: Dict[str, int] = {self.unk_token: 0}
        self.id_to_token: List[str] = [self.unk_token]
        self.token_id_to_width: Dict[int, int] = {0: self.max_bit_width}
        self.token_bitcodes: Dict[int, torch.Tensor] = {}

    def _normalize_span(self, span: str) -> str:
        cleaned = span.strip()
        if self.lower_case:
            cleaned = cleaned.lower()
        return cleaned or self.unk_token

    def _resize_vocab(self, vocab_size: int) -> None:
        """Resize vocab-dependent layers while preserving learned weights."""
        if vocab_size == self.vocab_size:
            return
        device = self.proj.weight.device
        old_token_embed = self.token_embed
        old_decode_proj = self.decode_proj
        copy_size = min(self.vocab_size, vocab_size)

        self.token_embed = nn.Embedding(vocab_size, self.embed_dim).to(device)
        self.decode_proj = nn.Linear(self.embed_dim, vocab_size).to(device)
        with torch.no_grad():
            self.token_embed.weight[:copy_size].copy_(old_token_embed.weight[:copy_size])
            self.decode_proj.weight[:copy_size].copy_(old_decode_proj.weight[:copy_size])
            self.decode_proj.bias[:copy_size].copy_(old_decode_proj.bias[:copy_size])

        self.vocab_size = vocab_size

    def set_vocabulary(
        self,
        tokens: Sequence[str],
        token_widths: Dict[str, int] | None = None,
    ) -> None:
        ordered_tokens = [self.unk_token]
        seen = {self.unk_token}
        for token in tokens:
            normalized = self._normalize_span(token)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_tokens.append(normalized)
            if len(ordered_tokens) >= self.max_vocab_size:
                break

        self._resize_vocab(len(ordered_tokens))
        self.id_to_token = ordered_tokens
        self.token_to_id = {token: idx for idx, token in enumerate(ordered_tokens)}

        token_widths = token_widths or {}
        self.token_id_to_width = {}
        for token, idx in self.token_to_id.items():
            width = int(token_widths.get(token, self.max_bit_width))
            width = max(1, min(self.max_bit_width, width))
            self.token_id_to_width[idx] = width

    def lookup_token_id(self, span: str) -> int:
        normalized = self._normalize_span(span)
        return self.token_to_id.get(normalized, 0)

    def decode_header(self, header_bits: torch.Tensor) -> int:
        width = 0
        for i in range(min(self.num_header_bits, header_bits.shape[0])):
            width |= int(header_bits[i].item()) << i
        return max(1, min(self.max_bit_width, width))

    def _stable_span_embedding(self, span: str, device: torch.device) -> torch.Tensor:
        digest = hashlib.blake2b(
            self._normalize_span(span).encode("utf-8"),
            digest_size=8,
        ).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        vector = torch.randn(self.embed_dim, generator=generator)
        return vector.to(device=device)

    def encode_token_ids(self, token_ids: torch.Tensor, bit_width: int) -> torch.Tensor:
        bit_width = max(1, min(self.max_bit_width, int(bit_width)))
        h = self.token_embed(token_ids)
        code_bits = self.quantize(h, bit_width)
        header = self.build_header(bit_width).to(token_ids.device)
        header = header.unsqueeze(0).expand(token_ids.shape[0], -1)
        return torch.cat([header, code_bits], dim=-1)

    def encode(self, span: str, bit_width: int) -> torch.Tensor:
        bit_width = max(1, min(self.max_bit_width, int(bit_width)))
        token_id = self.lookup_token_id(span)
        if token_id != 0:
            token_ids = torch.tensor([token_id], device=self.proj.weight.device)
            return self.encode_token_ids(token_ids, bit_width).squeeze(0).cpu()

        with torch.no_grad():
            h = self._stable_span_embedding(span, self.proj.weight.device).unsqueeze(0)
            code_bits = self.quantize(h, bit_width).squeeze(0).cpu()
        header = self.build_header(bit_width)
        return torch.cat([header, code_bits], dim=0)

    def decode(self, bits: torch.Tensor) -> str:
        if bits.numel() <= self.num_header_bits:
            return self.unk_token
        bit_width = self.decode_header(bits[: self.num_header_bits])
        total = self.total_bit_width(bit_width)
        truncated = bits[:total]
        embedded = self.embed(truncated)
        logits = self.decode_proj(embedded)
        token_id = int(torch.argmax(logits).item())
        if 0 <= token_id < len(self.id_to_token):
            return self.id_to_token[token_id]
        return self.unk_token

    def embed(self, bits: torch.Tensor) -> torch.Tensor:
        total_bw = bits.shape[-1]
        code_bits = bits[self.num_header_bits : total_bw]
        if code_bits.numel() == 0:
            return torch.zeros(self.embed_dim, device=bits.device, dtype=torch.float32)
        code = 2.0 * code_bits.float() - 1.0
        embedded = code @ self.code_embed[: code_bits.shape[0]]
        return embedded

    def quantize(self, h: torch.Tensor, bit_width: int) -> torch.Tensor:
        bit_width = max(1, min(self.max_bit_width, int(bit_width)))
        logits = self.proj(h)[..., :bit_width]
        z = torch.tanh(logits)
        bits = (z > 0).long()
        return bits

    def straight_through(self, h: torch.Tensor, bit_width: int) -> torch.Tensor:
        bit_width = max(1, min(self.max_bit_width, int(bit_width)))
        logits = self.proj(h)[..., :bit_width]
        z = torch.tanh(logits)
        bits = (z > 0).float()
        bits = bits + z - z.detach()
        return bits

    def forward(self, h: torch.Tensor, bit_width: int) -> tuple[torch.Tensor, torch.Tensor]:
        bit_width = max(1, min(self.max_bit_width, int(bit_width)))
        bits = self.straight_through(h, bit_width)
        embedded = bits @ self.code_embed[:bit_width]
        logits = self.decode_proj(embedded)
        return logits, bits

    def train_tokenizer(
        self,
        spans: Sequence[str],
        bit_widths: Sequence[int],
        epochs: int = 2,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> dict:
        """Fit LFQ codes from span supervision (paper methodology section 3.3)."""
        if len(spans) != len(bit_widths):
            raise ValueError("spans and bit_widths must have same length")
        if not spans:
            raise ValueError("train_tokenizer received no spans")

        normalized = [self._normalize_span(span) for span in spans]
        counts = Counter(normalized)
        width_sums: dict[str, float] = defaultdict(float)
        width_counts: dict[str, int] = defaultdict(int)
        for token, width in zip(normalized, bit_widths):
            bounded_width = max(1, min(self.max_bit_width, int(width)))
            width_sums[token] += float(bounded_width)
            width_counts[token] += 1

        sorted_tokens = [token for token, _ in counts.most_common(self.max_vocab_size - 1)]
        token_width_map = {
            token: int(round(width_sums[token] / max(width_counts[token], 1)))
            for token in sorted_tokens
        }
        self.set_vocabulary(sorted_tokens, token_width_map)

        train_ids = torch.tensor(
            [self.lookup_token_id(token) for token in normalized],
            dtype=torch.long,
        )
        train_widths = torch.tensor(
            [self.token_id_to_width[self.lookup_token_id(token)] for token in normalized],
            dtype=torch.long,
        )

        self.to(device)
        train_ids = train_ids.to(device)
        train_widths = train_widths.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)

        step_losses: List[float] = []
        for _ in range(max(1, epochs)):
            permutation = torch.randperm(train_ids.shape[0], device=device)
            for start in range(0, train_ids.shape[0], max(1, batch_size)):
                idx = permutation[start : start + max(1, batch_size)]
                batch_ids = train_ids[idx]
                batch_widths = train_widths[idx]
                hidden = self.token_embed(batch_ids)

                loss = torch.tensor(0.0, device=device)
                for width in torch.unique(batch_widths):
                    mask = batch_widths == width
                    if not torch.any(mask):
                        continue
                    logits, _ = self.forward(hidden[mask], int(width.item()))
                    width_loss = F.cross_entropy(logits, batch_ids[mask])
                    loss = loss + width_loss * (mask.float().mean())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step_losses.append(float(loss.detach().cpu().item()))

        self.eval()
        self.token_bitcodes = {}
        with torch.no_grad():
            for token_id, width in self.token_id_to_width.items():
                ids = torch.tensor([token_id], dtype=torch.long, device=device)
                bits = self.encode_token_ids(ids, width).squeeze(0).detach().cpu()
                self.token_bitcodes[token_id] = bits

        return {
            "num_spans": len(spans),
            "vocab_size": len(self.id_to_token),
            "avg_train_loss": sum(step_losses) / max(len(step_losses), 1),
        }

    def export_artifact(self, path: str | Path) -> Path:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "state_dict": self.state_dict(),
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "token_id_to_width": self.token_id_to_width,
            "token_bitcodes": {
                int(token_id): bits.tolist()
                for token_id, bits in self.token_bitcodes.items()
            },
        }
        torch.save(payload, artifact_path)
        return artifact_path

    def load_artifact(self, path: str | Path, map_location: str = "cpu") -> None:
        artifact = torch.load(Path(path), map_location=map_location)
        self.config = artifact.get("config", self.config)
        self.max_bit_width = int(self.config.get("max_bit_width", self.max_bit_width))
        self.num_header_bits = int(self.config.get("num_header_bits", self.num_header_bits))

        id_to_token = list(artifact["id_to_token"])
        token_widths = {
            id_to_token[int(token_id)]: int(width)
            for token_id, width in artifact["token_id_to_width"].items()
            if int(token_id) < len(id_to_token)
        }
        self.set_vocabulary(id_to_token[1:], token_widths)
        self.load_state_dict(artifact["state_dict"])
        self.token_bitcodes = {
            int(token_id): torch.tensor(bits, dtype=torch.long)
            for token_id, bits in artifact.get("token_bitcodes", {}).items()
        }
