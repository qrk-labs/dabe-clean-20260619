import hashlib
from typing import Dict, List

import lightning as L
import torch
import torch.nn.functional as F

from ..backbone.transformer import DABETransformer
from ..factories import build_encoder, build_router
from .contrastive import ContrastiveAlignmentLoss


class PretrainLightningModule(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.training_cfg = config.get("training", {})

        self.router = build_router(config)
        self.encoder = build_encoder(config)
        self.backbone = DABETransformer(config.get("backbone", {}))
        self.contrastive_loss = ContrastiveAlignmentLoss(
            temperature=self.training_cfg.get(
                "contrastive_temperature", config.get("contrastive_temperature", 0.07)
            ),
            margin=self.training_cfg.get("contrastive_margin", config.get("contrastive_margin", 0.5)),
        )

        self.recon_weight = self.training_cfg.get("recon_weight", config.get("recon_weight", 1.0))
        self.contrastive_weight = self.training_cfg.get(
            "contrastive_weight", config.get("contrastive_weight", 0.1)
        )

    def _stable_token_id(self, text: str) -> int:
        vocab_size = self.backbone.vocab_size
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) % vocab_size

    def _pad_span_encodings(self, span_encodings: List[torch.Tensor]) -> torch.Tensor:
        max_total_bw = max(encoding.shape[0] for encoding in span_encodings)
        return torch.stack(
            [
                F.pad(encoding.to(self.device), (0, max_total_bw - encoding.shape[0]))
                for encoding in span_encodings
            ]
        )

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        texts = batch["text"]
        langs = batch.get("lang", ["en"] * len(texts))

        recon_losses: List[torch.Tensor] = []
        all_hiddens: List[torch.Tensor] = []
        all_lang_ids: List[int] = []
        lang_to_idx: Dict[str, int] = {}
        total_spans = 0
        unique_targets: set[int] = set()

        for i, text in enumerate(texts):
            density_scores = self.router.score(text)
            spans = self.router.segment(text, density_scores)
            if not spans:
                continue

            span_texts = [span.text for span in spans]
            bit_widths = [span.bit_width for span in spans]

            span_encodings = []
            for span_text, bw in zip(span_texts, bit_widths):
                bits = self.encoder.encode(span_text, bw)
                span_encodings.append(bits)

            bits_tensor = self._pad_span_encodings(span_encodings)
            bw_tensor = torch.tensor(bit_widths, dtype=torch.long, device=self.device).unsqueeze(0)

            logits, hiddens = self.backbone(
                bits_tensor.unsqueeze(0),
                bit_widths=bw_tensor,
            )

            target_ids = [self._stable_token_id(span_text) for span_text in span_texts]
            recon_targets = torch.tensor(target_ids, dtype=torch.long, device=self.device)
            total_spans += len(span_texts)
            unique_targets.update(target_ids)
            recon_losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)), recon_targets))
            all_hiddens.append(hiddens.mean(dim=1).squeeze(0))

            lang = langs[i]
            if lang not in lang_to_idx:
                lang_to_idx[lang] = len(lang_to_idx)
            all_lang_ids.append(lang_to_idx[lang])

        if not recon_losses:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss": zero,
                "recon_loss": zero,
                "contrastive_loss": zero,
                "batch_num_spans": zero,
                "batch_unique_targets": zero,
                "batch_unique_langs": zero,
            }

        recon_loss = torch.stack(recon_losses).mean()
        hidden = torch.stack(all_hiddens, dim=0)
        lang_ids = torch.tensor(all_lang_ids, dtype=torch.long, device=self.device)
        contrastive_loss = self.contrastive_loss(hidden, lang_ids)

        loss = self.recon_weight * recon_loss + self.contrastive_weight * contrastive_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "contrastive_loss": contrastive_loss,
            "batch_num_spans": torch.tensor(float(total_spans), device=self.device),
            "batch_unique_targets": torch.tensor(float(len(unique_targets)), device=self.device),
            "batch_unique_langs": torch.tensor(float(len(set(all_lang_ids))), device=self.device),
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        batch_size = len(batch["text"])
        self.log_dict(
            {
                "train/loss": outputs["loss"],
                "train/recon_loss": outputs["recon_loss"],
                "train/contrastive_loss": outputs["contrastive_loss"],
                "train/num_spans": outputs["batch_num_spans"],
                "train/unique_targets": outputs["batch_unique_targets"],
                "train/unique_langs": outputs["batch_unique_langs"],
                "train_loss": outputs["loss"],
            },
            prog_bar=True,
            batch_size=batch_size,
        )
        return outputs["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        outputs = self(batch)
        batch_size = len(batch["text"])
        self.log_dict(
            {
                "val/loss": outputs["loss"],
                "val/recon_loss": outputs["recon_loss"],
                "val/contrastive_loss": outputs["contrastive_loss"],
                "val/num_spans": outputs["batch_num_spans"],
                "val/unique_targets": outputs["batch_unique_targets"],
                "val/unique_langs": outputs["batch_unique_langs"],
                "val_loss": outputs["loss"],
            },
            batch_size=batch_size,
        )
        return outputs

    def configure_optimizers(self):
        lr = self.training_cfg.get("learning_rate", self.config.get("learning_rate", 1e-4))
        wd = self.training_cfg.get("weight_decay", self.config.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_cfg.get("max_epochs", self.config.get("max_epochs", 10)),
        )
        return [optimizer], [scheduler]
