from dataclasses import dataclass
from typing import List, Optional

import torch
import wandb

from .backbone.transformer import DABETransformer
from .bitmask_encoder.base import BitmaskEncoder
from .density_router.base import DensityRouter, Span
from .factories import build_encoder, build_router
from .runtime.device import resolve_torch_device
from .training.contrastive import ContrastiveAlignmentLoss


@dataclass
class PipelineOutput:
    texts: List[str]
    spans: List[List[Span]]
    bitmasks: List[torch.Tensor]
    bit_widths: List[List[int]]
    backbone_logits: Optional[torch.Tensor] = None
    backbone_hidden: Optional[torch.Tensor] = None


class DABEPipeline:
    def __init__(self, config: dict):
        self.config = config
        requested_device = config.get("device", config.get("experiment", {}).get("device"))
        self.device = resolve_torch_device(requested_device)

        self.router: DensityRouter = build_router(config)
        self.encoder: BitmaskEncoder = build_encoder(config)

        self.backbone = DABETransformer(config.get("backbone", {})).to(self.device)
        self.contrastive_loss = ContrastiveAlignmentLoss(
            temperature=config.get("training", {}).get(
                "contrastive_temperature", config.get("contrastive_temperature", 0.07)
            ),
            margin=config.get("training", {}).get(
                "contrastive_margin", config.get("contrastive_margin", 0.5)
            ),
        )

    def forward(self, texts: List[str], langs: Optional[List[str]] = None) -> PipelineOutput:
        all_spans = []
        all_bitmasks = []
        all_bit_widths = []

        for text in texts:
            scores = self.router.score(text)
            spans = self.router.segment(text, scores)
            all_spans.append(spans)
            span_bitmasks = []
            span_bws = []
            max_total_bw = 0
            for span in spans:
                bits = self.encoder.encode(span.text, span.bit_width)
                span_bitmasks.append(bits)
                span_bws.append(span.bit_width)
                max_total_bw = max(max_total_bw, bits.shape[0])

            if span_bitmasks:
                padded = torch.stack(
                    [torch.nn.functional.pad(b, (0, max_total_bw - b.shape[0])) for b in span_bitmasks]
                )
            else:
                padded = torch.empty(0)
            all_bitmasks.append(padded)
            all_bit_widths.append(span_bws)

        return PipelineOutput(
            texts=texts,
            spans=all_spans,
            bitmasks=all_bitmasks,
            bit_widths=all_bit_widths,
        )

    def validate_shapes(self, batch_size: int = 2, seq_len: int = 4):
        max_bit_width = self.config["backbone"]["max_bit_width"]
        min_data_bw = self.config.get("router", {}).get("min_bit_width", 8)
        max_data_bw = self.config.get("router", {}).get("max_bit_width", max_bit_width)
        dummy_bits = torch.randint(0, 2, (batch_size, seq_len, max_bit_width), device=self.device)
        dummy_bw = torch.randint(min_data_bw, max_data_bw + 1, (batch_size, seq_len), device=self.device)

        logits, hidden = self.backbone(dummy_bits, bit_widths=dummy_bw)
        assert logits.shape == (batch_size, seq_len, self.config["backbone"]["vocab_size"]), (
            f"Expected logits shape ({batch_size}, {seq_len}, {self.config['backbone']['vocab_size']}), "
            f"got {logits.shape}"
        )
        assert hidden.shape == (batch_size, seq_len, self.config["backbone"]["hidden_dim"]), (
            f"Expected hidden shape ({batch_size}, {seq_len}, {self.config['backbone']['hidden_dim']}), "
            f"got {hidden.shape}"
        )
        return {"logits_shape": logits.shape, "hidden_shape": hidden.shape}

    def log_to_wandb(self, output: PipelineOutput, step: int):
        if wandb.run is None:
            return
        wandb.log({
            "pipeline/num_spans": sum(len(s) for s in output.spans),
            "pipeline/avg_bit_width": (
                sum(sum(bw) for bw in output.bit_widths) /
                max(sum(len(bw) for bw in output.bit_widths), 1)
            ),
            "pipeline/num_texts": len(output.texts),
        }, step=step)

    def run_dummy_forward(self):
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Le renard brun rapide saute par-dessus le chien paresseux.",
        ]
        langs = ["en", "fr"]

        output = self.forward(texts, langs)
        shapes = self.validate_shapes(batch_size=len(texts), seq_len=4)

        report = {
            "num_texts": len(texts),
            "num_spans": [len(s) for s in output.spans],
            "bit_widths": output.bit_widths,
            "shapes": shapes,
        }
        return report
