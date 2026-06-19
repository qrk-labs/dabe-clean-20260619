import logging
from typing import Optional

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset

log = logging.getLogger(__name__)


class MultilingualCorpus(TorchDataset):
    def __init__(
        self,
        source_langs: list[str],
        target_langs: Optional[list[str]] = None,
        dataset_name: str = "wikiann",
        synthetic_only: bool = False,
        split: str = "train",
        max_samples: int = 10000,
    ):
        self.source_langs = source_langs
        self.target_langs = target_langs or source_langs
        self.dataset_name = dataset_name
        self.synthetic_only = synthetic_only
        self.samples = []
        per_lang = max(1, max_samples // max(len(source_langs), 1))
        for lang in source_langs:
            if self.synthetic_only:
                self.samples.extend(self._fallback_samples(lang, per_lang))
                continue

            loaded = False
            dataset_candidates = [self.dataset_name, "unimelb-nlp/wikiann"]
            for dataset_id in dataset_candidates:
                try:
                    ds = load_dataset(dataset_id, lang, split=split, streaming=True)
                    for i, example in enumerate(ds):
                        if i >= per_lang:
                            break
                        self.samples.append(
                            {"text": example["tokens"], "lang": lang, "source_lang": lang}
                        )
                    loaded = True
                    break
                except Exception:
                    continue

            if not loaded:
                log.warning(
                    "Falling back to synthetic data for lang=%s split=%s (dataset=%s).",
                    lang,
                    split,
                    self.dataset_name,
                )
                self.samples.extend(self._fallback_samples(lang, per_lang))

    def _fallback_samples(self, lang: str, count: int) -> list[dict]:
        templates = [
            "dense bitmask encoding works",
            "adaptive width routing for language",
            "local metal run for dabe",
            "cross lingual alignment objective",
        ]
        samples = []
        for i in range(count):
            text = f"{templates[i % len(templates)]} {lang} {i}"
            samples.append({"text": text.split(), "lang": lang, "source_lang": lang})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class MultilingualDataModule(L.LightningDataModule):
    def __init__(
        self,
        source_langs: list[str],
        target_langs: Optional[list[str]] = None,
        dataset_name: str = "wikiann",
        synthetic_only: bool = False,
        batch_size: int = 8,
        max_samples: int = 10000,
        num_workers: int = 4,
    ):
        super().__init__()
        self.source_langs = source_langs
        self.target_langs = target_langs or source_langs
        self.dataset_name = dataset_name
        self.synthetic_only = synthetic_only
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        self.train_dataset = MultilingualCorpus(
            source_langs=self.source_langs,
            target_langs=self.target_langs,
            dataset_name=self.dataset_name,
            synthetic_only=self.synthetic_only,
            split="train",
            max_samples=self.max_samples,
        )
        self.val_dataset = MultilingualCorpus(
            source_langs=self.source_langs,
            target_langs=self.target_langs,
            dataset_name=self.dataset_name,
            synthetic_only=self.synthetic_only,
            split="validation",
            max_samples=max(self.batch_size * 8, self.max_samples // 10),
        )

    def collate_fn(self, batch: list[dict]) -> dict:
        texts = [item["text"] for item in batch]
        if isinstance(texts[0], list):
            texts = [" ".join(t) for t in texts]
        return {"text": texts, "lang": [item["lang"] for item in batch]}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
