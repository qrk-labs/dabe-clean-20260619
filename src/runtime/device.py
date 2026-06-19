from typing import Tuple

import torch


def resolve_torch_device(device_pref: str | None) -> torch.device:
    choice = (device_pref or "auto").lower()

    if choice == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device='mps' but MPS is unavailable on this machine.")
        return torch.device("mps")

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is unavailable on this machine.")
        return torch.device("cuda")

    return torch.device("cpu")


def resolve_trainer_accelerator(device_pref: str | None) -> Tuple[str, int]:
    device = resolve_torch_device(device_pref)
    if device.type == "mps":
        return "mps", 1
    if device.type == "cuda":
        return "gpu", 1
    return "cpu", 1
