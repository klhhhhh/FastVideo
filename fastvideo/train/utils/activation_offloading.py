from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.autograd.graph import saved_tensors_hooks


def activation_offload_context(
    *,
    enabled: bool,
    min_tensor_size_mb: int = 1,
) -> Any:
    if not enabled:
        return nullcontext()

    threshold_bytes = int(min_tensor_size_mb) * 1024 * 1024

    def pack_hook(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.device | None]:
        if not isinstance(tensor, torch.Tensor):
            return tensor, None

        tensor_size_bytes = tensor.numel() * tensor.element_size()
        if tensor.is_cuda and tensor_size_bytes >= threshold_bytes:
            return tensor.detach().cpu(), tensor.device

        return tensor, None

    def unpack_hook(packed: tuple[torch.Tensor, torch.device | None]) -> torch.Tensor:
        tensor, device = packed
        if device is not None:
            return tensor.to(device=device, non_blocking=True)
        return tensor

    return saved_tensors_hooks(pack_hook, unpack_hook)