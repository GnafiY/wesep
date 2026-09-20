"""Compatibility helpers for supported PyTorch versions."""

import torch


def register_load_state_dict_pre_hook(module, hook):
    """Register a load-state pre-hook across PyTorch versions."""
    register = getattr(module, "register_load_state_dict_pre_hook", None)
    if register is not None:
        return register(hook)

    register = getattr(module, "_register_load_state_dict_pre_hook", None)
    if register is None:
        raise RuntimeError(
            "The installed PyTorch version provides no load_state_dict "
            "pre-hook API.")
    return register(hook, with_module=True)


def create_cuda_grad_scaler(enabled):
    """Create a CUDA gradient scaler across PyTorch versions."""
    grad_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_autocast(enabled):
    """Create a CUDA autocast context across PyTorch versions."""
    autocast = getattr(getattr(torch, "amp", None), "autocast", None)
    if autocast is not None:
        return autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)
