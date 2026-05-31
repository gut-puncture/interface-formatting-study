from __future__ import annotations

import torch

from .access_vector import negative_intervention, projection_removal, random_vector_like, shuffled_pair_vector


def make_random_controls(vector: torch.Tensor, *, seeds: range | list[int] = range(5)) -> dict[str, torch.Tensor]:
    return {f"random_seed_{seed}": random_vector_like(vector, seed=int(seed)) for seed in seeds}


def make_direction_controls(vector: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "interface_formatting_study_positive": vector,
        "wrong_direction": -vector,
    }


def apply_control(hidden: torch.Tensor, vector: torch.Tensor, *, alpha: float, mode: str) -> torch.Tensor:
    if mode == "positive":
        return hidden + alpha * vector.to(device=hidden.device, dtype=hidden.dtype)
    if mode == "negative":
        return negative_intervention(hidden, vector, alpha)
    if mode == "projection_removal":
        return projection_removal(hidden, vector)
    raise ValueError(f"Unknown control mode: {mode}")


__all__ = [
    "make_random_controls",
    "make_direction_controls",
    "apply_control",
    "shuffled_pair_vector",
]
