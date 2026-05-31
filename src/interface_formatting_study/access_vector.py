from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import torch

from .utils import LABELS


def normalize_vector(vector: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= eps:
        raise ValueError("Cannot normalize a zero or near-zero access vector")
    return vector / norm


def normalized_deltas(clean: torch.Tensor, corrupt: torch.Tensor) -> torch.Tensor:
    if clean.shape != corrupt.shape:
        raise ValueError("Clean and corrupt activation tensors must have matching shapes")
    deltas = clean.float() - corrupt.float()
    norms = torch.linalg.vector_norm(deltas, dim=-1, keepdim=True)
    return deltas / (norms + 1e-8)


def label_balanced_average(
    deltas: torch.Tensor,
    labels: Sequence[str],
    *,
    secondary_groups: Sequence[str] | None = None,
) -> torch.Tensor:
    if deltas.ndim != 2:
        raise ValueError("deltas must have shape [n_examples, d_model]")
    if len(labels) != deltas.shape[0]:
        raise ValueError("Number of labels must match number of deltas")
    if secondary_groups is not None and len(secondary_groups) != len(labels):
        raise ValueError("secondary_groups length must match labels")

    grouped: dict[tuple[str, str | None], list[torch.Tensor]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label not in LABELS:
            raise ValueError(f"Invalid label: {label!r}")
        secondary = None if secondary_groups is None else str(secondary_groups[i])
        grouped[(label, secondary)].append(deltas[i])

    label_means: list[torch.Tensor] = []
    for label in LABELS:
        matching = [torch.stack(v).mean(dim=0) for (group_label, _), v in grouped.items() if group_label == label]
        if not matching:
            continue
        label_means.append(torch.stack(matching).mean(dim=0))
    if not label_means:
        raise ValueError("No valid label groups available for averaging")
    return normalize_vector(torch.stack(label_means).mean(dim=0))


def build_access_vector(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    labels: Sequence[str],
    *,
    secondary_groups: Sequence[str] | None = None,
) -> torch.Tensor:
    deltas = normalized_deltas(clean, corrupt)
    return label_balanced_average(deltas, labels, secondary_groups=secondary_groups)


def add_access_vector(hidden: torch.Tensor, vector: torch.Tensor, alpha: float) -> torch.Tensor:
    return hidden + float(alpha) * vector.to(device=hidden.device, dtype=hidden.dtype)


def negative_intervention(hidden: torch.Tensor, vector: torch.Tensor, alpha: float) -> torch.Tensor:
    return add_access_vector(hidden, vector, -float(alpha))


def projection_removal(hidden: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    direction = normalize_vector(vector).to(device=hidden.device, dtype=hidden.dtype)
    coeff = (hidden * direction).sum(dim=-1, keepdim=True)
    return hidden - coeff * direction


def shuffled_pair_vector(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    labels: Sequence[str],
    *,
    secondary_groups: Sequence[str] | None = None,
    seed: int = 0,
) -> torch.Tensor:
    if clean.shape[0] < 2:
        raise ValueError("Need at least two examples for shuffled-pair control")
    generator = torch.Generator().manual_seed(seed)
    n = clean.shape[0]
    for _ in range(100):
        perm = torch.randperm(n, generator=generator)
        if not bool(torch.any(perm == torch.arange(n))):
            break
    else:
        perm = torch.roll(torch.arange(n), shifts=1)
    if bool(torch.any(perm == torch.arange(n))):
        raise AssertionError("Failed to construct a derangement for shuffled-pair control")
    return build_access_vector(clean[perm], corrupt, labels, secondary_groups=secondary_groups)


def random_vector_like(vector: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device=vector.device).manual_seed(seed)
    noise = torch.randn(vector.shape, generator=generator, device=vector.device, dtype=vector.dtype)
    return normalize_vector(noise) * torch.linalg.vector_norm(vector)
