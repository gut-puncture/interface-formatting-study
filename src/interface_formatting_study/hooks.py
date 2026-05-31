from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Mapping

import torch


def find_transformer_blocks(model):
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
        ("model.decoder.layers", lambda m: m.model.decoder.layers),
    ]
    for _, getter in candidates:
        try:
            blocks = getter(model)
        except AttributeError:
            continue
        if blocks is not None:
            return blocks
    raise ValueError("Could not locate decoder transformer blocks on this model")


@dataclass
class ResidualCapture(AbstractContextManager):
    model: object
    layer: int
    position: int

    def __post_init__(self) -> None:
        self._handle = None
        self.value: torch.Tensor | None = None

    def __enter__(self):
        blocks = find_transformer_blocks(self.model)
        block = blocks[self.layer]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.value = hidden[:, self.position, :].detach().float().cpu()
            return output

        self._handle = block.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


@dataclass
class ResidualEdit(AbstractContextManager):
    model: object
    layer: int
    position: int
    edit_fn: Callable[[torch.Tensor], torch.Tensor]

    def __post_init__(self) -> None:
        self._handle = None

    def __enter__(self):
        blocks = find_transformer_blocks(self.model)
        block = blocks[self.layer]

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, self.position, :] = self.edit_fn(hidden[:, self.position, :])
                return (hidden, *output[1:])
            hidden = output.clone()
            hidden[:, self.position, :] = self.edit_fn(hidden[:, self.position, :])
            return hidden

        self._handle = block.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


@dataclass
class ResidualMultiEdit(AbstractContextManager):
    model: object
    layer: int
    edits: Mapping[int, Callable[[torch.Tensor], torch.Tensor]]

    def __post_init__(self) -> None:
        self._handle = None

    def __enter__(self):
        blocks = find_transformer_blocks(self.model)
        block = blocks[self.layer]

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                for position, edit_fn in sorted(self.edits.items()):
                    hidden[:, int(position), :] = edit_fn(hidden[:, int(position), :])
                return (hidden, *output[1:])
            hidden = output.clone()
            for position, edit_fn in sorted(self.edits.items()):
                hidden[:, int(position), :] = edit_fn(hidden[:, int(position), :])
            return hidden

        self._handle = block.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


@dataclass
class ResidualLayerRowMultiEdit(AbstractContextManager):
    model: object
    edits_by_layer: Mapping[int, Mapping[int, Mapping[int, torch.Tensor]]]

    def __post_init__(self) -> None:
        self._handles = []

    def __enter__(self):
        blocks = find_transformer_blocks(self.model)

        def make_hook(layer: int, row_edits: Mapping[int, Mapping[int, torch.Tensor]]):
            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    for row, edits in sorted(row_edits.items()):
                        for position, vector in sorted(edits.items()):
                            replacement = vector.to(device=hidden.device, dtype=hidden.dtype)
                            hidden[int(row), int(position), :] = replacement
                    return (hidden, *output[1:])
                hidden = output.clone()
                for row, edits in sorted(row_edits.items()):
                    for position, vector in sorted(edits.items()):
                        replacement = vector.to(device=hidden.device, dtype=hidden.dtype)
                        hidden[int(row), int(position), :] = replacement
                return hidden

            return hook

        for layer, row_edits in sorted(self.edits_by_layer.items()):
            self._handles.append(blocks[int(layer)].register_forward_hook(make_hook(int(layer), row_edits)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False


def replace_with(vector: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def edit(hidden_slice: torch.Tensor) -> torch.Tensor:
        replacement = vector.to(device=hidden_slice.device, dtype=hidden_slice.dtype)
        return replacement.expand_as(hidden_slice)

    return edit


def add_vector(vector: torch.Tensor, alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def edit(hidden_slice: torch.Tensor) -> torch.Tensor:
        direction = vector.to(device=hidden_slice.device, dtype=hidden_slice.dtype)
        return hidden_slice + float(alpha) * direction

    return edit


def remove_projection(vector: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def edit(hidden_slice: torch.Tensor) -> torch.Tensor:
        direction = vector.to(device=hidden_slice.device, dtype=hidden_slice.dtype)
        direction = direction / (torch.linalg.vector_norm(direction) + 1e-8)
        coeff = (hidden_slice * direction).sum(dim=-1, keepdim=True)
        return hidden_slice - coeff * direction

    return edit
