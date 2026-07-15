from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch


class BoundaryTokenizer:
    """Tiny tokenizer where prompt+completion tokenization can differ at the boundary."""

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.vocab: dict[str, int] = {"<pad>": 0}

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False):
        if not text:
            return []
        tokens: list[str] = []
        i = 0
        while i < len(text):
            if text.startswith("XA", i):
                tokens.append("XA")
                i += 2
            elif text[i] == " " and i + 1 < len(text):
                tokens.append(text[i : i + 2])
                i += 2
            elif text[i] == "\n":
                tokens.append("\\n")
                i += 1
            else:
                tokens.append(text[i])
                i += 1
        return [self._id(token) for token in tokens]

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


class RuleLogitModel(torch.nn.Module):
    """Predicts token logits from the previous input token with deterministic rules."""

    def __init__(self, tokenizer: BoundaryTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids=None, attention_mask=None, **_kwargs):
        vocab_size = max(256, len(self.tokenizer.vocab) + 64)
        logits = torch.full((*input_ids.shape, vocab_size), -20.0, device=input_ids.device)
        # For each next-token prediction, heavily favor the actual next token.
        for row in range(input_ids.shape[0]):
            for pos in range(input_ids.shape[1] - 1):
                next_id = int(input_ids[row, pos + 1])
                logits[row, pos, next_id] = 5.0
        return SimpleNamespace(logits=logits)


class CausalLabelModel(torch.nn.Module):
    """Context-independent causal logits suitable for fast/reference parity."""

    def __init__(self, tokenizer: BoundaryTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids=None, attention_mask=None, **_kwargs):
        vocab_size = max(256, len(self.tokenizer.vocab) + 64)
        logits = torch.full((*input_ids.shape, vocab_size), -10.0, device=input_ids.device)
        for rank, label in enumerate(("A", "B", "C", "D")):
            for token_id in self.tokenizer.encode(label, add_special_tokens=False):
                logits[..., token_id] = float(4 - rank)
        return SimpleNamespace(logits=logits)


class TinyBlock(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 1.0


class TinyHookModel(torch.nn.Module):
    def __init__(self, n_layers: int = 2, d_model: int = 3, vocab_size: int = 32):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([TinyBlock() for _ in range(n_layers)]))
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **_kwargs):
        hidden = self.embed(input_ids)
        states = [hidden]
        for block in self.model.layers:
            hidden = block(hidden)
            states.append(hidden)
        logits = self.lm_head(hidden)
        if output_hidden_states:
            return SimpleNamespace(logits=logits, hidden_states=tuple(states))
        return SimpleNamespace(logits=logits)


@pytest.fixture
def boundary_tokenizer():
    return BoundaryTokenizer()


@pytest.fixture
def rule_model(boundary_tokenizer):
    return RuleLogitModel(boundary_tokenizer)


@pytest.fixture
def causal_label_model(boundary_tokenizer):
    return CausalLabelModel(boundary_tokenizer)


@pytest.fixture
def tiny_hook_model():
    return TinyHookModel()
