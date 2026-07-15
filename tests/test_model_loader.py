from __future__ import annotations

from types import SimpleNamespace

import torch

from interface_formatting_study import model_loader


def test_loader_uses_pinned_transformers_torch_dtype_keyword(monkeypatch):
    observed = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            observed["device"] = device
            return self

    import transformers

    monkeypatch.setattr(model_loader, "choose_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())

    def fake_model(*args, **kwargs):
        observed.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model)

    model_loader.load_model_and_tokenizer("fake/model", revision="pinned")

    assert observed["torch_dtype"] == torch.float32
    assert "dtype" not in observed
    assert observed["revision"] == "pinned"
