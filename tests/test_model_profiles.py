from __future__ import annotations

from interface_formatting_study import model_runner
from interface_formatting_study.model_profiles import MECHANISTIC_LAYERS, get_model_profile


def test_two_model_profiles_are_pinned_and_isolated(tmp_path):
    phi = get_model_profile("phi")
    mistral = get_model_profile("mistral")

    assert phi.model_id == "microsoft/Phi-3.5-mini-instruct"
    assert phi.revision == "2fe192450127e6a83f7441aef6e3ca586c338b77"
    assert phi.slug == "phi-3.5-mini-instruct"
    assert mistral.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert mistral.revision == "c170c708c41dac9275d15a8fff4eca08d52bab71"
    assert mistral.slug == "mistral-7b-instruct-v0.3"

    for profile in (phi, mistral):
        assert profile.expected_layers == 32
        assert profile.trust_remote_code is False
        assert profile.attention_backend == "sdpa"
        assert profile.diagnostic_attention_backend == "eager"
        assert profile.mechanistic_layers == MECHANISTIC_LAYERS == (2, 5, 7, 9, 11, 14, 16, 18)
        assert profile.output_base(tmp_path) == tmp_path / "results" / "model_runs" / profile.slug


def test_profile_loader_uses_builtin_transformers_implementation_for_sdpa(monkeypatch):
    captured = {}

    def fake_loader(model_id, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return object(), object(), "cuda"

    monkeypatch.setattr(model_runner, "load_model_and_tokenizer", fake_loader)

    model_runner._load_profile_model(get_model_profile("phi"), backend="sdpa", local_files_only=True)

    assert captured["model_id"] == "microsoft/Phi-3.5-mini-instruct"
    assert captured["trust_remote_code"] is False
    assert captured["attn_implementation"] == "sdpa"
