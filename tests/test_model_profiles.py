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


def test_qwen_profile_is_pinned_for_the_three_model_followup(tmp_path):
    qwen = get_model_profile("qwen")

    assert qwen.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert qwen.revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert qwen.slug == "qwen2.5-1.5b-instruct"
    assert qwen.expected_layers == 28
    assert qwen.mechanistic_layers == (2, 4, 6, 8, 10, 12, 14, 16)
    assert qwen.output_base(tmp_path) == tmp_path / "results" / "model_runs" / qwen.slug


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
