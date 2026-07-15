from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MECHANISTIC_LAYERS = (2, 5, 7, 9, 11, 14, 16, 18)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    revision: str
    slug: str
    expected_layers: int = 32
    trust_remote_code: bool = False
    attention_backend: str = "sdpa"
    diagnostic_attention_backend: str = "eager"
    mechanistic_layers: tuple[int, ...] = MECHANISTIC_LAYERS

    @property
    def tokenizer_id(self) -> str:
        return self.model_id

    @property
    def tokenizer_revision(self) -> str:
        return self.revision

    def output_base(self, project_root: str | Path) -> Path:
        return Path(project_root) / "results" / "model_runs" / self.slug


MODEL_PROFILES: dict[str, ModelProfile] = {
    "phi": ModelProfile(
        name="phi",
        model_id="microsoft/Phi-3.5-mini-instruct",
        revision="2fe192450127e6a83f7441aef6e3ca586c338b77",
        slug="phi-3.5-mini-instruct",
    ),
    "mistral": ModelProfile(
        name="mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        slug="mistral-7b-instruct-v0.3",
    ),
}


def get_model_profile(name: str) -> ModelProfile:
    normalized = str(name).strip().lower()
    try:
        return MODEL_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown model profile {name!r}; choose one of {sorted(MODEL_PROFILES)}") from exc
