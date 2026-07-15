from __future__ import annotations

import torch


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(device: torch.device):
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_name: str,
    *,
    revision: str | None = None,
    tokenizer_name: str | None = None,
    tokenizer_revision: str | None = None,
    local_files_only: bool = False,
    attn_implementation: str | None = None,
    expected_layers: int | None = None,
    trust_remote_code: bool = True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .hooks import find_transformer_blocks

    device = choose_device()
    dtype = choose_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name,
        revision=tokenizer_revision or revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    model.to(device)
    if expected_layers is not None:
        observed_layers = len(find_transformer_blocks(model))
        if observed_layers != int(expected_layers):
            raise ValueError(
                f"Model {model_name}@{revision or 'default'} has {observed_layers} transformer blocks; "
                f"expected {expected_layers}"
            )
    return model, tokenizer, device
