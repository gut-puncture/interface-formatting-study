from __future__ import annotations

import torch

from interface_formatting_study.hooks import ResidualCapture, ResidualEdit, ResidualMultiEdit, add_vector, find_transformer_blocks, remove_projection, replace_with


def test_find_transformer_blocks(tiny_hook_model):
    blocks = find_transformer_blocks(tiny_hook_model)
    assert len(blocks) == 2


def test_phi_and_mistral_transformer_block_discovery():
    from transformers import MistralConfig, MistralForCausalLM, Phi3Config, Phi3ForCausalLM

    common = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    phi = Phi3ForCausalLM(Phi3Config(**common, original_max_position_embeddings=128))
    mistral = MistralForCausalLM(MistralConfig(**common))

    assert len(find_transformer_blocks(phi)) == 2
    assert len(find_transformer_blocks(mistral)) == 2


def test_hook_replaces_only_selected_position_and_is_removed(tiny_hook_model):
    input_ids = torch.tensor([[1, 2, 3]])
    baseline = tiny_hook_model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    replacement = torch.tensor([10.0, 20.0, 30.0])
    with ResidualEdit(tiny_hook_model, layer=0, position=1, edit_fn=replace_with(replacement)):
        patched = tiny_hook_model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    after = tiny_hook_model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]

    assert torch.allclose(after, baseline)
    assert torch.allclose(patched[:, 0, :], baseline[:, 0, :])
    assert not torch.allclose(patched[:, 1, :], baseline[:, 1, :])
    assert torch.allclose(patched[:, 2, :], baseline[:, 2, :])


def test_multi_edit_replaces_multiple_positions(tiny_hook_model):
    input_ids = torch.tensor([[1, 2, 3, 4]])
    baseline = tiny_hook_model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    edits = {
        1: replace_with(torch.tensor([10.0, 20.0, 30.0])),
        3: replace_with(torch.tensor([40.0, 50.0, 60.0])),
    }

    with ResidualMultiEdit(tiny_hook_model, layer=0, edits=edits):
        patched = tiny_hook_model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    after = tiny_hook_model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]

    assert torch.allclose(after, baseline)
    assert torch.allclose(patched[:, 0, :], baseline[:, 0, :])
    assert not torch.allclose(patched[:, 1, :], baseline[:, 1, :])
    assert torch.allclose(patched[:, 2, :], baseline[:, 2, :])
    assert not torch.allclose(patched[:, 3, :], baseline[:, 3, :])


def test_capture_uses_same_block_output_space_as_edit_hook(tiny_hook_model):
    input_ids = torch.tensor([[1, 2, 3]])
    with ResidualCapture(tiny_hook_model, layer=0, position=1) as capture:
        tiny_hook_model(input_ids=input_ids)
    replacement = capture.value[0]
    with ResidualEdit(tiny_hook_model, layer=0, position=1, edit_fn=replace_with(replacement)):
        patched = tiny_hook_model(input_ids=input_ids, output_hidden_states=True).hidden_states[1]
    assert torch.allclose(patched[0, 1], replacement)


def test_add_vector_preserves_shape_dtype_device():
    hidden = torch.ones(2, 3, dtype=torch.float32)
    edit = add_vector(torch.tensor([1.0, 2.0, 3.0]), alpha=0.5)
    out = edit(hidden)
    assert out.shape == hidden.shape
    assert out.dtype == hidden.dtype
    assert out.device == hidden.device
    assert torch.allclose(out[0], torch.tensor([1.5, 2.0, 2.5]))


def test_remove_projection_hook_edit():
    hidden = torch.tensor([[3.0, 4.0]])
    edit = remove_projection(torch.tensor([1.0, 0.0]))
    out = edit(hidden)
    assert torch.allclose(out, torch.tensor([[0.0, 4.0]]), atol=1e-6)
