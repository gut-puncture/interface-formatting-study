from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from interface_formatting_study.decision_binding_logit_lens import (
    PARITY_ATOL,
    audit_fixed_root_continuations,
    score_candidate_paths_cached,
    score_candidate_paths_cached_many,
    score_candidate_paths_scalar,
    tied_argmax_indices,
)


class ExactTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    all_special_ids = [0, 1]

    def __init__(self):
        self._pieces = [
            "Answer: ",
            "New York City",
            "New York",
            "Berlin",
            "A",
            "B",
            "C",
            "D",
            "New ",
            "York ",
            " City",
            "York",
            "é",
        ]
        self._id_by_piece = {piece: index + 2 for index, piece in enumerate(self._pieces)}
        self._piece_by_id = {value: key for key, value in self._id_by_piece.items()}

    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        # Candidate encodings deliberately expose a shared prefix.
        special = {
            "New York": [self._id_by_piece["New "], self._id_by_piece["York"]],
            "New York City": [
                self._id_by_piece["New "],
                self._id_by_piece["York"],
                self._id_by_piece[" City"],
            ],
        }
        if text in special:
            return special[text]
        output = []
        cursor = 0
        while cursor < len(text):
            matching = [piece for piece in self._pieces if text.startswith(piece, cursor)]
            if not matching:
                piece = text[cursor]
                if piece not in self._id_by_piece:
                    token_id = len(self._id_by_piece) + 2
                    self._id_by_piece[piece] = token_id
                    self._piece_by_id[token_id] = piece
                matching = [piece]
            piece = max(matching, key=len)
            output.append(self._id_by_piece[piece])
            cursor += len(piece)
        return output

    def decode(self, ids, **_kwargs):
        return "".join(self._piece_by_id[int(token_id)] for token_id in ids)


class AddBlock(torch.nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, hidden):
        return (hidden + self.amount,)


class TinyCachedMistral(torch.nn.Module):
    """Mistral-shaped causal model whose cache stores only sequence length."""

    def __init__(self, vocab_size=96, layers=2, hidden_size=5):
        super().__init__()
        torch.manual_seed(7)
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([AddBlock(index + 1) for index in range(layers)])
        self.model.norm = torch.nn.LayerNorm(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.forward_token_count = 0
        self.forward_batch_sizes = []
        self.logits_to_keep_values = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **_kwargs,
    ):
        del attention_mask
        self.logits_to_keep_values.append(_kwargs.get("logits_to_keep"))
        self.forward_token_count += int(input_ids.numel())
        self.forward_batch_sizes.append(int(input_ids.shape[0]))
        hidden = self.embed(input_ids)
        for block in self.model.layers:
            hidden = block(hidden)[0]
        logits = self.lm_head(self.model.norm(hidden))
        previous = 0
        if past_key_values is not None:
            previous = int(past_key_values[0][0].shape[-2])
        cache = None
        if use_cache:
            total = previous + input_ids.shape[1]
            batch = input_ids.shape[0]
            key = torch.zeros(batch, 1, total, 1)
            value = torch.zeros_like(key)
            cache = ((key, value),)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_continuation_audit_keeps_exact_surfaces_and_classifies_identity_hazards():
    tokenizer = ExactTokenizer()
    audit = audit_fixed_root_continuations(
        tokenizer,
        "Question\nAnswer: ",
        ["New York", "New York City", "A", "A"],
    )

    assert [entry.surface for entry in audit.candidates] == ["New York", "New York City", "A", "A"]
    assert all(tokenizer.decode(entry.token_ids) == entry.surface for entry in audit.candidates)
    assert audit.duplicate_surface_indices == ((2, 3),)
    assert audit.identical_token_indices == ((2, 3),)
    assert audit.prefix_collision_indices == ((0, 1),)
    assert audit.label_like_indices == (2, 3)
    assert not audit.identity_comparison_evaluable
    assert set(audit.identity_ineligibility_reasons) == {
        "duplicate_surface",
        "identical_token_sequence",
        "prefix_collision",
        "label_like_candidate",
    }


def test_logit_lens_batches_intermediate_layer_projections():
    tokenizer = ExactTokenizer()
    audit = audit_fixed_root_continuations(
        tokenizer,
        "Question\nAnswer: ",
        ["A", "B", "C", "D"],
    )
    model = TinyCachedMistral(layers=4)
    calls = 0

    def count_head_calls(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.lm_head.register_forward_hook(count_head_calls)
    try:
        score_candidate_paths_cached(model, audit, expected_layers=4)
    finally:
        handle.remove()

    # One native model projection, one batched intermediate-layer lens
    # projection, and one separately parity-anchored final-layer projection.
    assert calls == 3


def test_final_layer_projection_preserves_native_sequence_axis_for_bf16_parity():
    class RankSensitiveHead(torch.nn.Linear):
        def forward(self, hidden):
            logits = super().forward(hidden)
            if hidden.ndim == 2:
                logits = logits.clone()
                logits[:, 6] += 0.05
            return logits

    tokenizer = ExactTokenizer()
    audit = audit_fixed_root_continuations(
        tokenizer,
        "Question\nAnswer: ",
        ["A", "B", "C", "D"],
    )
    model = TinyCachedMistral(layers=4)
    replacement = RankSensitiveHead(
        model.lm_head.in_features,
        model.lm_head.out_features,
        bias=False,
    )
    replacement.weight.data.copy_(model.lm_head.weight.data)
    model.lm_head = replacement

    scored = score_candidate_paths_cached(model, audit, expected_layers=4)

    assert scored.max_final_native_difference < 1e-6


def test_continuation_audit_rejects_malformed_candidates_and_marks_empty_without_dropping_it():
    tokenizer = ExactTokenizer()
    with pytest.raises(ValueError, match="exactly four"):
        audit_fixed_root_continuations(tokenizer, "Answer: ", ["A"])

    audit = audit_fixed_root_continuations(tokenizer, "Answer: ", ["A", "B", "", "é"])
    assert len(audit.candidates) == 4
    assert audit.candidates[2].eligibility_reason == "empty_candidate"
    assert audit.candidates[3].utf8_bytes == len("é".encode("utf-8"))


def test_continuation_audit_marks_context_overflow_without_truncating_candidate():
    tokenizer = ExactTokenizer()
    audit = audit_fixed_root_continuations(
        tokenizer,
        "Answer: ",
        ["A", "Berlin", "New York", "New York City"],
        max_context_tokens=len(tokenizer.encode("Answer: ")) + 2,
    )

    assert audit.candidates[3].token_ids == tuple(tokenizer.encode("New York City"))
    assert audit.candidates[3].eligibility_reason == "context_overflow"
    assert not audit.candidates[3].eligible


def test_continuation_audit_rejects_label_tokens_that_change_at_prompt_boundary():
    class BoundaryMismatchTokenizer(ExactTokenizer):
        def decode(self, ids, **kwargs):
            decoded = super().decode(ids, **kwargs)
            if len(ids) > 1 and int(ids[-1]) == self._id_by_piece["A"]:
                return decoded[:-1] + " A"
            return decoded

    with pytest.raises(ValueError, match="prompt boundary"):
        audit_fixed_root_continuations(
            BoundaryMismatchTokenizer(), "Answer: ", ["A", "B", "C", "D"]
        )


def test_audit_disables_mistral_metaspace_prefix_for_fixed_root_continuations():
    class _Metaspace:
        def __repr__(self):
            return 'Metaspace(replacement="▁", prepend_scheme=first, split=False)'

    class _Backend:
        def __init__(self):
            self.pre_tokenizer = _Metaspace()

    class PrefixingTokenizer:
        all_special_ids = []

        def __init__(self):
            self.backend_tokenizer = _Backend()

        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            if text == "Answer: ":
                return [10, 11]
            no_prefix = "prepend_scheme=never" in repr(
                self.backend_tokenizer.pre_tokenizer
            )
            table = {letter: 20 + index for index, letter in enumerate("ABCD")}
            if text in table:
                return [table[text] if no_prefix else table[text] + 10]
            raise AssertionError(text)

        def decode(self, ids, **_kwargs):
            pieces = {
                10: "Answer:",
                11: " ",
                **{20 + i: value for i, value in enumerate("ABCD")},
                **{30 + i: " " + value for i, value in enumerate("ABCD")},
            }
            return "".join(pieces[int(value)] for value in ids)

    audit = audit_fixed_root_continuations(
        PrefixingTokenizer(), "Answer: ", ["A", "B", "C", "D"]
    )

    assert audit.label_token_ids == (20, 21, 22, 23)
    assert [candidate.token_ids for candidate in audit.candidates] == [
        (20,),
        (21,),
        (22,),
        (23,),
    ]


def test_audit_binds_exact_prompt_ids_when_tokenizer_drops_initial_space_on_decode():
    class LeadingSpaceDroppingTokenizer(ExactTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.startswith(" "):
                text = text[1:]
            return super().encode(text, add_special_tokens=add_special_tokens)

    audit = audit_fixed_root_continuations(
        LeadingSpaceDroppingTokenizer(), " Answer: ", ["A", "B", "C", "D"]
    )

    assert audit.prompt == " Answer: "
    assert audit.prompt_token_sha256


def test_scalar_oracle_scores_every_layer_and_sums_teacher_forced_path():
    tokenizer = ExactTokenizer()
    model = TinyCachedMistral()
    audit = audit_fixed_root_continuations(
        tokenizer,
        "Question\nAnswer: ",
        ["A", "Berlin", "New York", "New York City"],
    )

    scored = score_candidate_paths_scalar(model, audit, expected_layers=2)
    assert set(model.logits_to_keep_values) == {1}

    assert scored.first_token_logp.shape == (2, 4)
    assert scored.total_logp.shape == (2, 4)
    assert scored.mean_token_logp.shape == (2, 4)
    assert torch.allclose(scored.total_logp[:, 0], scored.first_token_logp[:, 0])
    assert torch.allclose(
        scored.mean_token_logp[:, 2],
        scored.total_logp[:, 2] / len(audit.candidates[2].token_ids),
    )
    path = audit.candidates[2].token_ids
    native_total = 0.0
    for offset, target in enumerate(path):
        prefix = (*audit.prompt_ids, *path[:offset])
        output = model(input_ids=torch.tensor([prefix]))
        native_total += float(
            torch.log_softmax(output.logits[0, -1].float(), dim=-1)[target].detach()
        )
    assert scored.total_logp[-1, 2] == pytest.approx(native_total)
    assert scored.max_final_native_difference <= PARITY_ATOL
    assert torch.isfinite(scored.total_logp).all()


def test_cached_trie_matches_scalar_and_reuses_shared_prefixes():
    tokenizer = ExactTokenizer()
    model = TinyCachedMistral()
    audit = audit_fixed_root_continuations(
        tokenizer,
        "Question\nAnswer: ",
        ["A", "Berlin", "New York", "New York City"],
    )
    scalar = score_candidate_paths_scalar(model, audit, expected_layers=2)
    scalar_tokens = model.forward_token_count

    model.forward_token_count = 0
    cached = score_candidate_paths_cached(model, audit, expected_layers=2)

    assert torch.allclose(cached.first_token_logp, scalar.first_token_logp, atol=PARITY_ATOL)
    assert torch.allclose(cached.total_logp, scalar.total_logp, atol=PARITY_ATOL)
    assert torch.allclose(cached.mean_token_logp, scalar.mean_token_logp, atol=PARITY_ATOL)
    assert cached.max_final_native_difference <= PARITY_ATOL
    assert model.forward_token_count < scalar_tokens


def test_cached_many_batches_equal_length_roots_and_matches_per_prompt_scores():
    tokenizer = ExactTokenizer()
    model = TinyCachedMistral()
    audits = [
        audit_fixed_root_continuations(
            tokenizer,
            prompt,
            ["A", "Berlin", "New York", "New York City"],
        )
        for prompt in ("Question\nAnswer: ", "Distinct\nAnswer: ")
    ]
    expected = [score_candidate_paths_cached(model, audit, expected_layers=2) for audit in audits]

    model.forward_batch_sizes.clear()
    observed = score_candidate_paths_cached_many(model, audits, expected_layers=2)

    assert model.forward_batch_sizes[0] == 2
    for actual, reference in zip(observed, expected, strict=True):
        assert torch.allclose(actual.letter_logp, reference.letter_logp, atol=PARITY_ATOL)
        assert torch.allclose(actual.total_logp, reference.total_logp, atol=PARITY_ATOL)


def test_cached_scorer_labels_root_and_branch_phases_for_telemetry():
    tokenizer = ExactTokenizer()
    model = TinyCachedMistral()
    audit = audit_fixed_root_continuations(
        tokenizer,
        "Question\nAnswer: ",
        ["A", "Berlin", "New York", "New York City"],
    )
    phases: list[str] = []

    score_candidate_paths_cached_many(
        model,
        [audit],
        expected_layers=2,
        phase_callback=phases.append,
    )

    assert phases[0] == "root"
    assert "branch" in phases
    assert phases[-1] == "idle"


def test_cached_many_requires_equal_root_token_lengths():
    tokenizer = ExactTokenizer()
    model = TinyCachedMistral()
    audits = [
        audit_fixed_root_continuations(tokenizer, "Answer: ", ["A", "B", "C", "D"]),
        audit_fixed_root_continuations(tokenizer, "xAnswer: ", ["A", "B", "C", "D"]),
    ]
    assert len(audits[0].prompt_ids) != len(audits[1].prompt_ids)
    with pytest.raises(ValueError, match="equal token length"):
        score_candidate_paths_cached_many(model, audits, expected_layers=2)


def test_tied_argmax_returns_all_exact_winners_without_order_breaking():
    assert tied_argmax_indices([1.0, 2.0, 2.0, -3.0]) == (1, 2)
    assert tied_argmax_indices([1.0, 2.0, 2.0 - 1e-8, -3.0]) == (1,)


def test_scorer_rejects_layer_drift_and_nonfinite_projections():
    tokenizer = ExactTokenizer()
    model = TinyCachedMistral()
    audit = audit_fixed_root_continuations(tokenizer, "Answer: ", ["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="transformer blocks"):
        score_candidate_paths_cached(model, audit, expected_layers=32)

    with torch.no_grad():
        model.lm_head.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        score_candidate_paths_cached(model, audit, expected_layers=2)


def test_cached_trie_matches_scalar_on_real_mistral_cache_api():
    from transformers import MistralConfig, MistralForCausalLM

    model = MistralForCausalLM(
        MistralConfig(
            vocab_size=96,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=1,
        )
    )
    tokenizer = ExactTokenizer()
    audits = [
        audit_fixed_root_continuations(
            tokenizer,
            prompt,
            ["A", "Berlin", "New York", "New York City"],
        )
        for prompt in ("Question\nAnswer: ", "Distinct\nAnswer: ")
    ]

    scalar = [
        score_candidate_paths_scalar(model, audit, expected_layers=2)
        for audit in audits
    ]
    cached = score_candidate_paths_cached_many(model, audits, expected_layers=2)

    for actual, expected in zip(cached, scalar, strict=True):
        assert torch.allclose(actual.total_logp, expected.total_logp, atol=PARITY_ATOL)
        assert torch.allclose(actual.letter_logp, expected.letter_logp, atol=PARITY_ATOL)


@pytest.mark.parametrize("architecture", ["phi", "qwen"])
def test_cached_trie_matches_scalar_and_final_native_on_cross_model_cache_apis(
    architecture,
):
    if architecture == "phi":
        from transformers import Phi3Config, Phi3ForCausalLM

        config_class, model_class = Phi3Config, Phi3ForCausalLM
    else:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config_class, model_class = Qwen2Config, Qwen2ForCausalLM
    model = model_class(
        config_class(
            vocab_size=96,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=1,
        )
    )
    tokenizer = ExactTokenizer()
    audits = [
        audit_fixed_root_continuations(
            tokenizer, prompt, ["A", "Berlin", "New York", "New York City"]
        )
        for prompt in ("Question\nAnswer: ", "Distinct\nAnswer: ")
    ]

    scalar = [
        score_candidate_paths_scalar(model, audit, expected_layers=2)
        for audit in audits
    ]
    cached = score_candidate_paths_cached_many(model, audits, expected_layers=2)

    for actual, expected in zip(cached, scalar, strict=True):
        assert actual.max_final_native_difference <= PARITY_ATOL
        assert torch.allclose(actual.total_logp, expected.total_logp, atol=PARITY_ATOL)
        assert torch.allclose(actual.mean_token_logp, expected.mean_token_logp, atol=PARITY_ATOL)
        assert torch.allclose(actual.letter_logp, expected.letter_logp, atol=PARITY_ATOL)
