from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from interface_formatting_study.causal_option_maps import (
    OptionRepresentation,
    OptionSlot,
    ParsedPrompt,
    SourceSpan,
    parse_prompt_options,
    transform_with_option_map,
)
from interface_formatting_study.decision_binding_content import (
    CandidateRanker,
    capture_candidate_states,
    evaluate_candidate_ranker,
    fit_candidate_ranker,
    locate_content_token_indices,
    metadata_candidate_features,
    prepare_candidate_sites,
)


_SUFFIX = "\n\nReturn only the letter (A, B, C, or D).\nAnswer: "


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return list(range(len(text)))

    def __call__(self, text: str, *, add_special_tokens=False, return_offsets_mapping=False):
        output = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return list(range(len(text.split())))

    def __call__(self, text: str, *, add_special_tokens=False, return_offsets_mapping=False):
        offsets = []
        cursor = 0
        for word in text.split():
            start = text.index(word, cursor)
            offsets.append((start, start + len(word)))
            cursor = start + len(word)
        output = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            output["offset_mapping"] = offsets
        return output


def _row(
    prompt: str,
    *,
    variant: int,
    manipulation: str,
    position_shift: int,
    label_shift: int,
    contents: list[int],
    labels: list[str],
    candidates: list[str],
) -> dict[str, object]:
    return {
        "work_key": f"letter|item-1|plain|{variant}",
        "item_id": "item-1",
        "wrapper_name": "plain",
        "split": "validation",
        "variant": variant,
        "manipulation": manipulation,
        "position_shift": position_shift,
        "label_shift": label_shift,
        "source_prompt_sha256": "",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "content_ids_by_position": contents,
        "labels_by_position": labels,
        "candidate_texts": candidates,
        "winner_position": 1,
        "winner_unique": True,
        "text_identity_ambiguous": False,
        "readout_role": "reader_gate",
    }


def test_prepare_candidate_sites_replays_exact_existing_transforms():
    candidates = ["Alpha answer", "Beta answer", "Gamma answer", "Delta answer"]
    source = "Question\nA) Alpha answer\nB) Beta answer\nC) Gamma answer\nD) Delta answer" + _SUFFIX
    parsed = parse_prompt_options(source, "plain", candidates)
    position = transform_with_option_map(parsed, source, position_shift=1, label_shift=0)
    label = transform_with_option_map(parsed, source, position_shift=0, label_shift=1)
    rows = pd.DataFrame(
        [
            _row(
                source,
                variant=0,
                manipulation="controlled_baseline",
                position_shift=0,
                label_shift=0,
                contents=[0, 1, 2, 3],
                labels=list("ABCD"),
                candidates=candidates,
            ),
            _row(
                position,
                variant=1,
                manipulation="position_only",
                position_shift=1,
                label_shift=0,
                contents=[3, 0, 1, 2],
                labels=list("DABC"),
                candidates=candidates,
            ),
            _row(
                label,
                variant=4,
                manipulation="label_only",
                position_shift=0,
                label_shift=1,
                contents=[0, 1, 2, 3],
                labels=list("BCDA"),
                candidates=candidates,
            ),
        ]
    )
    rows["source_prompt_sha256"] = rows.iloc[0].prompt_sha256
    applicability = pd.DataFrame(
        [{
            "item_id": "item-1",
            "wrapper_name": "plain",
            "split": "validation",
            "source_prompt_sha256": rows.iloc[0].prompt_sha256,
            "parse_provenance": "deterministic",
        }]
    )

    sites = prepare_candidate_sites(rows, applicability, option_maps={})

    assert len(sites) == 3
    assert sites["actual_content_ids_by_position"].tolist() == [
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [0, 1, 2, 3],
    ]
    for record in sites.to_dict("records"):
        for content_id, (start, end) in zip(
            record["actual_content_ids_by_position"], record["content_char_spans"], strict=True
        ):
            assert record["prompt"][start:end] == candidates[content_id]
        assert record["mapping_matches_ledger"] is True


def test_prepare_candidate_sites_uses_content_representation_not_symbolic_alias():
    source = (
        'OPTION_A="alpha"\nOPTION_B="beta"\nOPTION_C="gamma"\nOPTION_D="delta"\n'
        'printf "%s" "$OPTION_A $OPTION_B $OPTION_C $OPTION_D"' + _SUFFIX
    )
    labels = [SourceSpan(source.index(f"OPTION_{letter}") + 7, source.index(f"OPTION_{letter}") + 8) for letter in "ABCD"]
    payloads = [
        SourceSpan(source.index(value), source.index(value) + len(value))
        for value in ("alpha", "beta", "gamma", "delta")
    ]
    aliases = [
        SourceSpan(source.rindex(f"$OPTION_{letter}"), source.rindex(f"$OPTION_{letter}") + 8)
        for letter in "ABCD"
    ]
    alias_labels = [SourceSpan(span.end, span.end + 1) for span in aliases]
    parsed = ParsedPrompt(
        wrapper_name="shell_heredoc",
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        representations=(
            OptionRepresentation(tuple(OptionSlot(i, (labels[i],), (payloads[i],)) for i in range(4))),
            OptionRepresentation(
                tuple(OptionSlot(i, (alias_labels[i],), (aliases[i],)) for i in range(4))
            ),
        ),
        separable=True,
        not_applicable_reason="",
        provenance="audited",
    )
    row = _row(
        source,
        variant=0,
        manipulation="controlled_baseline",
        position_shift=0,
        label_shift=0,
        contents=[0, 1, 2, 3],
        labels=list("ABCD"),
        candidates=["alpha", "beta", "gamma", "delta"],
    )
    row.update(wrapper_name="shell_heredoc", source_prompt_sha256=parsed.source_sha256)
    applicability = pd.DataFrame([{
        "item_id": "item-1",
        "wrapper_name": "shell_heredoc",
        "split": "validation",
        "source_prompt_sha256": parsed.source_sha256,
        "parse_provenance": "gpt-5.6-luna:xhigh:thread",
    }])

    sites = prepare_candidate_sites(
        pd.DataFrame([row]), applicability, option_maps={("item-1", "shell_heredoc"): parsed}
    )

    assert sites.iloc[0].content_char_spans == [(span.start, span.end) for span in payloads]


def test_prepare_candidate_sites_corrects_literal_mapping_without_changing_prompt():
    candidates = ["Aorta", "Esophagus", "Trachea", "Pancreas"]
    prompt = (
        "Which allows air into the lungs? A=Trachea B=Esophagus C=Aorta D=Pancreas"
        + _SUFFIX
    )
    row = _row(
        prompt,
        variant=0,
        manipulation="controlled_baseline",
        position_shift=0,
        label_shift=0,
        contents=[0, 1, 2, 3],
        labels=list("ABCD"),
        candidates=candidates,
    )
    row.update(wrapper_name="key_equals", source_prompt_sha256=row["prompt_sha256"])
    applicability = pd.DataFrame([{
        "item_id": "item-1",
        "wrapper_name": "key_equals",
        "split": "validation",
        "source_prompt_sha256": row["prompt_sha256"],
        "parse_provenance": "legacy_deterministic",
    }])

    sites = prepare_candidate_sites(pd.DataFrame([row]), applicability, option_maps={})

    assert sites.iloc[0].prompt == prompt
    assert sites.iloc[0].actual_content_ids_by_position == [2, 1, 0, 3]
    assert sites.iloc[0].mapping_matches_ledger == False


def test_prepare_candidate_sites_does_not_confuse_substring_answer_with_other_options():
    candidates = ["2c", "c", "0.8c", "0.5c"]
    source = (
        'message { options: ["A) 2c", "B) c", "C) 0.8c", "D) 0.5c"] }'
        + _SUFFIX
    )
    prompt = (
        'message { options: ["C) 2c", "D) c", "A) 0.8c", "B) 0.5c"] }'
        + _SUFFIX
    )
    row = _row(
        prompt,
        variant=5,
        manipulation="label_only",
        position_shift=0,
        label_shift=2,
        contents=[0, 1, 2, 3],
        labels=list("CDAB"),
        candidates=candidates,
    )
    row.update(
        wrapper_name="protobuf_msg",
        source_prompt_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    baseline = _row(
        source,
        variant=0,
        manipulation="controlled_baseline",
        position_shift=0,
        label_shift=0,
        contents=[0, 1, 2, 3],
        labels=list("ABCD"),
        candidates=candidates,
    )
    baseline.update(wrapper_name="protobuf_msg", source_prompt_sha256=row["source_prompt_sha256"])
    applicability = pd.DataFrame([{
        "item_id": "item-1",
        "wrapper_name": "protobuf_msg",
        "split": "validation",
        "source_prompt_sha256": row["source_prompt_sha256"],
        "parse_provenance": "legacy_deterministic",
    }])

    sites = prepare_candidate_sites(pd.DataFrame([baseline, row]), applicability, option_maps={})

    observed = sites[sites["variant"] == 5].iloc[0]
    assert observed.actual_content_ids_by_position == [0, 1, 2, 3]
    assert [prompt[start:end] for start, end in observed.content_char_spans] == candidates


def test_locate_content_token_indices_uses_last_token_overlapping_payload():
    prompt = "A) alpha beta. B) gamma delta" + _SUFFIX
    spans = [(3, 14), (18, 29)]

    indices = locate_content_token_indices(WordTokenizer(), prompt, spans)

    assert len(indices) == 2
    offsets = WordTokenizer()(prompt, return_offsets_mapping=True)["offset_mapping"]
    assert offsets[indices[0]][1] == spans[0][1]
    assert offsets[indices[1]][1] == spans[1][1]


def test_locate_content_token_indices_rejects_token_crossing_payload_boundary():
    class CrossingTokenizer(CharacterTokenizer):
        def __call__(self, text: str, *, add_special_tokens=False, return_offsets_mapping=False):
            result = super().__call__(
                text, add_special_tokens=add_special_tokens, return_offsets_mapping=return_offsets_mapping
            )
            if return_offsets_mapping:
                result["offset_mapping"][6] = (6, 8)
            return result

    with pytest.raises(ValueError, match="crosses candidate payload boundary"):
        locate_content_token_indices(CrossingTokenizer(), "xxalpha, yy", [(2, 7)])


def test_prepare_candidate_sites_fails_closed_on_prompt_hash_drift():
    candidates = ["a", "b", "c", "d"]
    prompt = "A) a\nB) b\nC) c\nD) d" + _SUFFIX
    row = _row(
        prompt,
        variant=0,
        manipulation="controlled_baseline",
        position_shift=0,
        label_shift=0,
        contents=[0, 1, 2, 3],
        labels=list("ABCD"),
        candidates=candidates,
    )
    row["prompt_sha256"] = "0" * 64
    applicability = pd.DataFrame([{
        "item_id": "item-1",
        "wrapper_name": "plain",
        "split": "validation",
        "source_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "parse_provenance": "deterministic",
    }])

    with pytest.raises(ValueError, match="prompt checksum mismatch"):
        prepare_candidate_sites(pd.DataFrame([row]), applicability, option_maps={})


def test_shared_candidate_ranker_recovers_known_direction_without_position_identity():
    generator = np.random.default_rng(7)
    items, layers, hidden = 96, 3, 8
    targets = generator.integers(0, 4, size=items)
    values = generator.normal(size=(items, layers, 4, hidden)).astype(np.float32)
    direction = generator.normal(size=hidden).astype(np.float32)
    direction /= np.linalg.norm(direction)
    for item, target in enumerate(targets):
        values[item, 1, target] += 4.0 * direction

    ranker = fit_candidate_ranker(values, targets, l2=1e-2, max_iter=80)
    probabilities = evaluate_candidate_ranker(ranker, values)

    assert isinstance(ranker, CandidateRanker)
    assert probabilities.shape == (items, layers, 4)
    assert (probabilities[:, 1].argmax(axis=1) == targets).mean() > 0.95
    assert ranker.weights.shape == (layers, hidden)
    assert ranker.rms.shape == (layers,)
    assert np.isfinite(ranker.weights).all()


def test_shared_candidate_ranker_is_invariant_to_per_item_common_activation_shift():
    generator = np.random.default_rng(11)
    values = generator.normal(size=(40, 2, 4, 6)).astype(np.float32)
    targets = generator.integers(0, 4, size=40)
    ranker = fit_candidate_ranker(values, targets, l2=1e-2, max_iter=50)
    common = generator.normal(size=(40, 2, 1, 6)).astype(np.float32)

    before = evaluate_candidate_ranker(ranker, values)
    after = evaluate_candidate_ranker(ranker, values + common)

    np.testing.assert_allclose(before, after, atol=1e-5)


def test_metadata_controls_encode_only_the_declared_nuisance():
    frame = pd.DataFrame(
        {
            "labels_by_position": [list("ABCD"), list("BCDA")],
            "content_ids_by_position": [[0, 1, 2, 3], [3, 0, 1, 2]],
            "candidate_texts": [
                ["a", "medium", "long answer", "tiny"],
                ["a", "medium", "long answer", "tiny"],
            ],
        }
    )

    features = metadata_candidate_features(frame)

    assert set(features) == {"position", "label", "position_label", "answer_length"}
    assert features["position"].shape == (2, 1, 4, 4)
    assert features["label"].shape == (2, 1, 4, 4)
    assert features["position_label"].shape == (2, 1, 4, 8)
    assert features["answer_length"].shape == (2, 1, 4, 1)
    np.testing.assert_array_equal(features["position"][0, 0], np.eye(4))
    np.testing.assert_array_equal(features["label"][1, 0].argmax(axis=1), [1, 2, 3, 0])
    np.testing.assert_array_equal(
        features["answer_length"][1, 0, :, 0],
        [len("tiny"), len("a"), len("medium"), len("long answer")],
    )


def test_batched_candidate_capture_matches_scalar_layer_states(tiny_hook_model):
    class SmallCharacterTokenizer(CharacterTokenizer):
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False):
            return [(ord(character) % 15) + 1 for character in text]

    tokenizer = SmallCharacterTokenizer()
    prompts = ["A)a B)b C)c D)d" + _SUFFIX, "A)e B)f C)g D)h" + _SUFFIX]
    positions = [[2, 6, 10, 14], [2, 6, 10, 14]]

    captured = capture_candidate_states(
        tiny_hook_model,
        tokenizer,
        prompts,
        positions,
        batch_size=2,
        max_batch_tokens=512,
    )

    assert captured.activations.shape == (2, 2, 4, 3)
    assert captured.raw_log_probs.shape == (2, 4)
    assert captured.actual_tokens > 0
    for row, prompt in enumerate(prompts):
        direct = tiny_hook_model(
            input_ids=torch.tensor([tokenizer.encode(prompt)]), output_hidden_states=True
        )
        for layer in range(2):
            expected = direct.hidden_states[layer + 1][0, positions[row]].float()
            assert torch.allclose(captured.activations[row, layer], expected)
