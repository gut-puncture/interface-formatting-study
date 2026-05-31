from __future__ import annotations

import pytest
import torch

from interface_formatting_study.access_vector import (
    build_access_vector,
    label_balanced_average,
    negative_intervention,
    normalized_deltas,
    projection_removal,
    random_vector_like,
    shuffled_pair_vector,
)


def test_deltas_are_clean_minus_corrupt_and_normalized():
    clean = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    corrupt = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    deltas = normalized_deltas(clean, corrupt)
    assert torch.allclose(deltas, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), atol=1e-6)


def test_label_balanced_average_equalizes_imbalanced_labels():
    deltas = torch.vstack(
        [
            torch.tensor([[1.0, 0.0]]).repeat(90, 1),
            torch.tensor([[0.0, 1.0]]).repeat(10, 1),
            torch.tensor([[0.0, 1.0]]).repeat(10, 1),
            torch.tensor([[0.0, 1.0]]).repeat(10, 1),
        ]
    )
    labels = ["A"] * 90 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
    vector = label_balanced_average(deltas, labels)
    expected = torch.tensor([0.25, 0.75])
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(vector, expected, atol=1e-6)


def test_access_vector_norm_is_nonzero():
    clean = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    corrupt = torch.zeros_like(clean)
    labels = ["A", "B", "C", "D"]
    vector = build_access_vector(clean, corrupt, labels)
    assert torch.linalg.vector_norm(vector) > 0


def test_access_vector_cancellation_raises_instead_of_silent_zero():
    clean = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    corrupt = torch.zeros_like(clean)
    labels = ["A", "B", "C", "D"]
    with pytest.raises(ValueError, match="zero"):
        build_access_vector(clean, corrupt, labels)


def test_negative_intervention_uses_opposite_direction():
    hidden = torch.tensor([1.0, 1.0])
    vector = torch.tensor([0.5, 0.0])
    out = negative_intervention(hidden, vector, alpha=2.0)
    assert torch.allclose(out, torch.tensor([0.0, 1.0]))


def test_projection_removal_zeroes_vector_component_and_keeps_orthogonal_component():
    hidden = torch.tensor([3.0, 4.0])
    vector = torch.tensor([1.0, 0.0])
    out = projection_removal(hidden, vector)
    assert torch.allclose(out, torch.tensor([0.0, 4.0]), atol=1e-6)
    assert abs(float(torch.dot(out, vector))) < 1e-6


def test_random_vector_has_same_norm():
    vector = torch.tensor([3.0, 4.0])
    rand = random_vector_like(vector, seed=1)
    assert torch.linalg.vector_norm(rand) == torch.linalg.vector_norm(vector)


def test_shuffled_pair_vector_uses_different_pairs():
    clean = torch.eye(4)
    corrupt = torch.zeros_like(clean)
    labels = ["A", "B", "C", "D"]
    vector = shuffled_pair_vector(clean, corrupt, labels, seed=0)
    assert vector.shape == (4,)
