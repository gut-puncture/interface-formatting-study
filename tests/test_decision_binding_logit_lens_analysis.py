from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "analysis" / "analyze_decision_binding_logit_lens.py"
SPEC = importlib.util.spec_from_file_location("analyze_decision_binding_logit_lens", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _scores(winner: int, *, tie_with: int | None = None, margin: float = 2.0) -> list[float]:
    values = [-4.0, -4.0, -4.0, -4.0]
    values[winner] = margin
    if tie_with is not None:
        values[tie_with] = margin
    return values


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item_id in ("i1", "i2"):
        for contract in ("letter", "text"):
            for wrapper in ("plain", "json", "xml"):
                for layer in range(32):
                    # Letter trajectories diverge across wrappers. Candidate trajectories
                    # remain content-stable, so the candidate-minus-letter contrast is positive.
                    letter_winner = 0 if wrapper == "plain" else 1
                    candidate_winner = 0
                    # Displayed order is deliberately rotated for xml. The analyzer must
                    # map scores through content_ids_by_position before comparing winners.
                    content_ids = [0, 1, 2, 3] if wrapper != "xml" else [1, 0, 2, 3]

                    def displayed(content_scores: list[float]) -> list[float]:
                        return [content_scores[content_id] for content_id in content_ids]

                    rows.append(
                        {
                            "block_work_key": f"{item_id}|{wrapper}",
                            "item_id": item_id,
                            "split": "train" if item_id == "i1" else "validation",
                            "wrapper_name": wrapper,
                            "contract": contract,
                            "layer": layer,
                            "content_ids_by_position": content_ids,
                            "letter_raw_logps": displayed(_scores(letter_winner)),
                            "candidate_path_total_logps": displayed(_scores(candidate_winner)),
                            "candidate_mean_token_logps": displayed(_scores(candidate_winner)),
                            "candidate_first_token_logps": displayed(_scores(candidate_winner)),
                            "primary_contrast_evaluable": True,
                            "first_token_contrast_evaluable": True,
                            "letter_calibration_evaluable": True,
                            "confidence_stratum": "high" if item_id == "i1" else "low",
                            "position_susceptible": item_id == "i1",
                        }
                    )
    return pd.DataFrame(rows)


def test_argmax_sets_and_jaccard_preserve_exact_ties():
    assert MODULE.argmax_set([2.0, 2.0, 1.0, 0.0]) == frozenset({0, 1})
    assert MODULE.jaccard_agreement(frozenset({0, 1}), frozenset({1, 2})) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="finite four-score"):
        MODULE.argmax_set([1.0, np.nan, 0.0, 0.0])


def test_content_mapping_and_per_item_wrapper_aggregation_drive_primary_auc():
    result = MODULE.analyze_frame(_frame(), n_boot=40, seed=17)

    primary = result["primary_contrasts"].set_index("contract")
    assert set(primary.index) == {"letter", "text"}
    assert primary.loc["letter", "letter_auc"] == 0.0
    assert primary.loc["letter", "candidate_auc"] == 1.0
    assert primary.loc["letter", "contrast"] == 1.0
    assert primary.loc["letter", "n_items"] == 2
    assert primary.loc["letter", "n_pairs"] == 4

    trajectory = result["trajectories"]
    letter_layer = trajectory[
        (trajectory["contract"] == "letter")
        & (trajectory["score_variant"] == "candidate_total")
        & (trajectory["layer"] == 10)
    ].iloc[0]
    assert letter_layer["agreement"] == 1.0
    assert letter_layer["n_items"] == 2
    assert letter_layer["n_pairs"] == 4


def test_primary_population_is_identical_for_letter_and_candidate_scores():
    frame = _frame()
    frame.loc[frame["wrapper_name"] == "xml", "primary_contrast_evaluable"] = False
    frame.loc[frame["wrapper_name"] == "xml", "first_token_contrast_evaluable"] = False
    result = MODULE.analyze_frame(frame, n_boot=20, seed=3)

    primary = result["primary_contrasts"]
    assert set(primary["n_pairs"]) == {2}
    assert set(primary["n_items"]) == {2}

    separation = result["timing"][
        result["timing"]["timing_type"] == "plain_wrapped_separation_onset"
    ]
    assert set(separation["wrapper_name"]) == {"json"}


def test_first_token_secondary_uses_its_four_distinct_token_population_for_both_readouts():
    frame = _frame()
    frame.loc[frame["wrapper_name"] == "xml", "first_token_contrast_evaluable"] = False

    result = MODULE.analyze_frame(frame, n_boot=20, seed=7)

    first = result["secondary_contrasts"]
    first = first[first["score_variant"] == "candidate_first_token"]
    assert set(first["n_pairs"]) == {2}


def test_bootstrap_is_deterministic_and_holm_is_applied_only_to_two_primaries():
    first = MODULE.analyze_frame(_frame(), n_boot=50, seed=91)
    second = MODULE.analyze_frame(_frame(), n_boot=50, seed=91)

    pd.testing.assert_frame_equal(first["primary_contrasts"], second["primary_contrasts"])
    primary = first["primary_contrasts"]
    assert primary["holm_p_value"].notna().all()
    assert (primary["holm_p_value"] >= primary["p_value"]).all()
    assert "holm_p_value" not in first["secondary_contrasts"].columns
    pd.testing.assert_frame_equal(first["trajectories"], second["trajectories"])
    assert {
        "simultaneous_ci_low",
        "simultaneous_ci_high",
    } <= set(first["trajectories"].columns)
    assert (
        (first["trajectories"]["simultaneous_ci_low"] <= first["trajectories"]["agreement"])
        & (first["trajectories"]["agreement"] <= first["trajectories"]["simultaneous_ci_high"])
    ).all()


def test_stability_and_separation_timing_require_unique_persistent_winners():
    stable = {layer: _scores(1 if layer >= 7 else 0) for layer in range(32)}
    assert MODULE.stable_to_final_layer(stable) == (7, 6.0)

    tied = dict(stable)
    tied[12] = _scores(1, tie_with=2)
    assert MODULE.stable_to_final_layer(tied) == (13, 6.0)

    plain = {layer: _scores(0) for layer in range(32)}
    wrapped = {layer: _scores(0 if layer < 9 else 2) for layer in range(32)}
    assert MODULE.plain_wrapped_separation_onset(plain, wrapped) == 9
    wrapped[31] = _scores(0, tie_with=2)
    assert MODULE.plain_wrapped_separation_onset(plain, wrapped) is None


def test_validation_rejects_missing_layers_duplicates_and_non_permutations():
    frame = _frame()
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.analyze_frame(pd.concat([frame, frame.iloc[[0]]]), n_boot=5)

    with pytest.raises(ValueError, match="layers 0 through 31"):
        MODULE.analyze_frame(frame[frame["layer"] != 17], n_boot=5)

    malformed = frame.copy()
    malformed.at[0, "content_ids_by_position"] = [0, 0, 2, 3]
    with pytest.raises(ValueError, match="permutation"):
        MODULE.analyze_frame(malformed, n_boot=5)

    unknown_eligibility = frame.copy()
    unknown_eligibility["primary_contrast_evaluable"] = unknown_eligibility[
        "primary_contrast_evaluable"
    ].astype(object)
    unknown_eligibility.loc[0, "primary_contrast_evaluable"] = None
    with pytest.raises(ValueError, match="boolean and non-null"):
        MODULE.analyze_frame(unknown_eligibility, n_boot=5)


def test_secondary_and_frozen_strata_outputs_and_bounded_classification():
    result = MODULE.analyze_frame(_frame(), n_boot=30, seed=5)

    assert set(result["secondary_contrasts"]["score_variant"]) == {
        "candidate_mean",
        "candidate_first_token",
    }
    assert {"split", "confidence_stratum", "position_susceptible"} <= set(
        result["strata"]["stratum"]
    )
    assert set(result["strata"]["score_variant"]) == {"candidate_total", "candidate_mean"}
    assert result["strata"]["descriptive_only"].all()
    assert not result["strata"]["multiplicity_adjusted"].any()
    assert "holm_p_value" not in result["strata"].columns
    non_wrapper = result["strata"][result["strata"]["stratum"] != "wrapper_name"]
    assert set(non_wrapper["n_items"]) == {1}
    assert result["summary"]["conclusion"] == "uninformative"
    assert not result["quality_gates"]["all_passed"]
    assert not result["quality_gates"]["checks"]["coverage"]["passed"]

    uninformative = result["primary_contrasts"].copy()
    uninformative[["ci_low", "ci_high"]] = [-0.2, 0.2]
    assert MODULE.classify_conclusion(uninformative, result["secondary_contrasts"], result["strata"]) == "uninformative"


def test_formation_requires_total_and_token_mean_to_agree_in_direction():
    primary = pd.DataFrame(
        [
            {"contract": "letter", "contrast": -0.4, "ci_low": -0.6, "ci_high": -0.2},
            {"contract": "text", "contrast": 0.0, "ci_low": -0.2, "ci_high": 0.2},
        ]
    )
    secondary = pd.DataFrame(
        [
            {"contract": "letter", "score_variant": "candidate_mean", "contrast": 0.1, "ci_low": -0.1, "ci_high": 0.3},
            {"contract": "letter", "score_variant": "candidate_first_token", "contrast": 0.0, "ci_low": -0.1, "ci_high": 0.1},
        ]
    )
    assert MODULE.classify_conclusion(primary, secondary, pd.DataFrame()) == "uninformative"

    secondary.loc[secondary["score_variant"] == "candidate_mean", "contrast"] = -0.1
    assert (
        MODULE.classify_conclusion(primary, secondary, pd.DataFrame())
        == "consistent_with_wrapper_dependent_answer_formation"
    )


def test_heterogeneous_requires_large_opposite_total_strata_with_matching_mean_directions():
    primary = pd.DataFrame(
        [
            {"contract": "letter", "contrast": 0.0, "ci_low": -0.2, "ci_high": 0.2},
            {"contract": "text", "contrast": 0.0, "ci_low": -0.2, "ci_high": 0.2},
        ]
    )
    secondary = pd.DataFrame(
        [
            {"contract": "letter", "score_variant": "candidate_mean", "contrast": 0.0, "ci_low": -0.2, "ci_high": 0.2},
            {"contract": "letter", "score_variant": "candidate_first_token", "contrast": 0.0, "ci_low": -0.2, "ci_high": 0.2},
        ]
    )
    strata = pd.DataFrame(
        [
            {"stratum": "confidence_stratum", "value": "low", "contract": "letter", "score_variant": "candidate_total", "contrast": -0.3, "ci_low": -0.5, "ci_high": -0.1, "n_items": 100},
            {"stratum": "confidence_stratum", "value": "low", "contract": "letter", "score_variant": "candidate_mean", "contrast": -0.2, "ci_low": -0.4, "ci_high": 0.1, "n_items": 100},
            {"stratum": "confidence_stratum", "value": "high", "contract": "letter", "score_variant": "candidate_total", "contrast": 0.4, "ci_low": 0.2, "ci_high": 0.6, "n_items": 100},
            {"stratum": "confidence_stratum", "value": "high", "contract": "letter", "score_variant": "candidate_mean", "contrast": 0.1, "ci_low": -0.1, "ci_high": 0.3, "n_items": 100},
        ]
    )
    assert (
        MODULE.classify_conclusion(primary, secondary, strata)
        == "predeclared_stratum_heterogeneous"
    )

    strata.loc[
        (strata["value"] == "high") & (strata["score_variant"] == "candidate_mean"),
        "contrast",
    ] = -0.1
    assert MODULE.classify_conclusion(primary, secondary, strata) == "uninformative"


def test_quality_gates_freeze_coverage_fragility_direction_and_resolved_estimates():
    primary = pd.DataFrame(
        [
            {
                "contract": contract,
                "contrast": contrast,
                "ci_low": contrast - 0.1,
                "ci_high": contrast + 0.1,
                "holm_p_value": 0.02,
                "n_items": 2_000,
                "n_pairs": 10_000,
            }
            for contract, contrast in (("letter", 0.3), ("text", 0.2))
        ]
    )
    secondary = pd.DataFrame(
        [
            {
                "contract": contract,
                "score_variant": "candidate_mean",
                "contrast": contrast,
                "ci_low": contrast - 0.1,
                "ci_high": contrast + 0.1,
                "n_items": 2_000,
                "n_pairs": 10_000,
            }
            for contract, contrast in (("letter", 0.2), ("text", 0.1))
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "contract": contract,
                "score_variant": variant,
                "layer": 31,
                "plain_ambiguity_rate": 0.05,
                "wrapped_ambiguity_rate": 0.10,
            }
            for contract in ("letter", "text")
            for variant in ("letter_raw", "candidate_total")
        ]
    )

    passed = MODULE.evaluate_quality_gates(
        primary, secondary, diagnostics, population_items=2_401
    )
    assert passed["all_passed"]
    assert passed["policy"] == {
        "minimum_primary_items_per_contract": 200,
        "minimum_primary_item_fraction": 0.8,
        "maximum_final_layer_ambiguity_rate": 0.2,
        "ambiguity_reference": 0.04,
        "required_contracts": ["letter", "text"],
        "required_final_layer_readouts": ["candidate_total", "letter_raw"],
        "require_resolved_primary_estimates": True,
        "require_total_mean_direction_agreement_by_contract": True,
    }

    weak = primary.copy()
    weak["n_items"] = 1_900
    weak_gates = MODULE.evaluate_quality_gates(
        weak, secondary, diagnostics, population_items=2_401
    )
    assert not weak_gates["checks"]["coverage"]["passed"]

    fragile = diagnostics.copy()
    fragile.loc[
        (fragile["contract"] == "letter")
        & (fragile["score_variant"] == "candidate_total"),
        "wrapped_ambiguity_rate",
    ] = 0.21
    fragile_gates = MODULE.evaluate_quality_gates(
        primary, secondary, fragile, population_items=2_401
    )
    assert not fragile_gates["checks"]["numerical_fragility"]["passed"]

    reversed_mean = secondary.copy()
    reversed_mean.loc[reversed_mean["contract"] == "text", "contrast"] = -0.1
    direction_gates = MODULE.evaluate_quality_gates(
        primary, reversed_mean, diagnostics, population_items=2_401
    )
    assert not direction_gates["checks"]["total_mean_direction_agreement"]["passed"]

    unresolved = primary.copy()
    unresolved.loc[unresolved["contract"] == "text", "ci_high"] = np.nan
    unresolved_gates = MODULE.evaluate_quality_gates(
        unresolved, secondary, diagnostics, population_items=2_401
    )
    assert not unresolved_gates["checks"]["primary_estimates_resolved"]["passed"]


def test_calibrated_letter_is_a_separate_optional_secondary():
    frame = _frame()
    frame["letter_calibrated_logps"] = frame["letter_raw_logps"]
    frame.loc[frame["contract"] == "text", "letter_calibrated_logps"] = frame.loc[
        frame["contract"] == "text", "letter_calibrated_logps"
    ].map(lambda _: [np.nan] * 4)

    result = MODULE.analyze_frame(frame, n_boot=20, seed=19)

    calibrated = result["calibrated_letter"]
    assert set(calibrated["contract"]) == {"letter"}
    assert set(calibrated["score_variant"]) == {"letter_calibrated"}
    assert "letter_calibrated" not in set(result["primary_contrasts"]["score_variant"])
    assert MODULE.analyze_frame(_frame(), n_boot=5)["calibrated_letter"].empty


def test_calibrated_letter_uses_its_own_plain_wrapped_eligibility_and_receipts():
    frame = _frame()
    frame["letter_calibrated_logps"] = frame["letter_raw_logps"]
    text = frame["contract"] == "text"
    frame.loc[text, "letter_calibrated_logps"] = frame.loc[text, "letter_calibrated_logps"].map(
        lambda _: [np.nan] * 4
    )
    excluded = frame["wrapper_name"] == "xml"
    frame.loc[excluded, "letter_calibration_evaluable"] = False
    frame.loc[excluded, "letter_calibration_exclusion_reason"] = "missing_answer_prefix"
    frame.loc[excluded, "letter_calibrated_logps"] = frame.loc[
        excluded, "letter_calibrated_logps"
    ].map(lambda _: [np.nan] * 4)

    result = MODULE.analyze_frame(frame, n_boot=10, seed=23)

    calibrated = result["calibrated_letter"]
    assert set(calibrated["n_pairs"]) == {2}
    assert result["summary"]["letter_calibration_excluded_blocks"] == 2
    assert result["summary"]["exclusion_counts"]["letter_calibration_exclusion_reason"] == {
        "missing_answer_prefix": 2
    }
    assert set(result["primary_contrasts"]["n_pairs"]) == {4}


def test_calibrated_letter_nan_requires_explicit_calibration_ineligibility():
    frame = _frame()
    frame["letter_calibrated_logps"] = frame["letter_raw_logps"]
    frame.loc[frame["contract"] == "text", "letter_calibrated_logps"] = frame.loc[
        frame["contract"] == "text", "letter_calibrated_logps"
    ].map(lambda _: [np.nan] * 4)
    letter_row = frame.index[frame["contract"] == "letter"][0]
    frame.at[letter_row, "letter_calibrated_logps"] = [np.nan] * 4
    with pytest.raises(ValueError, match="letter_calibrated_logps.*calibration is evaluable"):
        MODULE.analyze_frame(frame, n_boot=5)

    frame.loc[
        (frame["block_work_key"] == frame.at[letter_row, "block_work_key"]),
        "letter_calibration_evaluable",
    ] = False
    result = MODULE.analyze_frame(frame, n_boot=5)
    assert result["summary"]["letter_calibration_excluded_blocks"] == 1


def test_diagnostics_report_jsd_margins_ties_and_fixed_ambiguity_rate():
    frame = _frame()
    # One exact tie and one 0.03 near tie in otherwise well-separated data.
    tie_mask = (
        (frame["item_id"] == "i1")
        & (frame["wrapper_name"] == "json")
        & (frame["contract"] == "letter")
        & (frame["layer"] == 4)
    )
    frame.loc[tie_mask, "candidate_path_total_logps"] = frame.loc[
        tie_mask, "candidate_path_total_logps"
    ].map(lambda _: [2.0, 2.0, -4.0, -4.0])
    near_mask = tie_mask.copy()
    near_mask = (
        (frame["item_id"] == "i2")
        & (frame["wrapper_name"] == "json")
        & (frame["contract"] == "letter")
        & (frame["layer"] == 4)
    )
    frame.loc[near_mask, "candidate_path_total_logps"] = frame.loc[
        near_mask, "candidate_path_total_logps"
    ].map(lambda _: [2.0, 1.97, -4.0, -4.0])

    diagnostics = MODULE.analyze_frame(frame, n_boot=10, seed=2)["diagnostics"]
    row = diagnostics[
        (diagnostics["contract"] == "letter")
        & (diagnostics["score_variant"] == "candidate_total")
        & (diagnostics["layer"] == 4)
    ].iloc[0]
    assert row["mean_jsd"] >= 0.0
    assert row["wrapped_tie_rate"] == pytest.approx(0.25)
    assert row["wrapped_ambiguity_rate"] == pytest.approx(0.5)
    assert row["ambiguity_threshold"] == 0.04
    assert row["plain_mean_margin"] == 6.0


def test_diagnostics_jsd_remains_finite_when_restricted_probabilities_underflow():
    frame = _frame()
    for column in (
        "letter_raw_logps",
        "candidate_path_total_logps",
        "candidate_mean_token_logps",
        "candidate_first_token_logps",
    ):
        frame[column] = frame.apply(
            lambda row: [
                (
                    [0.0, -1_000.0, -1_000.0, -1_000.0]
                    if row["wrapper_name"] == "plain"
                    else [-1_000.0, 0.0, -1_000.0, -1_000.0]
                )[content_id]
                for content_id in row["content_ids_by_position"]
            ],
            axis=1,
        )

    diagnostics = MODULE.analyze_frame(frame, n_boot=5, seed=2)["diagnostics"]

    assert np.isfinite(diagnostics["mean_jsd"]).all()
    assert np.allclose(diagnostics["mean_jsd"], np.log(2.0))


def test_diagnostics_jsd_remains_finite_when_mixture_midpoint_underflows():
    frame = _frame()
    for column in (
        "letter_raw_logps",
        "candidate_path_total_logps",
        "candidate_mean_token_logps",
        "candidate_first_token_logps",
    ):
        frame[column] = frame.apply(
            lambda row: [
                (
                    [0.0, -745.0, -1_000.0, -1_000.0]
                    if row["wrapper_name"] == "plain"
                    else [0.0, -1_000.0, -745.0, -1_000.0]
                )[content_id]
                for content_id in row["content_ids_by_position"]
            ],
            axis=1,
        )

    diagnostics = MODULE.analyze_frame(frame, n_boot=5, seed=2)["diagnostics"]

    assert np.isfinite(diagnostics["mean_jsd"]).all()
    assert (diagnostics["mean_jsd"] >= 0.0).all()
    assert (diagnostics["mean_jsd"] <= np.log(2.0)).all()


def test_masked_candidate_nans_are_allowed_but_eligible_or_letter_nans_fail():
    frame = _frame()
    excluded = frame["wrapper_name"] == "xml"
    frame.loc[excluded, "primary_contrast_evaluable"] = False
    frame.loc[excluded, "first_token_contrast_evaluable"] = False
    for column in (
        "candidate_path_total_logps",
        "candidate_mean_token_logps",
        "candidate_first_token_logps",
    ):
        frame.loc[excluded, column] = frame.loc[excluded, column].map(lambda _: [np.nan] * 4)
    result = MODULE.analyze_frame(frame, n_boot=5)
    assert result["summary"]["primary_excluded_blocks"] == 2
    assert result["summary"]["first_token_excluded_blocks"] == 2

    eligible_nan = _frame()
    eligible_nan.at[0, "candidate_path_total_logps"] = [np.nan] * 4
    with pytest.raises(ValueError, match="candidate_path_total_logps.*evaluable"):
        MODULE.analyze_frame(eligible_nan, n_boot=5)

    letter_nan = frame.copy()
    letter_nan.at[0, "letter_raw_logps"] = [np.nan] * 4
    with pytest.raises(ValueError, match="letter_raw_logps.*finite"):
        MODULE.analyze_frame(letter_nan, n_boot=5)


def test_summary_has_exact_population_and_exclusion_receipts():
    frame = _frame()
    frame.loc[frame["wrapper_name"] == "xml", "primary_contrast_evaluable"] = False
    frame.loc[frame["wrapper_name"] == "xml", "first_token_contrast_evaluable"] = False
    frame.loc[frame["wrapper_name"] == "xml", "primary_exclusion_reason"] = "surface_mismatch"

    summary = MODULE.analyze_frame(frame, n_boot=5)["summary"]

    assert summary["items"] == 2
    assert summary["blocks"] == 6
    assert summary["wrappers"] == ["json", "plain", "xml"]
    assert summary["wrapper_count"] == 3
    assert summary["layerwise_rows"] == 384
    assert summary["block_contracts"] == 12
    assert summary["primary_excluded_blocks"] == 2
    assert summary["exclusion_counts"] == {"primary_exclusion_reason": {"surface_mismatch": 2}}

    inconsistent = frame.copy()
    inconsistent.loc[inconsistent.index[0], "primary_exclusion_reason"] = "other"
    with pytest.raises(ValueError, match="Exclusion reason changes"):
        MODULE.analyze_frame(inconsistent, n_boot=5)


def test_cli_writes_deterministic_machine_readable_outputs(tmp_path):
    source = tmp_path / "layerwise_scores.parquet"
    _frame().to_parquet(source, index=False)

    outputs = MODULE.analyze(source, tmp_path / "out", n_boot=20, seed=11)

    assert set(outputs) == {
        "analysis_summary",
        "interpretation_memo",
        "quality_gates",
        "calibrated_letter",
        "diagnostics",
        "primary_contrasts",
        "secondary_contrasts",
        "trajectories",
        "timing",
        "timing_distributions",
        "trajectory_figure",
        "strata",
    }
    assert all(path.exists() for path in outputs.values())
    summary = json.loads(outputs["analysis_summary"].read_text())
    assert summary["bootstrap_samples"] == 20
    assert summary["bootstrap_seed"] == 11
    assert summary["pre_final_layers"] == list(range(31))
    gates = json.loads(outputs["quality_gates"].read_text())
    assert gates["final_conclusion"] == "uninformative"
    memo = outputs["interpretation_memo"].read_text()
    assert "descriptive" in memo.lower()
    assert "does not establish causation" in memo.lower()
    figure = outputs["trajectory_figure"]
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.stat().st_size > 10_000
    timing = pd.read_csv(outputs["timing_distributions"])
    assert set(timing["timing_type"]) == {
        "stable_to_final",
        "plain_wrapped_separation_onset",
    }
    assert {
        "n_total",
        "n_defined",
        "defined_rate",
        "median_layer",
        "q25_layer",
        "q75_layer",
    } <= set(timing.columns)

    repeat = MODULE.analyze(source, tmp_path / "repeat", n_boot=20, seed=11)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(outputs["trajectory_figure"]) == digest(repeat["trajectory_figure"])
