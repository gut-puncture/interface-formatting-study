"""Frozen descriptive analysis for the Mistral two-contract logit lens.

The verified merged input has one row per ``(block_work_key, contract, layer)``.
Four-score arrays are in displayed-option order; this module maps them to the
audited content identities before making any plain/wrapped comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


LAYERS = tuple(range(32))
PRE_FINAL_LAYERS = tuple(range(31))
SCORE_COLUMNS = {
    "letter_raw": "letter_raw_logps",
    "candidate_total": "candidate_path_total_logps",
    "candidate_mean": "candidate_mean_token_logps",
    "candidate_first_token": "candidate_first_token_logps",
}
CALIBRATED_COLUMN = "letter_calibrated_logps"
AMBIGUITY_THRESHOLD = 0.04
QUALITY_GATE_POLICY = {
    "minimum_primary_items_per_contract": 200,
    "minimum_primary_item_fraction": 0.8,
    "maximum_final_layer_ambiguity_rate": 0.2,
    "ambiguity_reference": AMBIGUITY_THRESHOLD,
    "required_contracts": ["letter", "text"],
    "required_final_layer_readouts": ["candidate_total", "letter_raw"],
    "require_resolved_primary_estimates": True,
    "require_total_mean_direction_agreement_by_contract": True,
}
FROZEN_STRATA_COLUMNS = (
    "wrapper_name",
    "split",
    "conflict_status",
    "confidence_stratum",
    "correctness_stratum",
    "position_susceptible",
    "label_susceptible",
    "susceptibility_stratum",
    "text_letter_status",
    "incorrect_wrapped_decision",
    "exact_surface_match",
    "audit_provenance",
)


def _four_finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (4,) or not np.isfinite(array).all():
        raise ValueError("Expected a finite four-score array")
    return array


def _four_allow_nan(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (4,) or np.isinf(array).any():
        raise ValueError("Expected a four-score array containing only finite values or NaN")
    return array


def argmax_set(values: Sequence[float]) -> frozenset[int]:
    """Return every exact maximizer; candidate order never breaks a tie."""

    array = _four_finite(values)
    maximum = float(array.max())
    return frozenset(int(index) for index in np.flatnonzero(array == maximum))


def jaccard_agreement(left: frozenset[int], right: frozenset[int]) -> float:
    if not left or not right:
        raise ValueError("Argmax sets must be non-empty")
    return len(left & right) / len(left | right)


def scores_by_content(values: Sequence[float], content_ids_by_position: Sequence[int]) -> list[float]:
    scores = _four_finite(values)
    content_ids = [int(value) for value in content_ids_by_position]
    if sorted(content_ids) != [0, 1, 2, 3]:
        raise ValueError("content_ids_by_position must be a permutation of 0, 1, 2, 3")
    mapped = np.empty(4, dtype=float)
    for position, content_id in enumerate(content_ids):
        mapped[content_id] = scores[position]
    return mapped.tolist()


def _margin(values: Sequence[float]) -> float:
    ordered = np.sort(_four_finite(values))
    return float(ordered[-1] - ordered[-2])


def _validate_trajectory(trajectory: Mapping[int, Sequence[float]]) -> dict[int, list[float]]:
    if set(trajectory) != set(LAYERS):
        raise ValueError("A trajectory must contain exactly layers 0 through 31")
    return {layer: _four_finite(trajectory[layer]).tolist() for layer in LAYERS}


def stable_to_final_layer(trajectory: Mapping[int, Sequence[float]]) -> tuple[int, float] | None:
    """Earliest pre-final layer with one winner persisting through layer 31."""

    scores = _validate_trajectory(trajectory)
    final = argmax_set(scores[31])
    if len(final) != 1:
        return None
    for start in PRE_FINAL_LAYERS:
        subsequent = [argmax_set(scores[layer]) for layer in range(start, 32)]
        if all(winners == final and len(winners) == 1 for winners in subsequent):
            return start, min(_margin(scores[layer]) for layer in range(start, 32))
    return None


def plain_wrapped_separation_onset(
    plain: Mapping[int, Sequence[float]],
    wrapped: Mapping[int, Sequence[float]],
) -> int | None:
    """Earliest layer from which two distinct final winners both persist."""

    plain_scores = _validate_trajectory(plain)
    wrapped_scores = _validate_trajectory(wrapped)
    plain_final = argmax_set(plain_scores[31])
    wrapped_final = argmax_set(wrapped_scores[31])
    if len(plain_final) != 1 or len(wrapped_final) != 1 or plain_final == wrapped_final:
        return None
    for start in PRE_FINAL_LAYERS:
        if all(
            argmax_set(plain_scores[layer]) == plain_final
            and argmax_set(wrapped_scores[layer]) == wrapped_final
            for layer in range(start, 32)
        ):
            return start
    return None


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "block_work_key",
        "item_id",
        "wrapper_name",
        "contract",
        "layer",
        "content_ids_by_position",
        "primary_contrast_evaluable",
        "first_token_contrast_evaluable",
        "letter_calibration_evaluable",
        *SCORE_COLUMNS.values(),
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Merged layerwise table is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Merged layerwise table is empty")
    table = frame.copy()
    table["layer"] = pd.to_numeric(table["layer"], errors="raise").astype(int)
    duplicate_key = ["block_work_key", "contract", "layer"]
    if table.duplicated(duplicate_key).any():
        raise ValueError("Merged layerwise table contains duplicate block-contract-layer rows")
    if set(table["contract"].astype(str)) != {"letter", "text"}:
        raise ValueError("Merged layerwise table must contain exactly letter and text contracts")
    for column in (
        "primary_contrast_evaluable",
        "first_token_contrast_evaluable",
        "letter_calibration_evaluable",
    ):
        eligibility = table[column]
        if eligibility.isna().any() or not eligibility.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"{column} must be boolean and non-null")
    if (table["first_token_contrast_evaluable"] & ~table["primary_contrast_evaluable"]).any():
        raise ValueError("first_token_contrast_evaluable requires primary_contrast_evaluable")
    for key, group in table.groupby(["block_work_key", "contract"], sort=False):
        if set(group["layer"]) != set(LAYERS) or len(group) != len(LAYERS):
            raise ValueError(f"Each block-contract must contain exactly layers 0 through 31: {key}")
        if group["item_id"].astype(str).nunique() != 1 or group["wrapper_name"].astype(str).nunique() != 1:
            raise ValueError(f"Block identity changes across layers: {key}")
        if group["primary_contrast_evaluable"].nunique(dropna=False) != 1:
            raise ValueError(f"Primary eligibility changes across layers: {key}")
        if group["first_token_contrast_evaluable"].nunique(dropna=False) != 1:
            raise ValueError(f"First-token eligibility changes across layers: {key}")
        if group["letter_calibration_evaluable"].nunique(dropna=False) != 1:
            raise ValueError(f"Letter-calibration eligibility changes across layers: {key}")
    for row in table.itertuples(index=False):
        content_ids = getattr(row, "content_ids_by_position")
        if sorted(int(value) for value in content_ids) != [0, 1, 2, 3]:
            raise ValueError("content_ids_by_position must be a permutation of 0, 1, 2, 3")
        try:
            _four_finite(getattr(row, "letter_raw_logps"))
        except ValueError as exc:
            raise ValueError("letter_raw_logps must always be finite") from exc
        for column in ("candidate_path_total_logps", "candidate_mean_token_logps"):
            values = getattr(row, column)
            if bool(row.primary_contrast_evaluable):
                try:
                    _four_finite(values)
                except ValueError as exc:
                    raise ValueError(f"{column} must be finite when primary contrast is evaluable") from exc
            else:
                _four_allow_nan(values)
        first_values = getattr(row, "candidate_first_token_logps")
        if bool(row.first_token_contrast_evaluable):
            try:
                _four_finite(first_values)
            except ValueError as exc:
                raise ValueError(
                    "candidate_first_token_logps must be finite when first-token contrast is evaluable"
                ) from exc
        else:
            _four_allow_nan(first_values)
        if CALIBRATED_COLUMN in table.columns:
            calibrated = getattr(row, CALIBRATED_COLUMN)
            if str(row.contract) == "letter" and bool(row.letter_calibration_evaluable):
                try:
                    _four_finite(calibrated)
                except ValueError as exc:
                    raise ValueError(
                        "letter_calibrated_logps must be finite when letter calibration is evaluable"
                    ) from exc
            else:
                _four_allow_nan(calibrated)
    return table


def _pair_rows(table: pd.DataFrame) -> pd.DataFrame:
    identity = ["item_id", "contract", "layer"]
    plain = table[table["wrapper_name"] == "plain"].copy()
    if plain.duplicated(identity).any():
        raise ValueError("Expected exactly one plain block per item-contract-layer")
    wrapped = table[table["wrapper_name"] != "plain"].copy()
    if wrapped.empty:
        raise ValueError("At least one wrapped block is required")
    pairs = wrapped.merge(plain, on=identity, suffixes=("", "_plain"), validate="many_to_one")
    if len(pairs) != len(wrapped):
        raise ValueError("Every wrapped row must have a matching plain row")
    pairs["pair_eligible"] = (
        pairs["primary_contrast_evaluable"]
        & pairs["primary_contrast_evaluable_plain"]
    )
    pairs["first_token_pair_evaluable"] = (
        pairs["pair_eligible"]
        & pairs["first_token_contrast_evaluable"]
        & pairs["first_token_contrast_evaluable_plain"]
    )
    pairs["letter_calibration_pair_evaluable"] = (
        pairs["letter_calibration_evaluable"]
        & pairs["letter_calibration_evaluable_plain"]
    )
    return pairs


def _active_score_columns(table: pd.DataFrame) -> dict[str, str]:
    columns = dict(SCORE_COLUMNS)
    if CALIBRATED_COLUMN in table.columns:
        columns["letter_calibrated"] = CALIBRATED_COLUMN
    return columns


def _variant_pair_evaluable(row, variant: str) -> bool:
    if variant == "letter_calibrated":
        return bool(row.letter_calibration_pair_evaluable)
    if variant == "candidate_first_token":
        return bool(row.first_token_pair_evaluable)
    return bool(row.pair_eligible)


def _agreement_rows(pairs: pd.DataFrame, score_columns: Mapping[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in pairs.itertuples(index=False):
        for variant, column in score_columns.items():
            if variant == "letter_calibrated" and str(row.contract) != "letter":
                continue
            if not _variant_pair_evaluable(row, variant):
                continue
            plain_values = scores_by_content(
                getattr(row, f"{column}_plain"), getattr(row, "content_ids_by_position_plain")
            )
            wrapped_values = scores_by_content(getattr(row, column), row.content_ids_by_position)
            rows.append(
                {
                    "item_id": str(row.item_id),
                    "wrapper_name": str(row.wrapper_name),
                    "contract": str(row.contract),
                    "layer": int(row.layer),
                    "score_variant": variant,
                    "agreement": jaccard_agreement(argmax_set(plain_values), argmax_set(wrapped_values)),
                }
            )
    if not rows:
        raise ValueError("No eligible plain/wrapped pairs remain")
    return pd.DataFrame(rows)


def _normalized_auc(layer_values: pd.Series) -> float:
    indexed = layer_values.sort_index()
    if tuple(int(value) for value in indexed.index) != PRE_FINAL_LAYERS:
        raise ValueError("AUC requires one value for each pre-final layer 0 through 30")
    values = indexed.to_numpy(dtype=float)
    return float((0.5 * values[0] + values[1:-1].sum() + 0.5 * values[-1]) / 30.0)


def _pair_aucs(agreements: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in agreements.groupby(
        ["item_id", "wrapper_name", "contract", "score_variant"], sort=True
    ):
        per_layer = group.groupby("layer")["agreement"].mean().reindex(PRE_FINAL_LAYERS)
        rows.append(
            {
                "item_id": keys[0],
                "wrapper_name": keys[1],
                "contract": keys[2],
                "score_variant": keys[3],
                "auc": _normalized_auc(per_layer),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(values: Iterable[float], *, n_boot: int, seed: int) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap requires at least one finite item estimate")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draws[index] = float(rng.choice(array, size=len(array), replace=True).mean())
    non_positive = int(np.count_nonzero(draws <= 0.0))
    non_negative = int(np.count_nonzero(draws >= 0.0))
    p_value = min(1.0, 2.0 * (min(non_positive, non_negative) + 1) / (n_boot + 1))
    return {
        "estimate": float(array.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_value": float(p_value),
    }


def _holm(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(values[index]) * (len(values) - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def _contrast_rows(
    pair_aucs: pd.DataFrame,
    candidate_variant: str,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for contract in ("letter", "text"):
        subset = pair_aucs[
            (pair_aucs["contract"] == contract)
            & pair_aucs["score_variant"].isin(["letter_raw", candidate_variant])
        ]
        paired = subset.pivot(
            index=["item_id", "wrapper_name"], columns="score_variant", values="auc"
        ).dropna()
        if paired.empty:
            raise ValueError(f"No paired item AUCs for {contract}/{candidate_variant}")
        wide = paired.groupby(level="item_id")[["letter_raw", candidate_variant]].mean()
        differences = wide[candidate_variant] - wide["letter_raw"]
        stat = _bootstrap(differences, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "contract": contract,
                "score_variant": candidate_variant,
                "letter_auc": float(wide["letter_raw"].mean()),
                "candidate_auc": float(wide[candidate_variant].mean()),
                "contrast": stat["estimate"],
                "ci_low": stat["ci_low"],
                "ci_high": stat["ci_high"],
                "p_value": stat["p_value"],
                "n_items": int(len(wide)),
                "n_pairs": int(len(paired)),
            }
        )
    return pd.DataFrame(rows)


def _trajectory_summary(
    agreements: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    item_layer = (
        agreements.groupby(["item_id", "contract", "score_variant", "layer"], as_index=False)
        .agg(agreement=("agreement", "mean"), n_pairs=("wrapper_name", "nunique"))
    )
    summary = (
        item_layer.groupby(["contract", "score_variant", "layer"], as_index=False)
        .agg(agreement=("agreement", "mean"), n_items=("item_id", "nunique"), n_pairs=("n_pairs", "sum"))
        .sort_values(["contract", "score_variant", "layer"])
        .reset_index(drop=True)
    )
    summary["simultaneous_ci_low"] = np.nan
    summary["simultaneous_ci_high"] = np.nan
    for (contract, variant), group in item_layer.groupby(["contract", "score_variant"], sort=True):
        matrix = group.pivot(index="item_id", columns="layer", values="agreement").reindex(columns=LAYERS)
        if matrix.isna().any().any():
            raise ValueError(f"Trajectory band requires complete item trajectories: {contract}/{variant}")
        values = matrix.to_numpy(dtype=float)
        estimate = values.mean(axis=0)
        rng = np.random.default_rng(seed)
        maximum_deviations = np.empty(n_boot, dtype=float)
        for draw_index in range(n_boot):
            sampled = rng.integers(0, len(values), size=len(values))
            draw = values[sampled].mean(axis=0)
            maximum_deviations[draw_index] = float(np.max(np.abs(draw - estimate)))
        radius = float(np.quantile(maximum_deviations, 0.95))
        mask = (summary["contract"] == contract) & (summary["score_variant"] == variant)
        summary.loc[mask, "simultaneous_ci_low"] = np.maximum(0.0, estimate - radius)
        summary.loc[mask, "simultaneous_ci_high"] = np.minimum(1.0, estimate + radius)
    return summary


def _restricted_probabilities(values: Sequence[float]) -> np.ndarray:
    scores = _four_finite(values)
    shifted = scores - scores.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def _jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    p = _restricted_probabilities(left)
    q = _restricted_probabilities(right)
    midpoint = 0.5 * (p + q)
    p_positive = p > 0.0
    q_positive = q > 0.0
    return float(
        0.5 * np.sum(p[p_positive] * np.log(p[p_positive] / midpoint[p_positive]))
        + 0.5 * np.sum(q[q_positive] * np.log(q[q_positive] / midpoint[q_positive]))
    )


def _diagnostic_summary(pairs: pd.DataFrame, score_columns: Mapping[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in pairs.itertuples(index=False):
        for variant, column in score_columns.items():
            if variant == "letter_calibrated" and str(row.contract) != "letter":
                continue
            if not _variant_pair_evaluable(row, variant):
                continue
            plain = scores_by_content(
                getattr(row, f"{column}_plain"), getattr(row, "content_ids_by_position_plain")
            )
            wrapped = scores_by_content(getattr(row, column), row.content_ids_by_position)
            plain_margin = _margin(plain)
            wrapped_margin = _margin(wrapped)
            rows.append(
                {
                    "item_id": str(row.item_id),
                    "wrapper_name": str(row.wrapper_name),
                    "contract": str(row.contract),
                    "score_variant": variant,
                    "layer": int(row.layer),
                    "jsd": _jensen_shannon(plain, wrapped),
                    "plain_margin": plain_margin,
                    "wrapped_margin": wrapped_margin,
                    "plain_tie": float(len(argmax_set(plain)) > 1),
                    "wrapped_tie": float(len(argmax_set(wrapped)) > 1),
                    "plain_ambiguous": float(plain_margin <= AMBIGUITY_THRESHOLD),
                    "wrapped_ambiguous": float(wrapped_margin <= AMBIGUITY_THRESHOLD),
                }
            )
    diagnostics = pd.DataFrame(rows)
    item_layer = (
        diagnostics.groupby(["item_id", "contract", "score_variant", "layer"], as_index=False)
        .agg(
            mean_jsd=("jsd", "mean"),
            plain_mean_margin=("plain_margin", "mean"),
            wrapped_mean_margin=("wrapped_margin", "mean"),
            plain_tie_rate=("plain_tie", "mean"),
            wrapped_tie_rate=("wrapped_tie", "mean"),
            plain_ambiguity_rate=("plain_ambiguous", "mean"),
            wrapped_ambiguity_rate=("wrapped_ambiguous", "mean"),
            n_pairs=("wrapper_name", "nunique"),
        )
    )
    result = (
        item_layer.groupby(["contract", "score_variant", "layer"], as_index=False)
        .agg(
            mean_jsd=("mean_jsd", "mean"),
            plain_mean_margin=("plain_mean_margin", "mean"),
            wrapped_mean_margin=("wrapped_mean_margin", "mean"),
            plain_tie_rate=("plain_tie_rate", "mean"),
            wrapped_tie_rate=("wrapped_tie_rate", "mean"),
            plain_ambiguity_rate=("plain_ambiguity_rate", "mean"),
            wrapped_ambiguity_rate=("wrapped_ambiguity_rate", "mean"),
            n_items=("item_id", "nunique"),
            n_pairs=("n_pairs", "sum"),
        )
    )
    result["ambiguity_threshold"] = AMBIGUITY_THRESHOLD
    return result.sort_values(["contract", "score_variant", "layer"]).reset_index(drop=True)


def _timing_summary(table: pd.DataFrame) -> pd.DataFrame:
    timing: list[dict[str, object]] = []
    mapped: dict[tuple[str, str, str, str], dict[int, list[float]]] = {}
    evaluable: set[tuple[str, str, str]] = set()
    first_token_evaluable: set[tuple[str, str, str]] = set()
    for (item, wrapper, contract), group in table.groupby(["item_id", "wrapper_name", "contract"], sort=True):
        if bool(group["primary_contrast_evaluable"].iloc[0]):
            evaluable.add((str(item), str(wrapper), str(contract)))
        if bool(group["primary_contrast_evaluable"].iloc[0]) and bool(
            group["first_token_contrast_evaluable"].iloc[0]
        ):
            first_token_evaluable.add((str(item), str(wrapper), str(contract)))
        for variant, column in SCORE_COLUMNS.items():
            if variant in {"candidate_total", "candidate_mean"} and (
                str(item), str(wrapper), str(contract)
            ) not in evaluable:
                continue
            if variant == "candidate_first_token" and (
                str(item), str(wrapper), str(contract)
            ) not in first_token_evaluable:
                continue
            trajectory = {
                int(row.layer): scores_by_content(getattr(row, column), row.content_ids_by_position)
                for row in group.itertuples(index=False)
            }
            mapped[(str(item), str(wrapper), str(contract), variant)] = trajectory
            result = stable_to_final_layer(trajectory)
            timing.append(
                {
                    "timing_type": "stable_to_final",
                    "item_id": str(item),
                    "wrapper_name": str(wrapper),
                    "contract": str(contract),
                    "score_variant": variant,
                    "layer": None if result is None else result[0],
                    "min_subsequent_margin": None if result is None else result[1],
                }
            )
    for item, wrapper, contract, variant in sorted(mapped):
        if wrapper == "plain":
            continue
        eligible_set = first_token_evaluable if variant == "candidate_first_token" else evaluable
        if (item, wrapper, contract) not in eligible_set or (item, "plain", contract) not in eligible_set:
            continue
        plain = mapped.get((item, "plain", contract, variant))
        if plain is None:
            raise ValueError(f"Missing plain trajectory for {item}/{contract}/{variant}")
        timing.append(
            {
                "timing_type": "plain_wrapped_separation_onset",
                "item_id": item,
                "wrapper_name": wrapper,
                "contract": contract,
                "score_variant": variant,
                "layer": plain_wrapped_separation_onset(plain, mapped[(item, wrapper, contract, variant)]),
                "min_subsequent_margin": None,
            }
        )
    return pd.DataFrame(timing)


def _strata_summary(
    pairs: pd.DataFrame,
    pair_aucs: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    metadata_columns = [
        column for column in FROZEN_STRATA_COLUMNS
        if column != "wrapper_name" and column in pairs.columns
    ]
    metadata = pairs[["item_id", "wrapper_name", "contract", *metadata_columns]].drop_duplicates()
    analysis_columns = ["wrapper_name", *metadata_columns]
    for column in analysis_columns:
        counts = metadata.groupby(["item_id", "wrapper_name", "contract"])[column].nunique(dropna=False)
        if (counts > 1).any():
            raise ValueError(f"Frozen stratum changes across layers: {column}")
    rows: list[dict[str, object]] = []
    for candidate_variant in ("candidate_total", "candidate_mean"):
        contrasts = pair_aucs[
            pair_aucs["score_variant"].isin(["letter_raw", candidate_variant])
        ].pivot(
            index=["item_id", "wrapper_name", "contract"],
            columns="score_variant",
            values="auc",
        ).dropna().reset_index()
        contrasts["contrast"] = contrasts[candidate_variant] - contrasts["letter_raw"]
        contrasts = contrasts.merge(
            metadata, on=["item_id", "wrapper_name", "contract"], validate="one_to_one"
        )
        for column in analysis_columns:
            for (contract, value), group in contrasts.groupby(
                ["contract", column], dropna=False, sort=True
            ):
                per_item = group.groupby("item_id")["contrast"].mean()
                stat = _bootstrap(per_item, n_boot=n_boot, seed=seed)
                rows.append(
                    {
                        "stratum": column,
                        "value": str(value),
                        "contract": contract,
                        "score_variant": candidate_variant,
                        "contrast": stat["estimate"],
                        "ci_low": stat["ci_low"],
                        "ci_high": stat["ci_high"],
                        "n_items": int(group["item_id"].nunique()),
                        "n_pairs": int(len(group)),
                        "descriptive_only": True,
                        "multiplicity_adjusted": False,
                    }
                )
    return pd.DataFrame(rows)


def _has_qualified_heterogeneity(strata: pd.DataFrame | None) -> bool:
    required = {
        "stratum",
        "value",
        "contract",
        "score_variant",
        "contrast",
        "ci_low",
        "ci_high",
        "n_items",
    }
    if strata is None or strata.empty or not required <= set(strata.columns):
        return False
    letter = strata[(strata["contract"] == "letter") & (strata["n_items"] >= 100)]
    total = letter[letter["score_variant"] == "candidate_total"]
    mean = letter[letter["score_variant"] == "candidate_mean"][
        ["stratum", "value", "contrast"]
    ].rename(columns={"contrast": "mean_contrast"})
    matched = total.merge(mean, on=["stratum", "value"], validate="one_to_one")
    matched["qualified_positive"] = (matched["ci_low"] > 0) & (matched["mean_contrast"] > 0)
    matched["qualified_negative"] = (matched["ci_high"] < 0) & (matched["mean_contrast"] < 0)
    for _, group in matched.groupby("stratum", sort=False):
        if group["qualified_positive"].any() and group["qualified_negative"].any():
            return True
    return False


def classify_conclusion(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    strata: pd.DataFrame | None = None,
) -> str:
    """Apply the bounded, predeclared result labels without inventing a layer."""

    required_contracts = {"letter", "text"}
    if set(primary["contract"].astype(str)) != required_contracts:
        return "uninformative"
    indexed = primary.set_index("contract")
    letter = indexed.loc["letter"]
    text = indexed.loc["text"]
    if (
        (float(letter["ci_low"]) > 0 and float(text["ci_high"]) < 0)
        or (float(letter["ci_high"]) < 0 and float(text["ci_low"]) > 0)
    ):
        return "contract_sensitive"
    if _has_qualified_heterogeneity(strata):
        return "predeclared_stratum_heterogeneous"
    letter_secondary = secondary[secondary["contract"] == "letter"].set_index("score_variant")
    mean_supports_positive = (
        "candidate_mean" in letter_secondary.index
        and float(letter_secondary.loc["candidate_mean", "contrast"]) >= 0
    )
    first_not_opposite = (
        "candidate_first_token" not in letter_secondary.index
        or float(letter_secondary.loc["candidate_first_token", "ci_high"]) >= 0
    )
    if float(letter["ci_low"]) > 0 and mean_supports_positive and first_not_opposite:
        return "consistent_with_later_output_binding_effects"
    mean_supports_negative = (
        "candidate_mean" in letter_secondary.index
        and float(letter_secondary.loc["candidate_mean", "contrast"]) < 0
    )
    if float(letter["ci_high"]) < 0 and mean_supports_negative:
        return "consistent_with_wrapper_dependent_answer_formation"
    return "uninformative"


def _finite_estimate_rows(frame: pd.DataFrame, required_columns: Sequence[str]) -> bool:
    if any(column not in frame.columns for column in required_columns):
        return False
    values = frame[list(required_columns)].apply(pd.to_numeric, errors="coerce")
    return bool(np.isfinite(values.to_numpy(dtype=float)).all())


def evaluate_quality_gates(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    population_items: int,
) -> dict[str, object]:
    """Evaluate the frozen gates that must pass before any positive label."""

    required_contracts = list(QUALITY_GATE_POLICY["required_contracts"])
    primary_contracts = (
        primary["contract"].astype(str)
        if "contract" in primary
        else pd.Series(dtype=str)
    )
    primary_shape_ok = (
        len(primary) == len(required_contracts)
        and primary_contracts.is_unique
        and set(primary_contracts) == set(required_contracts)
    )
    primary_resolved = primary_shape_ok and _finite_estimate_rows(
        primary,
        ("contrast", "ci_low", "ci_high", "holm_p_value", "n_items", "n_pairs"),
    )
    if primary_resolved:
        primary_resolved = bool(
            (primary["ci_low"] <= primary["contrast"]).all()
            and (primary["contrast"] <= primary["ci_high"]).all()
            and (primary["n_items"] > 0).all()
            and (primary["n_pairs"] > 0).all()
        )

    minimum_fraction = float(QUALITY_GATE_POLICY["minimum_primary_item_fraction"])
    minimum_count = int(QUALITY_GATE_POLICY["minimum_primary_items_per_contract"])
    required_items = max(minimum_count, int(np.ceil(population_items * minimum_fraction)))
    items_by_contract = (
        {
            str(row.contract): int(row.n_items)
            for row in primary[["contract", "n_items"]].itertuples(index=False)
        }
        if {"contract", "n_items"} <= set(primary.columns)
        else {}
    )
    coverage_passed = (
        primary_shape_ok
        and population_items > 0
        and all(
            items_by_contract.get(contract, -1) >= required_items
            for contract in required_contracts
        )
    )

    mean = secondary[
        secondary.get("score_variant", pd.Series(index=secondary.index, dtype=str)).astype(str)
        == "candidate_mean"
    ].copy()
    mean_contracts = (
        mean["contract"].astype(str)
        if "contract" in mean
        else pd.Series(dtype=str)
    )
    mean_shape_ok = (
        len(mean) == len(required_contracts)
        and mean_contracts.is_unique
        and set(mean_contracts) == set(required_contracts)
        and _finite_estimate_rows(mean, ("contrast",))
    )
    direction_details: dict[str, dict[str, object]] = {}
    direction_passed = bool(primary_resolved and mean_shape_ok)
    if direction_passed:
        totals = primary.set_index("contract")["contrast"]
        means = mean.set_index("contract")["contrast"]
        for contract in required_contracts:
            total_value = float(totals.loc[contract])
            mean_value = float(means.loc[contract])
            agrees = (
                total_value != 0.0
                and mean_value != 0.0
                and np.sign(total_value) == np.sign(mean_value)
            )
            direction_details[contract] = {
                "candidate_total_contrast": total_value,
                "candidate_mean_contrast": mean_value,
                "same_nonzero_direction": bool(agrees),
            }
            direction_passed = direction_passed and bool(agrees)

    required_readouts = list(QUALITY_GATE_POLICY["required_final_layer_readouts"])
    final_diagnostics = diagnostics[
        (pd.to_numeric(diagnostics.get("layer"), errors="coerce") == 31)
        & diagnostics.get(
            "contract", pd.Series(index=diagnostics.index, dtype=str)
        ).astype(str).isin(required_contracts)
        & diagnostics.get(
            "score_variant", pd.Series(index=diagnostics.index, dtype=str)
        ).astype(str).isin(required_readouts)
    ].copy()
    expected_final_rows = len(required_contracts) * len(required_readouts)
    fragility_shape_ok = (
        len(final_diagnostics) == expected_final_rows
        and not final_diagnostics.duplicated(["contract", "score_variant"]).any()
        and set(final_diagnostics["contract"].astype(str)) == set(required_contracts)
        and set(final_diagnostics["score_variant"].astype(str)) == set(required_readouts)
        and _finite_estimate_rows(
            final_diagnostics, ("plain_ambiguity_rate", "wrapped_ambiguity_rate")
        )
    )
    maximum_ambiguity = float(QUALITY_GATE_POLICY["maximum_final_layer_ambiguity_rate"])
    fragility_details: list[dict[str, object]] = []
    fragility_passed = fragility_shape_ok
    if fragility_shape_ok:
        for row in final_diagnostics.sort_values(["contract", "score_variant"]).itertuples():
            observed = max(float(row.plain_ambiguity_rate), float(row.wrapped_ambiguity_rate))
            passed = observed <= maximum_ambiguity
            fragility_details.append(
                {
                    "contract": str(row.contract),
                    "score_variant": str(row.score_variant),
                    "maximum_observed_ambiguity_rate": observed,
                    "passed": bool(passed),
                }
            )
            fragility_passed = fragility_passed and bool(passed)

    checks = {
        "primary_estimates_resolved": {
            "passed": bool(primary_resolved),
            "required_contracts": required_contracts,
        },
        "coverage": {
            "passed": bool(coverage_passed),
            "population_items": int(population_items),
            "required_items_per_contract": int(required_items),
            "items_by_contract": items_by_contract,
        },
        "total_mean_direction_agreement": {
            "passed": bool(direction_passed),
            "by_contract": direction_details,
        },
        "numerical_fragility": {
            "passed": bool(fragility_passed),
            "final_layer": 31,
            "ambiguity_reference": AMBIGUITY_THRESHOLD,
            "maximum_allowed_rate": maximum_ambiguity,
            "rows": fragility_details,
        },
    }
    return {
        "schema_version": 1,
        "policy": dict(QUALITY_GATE_POLICY),
        "checks": checks,
        "all_passed": bool(all(check["passed"] for check in checks.values())),
    }


def _population_receipts(table: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, object]:
    block_columns = [
        "block_work_key",
        "item_id",
        "wrapper_name",
        "primary_contrast_evaluable",
        "first_token_contrast_evaluable",
        "letter_calibration_evaluable",
    ]
    blocks = table[block_columns].drop_duplicates()
    if blocks.duplicated("block_work_key").any():
        raise ValueError("Block eligibility or identity changes across contracts")
    exclusion_counts: dict[str, dict[str, int]] = {}
    for column in sorted(value for value in table.columns if value.endswith("_exclusion_reason")):
        distinct = table.groupby("block_work_key")[column].nunique(dropna=False)
        if (distinct > 1).any():
            raise ValueError(f"Exclusion reason changes within a block: {column}")
        reasons = table[["block_work_key", column]].drop_duplicates()
        reasons = reasons[reasons[column].notna() & (reasons[column].astype(str) != "")]
        exclusion_counts[column] = {
            str(reason): int(count)
            for reason, count in reasons[column].astype(str).value_counts().sort_index().items()
        }
    wrappers = sorted(blocks["wrapper_name"].astype(str).unique())
    return {
        "items": int(blocks["item_id"].nunique()),
        "blocks": int(len(blocks)),
        "block_contracts": int(table[["block_work_key", "contract"]].drop_duplicates().shape[0]),
        "layerwise_rows": int(len(table)),
        "wrappers": wrappers,
        "wrapper_count": int(len(wrappers)),
        "blocks_by_wrapper": {
            str(wrapper): int(count)
            for wrapper, count in blocks["wrapper_name"].astype(str).value_counts().sort_index().items()
        },
        "primary_eligible_blocks": int(blocks["primary_contrast_evaluable"].sum()),
        "primary_excluded_blocks": int((~blocks["primary_contrast_evaluable"]).sum()),
        "first_token_eligible_blocks": int(blocks["first_token_contrast_evaluable"].sum()),
        "first_token_excluded_blocks": int((~blocks["first_token_contrast_evaluable"]).sum()),
        "letter_calibration_eligible_blocks": int(blocks["letter_calibration_evaluable"].sum()),
        "letter_calibration_excluded_blocks": int((~blocks["letter_calibration_evaluable"]).sum()),
        "eligible_wrapped_pairs": int(
            pairs.loc[pairs["pair_eligible"], ["item_id", "wrapper_name", "contract"]]
            .drop_duplicates()
            .shape[0]
        ),
        "exclusion_counts": exclusion_counts,
    }


def _timing_distributions(timing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (timing_type, contract, variant), group in timing.groupby(
        ["timing_type", "contract", "score_variant"], sort=True
    ):
        layers = pd.to_numeric(group["layer"], errors="coerce").dropna().astype(float)
        margins = pd.to_numeric(group["min_subsequent_margin"], errors="coerce").dropna().astype(float)
        rows.append(
            {
                "timing_type": timing_type,
                "contract": contract,
                "score_variant": variant,
                "n_total": int(len(group)),
                "n_defined": int(len(layers)),
                "defined_rate": float(len(layers) / len(group)),
                "q25_layer": float(layers.quantile(0.25)) if len(layers) else float("nan"),
                "median_layer": float(layers.median()) if len(layers) else float("nan"),
                "q75_layer": float(layers.quantile(0.75)) if len(layers) else float("nan"),
                "mean_min_subsequent_margin": float(margins.mean()) if len(margins) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _write_trajectory_figure(
    trajectories: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {"letter_raw": "#1f77b4", "candidate_total": "#d62728"}
    labels = {"letter_raw": "A/B/C/D", "candidate_total": "candidate path total"}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for column, contract in enumerate(("letter", "text")):
            agreement_axis = axes[0, column]
            for variant in ("letter_raw", "candidate_total"):
                data = trajectories[
                    (trajectories["contract"] == contract)
                    & (trajectories["score_variant"] == variant)
                ].sort_values("layer")
                agreement_axis.plot(
                    data["layer"], data["agreement"], color=colors[variant], label=labels[variant]
                )
                agreement_axis.fill_between(
                    data["layer"].to_numpy(dtype=float),
                    data["simultaneous_ci_low"].to_numpy(dtype=float),
                    data["simultaneous_ci_high"].to_numpy(dtype=float),
                    color=colors[variant],
                    alpha=0.14,
                    linewidth=0,
                )
            agreement_axis.axvline(31, color="0.5", linewidth=0.8, linestyle=":")
            agreement_axis.set_ylim(0.0, 1.0)
            agreement_axis.set_ylabel("Plain/wrapped winner agreement")
            agreement_axis.set_title(f"{contract.capitalize()} output contract")
            agreement_axis.grid(alpha=0.2)
            agreement_axis.legend(loc="lower left", frameon=False)

            diagnostic_axis = axes[1, column]
            margin_axis = diagnostic_axis.twinx()
            for variant in ("letter_raw", "candidate_total"):
                data = diagnostics[
                    (diagnostics["contract"] == contract)
                    & (diagnostics["score_variant"] == variant)
                ].sort_values("layer")
                diagnostic_axis.plot(
                    data["layer"],
                    data["mean_jsd"],
                    color=colors[variant],
                    label=f"{labels[variant]} JSD",
                )
                margin_axis.plot(
                    data["layer"],
                    data["plain_mean_margin"],
                    color=colors[variant],
                    linestyle="--",
                    alpha=0.65,
                    label=f"{labels[variant]} plain margin",
                )
                margin_axis.plot(
                    data["layer"],
                    data["wrapped_mean_margin"],
                    color=colors[variant],
                    linestyle=":",
                    alpha=0.9,
                    label=f"{labels[variant]} wrapped margin",
                )
            diagnostic_axis.axvline(31, color="0.5", linewidth=0.8, linestyle=":")
            diagnostic_axis.set_xlabel("Transformer layer")
            diagnostic_axis.set_ylabel("Restricted four-way JSD")
            margin_axis.set_ylabel("Top-two log-probability margin")
            diagnostic_axis.grid(alpha=0.2)
            diagnostic_axis.legend(loc="upper left", frameon=False)
            margin_axis.legend(loc="upper right", frameon=False)

        figure.suptitle("Mistral two-contract logit-lens trajectories", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(
            output_path,
            dpi=160,
            facecolor="white",
            metadata={"Software": "interface-formatting-study"},
        )
        plt.close(figure)


def analyze_frame(frame: pd.DataFrame, *, n_boot: int = 5000, seed: int = 1729) -> dict[str, object]:
    table = _validate_frame(frame)
    pairs = _pair_rows(table)
    score_columns = _active_score_columns(table)
    agreements = _agreement_rows(pairs, score_columns)
    pair_aucs = _pair_aucs(agreements)
    primary = _contrast_rows(
        pair_aucs, "candidate_total", n_boot=n_boot, seed=seed
    )
    primary["holm_p_value"] = _holm(primary["p_value"].tolist())
    secondary = pd.concat(
        [
            _contrast_rows(pair_aucs, variant, n_boot=n_boot, seed=seed)
            for variant in ("candidate_mean", "candidate_first_token")
        ],
        ignore_index=True,
    )
    all_trajectories = _trajectory_summary(agreements, n_boot=n_boot, seed=seed)
    calibrated_letter = all_trajectories[
        all_trajectories["score_variant"] == "letter_calibrated"
    ].copy()
    if not calibrated_letter.empty:
        calibrated_auc = _normalized_auc(
            calibrated_letter.set_index("layer").loc[list(PRE_FINAL_LAYERS), "agreement"]
        )
        calibrated_letter["agreement_auc"] = calibrated_auc
    trajectories = all_trajectories[
        all_trajectories["score_variant"] != "letter_calibrated"
    ].reset_index(drop=True)
    diagnostics = _diagnostic_summary(pairs, score_columns)
    timing = _timing_summary(table)
    strata = _strata_summary(pairs, pair_aucs, n_boot=n_boot, seed=seed)
    population_receipts = _population_receipts(table, pairs)
    quality_gates = evaluate_quality_gates(
        primary,
        secondary,
        diagnostics,
        population_items=int(population_receipts["items"]),
    )
    ungated_conclusion = classify_conclusion(primary, secondary, strata)
    conclusion = (
        ungated_conclusion if quality_gates["all_passed"] else "uninformative"
    )
    quality_gates["ungated_conclusion"] = ungated_conclusion
    quality_gates["final_conclusion"] = conclusion
    quality_gates["positive_interpretation_allowed"] = bool(
        quality_gates["all_passed"] and conclusion != "uninformative"
    )
    summary = {
        "analysis_version": "decision_binding_logit_lens_v1",
        "bootstrap_samples": int(n_boot),
        "bootstrap_seed": int(seed),
        "pre_final_layers": list(PRE_FINAL_LAYERS),
        "final_parity_layer": 31,
        "conclusion": conclusion,
        "quality_gates_passed": bool(quality_gates["all_passed"]),
        "claim_boundary": (
            "Descriptive intermediate-state decodability only; not causal control, literal thought, "
            "factual understanding, or a unique mechanism."
        ),
        **population_receipts,
    }
    return {
        "summary": summary,
        "primary_contrasts": primary,
        "secondary_contrasts": secondary,
        "trajectories": trajectories,
        "calibrated_letter": calibrated_letter,
        "diagnostics": diagnostics,
        "timing": timing,
        "strata": strata,
        "quality_gates": quality_gates,
    }


def _interpretation_memo(
    summary: Mapping[str, object], gates: Mapping[str, object]
) -> str:
    gate_status = "passed" if gates["all_passed"] else "did not all pass"
    failed = sorted(
        str(name)
        for name, receipt in gates["checks"].items()
        if not receipt["passed"]
    )
    failure_sentence = "" if not failed else f" Failed gates: {', '.join(failed)}."
    return (
        "# Bounded Interpretation\n\n"
        f"Predeclared conclusion: `{summary['conclusion']}`.\n\n"
        f"The frozen quality gates {gate_status}.{failure_sentence} Any positive pattern "
        "label is suppressed "
        "unless primary estimates are resolved, coverage is adequate, total and token-mean "
        "scores agree in direction under both contracts, and final-layer numerical ambiguity "
        "stays below the predeclared limit.\n\n"
        "This experiment is descriptive evidence about trajectories exposed by the final "
        "normalization and output head. It does not establish causation, literal thought, "
        "factual understanding, behavioral control, or a unique mechanism. Full candidate "
        "paths measure teacher-forced continuation compatibility; they are not information "
        "contained entirely at the original Answer: position.\n"
    )


def analyze(input_path: Path, output_dir: Path, *, n_boot: int = 5000, seed: int = 1729) -> dict[str, Path]:
    result = analyze_frame(pd.read_parquet(input_path), n_boot=n_boot, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "analysis_summary": output_dir / "analysis_summary.json",
        "interpretation_memo": output_dir / "interpretation_memo.md",
        "quality_gates": output_dir / "quality_gates.json",
        "calibrated_letter": output_dir / "calibrated_letter.csv",
        "diagnostics": output_dir / "diagnostics.csv",
        "primary_contrasts": output_dir / "primary_contrasts.csv",
        "secondary_contrasts": output_dir / "secondary_contrasts.csv",
        "trajectories": output_dir / "trajectories.csv",
        "timing": output_dir / "timing.csv",
        "timing_distributions": output_dir / "timing_distributions.csv",
        "trajectory_figure": output_dir / "trajectory_2x2.png",
        "strata": output_dir / "strata.csv",
    }
    outputs["analysis_summary"].write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["quality_gates"].write_text(
        json.dumps(result["quality_gates"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["interpretation_memo"].write_text(
        _interpretation_memo(result["summary"], result["quality_gates"]),
        encoding="utf-8",
    )
    for name in (
        "calibrated_letter",
        "diagnostics",
        "primary_contrasts",
        "secondary_contrasts",
        "trajectories",
        "timing",
        "strata",
    ):
        result[name].to_csv(outputs[name], index=False)
    _timing_distributions(result["timing"]).to_csv(outputs["timing_distributions"], index=False)
    _write_trajectory_figure(
        result["trajectories"], result["diagnostics"], outputs["trajectory_figure"]
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Verified merged layerwise_scores.parquet")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    outputs = analyze(args.input, args.output_dir, n_boot=args.bootstrap_samples, seed=args.seed)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
