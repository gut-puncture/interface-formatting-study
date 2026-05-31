from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

REQUIRED_FINAL_ARTIFACTS = [
    "results/metadata/dataset_audit.json",
    "results/metadata/wrapper_audit.csv",
    "results/metadata/split_manifest.json",
    "results/metadata/experiment_manifest.json",
    "results/raw/behavioral_scores.parquet",
    "results/processed/conflict_pairs.parquet",
    "results/processed/attention_diagnostics.parquet",
    "results/processed/vanilla_convergence.parquet",
    "results/processed/focused_patching_controls.parquet",
    "results/tables/table1_dataset_wrapper_audit.csv",
    "results/tables/table2_behavioral_conflicts.csv",
    "results/tables/table_attention_diagnostics.csv",
    "results/tables/table_vanilla_convergence.csv",
    "results/tables/table_focused_controls.csv",
    "results/figures/wrapper_accuracy_heatmap.png",
    "results/figures/format_conflict_distribution.png",
    "paper/main.tex",
    "paper/references.bib",
]

LEGACY_PAPER_ARTIFACTS = [
    "paper/tables/table3_semantic_patching.csv",
    "paper/tables/table3_semantic_patching.tex",
]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_final_artifacts(root: str | Path = ".") -> list[str]:
    base = Path(root)
    return [path for path in REQUIRED_FINAL_ARTIFACTS if not (base / path).exists()]


def _require_non_null(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{name} has null provenance values in {column}")


def validate_final_artifacts(root: str | Path = ".") -> None:
    base = Path(root)
    missing = missing_final_artifacts(base)
    if missing:
        raise FileNotFoundError("Missing final artifacts:\n" + "\n".join(missing))

    manifest = json.loads((base / "results/metadata/experiment_manifest.json").read_text())
    if not manifest.get("dataset", {}).get("audit"):
        raise ValueError("experiment_manifest.json is missing dataset audit metadata")
    if not manifest.get("split_manifest"):
        raise ValueError("experiment_manifest.json is missing split metadata")

    expected_rows = int(manifest["dataset"]["audit"]["num_rows"])
    behavioral = pd.read_parquet(base / "results/raw/behavioral_scores.parquet")
    if len(behavioral) != expected_rows:
        raise ValueError(f"behavioral_scores.parquet has {len(behavioral)} rows; expected {expected_rows}")
    attention = pd.read_parquet(base / "results/processed/attention_diagnostics.parquet")
    if attention.empty:
        raise ValueError("attention_diagnostics.parquet is empty")
    convergence = pd.read_parquet(base / "results/processed/vanilla_convergence.parquet")
    if convergence.empty:
        raise ValueError("vanilla_convergence.parquet is empty")
    controls = pd.read_parquet(base / "results/processed/focused_patching_controls.parquet")
    resolved_controls = controls[~controls.get("skipped", pd.Series(False, index=controls.index)).astype(bool)]
    if resolved_controls.empty:
        raise ValueError("focused_patching_controls.parquet has no resolved control rows")
    required_conditions = {
        "same_item_clean_to_corrupt",
        "vanilla_to_corrupt",
        "cross_item_clean_to_corrupt",
        "same_label_cross_item_clean_to_corrupt",
        "different_label_cross_item_clean_to_corrupt",
    }
    observed_conditions = set(resolved_controls["condition"].astype(str))
    missing_conditions = required_conditions - observed_conditions
    if missing_conditions:
        raise ValueError(f"focused_patching_controls.parquet is missing conditions: {sorted(missing_conditions)}")
    allowed_anchors = {"options_end", "all_option_ends"}
    observed_anchors = set(resolved_controls["anchor"].astype(str))
    if not observed_anchors.issubset(allowed_anchors):
        raise ValueError(f"focused controls include non-focused anchors: {sorted(observed_anchors - allowed_anchors)}")

    table1 = pd.read_csv(base / "results/tables/table1_dataset_wrapper_audit.csv")
    if set(table1.get("behavioral_run_type", [])) != {"final"}:
        raise ValueError("table1_dataset_wrapper_audit.csv is not marked as final")
    table2 = pd.read_csv(base / "results/tables/table2_behavioral_conflicts.csv")
    if set(table2.get("run_type", [])) != {"final"}:
        raise ValueError("table2_behavioral_conflicts.csv is not marked as final")
    for table_path in [
        "results/tables/table_attention_diagnostics.csv",
        "results/tables/table_vanilla_convergence.csv",
        "results/tables/table_focused_controls.csv",
    ]:
        table = pd.read_csv(base / table_path)
        if table.empty:
            raise ValueError(f"{table_path} is empty")

    artifact_hashes = manifest.get("artifact_hashes", {})
    for relative_path in REQUIRED_FINAL_ARTIFACTS:
        if relative_path == "results/metadata/experiment_manifest.json":
            continue
        expected_hash = artifact_hashes.get(relative_path)
        if expected_hash is None:
            raise ValueError(f"experiment_manifest.json is missing artifact hash for {relative_path}")
        actual_hash = _sha256_file(base / relative_path)
        if actual_hash != expected_hash:
            raise ValueError(f"artifact hash mismatch for {relative_path}")


def _stage_csv_as_latex(src: Path, dest: Path) -> None:
    frame = pd.read_csv(src)
    latex = frame.to_latex(index=False, escape=True)
    dest.write_text(latex)


def stage_paper_assets(root: str | Path = ".") -> None:
    base = Path(root)
    for relative_path in LEGACY_PAPER_ARTIFACTS:
        stale = base / relative_path
        if stale.exists():
            stale.unlink()
    figure_paths = [
        "results/figures/wrapper_accuracy_heatmap.png",
        "results/figures/format_conflict_distribution.png",
    ]
    table_paths = [
        "results/tables/table1_dataset_wrapper_audit.csv",
        "results/tables/table2_behavioral_conflicts.csv",
        "results/tables/table_attention_diagnostics.csv",
        "results/tables/table_vanilla_convergence.csv",
        "results/tables/table_focused_controls.csv",
    ]
    for relative_path in figure_paths:
        src = base / relative_path
        dest = base / "paper/figures" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    for relative_path in table_paths:
        src = base / relative_path
        dest = base / "paper/tables" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        _stage_csv_as_latex(src, dest.with_suffix(".tex"))


def build_pdf(root: str | Path = ".") -> bool:
    base = Path(root)
    if shutil.which("latexmk"):
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=base / "paper", check=True)
        return True
    if shutil.which("pdflatex"):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=base / "paper", check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=base / "paper", check=True)
        return True
    raise RuntimeError("PDF compilation requested, but neither latexmk nor pdflatex is available")


def prepare_deliverables(*, root: str | Path = ".", compile_pdf: bool = True) -> dict[str, object]:
    validate_final_artifacts(root)
    stage_paper_assets(root)
    pdf_built = build_pdf(root) if compile_pdf else False
    return {"staged_assets": True, "pdf_built": pdf_built}
