#!/usr/bin/env python3
"""Assemble the supported FORM manuscript figures and tables from existing result leaves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_leaf(source: Path, destination: Path, content_id: str, kind: str, status: str, root: Path, config_sha256: str, rows: list[dict]) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required {content_id} leaf is missing: {source}")
    producer = {
        "m_method": "form_method.py",
        "b_group_level": "form_validation_b_group_level.py",
        "c_backfit": "form_validation_c_backfit.py",
        "d_descriptive": "form_validation_d_descriptive.py",
    }.get(source.relative_to(root).parts[0], "unknown")
    shutil.copy2(source, destination)
    rows.append({"content_id": content_id, "kind": kind, "file": str(destination.relative_to(destination.parents[1])),
                 "status": status, "producer": producer, "config_sha256": config_sha256,
                 "source": str(source), "source_sha256": _sha256(source),
                 "output_sha256": _sha256(destination)})


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_sha256 = _sha256(args.config.resolve())
    root = (args.config.parent / cfg["output_root"]).resolve()
    final = root / "f_final_figures_and_tables"
    staging = root / ".f_final_figures_and_tables_staging"
    if staging.exists():
        shutil.rmtree(staging)
    if final.exists() and any(final.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{final} is non-empty; use --overwrite after reviewing it.")
    if final.exists() and args.overwrite:
        shutil.rmtree(final)
    final = staging
    final.mkdir(parents=True, exist_ok=True)
    figures, tables, metadata = (final / name for name in ("figures", "tables", "metadata"))
    for directory in (figures, tables, metadata):
        directory.mkdir(exist_ok=True)
    manifest: list[dict] = []

    leaves = [
        ("F01", "figure", root / "m_method/figure_literature_mds.svg", figures / "F01_literature_mds.svg", "implemented; MATLAB-like s-stress MDS selected view and Custo A-E maps"),
        ("F03", "figure", root / "m_method/figure_spatial_spectrum_selectivity.svg", figures / "F03_spatial_spectrum_selectivity.svg", "implemented; controlled fixed-FORM spectral simulation"),
        ("F04", "figure", root / "m_method/figure_literature_conformity.svg", figures / "F04_literature_conformity.svg", "implemented"),
        ("F05", "figure", root / "b_group_level/figure_F05_combined.svg", figures / "F05_combined_criteria_and_correspondence.svg", "implemented; recording-wise criteria and corrected full-montage similarity"),
        ("F06", "figure", root / "d_descriptive/group/figure_lemon_conformity.svg", figures / "F06_lemon_conformity.svg", "implemented"),
        ("F07", "figure", root / "m_method/orientation_vectors_isometric.svg", figures / "F07_orientation_vectors_isometric.svg", "implemented; isometric view"),
        ("F07", "figure", root / "m_method/orientation_vectors_lateral.svg", figures / "F07_orientation_vectors_lateral.svg", "implemented; lateral view"),
        ("F07", "figure", root / "m_method/orientation_vectors_top.svg", figures / "F07_orientation_vectors_top.svg", "implemented; top view"),
        ("F07", "figure", root / "m_method/orientation_field_family.svg", figures / "F07_orientation_field_family.svg", "implemented; historical dense grid"),
        ("F07", "figure", root / "m_method/orientation_literature_angles.svg", figures / "F07_orientation_literature_angles.svg", "implemented; canonical topomaps at right"),
    ]
    for content_id, kind, source, destination, status in leaves:
        copy_leaf(source, destination, content_id, kind, status, root, config_sha256, manifest)

    copy_leaf(
        root / "d_descriptive/group/figure_td_gfp_example_sub-010002_EC.svg",
        figures / "F08_td_gfp_example.svg", "F08", "figure",
        "implemented; fixed Stage-D EC exemplar", root, config_sha256, manifest,
    )
    copy_leaf(
        root / "d_descriptive/group/figure_td_gfp_example_sub-010002_EC_fits.csv",
        tables / "F08_td_gfp_example_fits.csv", "F08", "table",
        "unrounded Stage-D source fits", root, config_sha256, manifest,
    )
    copy_leaf(
        root / "d_descriptive/group/figure_td_gfp_example_sub-010002_EC_metadata.json",
        metadata / "F08_td_gfp_example_metadata.json", "F08", "metadata",
        "Stage-D figure provenance", root, config_sha256, manifest,
    )

    for model in ("M1", "M2", "M3"):
        copy_leaf(root / f"d_descriptive/group/figure_assignment_ceiling_{model}.svg", figures / f"F09_assignment_ceiling_{model}.svg", "F09", "figure", "implemented; model explicit", root, config_sha256, manifest)
        mad_status = "implemented; raw correlation-scale MAD (primary)" if model == "M2" else "implemented; model-native MAD (sensitivity)"
        copy_leaf(root / f"d_descriptive/group/figure_beta_mad_{model}_model_native.svg", figures / f"F10_beta_mad_{model}.svg", "F10", "figure", mad_status, root, config_sha256, manifest)

    table_leaves = [
        ("T01", root / "c_backfit/gev/gev_summary.csv", tables / "T01_gev_summary.csv", "canonical conventional GEV source table"),
        ("S03", root / "c_backfit/gev/table_S3_global_explained_variance.csv", tables / "S03_global_explained_variance.csv", "six-row Supplementary Table S3 source"),
        ("T02", root / "d_descriptive/group/table_td_gfp_group.html", tables / "T02_td_gfp_group.html", "participant empirical intervals and Bonferroni descriptive counts"),
        ("T02", root / "d_descriptive/group/td_gfp_component_summary.csv", tables / "T02_td_gfp_component_summary.csv", "source summary"),
        ("T02", root / "d_descriptive/group/td_gfp_participant_fits.csv", tables / "T02_td_gfp_participant_fits.csv", "unrounded source rows"),
        ("F04", root / "m_method/literature_form_map_rows.csv", tables / "F04_literature_form_map_rows.csv", "unrounded source rows"),
        ("F04", root / "m_method/literature_rho2_table_k4_k8.csv", tables / "F04_literature_rho2_table_k4_k8.csv", "paper table: literature summaries and exact meta rho squared"),
        ("F04", root / "m_method/literature_rho2_table_k5.csv", tables / "F04_literature_rho2_table_k5.csv", "paper table: K5 literature summaries and exact meta rho squared"),
        ("F04", root / "m_method/literature_rho2_summary_k4_k8.csv", tables / "F04_literature_rho2_summary_k4_k8.csv", "unrounded K4-K8 source statistics"),
        ("F06", root / "d_descriptive/group/group_map_fit_r2.csv", tables / "F06_group_map_fit_r2.csv", "unrounded source rows"),
        ("F09", root / "d_descriptive/group/beta1_model_comparison.csv", tables / "F09_F10_beta_model_comparison.csv", "unrounded source rows"),
    ]
    for content_id, source, destination, status in table_leaves:
        copy_leaf(source, destination, content_id, "table", status, root, config_sha256, manifest)

    for name in ("FORM_RERUN_AUDIT.md", "FORM_PLAN_REAUDIT.md"):
        source = SCRIPT_DIR / name
        if source.is_file():
            shutil.copy2(source, metadata / name)
    with (metadata / "publication_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["content_id", "kind", "file", "status", "producer", "config_sha256", "source", "source_sha256", "output_sha256"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(manifest)
    (metadata / "assembly_provenance.json").write_text(json.dumps({
        "config_file": str(args.config.resolve()), "config_sha256": config_sha256,
        "leaf_count": len(manifest),
    }, indent=2) + "\n", encoding="utf-8")
    final.rename(root / "f_final_figures_and_tables")
    print(f"Wrote {root / 'f_final_figures_and_tables'}")


if __name__ == "__main__":
    main()
