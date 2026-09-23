"""Write all Stage-D descriptive FORM outputs.

Run after backfitting. D reads the canonical subject caches for map-dependent
tables and figures and optionally runs the spatial-frequency and finite-template rho-ceiling descriptive branch. That branch rebuilds
the recording geometry only to verify and summarize the canonical FORM cache;
they do not modify upstream artifacts.
"""

import argparse
import hashlib
import atexit
import json
import shutil
from pathlib import Path
import threading
from traceback import format_exc

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import yaml
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import form_method_utils as meed
import form_validation_abc_utils as mvu
import form_validation_d_utils as mvdu


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yml"
D_ANALYSIS_SCHEMA = "form-empirical-analysis-v3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_ram(n_bytes: int) -> str:
    return f"{n_bytes / 1024 ** 3:.2f} GiB"


def _start_ram_monitor(interval_seconds: float):
    """Periodically report main and recursively spawned-process RSS."""
    stop_event = threading.Event()
    main_process = psutil.Process()

    def report():
        while not stop_event.is_set():
            try:
                worker_rss = 0
                worker_count = 0
                for process in main_process.children(recursive=True):
                    try:
                        worker_rss += process.memory_info().rss
                        worker_count += 1
                    except psutil.Error:
                        pass
                system_memory = psutil.virtual_memory()
                tqdm.write(
                    "RAM | "
                    f"main={_format_ram(main_process.memory_info().rss)} | "
                    f"spawned={_format_ram(worker_rss)} ({worker_count} processes) | "
                    f"system={_format_ram(system_memory.used)}/{_format_ram(system_memory.total)}"
                )
            except psutil.Error:
                pass
            stop_event.wait(max(float(interval_seconds), 0.1))

    thread = threading.Thread(target=report, name="ram-monitor", daemon=True)
    thread.start()
    atexit.register(stop_event.set)
    return stop_event, thread


def _work_realdata(
    row,
    *,
    meta_json: str,
    stage_dir: str,
    sfreq_expected: float,
    sfreq_tolerance: float,
    interleaved_block_seconds: float,
    interleaved_n_blocks: int,
    interleaved_first_condition: str,
    base_k: int,
    map_k_min: int,
    map_k_max: int,
    map_source: str,
    group_maps_file: str,
    time_start_s: float,
    time_end_s: float,
    write_subject_outputs: bool,
    write_exemplary_plots: bool,
    recompute: bool,
):
    recording_id = str(row["recording_id"])
    subject = str(row["subject"])
    condition = str(row["condition"])
    eeg_path = str(row["fif_path"])
    source_layout = str(row.get("source_layout", "condition_files"))
    recording_dir = Path(stage_dir) / "recordings" / recording_id
    map_fit_csv = recording_dir / f"{recording_id}_map_fit_r2.csv"
    summary_file = recording_dir / f"{recording_id}_summary.json"

    status = {
        "recording_id": recording_id,
        "subject": subject,
        "condition": condition,
        "eeg_path": eeg_path,
        "sample_cache_file": str(row["sample_cache_file"]),
        "model_dir": str(row["model_dir"]),
        "recording_dir": str(recording_dir),
        "map_fit_csv": str(map_fit_csv),
        "summary_file": str(summary_file),
        "success": None,
        "error": None,
        "traceback": None,
    }
    core_required = [
        map_fit_csv, summary_file,
        recording_dir / f"{recording_id}_sensor_maps.csv",
        recording_dir / f"{recording_id}_dipole_maps.csv",
    ]
    if write_subject_outputs:
        core_required.extend([
            recording_dir / "figure_1A_gfp_td_relationships.png",
            recording_dir / "figure_1B_gfp_peak_td_relationships.png",
            recording_dir / "figure_1C_td_component_decomposition.svg",
            recording_dir / "item1c_layout_v10.txt",
            recording_dir / "figure_3A_microstate_likeness.png",
            recording_dir / "table_1_decomposition_qc.csv",
            recording_dir / "figure_1E_td_component_loglog_hexbin.svg",
            recording_dir / "table_3_target_summary.csv",
            recording_dir / "figure_1F_gfp_component_loglog_hexbin.svg",
            recording_dir / "item1_component_hexbin_layout_v2.txt",
        ])
    if write_exemplary_plots:
        core_required.extend([
            recording_dir / "figure_S3_exemplary_f_meed_transition_pairs.svg",
            recording_dir / "figure_S4_exemplary_gfp_rho_winning_corr_cases.svg",
        ])
    if not recompute and all(path.exists() for path in core_required):
        try:
            cached = json.loads(summary_file.read_text(encoding="utf-8"))
            if cached.get("success", True):
                status["n_analysis_samples"] = cached.get("n_analysis_samples", np.nan)
                status["success"] = True
                status["cache_reused"] = True
                return status
        except (OSError, json.JSONDecodeError):
            pass
    try:
        with threadpool_limits(limits=1):
            meta_level, template_ch_names = mvu.load_meta_level(meta_json, int(base_k))
            raw = mvu.load_analysis_raw(
                eeg_path,
                source_layout,
                condition,
                template_ch_names,
                sfreq_expected,
                sfreq_tolerance,
                interleaved_block_seconds,
                interleaved_n_blocks,
                interleaved_first_condition,
            )
            if map_source == "subject":
                levels_by_k = mvu.load_cached_levels(
                    row["model_dir"],
                    meta_json,
                    range(int(map_k_min), int(map_k_max) + 1),
                    raw.info,
                )
            else:
                if map_source == "group":
                    with np.load(group_maps_file, allow_pickle=False) as cache:
                        maps = cache["group_maps"].astype(float)
                        labels = cache["labels"].astype(str).tolist()
                    selected_level = mvu.build_template_level("group", maps, labels, raw.info)
                else:
                    selected_level = meta_level
                levels_by_k = {int(base_k): {"meta": meta_level, "subject": selected_level}}
            if int(base_k) not in levels_by_k:
                raise RuntimeError(
                    f"Cached recording model K={base_k} is unavailable for {recording_id}."
                )
            sample_df = mvdu.sample_cache_to_dataframe(
                row["sample_cache_file"],
                subject=subject,
                condition=condition,
                recording_id=recording_id,
            )
            selected_level = levels_by_k[int(base_k)]["subject"]
            sample_df = mvdu.add_selected_map_columns(
                sample_df,
                raw.get_data(picks=raw.ch_names),
                selected_level,
            )
            analysis_df = sample_df.loc[
                sample_df["time"].between(float(time_start_s), float(time_end_s), inclusive="both")
            ].copy()

            recording_dir.mkdir(parents=True, exist_ok=True)
            mvdu.export_sensor_and_dipole_maps(recording_id, levels_by_k, recording_dir)
            mvdu.export_map_fit_r2_table(recording_id, levels_by_k, recording_dir)

            if write_subject_outputs:
                mvdu.save_item_outputs(
                    recording_id,
                    raw,
                    analysis_df,
                    recording_dir,
                    write_exemplary_plots=bool(write_exemplary_plots),
                )

            for legacy_dir_name in [
                "map_tables",
                "item1_zanesco_td",
                "item2_theta_phi",
                "item3_nonredundancy",
                "exemplary_plots",
            ]:
                legacy_dir = recording_dir / legacy_dir_name
                if legacy_dir.exists():
                    shutil.rmtree(legacy_dir)

        status["n_analysis_samples"] = int(len(analysis_df))
        status["success"] = True
        summary = {
            key: value for key, value in status.items()
            if key not in {"eeg_path", "sample_cache_file", "model_dir", "recording_dir", "map_fit_csv", "summary_file"}
        }
        mvu.write_json(summary, summary_file)
    except Exception as error:
        status["success"] = False
        status["error"] = repr(error)
        status["traceback"] = format_exc()
    return status



def _work_closed_forms(
    row: dict,
    cfg: dict,
    base: Path,
    outdir: Path,
    group_template_sets: dict[int, tuple[np.ndarray, list[str]]] | None,
    *,
    recompute: bool,
    refresh_figures_subjects: bool,
) -> None:
    """Run the former Stage-F descriptive branches inside Stage D."""
    record = str(row["recording_id"])
    meta_json = base / Path(cfg["meta_json"])
    descriptive = cfg["d_descriptive"]
    record_dir = outdir / "recordings" / record
    beta1_cache_required = [
        record_dir / "beta1_condition_fits.csv",
        record_dir / "beta1_plot_samples.csv",
        record_dir / "beta1_candidate_map_qc.csv",
        record_dir / "beta1_native_samples.npz",
    ]
    beta1_figure_required = [
        record_dir / "figure_4A_rho_scaling.svg",
        *(record_dir / f"figure_4A_rho_scaling_{model}.svg" for model in ("M1", "M2", "M3")),
    ]
    spatial_cache_required = [
        record_dir / "spatial_frequency_summary.csv",
        record_dir / "spatial_frequency_validation.csv",
        record_dir / "spatial_frequency_layout_v2.txt",
    ]
    beta1_cache_complete = all(path.exists() for path in beta1_cache_required)
    spatial_cache_complete = all(path.exists() for path in spatial_cache_required)

    # Fast figure-refresh path: never reopen EEG or recompute FORM when the
    # numerical recording leaves already exist.
    if not recompute and beta1_cache_complete and spatial_cache_complete:
        if refresh_figures_subjects or not all(path.exists() for path in beta1_figure_required):
            mvdu.write_beta1_recording_figures(record_dir)
        return {"recording_id": record, "subject": str(row["subject"]), "beta1": mvdu.load_beta1_recording_outputs(record_dir)}
    base_k = int(descriptive["base_k"])
    _, channels = mvu.load_meta_level(meta_json, base_k)
    raw = mvu.load_analysis_raw(
        row["fif_path"], row["source_layout"], row["condition"], channels,
        cfg["sfreq_expected"], cfg["sfreq_tolerance"], cfg["interleaved_block_seconds"],
        cfg["interleaved_n_blocks"], cfg["interleaved_first_condition"],
    )
    # Match the Stage-A cache construction exactly when checking FORM geometry.
    subject_level = cfg["a_subject_level"]
    block_id, _, _, discontinuities = mvu.detect_condition_blocks(
        raw, subject_level["block_seconds"], subject_level["expected_blocks_per_condition"],
        subject_level["annotation_boundary_regex"],
    )
    metadata_file = Path(row["sample_cache_file"]).parent.parent / "metadata" / f"{record}_samplewise.json"
    with metadata_file.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    position = np.array([channel["loc"][:3] for channel in raw.info["chs"]], dtype=float)
    sensor_directions = position - np.asarray(metadata["sphere_origin_m"], dtype=float)[None, :]
    sensor_directions /= float(metadata["sphere_radius_m"])
    sensor_directions /= np.linalg.norm(sensor_directions, axis=1, keepdims=True)
    centered_directions = sensor_directions - sensor_directions.mean(axis=0, keepdims=True)
    inverse_sqrt, _ = meed.inv_sqrtm(centered_directions.T @ centered_directions)
    geometry = {"S": sensor_directions, "S_orth": centered_directions @ inverse_sqrt}
    core, unit_maps = mvu.compute_form_samplewise(
        raw.get_data(picks=raw.ch_names), geometry["S_orth"], block_id, discontinuities,
        subject_level["chunk_samples"],
    )
    cache = mvu.load_sample_cache(row["sample_cache_file"])
    cached_q = cache["form_q"].astype(float)
    cached_rho = cache["rho"].astype(float)
    sample_index = cache["sample_index"].astype(np.int64)
    time_s = cache["time_s"].astype(float)
    if np.max(np.abs(cached_q - core["form_q"])) > 2e-6 or np.max(np.abs(cached_rho - core["rho"])) > 2e-6:
        raise RuntimeError(f"{record}: canonical cache disagrees with recomputed FORM coordinates")

    record_dir.mkdir(parents=True, exist_ok=True)
    if group_template_sets is None:
        raise RuntimeError("Group-template sets were not loaded.")
    if recompute or not spatial_cache_complete:
        decomposition = mvdu.spatial_frequency_decomposition(
            unit_maps, geometry["S_orth"], geometry["S"], cached_rho ** 2,
            core["td2"], core["td2_form"], core["td2_off_form"], core["td2_rho"], core["td2_psi"],
            core["valid_transition"], lmax=int(descriptive["spatial_lmax"]),
        )
        summary = mvdu.summarize_spatial_frequency(decomposition, recording_id=record)
        validation = pd.DataFrame([{**decomposition["validation"], "recording_id": record}])
        summary.to_csv(record_dir / "spatial_frequency_summary.csv", index=False)
        validation.to_csv(record_dir / "spatial_frequency_validation.csv", index=False)
        fig = mvdu.plot_spatial_frequency_summary(summary)
        fig.savefig(record_dir / "figure_S1_spatial_frequency.png", dpi=600, bbox_inches="tight")
        (record_dir / "spatial_frequency_layout_v2.txt").write_text("S1 layout v2: physical-colour decomposition panels\n", encoding="utf-8")
        import matplotlib.pyplot as plt
        plt.close(fig)
    levels = {
        k: mvu.build_template_level("group_template", maps, labels, raw.info)
        for k, (maps, labels) in group_template_sets.items()
    }
    if recompute or not beta1_cache_complete:
        beta1_contributions = mvdu.write_beta1_recording_outputs(
            record_dir, levels, cached_q, cached_rho, unit_maps, sample_index, time_s,
            int(descriptive["beta1_plot_rows"]), subject=str(row["subject"]),
            condition=str(row["condition"]),
        )
    else:
        beta1_contributions = mvdu.load_beta1_recording_outputs(record_dir)
        if refresh_figures_subjects or not all(path.exists() for path in beta1_figure_required):
            mvdu.write_beta1_recording_figures(record_dir)
    return {"recording_id": record, "subject": str(row["subject"]), "beta1": beta1_contributions}


def _write_beta1_group_figures(group_dir: Path, outputs: dict, comparison: pd.DataFrame, model_tests: pd.DataFrame, residual_samples: pd.DataFrame) -> None:
    """Redraw group Figure 4 leaves without modifying cached tables."""
    import matplotlib.pyplot as plt

    for model in ("M1", "M2", "M3"):
        figure = mvdu.plot_beta1_group_model_scaling(outputs["render_samples"], model, n_rows=2, n_cols=6)
        path = group_dir / f"figure_assignment_ceiling_{model}.svg"
        figure.savefig(path, format="svg", bbox_inches=None)
        meed.assert_svg_width(path)
        if model == "M1":
            figure.savefig(group_dir / "figure_4A_rho_scaling.svg", bbox_inches="tight")
        plt.close(figure)
        figure = mvdu.plot_beta1_model_boxplots(comparison, model_tests, model)
        path = group_dir / f"figure_beta_mad_{model}_model_native.svg"
        figure.savefig(path, format="svg", bbox_inches=None)
        meed.assert_svg_width(path)
        mad_column = "raw_residual_mad" if model == "M2" else "model_native_residual_mad"
        (group_dir / f"figure_beta_mad_{model}_model_native.json").write_text(json.dumps({"model": model, "mad_column": mad_column, "fit_population": "full native subject/K comparison"}, indent=2) + "\n")
        plt.close(figure)
        figure = mvdu.plot_beta1_model_residual_diagnostics(residual_samples, model)
        figure.savefig(group_dir / f"figure_4C_beta1_residual_diagnostics_{model}.svg", bbox_inches="tight")
        plt.close(figure)

    for filename, figure in [
        ("figure_4B_beta1_by_k", mvdu.plot_beta1_group_figure(
            outputs["subject"], outputs["summary"], outputs["tests"]
        )),
        ("figure_4C_beta1_residual_diagnostics", mvdu.plot_beta1_residual_diagnostics(
            outputs["subject"], outputs["render_samples"]
        )),
    ]:
        figure.savefig(group_dir / f"{filename}.svg", bbox_inches="tight")
        plt.close(figure)

    # Group Figure 4 is SVG-only. Remove stale PNGs from older runs.
    for stale_png in group_dir.glob("figure_4*.png"):
        stale_png.unlink()


def _parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only-id", default=None)
    return parser.parse_args()


def main():
    cli = _parse_cli()
    with cli.config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    ram_stop, ram_thread = _start_ram_monitor(cfg["ram_monitor_interval_seconds"])

    base = cli.config.resolve().parent
    output_root = base / Path(cfg["output_root"])
    meta_json = base / Path(cfg["meta_json"])

    descriptive = cfg["d_descriptive"]
    recompute = bool(descriptive.get("recompute", False))
    refresh_figures_subjects = bool(descriptive.get("refresh_figures_subjects", False))
    refresh_figures_group = bool(descriptive.get("refresh_figures_group", False))
    map_source = "group"
    summaries_dir = output_root / "a_subject_level" / "summaries"
    group_maps_file = output_root / "b_group_level" / "group_maps_A-E.npz"
    if not summaries_dir.exists():
        raise FileNotFoundError(
            f"Canonical real-data summaries not found: {summaries_dir}. "
            "Run form_validation_a_subject_level.py first."
        )
    if map_source == "group" and not group_maps_file.exists():
        raise FileNotFoundError(f"Group maps not found: {group_maps_file}. Run B first.")
    print(f"[D] Loading subject-level summaries with {map_source} maps.")
    cache_index = mvu.filter_only_id(mvu.load_recording_jsons(summaries_dir, "*_summary.json"), cli.only_id)
    if cache_index.empty:
        raise RuntimeError("No cached recordings remain for real-data outputs.")

    # A summaries may have been created from a different workspace drive. Rebuild
    # filesystem locations from this configuration while retaining their cached
    # recording-level analysis values and model-selection results.
    data_root = base / Path(cfg["data_root"])
    exclude_ids = (
        base / Path(cfg["preclustering_exclude_ids"])
        if cfg.get("preclustering_exclude_ids")
        else None
    )
    inventory = mvu.discover_recordings(
        data_root,
        cfg["fif_glob"],
        cfg["subject_regex"],
        cfg["condition_regex"],
        cfg["input_layout"],
        cfg["interleaved_n_blocks"],
        cfg["interleaved_first_condition"],
        cfg["require_both_conditions"],
        exclude_ids,
        cfg.get("max_subjects", 0),
    )
    current_inputs = inventory.loc[inventory["included"], [
        "recording_id", "fif_path", "source_layout", "source_block_indices",
    ]]
    cache_index = cache_index.drop(
        columns=["fif_path", "source_layout", "source_block_indices", "sample_cache_file", "model_dir"],
        errors="ignore",
    ).merge(current_inputs, on="recording_id", how="left", validate="one_to_one")
    missing_inputs = cache_index["fif_path"].isna()
    if missing_inputs.any():
        missing_ids = cache_index.loc[missing_inputs, "recording_id"].tolist()
        raise RuntimeError(
            "Cached recordings are absent from the current configured FIF inventory: "
            + ", ".join(missing_ids[:10])
        )
    cache_index["sample_cache_file"] = cache_index["recording_id"].map(
        lambda recording_id: str(
            output_root / "a_subject_level" / "samples" / f"{recording_id}_samplewise.npz"
        )
    )
    cache_index["model_dir"] = cache_index["recording_id"].map(
        lambda recording_id: str(output_root / "a_subject_level" / "models" / recording_id)
    )

    # Generate recording-level diagnostics from the canonical sample/model caches.
    stage_dir = output_root / "d_descriptive"
    stage_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = 1 if cli.only_id is not None else mvu.effective_n_jobs(descriptive["n_jobs_recordings"], len(cache_index))
    write_exemplary = bool(descriptive["write_exemplary_plots"] or cli.only_id is not None)
    if descriptive.get("write_recording_outputs", True):
        print(f"[D] Writing descriptive outputs with {n_jobs} worker(s).")
        results = []
        with tqdm(total=len(cache_index), desc="D recordings", position=0) as progress:
            for result in Parallel(
                n_jobs=n_jobs,
                backend="loky",
                return_as="generator_unordered",
            )(
                delayed(_work_realdata)(
                    row,
                    meta_json=str(meta_json),
                    stage_dir=str(stage_dir),
                    sfreq_expected=cfg["sfreq_expected"],
                    sfreq_tolerance=cfg["sfreq_tolerance"],
                    interleaved_block_seconds=cfg["interleaved_block_seconds"],
                    interleaved_n_blocks=cfg["interleaved_n_blocks"],
                    interleaved_first_condition=cfg["interleaved_first_condition"],
                    base_k=descriptive["base_k"],
                    map_k_min=descriptive["map_k_min"],
                    map_k_max=descriptive["map_k_max"],
                    map_source=map_source,
                    group_maps_file=str(group_maps_file),
                    time_start_s=descriptive["time_start_s"],
                    time_end_s=descriptive["time_end_s"],
                    write_subject_outputs=descriptive["write_subject_outputs"],
                    write_exemplary_plots=write_exemplary,
                    recompute=bool(descriptive.get("recompute", False)),
                )
                for row in cache_index.to_dict(orient="records")
            ):
                results.append(result)
                progress.update()
        status = pd.DataFrame(results)
    else:
        print("[D] Skipping recording-level outputs; using canonical caches for cohort outputs.")
        status = cache_index.copy()
        status["eeg_path"] = status["fif_path"]
        status["success"] = True
        status["error"] = None

    status = status.sort_values(["subject", "condition"])
    successful = status.loc[status["success"].eq(True)].copy()

    group_dir = stage_dir / "group"
    analysis_provenance = group_dir / "d_analysis_provenance.json"
    analysis_identity = {
        "schema": D_ANALYSIS_SCHEMA,
        "config_sha256": _sha256_file(cli.config.resolve()),
        "group_maps_sha256": _sha256_file(group_maps_file),
        "stage_c_association_schema": mvdu.STAGE_C_ASSOCIATION_SCHEMA,
        "assignment_target": "instantaneous canonical pooled K=5 A-E correlation on Stage-C valid maps",
        "m2_primary_residual": "raw correlation-scale MAD of r - beta*rho",
        "m3_mean": "exact M2 beta with NNLS residual variance decomposition",
    }
    try:
        analysis_cache_current = json.loads(analysis_provenance.read_text(encoding="utf-8")) == analysis_identity
    except (OSError, json.JSONDecodeError):
        analysis_cache_current = False
    if descriptive["write_group_outputs"] and cli.only_id is None and not successful.empty:
        print("[D] Group outputs | starting cohort-level summaries.")
        map_fit_required = [
            group_dir / "group_map_fit_r2.csv", group_dir / "condition_group_maps_raw.npz",
            group_dir / "condition_to_group_matching.csv", group_dir / "condition_group_maps_A-E.npz",
            group_dir / "figure_condition_to_group_matching_diagnostic.svg", group_dir / "figure_condition_group_raw_maps.svg",
        ]
        item3_required = [
            group_dir / "table_3_within_subject_model_comparison.csv", group_dir / "table_3_loso_model_comparison.csv",
            group_dir / "table_3_softmax_lambda_sweep.csv", group_dir / "table_3_sample_predictions_loso.csv",
            group_dir / "table_3D_group_exemplar_samples.csv", group_dir / "figure_3B_winning_template_prediction_performance.svg",
            group_dir / "figure_3C_softmax_confidence_sweep.svg", group_dir / "figure_3D_winning_template_exemplar_topomaps.svg",
        ]
        if map_source == "group":
            group = cfg["b_group_level"]
            if recompute or not all(path.exists() for path in map_fit_required):
                print("[D] Group outputs | fitting EC/EO-specific K=5 maps and writing map-fit outputs.")
                mvdu.save_group_map_fit_r2_outputs(
                    successful, group_dir, meta_json=meta_json, group_maps_file=group_maps_file,
                    selected_k=int(group["selected_k"]), n_init=int(group["n_init"]), max_iter=int(group["max_iter"]),
                    tol=float(group["tol"]), random_seed=int(cfg["random_seed"]), n_jobs=int(group["n_jobs_clustering"]),
                )
            else:
                print("[D] Group outputs | redrawing F06 from valid map-fit leaves.")
                with np.load(group_dir / "condition_group_maps_A-E.npz") as maps_file:
                    condition_maps = {condition: maps_file[condition] for condition in ("EC", "EO")}
                meta_level, _ = mvdu.load_meta_level(meta_json, int(descriptive["base_k"]))
                figure = mvdu.plot_group_map_fit_r2_summary(pd.read_csv(group_dir / "group_map_fit_r2.csv"), condition_maps=condition_maps, info=meta_level.info)
                figure.savefig(group_dir / "figure_lemon_conformity.svg", format="svg", bbox_inches=None)
                meed.assert_svg_width(group_dir / "figure_lemon_conformity.svg")
                plt.close(figure)
        if recompute or not analysis_cache_current or not all(path.exists() for path in item3_required):
            print("[D] Group outputs | writing Item 3 model-comparison outputs.")
            mvdu.save_group_item3_ml_outputs(successful, group_dir, progress=lambda message: print(f"[D] Group Item 3 | {message}"))
        else:
            print("[D] Group outputs | reusing complete Item 3 leaves.")
        # Publication leaves consume existing sample caches and never refit K-means.
        td_fits = mvdu.write_td_gfp_group_outputs(successful.to_dict("records"), group_dir)
        example = successful.loc[successful["recording_id"].eq("sub-010002_EC")]
        if not example.empty:
            row = example.iloc[0]
            sample_df = mvdu.sample_cache_to_dataframe(row["sample_cache_file"], subject=str(row["subject"]), condition=str(row["condition"]), recording_id=str(row["recording_id"]))
            figure, fits, metadata = mvdu.plot_td_gfp_example(sample_df)
            path = group_dir / "figure_td_gfp_example_sub-010002_EC.svg"
            figure.savefig(path, format="svg", bbox_inches=None); meed.assert_svg_width(path); plt.close(figure)
            fits.to_csv(group_dir / "figure_td_gfp_example_sub-010002_EC_fits.csv", index=False)
            (group_dir / "figure_td_gfp_example_sub-010002_EC_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("[D] Group outputs | complete.")

    # Evaluate finite-dictionary ceiling fractions at the participant level.
    # Figure refresh remains independent from numerical recomputation.
    beta1_group_cache_required = [
        group_dir / "beta1_condition_fits.csv",
        group_dir / "beta1_subject_fits.csv",
        group_dir / "beta1_group_summary.csv",
        group_dir / "beta1_group_tests.csv",
        group_dir / "beta1_condition_summary.csv",
        group_dir / "beta1_candidate_map_qc.csv",
        group_dir / "beta1_model_comparison.csv",
        group_dir / "beta1_model_tests.csv",
    ]
    beta1_group_figure_required = [
        *(group_dir / f"figure_4A_rho_scaling_{model}.svg" for model in ("M1", "M2", "M3")),
        *(group_dir / f"figure_4B_beta1_by_k_{model}.svg" for model in ("M1", "M2", "M3")),
        *(group_dir / f"figure_4C_beta1_residual_diagnostics_{model}.svg" for model in ("M1", "M2", "M3")),
        group_dir / "figure_4A_rho_scaling.svg",
        group_dir / "figure_4B_beta1_by_k.svg",
        group_dir / "figure_4C_beta1_residual_diagnostics.svg",
    ]

    beta1_group_cache_complete = False
    if not recompute and analysis_cache_current and all(path.exists() for path in beta1_group_cache_required):
        try:
            beta1_existing_subjects = pd.read_csv(group_dir / "beta1_subject_fits.csv")
            beta1_model_comparison = pd.read_csv(group_dir / "beta1_model_comparison.csv")
            expected_subjects = set(cache_index["subject"].astype(str))
            expected_k = set(int(k) for k in descriptive["group_k_values"])
            beta1_group_cache_complete = (
                "model" in beta1_existing_subjects
                and beta1_existing_subjects["model"].eq("M2").all()
                and set(beta1_existing_subjects["subject"].astype(str)) == expected_subjects
                and set(beta1_existing_subjects["k"].astype(int)) == expected_k
                and len(beta1_existing_subjects) == len(expected_subjects) * len(expected_k)
                and set(beta1_model_comparison["subject"].astype(str)) == expected_subjects
                and set(beta1_model_comparison["k"].astype(int)) == expected_k
                and set(beta1_model_comparison["model"].astype(str)) == {"M1", "M2", "M3"}
                and len(beta1_model_comparison) == len(expected_subjects) * len(expected_k) * 3
            )
        except (OSError, ValueError, KeyError):
            beta1_group_cache_complete = False

    need_recording_pass = recompute or refresh_figures_subjects or not beta1_group_cache_complete
    all_contributions = None
    if not cache_index.empty and need_recording_pass:
        group_template_sets = mvdu.load_group_template_sets(cfg, base)
        all_contributions = []
        closed_rows = cache_index.sort_values(["subject", "condition"]).to_dict(orient="records")
        mode = "recomputing" if recompute or not beta1_group_cache_complete else "refreshing subject figures from cache"
        print(f"[D] Rho-ceiling beta1 | {mode} for {len(closed_rows)} recording-condition files with {n_jobs} worker(s).")
        with tqdm(total=len(closed_rows), desc="D rho-ceiling recordings", position=0) as progress:
            for result in Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
                delayed(_work_closed_forms)(
                    row, cfg, base, stage_dir, group_template_sets,
                    recompute=recompute or not analysis_cache_current,
                    refresh_figures_subjects=refresh_figures_subjects,
                )
                for row in closed_rows
            ):
                all_contributions.extend(result["beta1"])
                progress.update()

    need_group_compute = recompute or not analysis_cache_current or not beta1_group_cache_complete
    if cli.only_id is None and not cache_index.empty and need_group_compute:
        group_dir.mkdir(parents=True, exist_ok=True)
        if all_contributions is None:
            raise RuntimeError("Rho-ceiling group recomputation requires recording contributions.")
        mvdu.write_spatial_group_summary(stage_dir)
        outputs = mvdu.finalize_beta1_group_outputs(all_contributions)
        outputs["detailed"].to_csv(group_dir / "beta1_condition_fits.csv", index=False)
        outputs["subject"].to_csv(group_dir / "beta1_subject_fits.csv", index=False)
        outputs["summary"].to_csv(group_dir / "beta1_group_summary.csv", index=False)
        outputs["tests"].to_csv(group_dir / "beta1_group_tests.csv", index=False)
        outputs["condition_summary"].to_csv(group_dir / "beta1_condition_summary.csv", index=False)
        outputs["candidate_summary"].to_csv(group_dir / "beta1_candidate_map_qc.csv", index=False)
        comparison = mvdu.beta1_model_comparison_from_native(all_contributions)
        comparison.to_csv(group_dir / "beta1_model_comparison.csv", index=False)
        model_tests = mvdu.beta1_model_comparison_tests(comparison)
        model_tests.to_csv(group_dir / "beta1_model_tests.csv", index=False)
        residual_samples = mvdu.beta1_native_residual_render_samples(all_contributions, comparison)
        _write_beta1_group_figures(group_dir, outputs, comparison, model_tests, residual_samples)
        print("[D] Rho-ceiling beta1 | caches, tests, QC, and figures complete.")
    elif cli.only_id is None and not cache_index.empty and (
        refresh_figures_group or not all(path.exists() for path in beta1_group_figure_required)
    ):
        # Pure group-figure refresh: reuse saved recording and group numerical leaves.
        if all_contributions is None:
            all_contributions = []
            for row in cache_index.sort_values(["subject", "condition"]).to_dict(orient="records"):
                record_dir = stage_dir / "recordings" / str(row["recording_id"])
                all_contributions.extend(mvdu.load_beta1_recording_outputs(record_dir))
        outputs = mvdu.finalize_beta1_group_outputs(all_contributions)
        comparison = pd.read_csv(group_dir / "beta1_model_comparison.csv")
        model_tests = pd.read_csv(group_dir / "beta1_model_tests.csv")
        residual_samples = mvdu.beta1_native_residual_render_samples(all_contributions, comparison)
        _write_beta1_group_figures(group_dir, outputs, comparison, model_tests, residual_samples)
        print("[D] Rho-ceiling beta1 | refreshed group figures from cached numerical leaves.")
    elif cli.only_id is None and not cache_index.empty:
        print("[D] Rho-ceiling beta1 | reusing complete recording and group leaves.")

    failures = status.loc[status["success"].ne(True)]
    if cli.only_id is None and descriptive["write_group_outputs"] and failures.empty:
        group_dir.mkdir(parents=True, exist_ok=True)
        analysis_provenance.write_text(json.dumps(analysis_identity, indent=2) + "\n", encoding="utf-8")
    print(
        f"Real-data output stage complete: {len(successful)} successful, {len(failures)} failed. "
        f"Per-recording summaries are in d_descriptive/recordings/."
    )
    if not failures.empty:
        print(failures[["recording_id", "error"]].to_string(index=False))
    ram_stop.set()
    ram_thread.join(timeout=1)


if __name__ == "__main__":
    main()
