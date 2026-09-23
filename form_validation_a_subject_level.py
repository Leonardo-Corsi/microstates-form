"""Build the canonical per-recording cache shared by FORM real-data and effects analyses.

This is the only expensive recording-level stage. For every subject-condition it
loads the aligned preprocessed FIF once, caches GFP peaks, fits native
Pycrostates solutions K=1..12, computes the recording-intrinsic FORM/GFP/TD
samplewise quantities, and stores fixed reference-K meta/recording-map associations.
"""

import argparse
import atexit
import json
from pathlib import Path
import threading
from traceback import format_exc

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import psutil
import yaml
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import form_method_utils as meed
import form_validation_abc_utils as mvud


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yml"

CLUSTERING_IMPLEMENTATION = "Pycrostates ModKMeans"
SAMPLE_CACHE_SCHEMA = 4


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


def _load_completed_status(
    row,
    stage_dir: str,
    k_min: int,
    k_max: int,
    base_k: int,
    clustering_implementation: str,
):
    """Return cached status when one recording's complete artifact set exists."""
    recording_id = str(row["recording_id"])
    stage_path = Path(stage_dir)
    summary_file = stage_path / "summaries" / f"{recording_id}_summary.json"
    sample_file = stage_path / "samples" / f"{recording_id}_samplewise.npz"
    metadata_file = stage_path / "metadata" / f"{recording_id}_samplewise.json"
    peak_file = stage_path / "gfp_peaks" / f"{recording_id}_gfp_peaks.npz"
    criteria_file = stage_path / "criteria" / f"{recording_id}_criteria.csv"
    votes_file = stage_path / "votes" / f"{recording_id}_votes.csv"
    model_dir = stage_path / "models" / recording_id
    model_files = [model_dir / f"k-{k:02d}_modkmeans.fif" for k in range(k_min, k_max + 1)]
    required = [summary_file, sample_file, metadata_file, peak_file, criteria_file, votes_file, *model_files]
    if not all(path.exists() for path in required):
        return None
    with summary_file.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    with metadata_file.open("r", encoding="utf-8") as stream:
        sample_metadata = json.load(stream)
    selected_k = summary.get("selected_k")
    if (
        str(summary.get("recording_id")) != recording_id
        or not isinstance(selected_k, (int, float))
        or int(selected_k) not in range(k_min, k_max + 1)
        or summary.get("clustering_implementation") != clustering_implementation
        or int(sample_metadata.get("base_k", -1)) != int(base_k)
        or int(sample_metadata.get("sample_cache_schema", -1)) != SAMPLE_CACHE_SCHEMA
    ):
        return None
    status = dict(summary)
    status.update(
        fif_path=str(row["fif_path"]),
        source_layout=str(row.get("source_layout", "condition_files")),
        source_block_indices=str(row.get("source_block_indices", "")),
        sample_cache_file=str(sample_file), metadata_file=str(metadata_file),
        peak_file=str(peak_file), model_dir=str(model_dir), criteria_file=str(criteria_file),
        votes_file=str(votes_file), summary_file=str(summary_file), success=True,
        error=None, traceback=None, reused_artifacts=True,
    )
    return status


def _fit_recording_model(
    peak_maps: np.ndarray,
    *,
    meta_json: str,
    base_k: int,
    model_file: str,
    k: int,
    n_init: int,
    max_iter: int,
    tol: float,
    random_seed: int,
    recording_id: str,
    n_jobs_clustering: int,
):
    """Fit and save one native Pycrostates recording x K model."""
    with threadpool_limits(limits=1):
        meta = mvud.load_meta_template(meta_json, k=int(base_k))
        seed = mvud.stable_seed(random_seed, recording_id, "recording", k)
        model = mvud.fit_modkmeans(
            mvud.make_chdata(peak_maps, meta["info"]),
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            n_jobs=n_jobs_clustering,
        )
        mvud.save_cluster_model(model, model_file, overwrite=True)
    return k


def _ensure_peak_cache(
    row,
    *,
    meta_json: str,
    stage_dir: str,
    base_k: int,
    sfreq_expected: float,
    sfreq_tolerance: float,
    interleaved_block_seconds: float,
    interleaved_n_blocks: int,
    interleaved_first_condition: str,
    min_peak_distance: int,
    overwrite: bool,
):
    """Prepare one reusable GFP-peak node before global K-task scheduling."""
    recording_id = str(row["recording_id"])
    peak_file = Path(stage_dir) / "gfp_peaks" / f"{recording_id}_gfp_peaks.npz"
    meta = mvud.load_meta_template(meta_json, k=int(base_k))
    if peak_file.exists() and not overwrite:
        try:
            with np.load(peak_file, allow_pickle=False) as cache:
                channels = cache["ch_names"].astype(str).tolist()
                _ = cache["peak_maps"].shape
            if channels == meta["ch_names"]:
                return str(peak_file)
        except (KeyError, OSError, ValueError, EOFError):
            # A partial/corrupt archive is not a cache; rebuild it from FIF.
            pass
    raw = mvud.load_analysis_raw(
        str(row["fif_path"]),
        str(row.get("source_layout", "condition_files")),
        str(row["condition"]),
        meta["ch_names"],
        sfreq_expected,
        sfreq_tolerance,
        interleaved_block_seconds,
        interleaved_n_blocks,
        interleaved_first_condition,
    )
    _, peak_indices, peak_maps, continuous_gfp = mvud.extract_recording_gfp_peaks(
        raw, min_peak_distance
    )
    peak_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        peak_file,
        peak_indices=peak_indices,
        peak_maps=peak_maps.astype(np.float32),
        peak_gfp=continuous_gfp[peak_indices].astype(np.float32),
        ch_names=np.asarray(meta["ch_names"], dtype="U"),
        sfreq=np.asarray(float(raw.info["sfreq"])),
        n_samples=np.asarray(int(raw.n_times), dtype=np.int64),
        duration_seconds=np.asarray(float(raw.n_times / raw.info["sfreq"])),
    )
    return str(peak_file)


def _fit_recording_k_task(
    peak_file: str,
    **kwargs,
):
    """One global queue item: one recording x K model fit."""
    with np.load(peak_file, allow_pickle=False) as cache:
        peak_maps = cache["peak_maps"].astype(float)
    return _fit_recording_model(peak_maps, **kwargs)


def _work_cache(
    row,
    *,
    meta_json: str,
    stage_dir: str,
    sfreq_expected: float,
    sfreq_tolerance: float,
    interleaved_block_seconds: float,
    interleaved_n_blocks: int,
    interleaved_first_condition: str,
    k_min: int,
    k_max: int,
    n_init: int,
    max_iter: int,
    tol: float,
    n_jobs_clustering: int,
    min_peak_distance: int,
    criteria_corr_threshold: float,
    criteria_min_k: int,
    criteria_max_maps: int,
    criteria_max_pairs: int,
    base_k: int,
    block_seconds: float,
    expected_blocks: int,
    annotation_boundary_regex: str,
    chunk_samples: int,
    random_seed: int,
    overwrite: bool,
    models_ready: bool = False,
    reuse_peak_cache: bool = False,
):
    """Edge: aligned FIF -> recording models plus canonical samplewise cache."""
    recording_id = str(row["recording_id"])
    subject = str(row["subject"])
    condition = str(row["condition"])
    fif_path = str(row["fif_path"])
    source_layout = str(row.get("source_layout", "condition_files"))
    stage_path = Path(stage_dir)

    sample_file = stage_path / "samples" / f"{recording_id}_samplewise.npz"
    metadata_file = stage_path / "metadata" / f"{recording_id}_samplewise.json"
    peak_file = stage_path / "gfp_peaks" / f"{recording_id}_gfp_peaks.npz"
    model_dir = stage_path / "models" / recording_id
    criteria_file = stage_path / "criteria" / f"{recording_id}_criteria.csv"
    votes_file = stage_path / "votes" / f"{recording_id}_votes.csv"
    summary_file = stage_path / "summaries" / f"{recording_id}_summary.json"
    clustering_implementation = CLUSTERING_IMPLEMENTATION
    prior_summary = None
    if summary_file.exists():
        with summary_file.open("r", encoding="utf-8") as stream:
            prior_summary = json.load(stream)
    refresh_clustering_outputs = (
        overwrite
        or prior_summary is None
        or prior_summary.get("clustering_implementation") != clustering_implementation
    )

    status = {
        "recording_id": recording_id,
        "subject": subject,
        "condition": condition,
        "fif_path": fif_path,
        "source_layout": source_layout,
        "source_block_indices": str(row.get("source_block_indices", "")),
        "sample_cache_file": str(sample_file),
        "metadata_file": str(metadata_file),
        "peak_file": str(peak_file),
        "model_dir": str(model_dir),
        "criteria_file": str(criteria_file),
        "votes_file": str(votes_file),
        "summary_file": str(summary_file),
        "success": None,
        "error": None,
        "traceback": None,
    }

    try:
        with threadpool_limits(limits=1):
            meta = mvud.load_meta_template(meta_json, k=int(base_k))
            model_paths = {
                k: model_dir / f"k-{k:02d}_modkmeans.fif"
                for k in range(int(k_min), int(k_max) + 1)
            }
            raw = None

            if peak_file.exists() and (not overwrite or reuse_peak_cache):
                with np.load(peak_file, allow_pickle=False) as cache:
                    peak_indices = cache["peak_indices"].astype(np.int64)
                    peak_maps = cache["peak_maps"].astype(float)
                    n_samples = int(cache["n_samples"])
                    duration_seconds = float(cache["duration_seconds"])
                    cached_channels = cache["ch_names"].astype(str).tolist()
                if cached_channels != meta["ch_names"]:
                    raise RuntimeError(
                        f"Cached GFP-peak channels differ from the meta template for {recording_id}."
                    )
                peak_data = mvud.make_chdata(peak_maps, meta["info"])
            else:
                raw = mvud.load_analysis_raw(
                    fif_path,
                    source_layout,
                    condition,
                    meta["ch_names"],
                    sfreq_expected,
                    sfreq_tolerance,
                    interleaved_block_seconds,
                    interleaved_n_blocks,
                    interleaved_first_condition,
                )
                peak_data, peak_indices, peak_maps, continuous_gfp = mvud.extract_recording_gfp_peaks(
                    raw, min_peak_distance
                )
                n_samples = int(raw.n_times)
                duration_seconds = float(raw.n_times / raw.info["sfreq"])
                peak_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    peak_file,
                    peak_indices=peak_indices,
                    peak_maps=peak_maps.astype(np.float32),
                    peak_gfp=continuous_gfp[peak_indices].astype(np.float32),
                    ch_names=np.asarray(meta["ch_names"], dtype="U"),
                    sfreq=np.asarray(float(raw.info["sfreq"])),
                    n_samples=np.asarray(n_samples, dtype=np.int64),
                    duration_seconds=np.asarray(duration_seconds),
                )

            pending_k = [
                k for k, model_file in model_paths.items()
                if not models_ready and (refresh_clustering_outputs or not model_file.exists())
            ]
            fit_tasks = [
                delayed(_fit_recording_model)(
                    peak_maps,
                    meta_json=meta_json,
                    base_k=base_k,
                    model_file=str(model_paths[k]),
                    k=k,
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tol,
                    random_seed=random_seed,
                    recording_id=recording_id,
                    n_jobs_clustering=n_jobs_clustering,
                )
                for k in pending_k
            ]
            if pending_k:
                Parallel(n_jobs=1, backend="loky")(fit_tasks)
            models_by_k = {
                k: mvud.read_cluster_model(model_file)
                for k, model_file in model_paths.items()
            }
            # Record how the fixed reference-K subject solution maps onto the
            # canonical meta-template labels. This is provenance metadata for
            # the sample cache; continuous FORM remains anchored to meta maps.
            _, subject_matching = mvud.match_group_maps_to_meta(
                np.asarray(models_by_k[int(base_k)].cluster_centers_, dtype=float),
                meta["maps"],
                meta["labels"],
            )

            criteria, votes, selected_k = mvud.compute_zanesco_metacriterion(
                models_by_k,
                peak_maps,
                corr_threshold=criteria_corr_threshold,
                criteria_min_k=criteria_min_k,
                max_maps=criteria_max_maps,
                max_pairs=criteria_max_pairs,
                random_seed=mvud.stable_seed(random_seed, recording_id, "criteria"),
            )
            criteria_file.parent.mkdir(parents=True, exist_ok=True)
            votes_file.parent.mkdir(parents=True, exist_ok=True)
            criteria.to_csv(criteria_file, index=False)
            votes.to_csv(votes_file, index=False)

            reuse_sample_cache = sample_file.exists() and metadata_file.exists() and not refresh_clustering_outputs
            if reuse_sample_cache:
                with metadata_file.open("r", encoding="utf-8") as stream:
                    sample_metadata = json.load(stream)
                reuse_sample_cache = (
                    int(sample_metadata.get("base_k", -1)) == int(base_k)
                    and int(sample_metadata.get("sample_cache_schema", -1)) == SAMPLE_CACHE_SCHEMA
                )
            if not reuse_sample_cache:
                if raw is None:
                    raw = mvud.load_analysis_raw(
                        fif_path,
                        source_layout,
                        condition,
                        meta["ch_names"],
                        sfreq_expected,
                        sfreq_tolerance,
                        interleaved_block_seconds,
                        interleaved_n_blocks,
                        interleaved_first_condition,
                    )
                data = np.asarray(raw.get_data(picks=raw.ch_names), dtype=float)
                block_id, boundaries, block_source, discontinuities = mvud.detect_condition_blocks(
                    raw,
                    block_seconds,
                    expected_blocks,
                    annotation_boundary_regex,
                )
                geometry = meed.sensor_geometry(raw.info)
                core, unit_maps = mvud.compute_form_samplewise(
                    data,
                    geometry["S_orth"],
                    block_id,
                    discontinuities,
                    chunk_samples,
                )

                meta_assoc = mvud.compute_microstate_association(unit_maps, meta["maps"])
                meta_dist = mvud.compute_template_geometry_distances(
                    core["form_q"],
                    core["theta_deg"],
                    core["phi_deg"],
                    meta["maps"],
                    geometry["S_orth"],
                )


                local_gfp_peak = np.zeros(raw.n_times, dtype=bool)
                local_gfp_peak[peak_indices] = True
                sample_index = np.arange(raw.n_times, dtype=np.int64)
                time_s = sample_index.astype(np.float64) / float(raw.info["sfreq"])

                sample_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    sample_file,
                    **core,
                    sample_cache_schema=np.asarray(SAMPLE_CACHE_SCHEMA, dtype=np.int16),
                    sample_index=sample_index,
                    time_s=time_s.astype(np.float32),
                    local_gfp_peak=local_gfp_peak,
                    corr_meta_signed=meta_assoc["corr_signed"],
                    corr_meta_abs=meta_assoc["corr_abs"],
                    max_meta_corr=meta_assoc["max_abs_corr"],
                    second_meta_corr=meta_assoc["second_abs_corr"],
                    corr_meta_margin=meta_assoc["corr_margin"],
                    corr_meta_margin_relative=meta_assoc["corr_margin_relative"],
                    corr_meta_entropy_abs=meta_assoc["corr_entropy_abs"],
                    corr_meta_entropy_r2=meta_assoc["corr_entropy_r2"],
                    psi_meta_deg=meta_dist["psi_deg"],
                    theta_meta_distance_deg=meta_dist["theta_distance_deg"],
                    phi_meta_distance_deg=meta_dist["phi_distance_deg"],
                    ch_names=np.asarray(meta["ch_names"], dtype="U"),
                    meta_labels=np.asarray(meta["labels"], dtype="U"),
                )

                finite_closure = core["td_closure"][np.isfinite(core["td_closure"])]
                identity_mask = np.isfinite(core["td_form"]) & np.isfinite(
                    core["dipole_delta_norm"]
                )
                identity_error = (
                    core["td_form"][identity_mask] - core["dipole_delta_norm"][identity_mask]
                )
                sample_metadata = {
                    "sample_cache_schema": SAMPLE_CACHE_SCHEMA,
                    "recording_id": recording_id,
                    "subject": subject,
                    "condition": condition,
                    "sfreq": float(raw.info["sfreq"]),
                    "n_channels": len(raw.ch_names),
                    "n_samples": int(raw.n_times),
                    "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
                    "ch_names": list(raw.ch_names),
                    "block_boundaries_samples": boundaries.tolist(),
                    "temporal_discontinuity_samples": discontinuities.tolist(),
                    "n_temporal_discontinuities": int(discontinuities.size),
                    "block_source": block_source,
                    "n_blocks": int(np.unique(block_id).size),
                    "sphere_origin_m": np.asarray(geometry["origin_m"], dtype=float).tolist(),
                    "sphere_radius_m": float(geometry["radius_m"]),
                    "geometry_condition_number": float(geometry["M_condition"]),
                    "s_orth_residual_fro": float(geometry["S_orth_residual_fro"]),
                    "base_k": int(base_k),
                    "base_k_subject_meta_matching": subject_matching.to_dict(orient="records"),
                    "maximum_absolute_td_closure": (
                        float(np.max(np.abs(finite_closure))) if finite_closure.size else None
                    ),
                    "maximum_absolute_td_meed_minus_dipole_delta_norm": (
                        float(np.max(np.abs(identity_error))) if identity_error.size else None
                    ),
                    "psiD_definition": "signed angle arccos(dot(qhat_t,qhat_t-1)) in degrees",
                    "dipole_angle_antipodal_definition": "arccos(abs(dot(qhat_t,qhat_t-1))) in degrees",
                    "f_meed_definition": "TD_FORM^2 / TD^2 where TD > 0",
                    "corr_meta_entropy_r2_definition": (
                        f"normalized Shannon entropy of squared absolute K={base_k} meta correlations"
                    ),
                    "corr_meta_entropy_abs_definition": (
                        f"normalized Shannon entropy of absolute K={base_k} meta correlations"
                    ),
                }
                mvud.write_json(sample_metadata, metadata_file)

            selected_model = models_by_k[int(selected_k)]
            summary = {
                "recording_id": recording_id,
                "subject": subject,
                "condition": condition,
                "n_samples": int(sample_metadata["n_samples"]),
                "duration_seconds": float(sample_metadata["duration_seconds"]),
                "sfreq": float(sample_metadata["sfreq"]),
                "n_blocks": int(sample_metadata["n_blocks"]),
                "block_source": str(sample_metadata["block_source"]),
                "n_temporal_discontinuities": int(sample_metadata["n_temporal_discontinuities"]),
                "n_gfp_peaks": int(peak_indices.size),
                "selected_k": int(selected_k),
                "selected_model_GEV_fraction": float(selected_model.GEV_),
                "selected_model_GEV_percent": 100.0 * float(selected_model.GEV_),
                "selection_implementation": "Zanesco seven-criterion metacriterion",
                "clustering_implementation": clustering_implementation,
                "criteria_assignment_threshold": float(criteria_corr_threshold),
            }
            mvud.write_json(summary, summary_file)
            status.update(summary)
            status["success"] = True
    except Exception as error:
        status["success"] = False
        status["error"] = repr(error)
        status["traceback"] = format_exc()
    return status


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
    recording = cfg["a_subject_level"]

    data_root = base / Path(cfg["data_root"])
    output_root = base / Path(cfg["output_root"])
    meta_json = base / Path(cfg["meta_json"])
    exclude_ids = base / Path(cfg["preclustering_exclude_ids"]) if cfg.get("preclustering_exclude_ids") else None

    stage_dir = output_root / "a_subject_level"
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Discover one common-average-referenced recording per subject-condition.
    inventory = mvud.discover_recordings(
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
    selected = mvud.filter_only_id(inventory.loc[inventory["included"]].copy(), cli.only_id)
    if selected.empty:
        raise RuntimeError("No included recordings remain after inventory and --only-id filtering.")

    # Cache GFP peaks, FORM/TD quantities and native clustering models per recording.
    cached_results = (
        [None] * len(selected)
        if cfg["overwrite"]
        else [
            _load_completed_status(
                row, str(stage_dir), int(recording["k_min"]), int(recording["k_max"]),
                int(recording["subject_reference_k"]),
                CLUSTERING_IMPLEMENTATION,
            )
            for row in selected.to_dict(orient="records")
        ]
    )
    pending_rows = [
        row for row, cached in zip(selected.to_dict(orient="records"), cached_results)
        if cached is None
    ]
    n_jobs_tasks = int(recording["n_jobs_tasks"])
    n_jobs_clustering = int(recording["n_jobs_clustering"])
    n_jobs_finalization = int(recording["n_jobs_finalization"])
    if cli.only_id is not None:
        n_jobs_tasks = n_jobs_clustering = n_jobs_finalization = 1
    if n_jobs_tasks < 1:
        raise ValueError("a_subject_level.n_jobs_tasks must be at least one.")
    if n_jobs_finalization < 1:
        raise ValueError("a_subject_level.n_jobs_finalization must be at least one.")
    if n_jobs_tasks > 1 and n_jobs_clustering > 1:
        raise ValueError(
            "Use either global a_subject_level.n_jobs_tasks or internal "
            "a_subject_level.n_jobs_clustering above one, not both."
        )
    worker_kwargs = {
        "meta_json": str(meta_json),
        "stage_dir": str(stage_dir),
        "sfreq_expected": cfg["sfreq_expected"],
        "sfreq_tolerance": cfg["sfreq_tolerance"],
        "interleaved_block_seconds": cfg["interleaved_block_seconds"],
        "interleaved_n_blocks": cfg["interleaved_n_blocks"],
        "interleaved_first_condition": cfg["interleaved_first_condition"],
        "k_min": recording["k_min"],
        "k_max": recording["k_max"],
        "n_init": recording["n_init"],
        "max_iter": recording["max_iter"],
        "tol": recording["tol"],
        "n_jobs_clustering": n_jobs_clustering,
        "min_peak_distance": recording["min_peak_distance"],
        "criteria_corr_threshold": recording["criteria_corr_threshold"],
        "criteria_min_k": recording["criteria_min_k"],
        "criteria_max_maps": recording["criteria_max_maps"],
        "criteria_max_pairs": recording["criteria_max_pairs"],
        "base_k": recording["subject_reference_k"],
        "block_seconds": recording["block_seconds"],
        "expected_blocks": recording["expected_blocks_per_condition"],
        "annotation_boundary_regex": recording["annotation_boundary_regex"],
        "chunk_samples": recording["chunk_samples"],
        "random_seed": cfg["random_seed"],
        "overwrite": cfg["overwrite"],
    }
    results = [cached for cached in cached_results if cached is not None]
    print(
        f"[A] Preparing GFP-peak caches for {len(pending_rows)} recording(s) "
        f"with {n_jobs_tasks} worker(s)."
    )
    peak_files = {}
    prepare_kwargs = {
        key: worker_kwargs[key]
        for key in (
            "meta_json", "stage_dir", "base_k", "sfreq_expected", "sfreq_tolerance",
            "interleaved_block_seconds", "interleaved_n_blocks", "interleaved_first_condition",
            "min_peak_distance", "overwrite",
        )
    }
    with tqdm(total=len(pending_rows), desc="A peak caches", position=0) as progress:
        prepared_caches = Parallel(
            n_jobs=mvud.effective_n_jobs(n_jobs_tasks, len(pending_rows)),
            backend="loky",
            return_as="generator",
        )(
            delayed(_ensure_peak_cache)(row, **prepare_kwargs)
            for row in pending_rows
        )
        for row, peak_file in zip(pending_rows, prepared_caches):
            peak_files[str(row["recording_id"])] = peak_file
            progress.update()

    fit_items = []
    for row in pending_rows:
        recording_id = str(row["recording_id"])
        model_dir = Path(worker_kwargs["stage_dir"]) / "models" / recording_id
        summary_file = Path(worker_kwargs["stage_dir"]) / "summaries" / f"{recording_id}_summary.json"
        refresh = bool(worker_kwargs["overwrite"])
        if summary_file.exists() and not refresh:
            with summary_file.open("r", encoding="utf-8") as stream:
                prior = json.load(stream)
            refresh = (
                prior.get("clustering_implementation") != CLUSTERING_IMPLEMENTATION
            )
        for k in range(int(recording["k_min"]), int(recording["k_max"]) + 1):
            model_file = model_dir / f"k-{k:02d}_modkmeans.fif"
            if refresh or not model_file.exists():
                fit_items.append((recording_id, peak_files[recording_id], k, str(model_file)))

    total_model_tasks = len(selected) * (int(recording["k_max"]) - int(recording["k_min"]) + 1)
    reusable_model_tasks = total_model_tasks - len(fit_items)
    n_fit_workers = mvud.effective_n_jobs(n_jobs_tasks, len(fit_items))
    print(
        f"[A] K-model tasks: {total_model_tasks} total, {reusable_model_tasks} reusable, "
        f"{len(fit_items)} queued with {n_fit_workers} worker(s)."
    )
    with tqdm(
        total=total_model_tasks,
        initial=reusable_model_tasks,
        desc="A global K fits",
        position=0,
    ) as progress:
        for _ in Parallel(
            n_jobs=n_fit_workers,
            backend="loky",
            return_as="generator_unordered",
        )(
            delayed(_fit_recording_k_task)(
                peak_file,
                meta_json=worker_kwargs["meta_json"],
                base_k=worker_kwargs["base_k"],
                model_file=model_file,
                k=k,
                n_init=worker_kwargs["n_init"],
                max_iter=worker_kwargs["max_iter"],
                tol=worker_kwargs["tol"],
                random_seed=worker_kwargs["random_seed"],
                recording_id=recording_id,
                n_jobs_clustering=worker_kwargs["n_jobs_clustering"],
            )
            for recording_id, peak_file, k, model_file in fit_items
        ):
            progress.update()

    print("[A] Finalizing atomized criteria, sample caches, and summaries.")
    with tqdm(total=len(pending_rows), desc="A recording caches", position=0) as progress:
        for result in Parallel(
            n_jobs=mvud.effective_n_jobs(n_jobs_finalization, len(pending_rows)),
            backend="loky",
            return_as="generator_unordered",
        )(
            delayed(_work_cache)(
                row,
                models_ready=True,
                reuse_peak_cache=True,
                **worker_kwargs,
            )
            for row in pending_rows
        ):
            results.append(result)
            progress.update()

    suffix = f"_{mvud.normalize_only_id(cli.only_id)}" if cli.only_id is not None else ""
    status = pd.DataFrame(results).sort_values(["subject", "condition"])
    successful = status.loc[status["success"].eq(True)].copy()
    failures = status.loc[status["success"].ne(True)]
    print(
        f"Real-data cache stage complete: {len(successful)} successful, {len(failures)} failed. "
        "Per-recording summaries are in a_subject_level/summaries/."
    )
    if not failures.empty:
        print(failures[["recording_id", "error", "traceback"]].to_string(index=False))
    ram_stop.set()
    ram_thread.join(timeout=1)


if __name__ == "__main__":
    main()
