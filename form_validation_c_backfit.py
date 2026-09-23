"""Backfit canonical pooled A-E maps and write conventional GEV outputs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from traceback import format_exc
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import form_validation_abc_utils as mvud

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_SCHEMA = "instantaneous-pooled-association-and-gev-v5"
LABELS = ("A", "B", "C", "D", "E")


def _sha256(path: str | Path) -> str:
    """Return a content identity for one upstream cache or model artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gev_row(corr_abs, weights, valid, labels, *, level, recording_id, subject, condition):
    """GEV is GFP-weighted winning correlation squared, attributed to A-E."""
    corr_abs, weights = np.asarray(corr_abs, float), np.asarray(weights, float)
    valid = np.asarray(valid, bool) & np.isfinite(weights) & (weights >= 0)
    valid &= np.isfinite(corr_abs).all(axis=1)
    if not valid.any(): raise RuntimeError(f"No valid maps for {level} GEV in {recording_id}")
    denominator = float(weights[valid].sum())
    if denominator <= 0: raise RuntimeError(f"Non-positive GEV denominator for {recording_id}")
    winner, values = np.argmax(corr_abs, axis=1), np.zeros(len(labels))
    for i in range(len(labels)):
        take = valid & (winner == i)
        values[i] = np.sum(weights[take] * corr_abs[take, i] ** 2) / denominator
    row = {"level": level, "recording_id": recording_id, "subject": subject, "condition": condition,
           "n_candidates": int(valid.sum()), "weight_sum": denominator, "GEV_Total": float(values.sum())}
    row.update({f"GEV_{label}": float(value) for label, value in zip(labels, values)})
    return row


def _subject_maps(model_file, group_maps, labels):
    """Match each fixed K=5 recording model to the one pooled A-E dictionary."""
    model = mvud.read_cluster_model(model_file)
    return mvud.match_group_maps_to_meta(np.asarray(model.cluster_centers_, float), group_maps,
        list(labels), raw_labels=list(model.cluster_names))


def _work(row, *, maps_file, stage_dir, cfg, overwrite):
    recording_id, subject, condition = (str(row[k]) for k in ("recording_id", "subject", "condition"))
    stage_dir, association_file = Path(stage_dir), Path(stage_dir) / "associations" / f"{recording_id}_group_A-E.npz"
    metadata_file = Path(stage_dir) / "metadata" / f"{recording_id}_group_A-E.json"
    status = {"recording_id": recording_id, "subject": subject, "condition": condition, "success": False, "error": None, "traceback": None}
    try:
        source_identity = {
            "group_maps_sha256": _sha256(maps_file),
            "sample_cache_sha256": _sha256(row["sample_cache_file"]),
            "subject_k5_model_sha256": _sha256(row["subject_k5_model_file"]),
        }
        if association_file.exists() and metadata_file.exists() and not overwrite:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata.get("cache_schema") == CACHE_SCHEMA and all(metadata.get(key) == value for key, value in source_identity.items()):
                status.update(metadata); status["success"] = True; return status
        with np.load(maps_file, allow_pickle=False) as bank:
            group_maps, labels, ch_names = bank["group_maps"].astype(float), bank["labels"].astype(str).tolist(), bank["ch_names"].astype(str).tolist()
        if tuple(labels) != LABELS: raise RuntimeError(f"Canonical group labels must be A-E, got {labels}")
        with threadpool_limits(limits=1):
            raw = mvud.load_analysis_raw(str(row["fif_path"]), str(row.get("source_layout", "condition_files")), condition, ch_names,
                cfg["sfreq_expected"], cfg["sfreq_tolerance"], cfg["interleaved_block_seconds"], cfg["interleaved_n_blocks"], cfg["interleaved_first_condition"])
            data = np.asarray(raw.get_data(picks=raw.ch_names), float)
            unit, gfp, valid = mvud._center_unit_maps(data)
            association = mvud.compute_microstate_association(unit, group_maps)
        with np.load(row["sample_cache_file"], allow_pickle=False) as shared:
            if not np.array_equal(shared["valid_topography"].astype(bool), valid): raise RuntimeError("Stage-A and Stage-C valid maps differ")
            block_id = shared["block_id"].astype(np.int16)
        backfit = _gev_row(association["corr_abs"], gfp ** 2, valid, labels, level="backfit_k5", recording_id=recording_id, subject=subject, condition=condition)
        maps, matching = _subject_maps(row["subject_k5_model_file"], group_maps, labels)
        with np.load(row["peak_file"], allow_pickle=False) as peak: peak_maps, peak_gfp = peak["peak_maps"].astype(float), peak["peak_gfp"].astype(float)
        peak_unit, _, peak_valid = mvud._center_unit_maps(peak_maps)
        subject_gev = _gev_row(mvud.compute_microstate_association(peak_unit, maps)["corr_abs"], peak_gfp ** 2, peak_valid, labels, level="subject_k5", recording_id=recording_id, subject=subject, condition=condition)
        association_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(association_file, corr_abs=association["corr_abs"].astype(np.float32), corr_signed=association["corr_signed"].astype(np.float32), max_abs_corr=association["max_abs_corr"].astype(np.float32), second_abs_corr=association["second_abs_corr"].astype(np.float32), corr_margin=association["corr_margin"].astype(np.float32), winning_label=np.argmax(association["corr_abs"], axis=1).astype(np.int8), gfp=gfp.astype(np.float32), valid_topography=valid, block_id=block_id, labels=np.asarray(labels, dtype="U"), ch_names=np.asarray(ch_names, dtype="U"))
        metadata = {"recording_id": recording_id, "subject": subject, "condition": condition, "cache_schema": CACHE_SCHEMA, **source_identity, "sfreq": float(raw.info["sfreq"]), "n_samples": int(raw.n_times), "association": "absolute correlation to canonical pooled K=5 A-E before postprocessing", "gev_definition": "sum(GFP_squared * winning_correlation_squared) / sum(GFP_squared)", "subject_k5_matching": matching.to_dict(orient="records"), "backfit_gev": backfit, "subject_k5_gev": subject_gev}
        metadata_file.parent.mkdir(parents=True, exist_ok=True); mvud.write_json(metadata, metadata_file)
        status.update(metadata); status["success"] = True
    except Exception as error: status.update(error=repr(error), traceback=format_exc())
    return status


def _group_rows(pooled_file, group_maps, labels):
    with np.load(pooled_file, allow_pickle=False) as pooled:
        maps, conditions = pooled["maps"].astype(float), pooled["source_condition"].astype(str)
    unit, norms, valid = mvud._center_unit_maps(maps); association = mvud.compute_microstate_association(unit, group_maps)
    return [_gev_row(association["corr_abs"], norms ** 2, valid & (conditions == condition), labels, level="group_k5_centroids", recording_id="pooled_group", subject="pooled_group", condition=condition) for condition in ("EC", "EO")]


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yml"); parser.add_argument("--only-id"); cli = parser.parse_args()
    cfg, base = yaml.safe_load(cli.config.read_text(encoding="utf-8")), cli.config.resolve().parent; root = (base / cfg["output_root"]).resolve(); stage = root / "c_backfit"; gev = stage / "gev"
    index = mvud.filter_only_id(mvud.load_recording_jsons(root / "a_subject_level" / "summaries", "*_summary.json"), cli.only_id)
    inventory = mvud.discover_recordings(base / cfg["data_root"], cfg["fif_glob"], cfg["subject_regex"], cfg["condition_regex"], cfg["input_layout"], cfg["interleaved_n_blocks"], cfg["interleaved_first_condition"], cfg["require_both_conditions"], base / cfg["preclustering_exclude_ids"] if cfg.get("preclustering_exclude_ids") else None, cfg.get("max_subjects", 0))
    index = index.drop(columns=["fif_path", "source_layout", "source_block_indices", "sample_cache_file", "peak_file"], errors="ignore").merge(inventory.loc[inventory.included, ["recording_id", "fif_path", "source_layout"]], on="recording_id", how="left", validate="one_to_one")
    if index.empty or index.fif_path.isna().any(): raise RuntimeError("Stage-A summaries and configured recordings do not agree")
    index["sample_cache_file"] = index.recording_id.map(lambda x: str(root / "a_subject_level" / "samples" / f"{x}_samplewise.npz")); index["peak_file"] = index.recording_id.map(lambda x: str(root / "a_subject_level" / "gfp_peaks" / f"{x}_gfp_peaks.npz")); index["subject_k5_model_file"] = index.recording_id.map(lambda x: str(root / "a_subject_level" / "models" / x / "k-05_modkmeans.fif"))
    maps_file = root / "b_group_level" / "group_maps_A-E.npz"
    if not maps_file.is_file(): raise FileNotFoundError("Run Stage B before Stage C")
    stage.mkdir(parents=True, exist_ok=True); n_jobs = 1 if cli.only_id is not None else mvud.effective_n_jobs(cfg["c_backfit"]["n_jobs_recordings"], len(index))
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_work)(row, maps_file=str(maps_file), stage_dir=str(stage), cfg=cfg, overwrite=cfg["overwrite"]) for row in tqdm(index.to_dict(orient="records"), desc="C instantaneous backfit"))
    status = pd.DataFrame(results).sort_values(["subject", "condition"])
    status_file = stage / ("backfit_status.csv" if cli.only_id is None else f"backfit_status_{cli.only_id}.csv")
    status.to_csv(status_file, index=False)
    if not status.success.all(): raise RuntimeError(f"Stage C failed; see {status_file}")
    if cli.only_id is not None:
        print(f"Wrote scoped Stage-C associations for {cli.only_id}; cohort GEV tables were not changed.")
        return
    gev.mkdir(exist_ok=True); recording = pd.DataFrame([item[key] for item in results for key in ("subject_k5_gev", "backfit_gev")]).sort_values(["level", "subject", "condition"]); recording.to_csv(gev / "gev_recording_k5.csv", index=False)
    index[["recording_id", "subject", "condition", "selected_k", "selected_model_GEV_fraction", "selected_model_GEV_percent"]].to_csv(gev / "gev_selected_k_qc.csv", index=False)
    with np.load(maps_file, allow_pickle=False) as bank:
        group_maps, labels = bank["group_maps"].astype(float), bank["labels"].astype(str).tolist()
    pooled_file = root / "b_group_level" / "pooled_recording_centroids.npz"
    group = pd.DataFrame(_group_rows(pooled_file, group_maps, labels))
    with np.load(pooled_file, allow_pickle=False) as pooled:
        pooled_maps = pooled["maps"].astype(float)
    pooled_unit, pooled_norms, pooled_valid = mvud._center_unit_maps(pooled_maps)
    pooled_assoc = mvud.compute_microstate_association(pooled_unit, group_maps)
    pooled_total = _gev_row(pooled_assoc["corr_abs"], pooled_norms ** 2, pooled_valid, labels,
        level="group_k5_centroids", recording_id="pooled_group", subject="pooled_group", condition="pooled")["GEV_Total"]
    native_total = float(mvud.read_cluster_model(root / "b_group_level" / "models" / "k-05_modkmeans.fif").GEV_)
    if not np.isclose(pooled_total, native_total, atol=1e-6):
        raise RuntimeError(f"Pooled-centroid GEV disagrees with native Stage-B model ({pooled_total} vs {native_total})")
    group.to_csv(gev / "gev_group_condition.csv", index=False)
    gev_summary = pd.concat([recording, group], ignore_index=True)
    gev_summary.to_csv(gev / "gev_summary.csv", index=False)
    gev_columns = [f"GEV_{label}" for label in LABELS] + ["GEV_Total"]
    table_s3 = gev_summary.groupby(["level", "condition"], as_index=False)[gev_columns].mean()
    table_s3["level"] = table_s3["level"].map({
        "subject_k5": "Subject-level clustering",
        "group_k5_centroids": "Group-level clustering",
        "backfit_k5": "Backfitting",
    })
    table_s3 = table_s3.rename(columns={"level": "GEV level", **{f"GEV_{label}": label for label in LABELS}, "GEV_Total": "Total"})
    table_s3 = table_s3.round(3)
    table_s3.to_csv(gev / "table_S3_global_explained_variance.csv", index=False, float_format="%.3f")
    (gev / "gev_provenance.json").write_text(json.dumps({"schema": CACHE_SCHEMA, "dictionary": "canonical pooled condition-insensitive K=5 A-E", "group_maps_sha256": _sha256(maps_file), "pooled_centroids_sha256": _sha256(pooled_file), "class_matching": "fixed subject K=5 Hungarian-matched to pooled A-E", "levels": {"subject_k5": "GFP peaks, GFP squared", "group_k5_centroids": "actual Stage-B centroids, centered-map energy squared", "backfit_k5": "continuous valid samples, GFP squared"}, "pooled_native_gev": native_total, "pooled_stage_c_gev": pooled_total}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote instantaneous associations and conventional GEV to {stage}")

if __name__ == "__main__": main()
