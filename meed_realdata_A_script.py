"""Run Problem A outputs for MEED real-data analysis.

This script is intentionally simple. It processes cleaned EEG recordings and
writes one subject folder per recording:

results-realdata-A-v1/<subject>/
    cache/
        <subject>_samplewise_characterization.csv
        <subject>_sensor_maps.csv
        <subject>_dipole_maps.csv
        levels_by_k.pkl
    item1_zanesco_td/
    item2_theta_phi/
    item3_nonredundancy/

Start with MAX_SUBJECTS = 1. Increase only after inspecting the first outputs.
"""

import argparse
import os
import re
import warnings
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import utils_realdata_A_v1 as ua  # type: ignore
import yaml

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 120

with open(Path(__file__).with_name("config.yml"), encoding="utf-8") as f:
    PROJECT_PATHS = yaml.safe_load(f)["paths"]

DATA_ROOT = Path(PROJECT_PATHS["data_root"])
JSON_PATH = Path("metamaps_export_lemon.json")
OUTDIR = Path(PROJECT_PATHS["results_root"]) / "results-realdata-A"
FIF_OR_EDF = 'FIF'  # Change to 'FIF' if using .fif files

BASE_K = 4
COMPUTE_ALL_K = False
ALL_K_VALUES = range(4, 9)
K_VALUES = tuple(ALL_K_VALUES) if COMPUTE_ALL_K else (BASE_K,)
RANDOM_STATE = 0
MAX_SUBJECTS = 2000
TIME_WINDOW = (0.0, 3600.0)
N_JOBS = 3
N_JOBS = min(N_JOBS, os.cpu_count() - 1) if os.cpu_count() > 1 else 1
N_JOBS = min(N_JOBS, MAX_SUBJECTS)  # Don't use more jobs than subjects

FORCE_REFIT_MAPS = False
FORCE_REBUILD_SAMPLEWISE = False
FORCE_REBUILD_ITEMS = False
FORCE_REBUILD_ABSTRACT_FIGURE = False
RUN_FIT = False


def _normalize_only_id(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        raise ValueError(f"Could not parse subject id from {value!r}")
    return f"sub-{digits.zfill(6)}"


def process_subject(eeg_path: Path, template_ch_names: list[str], *, write_exemplary_plots: bool = False):
    sub = ua.parse_subject_id(eeg_path)
    subject_dir = OUTDIR / sub
    cache_dir = subject_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {sub}")
    print(f"EEG: {eeg_path}")

    raw = ua.load_and_align_raw(eeg_path, template_ch_names)

    levels_cache = cache_dir / "levels_by_k.pkl"
    meta_base, _ = ua.load_meta_level(JSON_PATH, BASE_K)
    if levels_cache.exists() and (not RUN_FIT or not FORCE_REFIT_MAPS):
        print(f"Loading cached subject maps for k={list(K_VALUES)}")
        levels_by_k = pd.read_pickle(levels_cache)
    else:
        if not RUN_FIT:
            print(f"No cached subject maps for {sub}; continuing with meta-only processing because RUN_FIT is False")
            levels_by_k = {BASE_K: {"meta": meta_base, "subject": None}}
        else:
            print(f"Fitting subject maps for k={list(K_VALUES)}")
            levels_by_k = ua.fit_subject_maps_all_k(
                raw,
                JSON_PATH,
                k_values=K_VALUES,
                random_state=RANDOM_STATE,
            )
            pd.to_pickle(levels_by_k, levels_cache)

    sensor_csv = cache_dir / f"{sub}_sensor_maps.csv"
    dipole_csv = cache_dir / f"{sub}_dipole_maps.csv"
    map_fit_csv = cache_dir / f"{sub}_map_fit_r2.csv"
    if RUN_FIT and (FORCE_REFIT_MAPS or not sensor_csv.exists() or not dipole_csv.exists()):
        print("Writing sensor and dipole map tables")
        ua.export_sensor_and_dipole_maps(sub, levels_by_k, cache_dir)
    if RUN_FIT and (FORCE_REFIT_MAPS or not map_fit_csv.exists()):
        print("Writing map fit R^2 table")
        ua.export_map_fit_r2_table(sub, levels_by_k, cache_dir)

    sample_csv = cache_dir / f"{sub}_samplewise_characterization.csv"
    if sample_csv.exists() and not FORCE_REBUILD_SAMPLEWISE:
        print("Loading cached samplewise characterization")
        sample_df = pd.read_csv(sample_csv)
    else:
        print("Computing samplewise characterization")
        sample_df = ua.compute_samplewise_characterization(
            raw,
            levels_by_k[BASE_K]["meta"],
            levels_by_k[BASE_K].get("subject"),
            sub=sub,
        )
        sample_df.to_csv(sample_csv, index=False)

    analysis_df = sample_df.copy()
    if TIME_WINDOW is not None:
        _t0, _t1 = TIME_WINDOW
        analysis_df = sample_df.query("@t0 <= time <= @t1").copy()
    analysis_csv = sample_csv

    if FORCE_REBUILD_ITEMS:
        print("Writing Problem A item outputs")
        ua.save_item_outputs(
            sub,
            raw,
            analysis_df,
            levels_by_k,
            subject_dir,
            base_k=BASE_K,
            write_exemplary_plots=write_exemplary_plots,
        )
    elif write_exemplary_plots:
        print("Writing subject exemplary plots")
        ua.save_exemplary_subject_plots(sub, raw, analysis_df, subject_dir)

    print(f"Done: {subject_dir}")
    return {
        "sub": sub,
        "eeg_path": str(eeg_path),
        "subject_dir": subject_dir,
        "cache_dir": cache_dir,
        "sample_csv": sample_csv,
        "analysis_csv": analysis_csv,
        "sensor_csv": sensor_csv,
        "dipole_csv": dipole_csv,
        "map_fit_csv": map_fit_csv if map_fit_csv.exists() else None,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--only-id", dest="only_id", default=None)
    args = parser.parse_args()
    only_sub = _normalize_only_id(args.only_id)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    _meta4, template_ch_names = ua.load_meta_level(JSON_PATH, BASE_K)

    fmt = str(FIF_OR_EDF).strip().lower()
    if fmt not in {"fif", "edf"}:
        raise ValueError(f"FIF_OR_EDF must be 'FIF' or 'EDF', got {FIF_OR_EDF!r}")
    pattern = "*.fif" if fmt == "fif" else "*.edf"
    eeg_files = [Path(p) for p in glob(str(DATA_ROOT / "**" / pattern), recursive=True)]
    if only_sub is not None:
        eeg_files = [p for p in eeg_files if ua.parse_subject_id(p) == only_sub]
    eeg_files = sorted(eeg_files)[:MAX_SUBJECTS]

    print(f"Found {len(eeg_files)} EEG files to process")
    if len(eeg_files) == 0:
        print(f"No EEG files found in {DATA_ROOT.resolve() if DATA_ROOT.exists() else DATA_ROOT}")
        return
    if only_sub is not None:
        print(f"Restricted to subject: {only_sub}")

    from joblib import Parallel, delayed
    outputs = Parallel(n_jobs=N_JOBS)(delayed(process_subject)(eeg_path, template_ch_names, write_exemplary_plots=(only_sub is not None)) 
                        for eeg_path in eeg_files)
    
    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv(OUTDIR / "processed_subjects.csv", index=False)
    if only_sub is None and RUN_FIT:
        print("Writing group map-fit R^2 outputs")
        ua.save_group_map_fit_r2_outputs(outputs_df, OUTDIR)
    if only_sub is None and FORCE_REBUILD_ABSTRACT_FIGURE:
        print("Writing abstract group outputs")
        ua.save_abstract_group_outputs(outputs_df, OUTDIR)
    if only_sub is None and FORCE_REBUILD_ITEMS:
        print("Writing group Item 3 ML outputs")
        ua.save_group_item3_ml_outputs(outputs_df, OUTDIR)
    print(f"\nWrote run index: {OUTDIR / 'processed_subjects.csv'}")


if __name__ == "__main__":
    main()
