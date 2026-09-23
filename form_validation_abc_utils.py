"""Shared A/B/C mechanics and the primitive helpers consumed by Stage D.

Stage scripts own paths, caches, parallel work, and LEMON-specific interpretation.
This module owns loading, clustering, FORM geometry, and sequence construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import os
from pathlib import Path
import json
from collections.abc import Mapping
import re
import warnings
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment, least_squares
from scipy.signal import find_peaks
from scipy.special import expit
from scipy.stats import mannwhitneyu, pearsonr, pointbiserialr, t as student_t, ttest_rel
from sklearn.metrics import silhouette_score
import statsmodels.formula.api as smf

import form_method_utils as u
import form_method_utils as meed

# Legacy Stage-A caches remain valid inputs. New writers use FORM field names;
# duplicate old/new fields must agree exactly, including dtype and NaN locations.
CACHE_RENAMES = {
    "meed_q": "form_q",
    "td_meed": "td_form",
    "td_nonmeed": "td_off_form",
    "td2_meed": "td2_form",
    "td2_nonmeed": "td2_off_form",
    "f_meed": "f_form",
    "td_meed_internal_closure": "td_form_internal_closure",
}

def canonical_cache_names(payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return FORM cache keys while rejecting conflicting legacy duplicates."""
    result = dict(payload)
    for old, new in CACHE_RENAMES.items():
        if old not in result:
            continue
        if new in result:
            first, second = np.asarray(result[old]), np.asarray(result[new])
            if first.dtype != second.dtype or not np.array_equal(first, second, equal_nan=True):
                raise ValueError(f"Conflicting cache values under {old!r} and {new!r}")
        else:
            result[new] = result[old]
        del result[old]
    return result


def load_sample_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load one sample cache through the narrow legacy-name adapter."""
    with np.load(path, allow_pickle=False) as cache:
        return canonical_cache_names({key: cache[key] for key in cache.files})


LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


@dataclass
class TemplateLevel:
    name: str
    B: np.ndarray
    labels: list[str]
    info: mne.Info
    geom: dict
    meed: dict
    model: object | None = None

    @property
    def U(self) -> np.ndarray:
        return np.asarray(self.meed["U"], dtype=float)

    @property
    def D(self) -> np.ndarray:
        return np.asarray(self.meed["D"], dtype=float)

    @property
    def Q(self) -> np.ndarray:
        return np.asarray(self.meed["Q"], dtype=float)


# ---------------------------------------------------------------------
# Loading, alignment, and map fitting
# ---------------------------------------------------------------------


def standard_names_picker(rawobj, set_montage: bool = True):
    old_to_new = {"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"}
    montage = mne.channels.make_standard_montage("standard_1020")
    std = {name.lower(): name for name in montage.ch_names}

    keep = []
    rename = {}
    used = set()
    for ch in rawobj.ch_names:
        key = ch.strip().lower()
        new_name = old_to_new.get(key, std.get(key))
        if new_name is None or new_name in used:
            continue
        keep.append(ch)
        rename[ch] = new_name
        used.add(new_name)

    raw = rawobj.copy().pick(keep)
    raw.rename_channels(rename)
    if set_montage:
        raw.set_montage(montage, on_missing="raise")
    return raw


def get_fs_from_time_column(time_values, eps: float = 1e-12) -> float:
    values = np.asarray(time_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("Cannot infer sampling frequency from fewer than two finite time values")
    dt = np.diff(values)
    dt = dt[np.isfinite(dt) & (dt > eps)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling frequency from a constant/non-increasing time column")
    return float(1.0 / np.median(dt))


def load_meta_level(json_path: str | Path, k: int) -> tuple[TemplateLevel, list[str]]:
    ds = u.load_microstates(str(json_path), int(k))
    B = u.center_l2_cols(np.asarray(ds["B"], dtype=float))
    labels = [str(x) for x in ds["labels"]]
    info = ds["info"]
    level = build_template_level("meta", B, labels, info)
    return level, list(ds["ch_names"])


def build_template_level(name: str, B: np.ndarray, labels: list[str], info: mne.Info, model=None) -> TemplateLevel:
    B = u.center_l2_cols(np.asarray(B, dtype=float))
    geom = u.sensor_geometry(info)
    if hasattr(u, "fixed_meed"):
        meed = u.fixed_meed(B, geom["S_orth"])
    else:
        Q, Bproj, Bmeed, df = u.project_fixed_dipole(B, geom["S_orth"])
        U = np.column_stack([u.unit_from_angles(th, ph) for th, ph in zip(df["theta_deg"], df["phi_deg"])])
        D = np.vstack([df["theta_deg"].to_numpy(float), df["phi_deg"].to_numpy(float)])
        meed = {"Q": Q, "U": U, "D": D, "B_projection": Bproj, "B_meed": Bmeed, "df": df}
    return TemplateLevel(name=name, B=B, labels=list(labels), info=info.copy(), geom=geom, meed=meed, model=model)


def align_maps_to_reference(B: np.ndarray, B_ref: np.ndarray, labels: list[str], ref_labels: list[str]):
    B = u.center_l2_cols(B)
    B_ref = u.center_l2_cols(B_ref)
    C = B_ref.T @ B
    rows, cols = linear_sum_assignment(-np.abs(C))
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    B_aligned = B[:, cols].copy()
    signs = np.sign(np.sum(B_ref[:, rows] * B_aligned, axis=0))
    signs[signs == 0] = 1
    B_aligned *= signs[None, :]
    labels_aligned = [ref_labels[i] if i < len(ref_labels) else labels[j] for i, j in zip(rows, cols)]
    return B_aligned, labels_aligned


def load_cached_levels(
    model_dir: str | Path,
    json_path: str | Path,
    k_values,
    info: mne.Info,
):
    """Build the old TemplateLevel view from canonical cached Pycrostates models."""
    model_dir = Path(model_dir)
    levels_by_k = {}
    for k in k_values:
        meta_level, _ = load_meta_level(json_path, int(k))
        model_file = model_dir / f"k-{int(k):02d}_modkmeans.fif"
        if not model_file.exists():
            continue
        model = read_cluster_model(model_file)
        B = u.center_l2_cols(np.asarray(model.cluster_centers_, dtype=float).T)
        labels = [chr(ord("A") + i) for i in range(B.shape[1])]
        B, labels = align_maps_to_reference(B, meta_level.B, labels, meta_level.labels)
        subject_level = build_template_level("subject", B, labels, info, model=model)
        levels_by_k[int(k)] = {"meta": meta_level, "subject": subject_level}
    return levels_by_k


# ---------------------------------------------------------------------
# Shared data, clustering, and samplewise mechanics
# ---------------------------------------------------------------------

CONDITIONS = ("EC", "EO")



def require_pycrostates() -> dict[str, Any]:
    """Import the Pycrostates objects used by the stage with one clear error."""
    try:
        from pycrostates.cluster import ModKMeans
        from pycrostates.io import ChData, read_cluster
        from pycrostates.metrics import davies_bouldin_score, dunn_score, silhouette_score
        from pycrostates.preprocessing import extract_gfp_peaks
    except ImportError as error:
        raise RuntimeError(
            "This stage requires pycrostates==0.6.1. Install the requirements "
            "in the active Python environment before running the clustering scripts."
        ) from error
    return {
        "ModKMeans": ModKMeans,
        "ChData": ChData,
        "read_cluster": read_cluster,
        "extract_gfp_peaks": extract_gfp_peaks,
        "silhouette_score": silhouette_score,
        "davies_bouldin_score": davies_bouldin_score,
        "dunn_score": dunn_score,
    }


def stable_seed(base_seed: int, *tokens: object) -> int:
    text = "|".join([str(int(base_seed)), *(str(token) for token in tokens)])
    digest = sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def parse_subject_condition(
    path: str | Path,
    subject_regex: str,
    condition_regex: str,
) -> tuple[str, str]:
    text = str(path).replace("\\", "/")
    subject_match = re.search(subject_regex, text)
    condition_matches = re.findall(condition_regex, text)
    if subject_match is None:
        raise ValueError(f"Could not parse a subject identifier from {path}")
    if not condition_matches:
        raise ValueError(f"Could not parse EC or EO from {path}")
    parsed_conditions = [
        str(match[0] if isinstance(match, tuple) else match).upper()
        for match in condition_matches
    ]
    unique_conditions = sorted(set(parsed_conditions) & set(CONDITIONS))
    if len(unique_conditions) != 1:
        raise ValueError(
            f"Expected exactly one condition token in {path}, got {unique_conditions}"
        )
    subject = str(subject_match.group(1) if subject_match.groups() else subject_match.group(0))
    subject = subject.lower().replace("sub-", "sub-")
    return subject, unique_conditions[0]


def parse_subject_id(path: str | Path, subject_regex: str) -> str:
    text = str(path).replace("\\", "/")
    subject_match = re.search(subject_regex, text)
    if subject_match is None:
        raise ValueError(f"Could not parse a subject identifier from {path}")
    subject = str(
        subject_match.group(1) if subject_match.groups() else subject_match.group(0)
    )
    return subject.lower()


def read_id_set(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Exclusion ID file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        preferred = [
            column
            for column in table.columns
            if column.lower() in {"subject", "subject_id", "participant_id", "id"}
        ]
        values = table[preferred[0]] if preferred else table.iloc[:, 0]
        return {str(value).strip().lower() for value in values.dropna() if str(value).strip()}
    with path.open("r", encoding="utf-8") as stream:
        return {line.strip().lower() for line in stream if line.strip()}


def discover_recordings(
    data_root: Path,
    fif_glob: str,
    subject_regex: str,
    condition_regex: str,
    input_layout: str,
    interleaved_n_blocks: int,
    interleaved_first_condition: str,
    require_both_conditions: bool,
    preclustering_exclude_ids: Path | None,
    max_subjects: int,
) -> pd.DataFrame:
    """Build subject-condition tasks from separate or interleaved FIF files."""
    files = sorted(path for path in data_root.glob(fif_glob) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No FIF files found under {data_root} with glob {fif_glob!r}."
        )
    excluded_ids = read_id_set(preclustering_exclude_ids)
    parsed_files: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for path in files:
        try:
            subject = parse_subject_id(path, subject_regex)
        except ValueError as error:
            parse_errors.append(str(error))
            continue
        try:
            _, condition = parse_subject_condition(path, subject_regex, condition_regex)
        except ValueError:
            condition = None
        parsed_files.append(
            {"subject": subject, "condition": condition, "fif_path": str(path)}
        )

    if not parsed_files:
        details = "\n".join(parse_errors[:10])
        raise FileNotFoundError(
            f"No files with parseable subject identifiers were found under {data_root}."
            + (f"\nFirst parse errors:\n{details}" if details else "")
        )

    requested_layout = str(input_layout).strip().lower()
    condition_presence = [item["condition"] is not None for item in parsed_files]
    if requested_layout == "auto":
        if all(condition_presence):
            resolved_layout = "condition_files"
        elif not any(condition_presence):
            resolved_layout = "interleaved_blocks"
        else:
            raise RuntimeError(
                "Automatic input-layout detection found a mixture of condition-labelled "
                "and unlabelled FIF files. Set input_layout explicitly or separate the roots."
            )
    else:
        resolved_layout = requested_layout

    rows: list[dict[str, Any]] = []
    if resolved_layout == "condition_files":
        missing_condition = [item["fif_path"] for item in parsed_files if item["condition"] is None]
        if missing_condition:
            raise RuntimeError(
                "input_layout='condition_files' requires EC or EO in every path. First "
                f"unlabelled file: {missing_condition[0]}"
            )
        for item in parsed_files:
            subject = str(item["subject"])
            condition = str(item["condition"])
            rows.append(
                {
                    "recording_id": f"{subject}_{condition}",
                    "subject": subject,
                    "condition": condition,
                    "fif_path": str(item["fif_path"]),
                    "source_layout": "condition_files",
                    "source_block_indices": "",
                    "preclustering_excluded": subject.lower() in excluded_ids,
                }
            )
    elif resolved_layout == "interleaved_blocks":
        source_table = pd.DataFrame(parsed_files).sort_values(["subject", "fif_path"])
        duplicated_subjects = source_table.duplicated("subject", keep=False)
        if duplicated_subjects.any():
            duplicate_rows = source_table.loc[duplicated_subjects, ["subject", "fif_path"]]
            raise RuntimeError(
                "Interleaved layout requires one source FIF per subject. Resolve these files:\n"
                + duplicate_rows.to_string(index=False)
            )
        condition_order = (
            ("EC", "EO") if str(interleaved_first_condition).upper() == "EC" else ("EO", "EC")
        )
        block_indices = {
            condition_order[0]: list(range(0, int(interleaved_n_blocks), 2)),
            condition_order[1]: list(range(1, int(interleaved_n_blocks), 2)),
        }
        for item in source_table.to_dict(orient="records"):
            subject = str(item["subject"])
            for condition in CONDITIONS:
                rows.append(
                    {
                        "recording_id": f"{subject}_{condition}",
                        "subject": subject,
                        "condition": condition,
                        "fif_path": str(item["fif_path"]),
                        "source_layout": "interleaved_blocks",
                        "source_block_indices": ",".join(
                            str(index) for index in block_indices[condition]
                        ),
                        "preclustering_excluded": subject.lower() in excluded_ids,
                    }
                )
    else:
        raise ValueError(f"Unsupported input layout: {resolved_layout}")

    inventory = pd.DataFrame(rows).sort_values(["subject", "condition", "fif_path"])
    duplicates = inventory.duplicated(["subject", "condition"], keep=False)
    if duplicates.any():
        duplicate_rows = inventory.loc[duplicates, ["subject", "condition", "fif_path"]]
        raise RuntimeError(
            "Multiple FIF files map to the same subject-condition. Resolve these before "
            "clustering:\n" + duplicate_rows.to_string(index=False)
        )

    condition_counts = inventory.groupby("subject")["condition"].nunique()
    inventory["has_both_conditions"] = inventory["subject"].map(condition_counts).eq(2)
    inventory["included"] = ~inventory["preclustering_excluded"]
    inventory["exclusion_reason"] = np.where(
        inventory["preclustering_excluded"], "preclustering_id_list", ""
    )
    if require_both_conditions:
        missing_pair = ~inventory["has_both_conditions"]
        inventory.loc[missing_pair, "included"] = False
        inventory.loc[missing_pair & (inventory["exclusion_reason"] == ""), "exclusion_reason"] = (
            "missing_ec_or_eo"
        )

    if max_subjects > 0:
        included_subjects = (
            inventory.loc[inventory["included"], "subject"].drop_duplicates().tolist()
        )
        kept_subjects = set(included_subjects[: int(max_subjects)])
        limited = inventory["included"] & ~inventory["subject"].isin(kept_subjects)
        inventory.loc[limited, "included"] = False
        inventory.loc[limited & (inventory["exclusion_reason"] == ""), "exclusion_reason"] = (
            "max_subjects_limit"
        )

    inventory["parse_error_count"] = len(parse_errors)
    inventory["input_layout_requested"] = requested_layout
    inventory["input_layout_resolved"] = resolved_layout
    return inventory.reset_index(drop=True)


def load_meta_template(meta_json: str | Path, k: int = 5) -> dict[str, Any]:
    template = meed.load_microstates(str(meta_json), int(k), warn=False)
    return {
        "maps": center_l2_columns(np.asarray(template["B"], dtype=float)),
        "labels": [str(label) for label in template["labels"]],
        "ch_names": [str(name) for name in template["ch_names"]],
        "info": template["info"].copy(),
    }


def standardize_channel_names(raw: mne.io.BaseRaw) -> None:
    mapping: dict[str, str] = {}
    for name in raw.ch_names:
        cleaned = str(name).strip()
        cleaned = re.sub(r"^(Green|Yellow|Red|Blue|White|Black)_[0-9]+_", "", cleaned)
        cleaned = cleaned.replace(" ", "")
        mapping[name] = cleaned
    raw.rename_channels(mapping)


def load_aligned_raw(
    fif_path: str | Path,
    template_ch_names: list[str],
    sfreq_expected: float,
    sfreq_tolerance: float,
) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    standardize_channel_names(raw)
    eeg_names = [
        name
        for name, channel_type in zip(raw.ch_names, raw.get_channel_types())
        if channel_type == "eeg"
    ]
    missing = [name for name in template_ch_names if name not in eeg_names]
    if missing:
        raise RuntimeError(
            f"{fif_path} is missing {len(missing)} required LEMON template channels: "
            + ", ".join(missing)
        )
    raw.pick(template_ch_names)
    raw.reorder_channels(template_ch_names)
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - float(sfreq_expected)) > float(sfreq_tolerance):
        raise RuntimeError(
            f"Unexpected sampling frequency in {fif_path}: {sfreq} Hz; "
            f"expected {sfreq_expected} +/- {sfreq_tolerance} Hz."
        )
    return raw


def load_analysis_raw(
    fif_path: str | Path,
    source_layout: str,
    condition: str,
    template_ch_names: list[str],
    sfreq_expected: float,
    sfreq_tolerance: float,
    interleaved_block_seconds: float,
    interleaved_n_blocks: int,
    interleaved_first_condition: str,
) -> mne.io.BaseRaw:
    """Load one canonical condition recording from either supported layout."""
    raw = load_aligned_raw(
        fif_path,
        template_ch_names,
        sfreq_expected,
        sfreq_tolerance,
    )
    layout = str(source_layout).strip().lower()
    if layout == "condition_files":
        return raw
    if layout != "interleaved_blocks":
        raise ValueError(f"Unsupported source_layout {source_layout!r}")

    condition = str(condition).upper()
    first_condition = str(interleaved_first_condition).upper()
    if condition not in CONDITIONS or first_condition not in CONDITIONS:
        raise ValueError("condition and interleaved_first_condition must be EC or EO")
    block_samples = int(round(float(interleaved_block_seconds) * raw.info["sfreq"]))
    required_samples = int(interleaved_n_blocks) * block_samples
    if raw.n_times < required_samples:
        raise RuntimeError(
            f"Interleaved source {fif_path} contains {raw.n_times} samples but "
            f"{required_samples} are required for {interleaved_n_blocks} blocks of "
            f"{interleaved_block_seconds} s."
        )
    if raw.n_times > required_samples + 1:
        raise RuntimeError(
            f"Interleaved source {fif_path} contains {raw.n_times} samples, exceeding "
            f"the configured {required_samples} by more than one sample. Correct the "
            "interleaved block settings rather than silently cropping the recording."
        )

    condition_order = ("EC", "EO") if first_condition == "EC" else ("EO", "EC")
    parity = 0 if condition == condition_order[0] else 1
    selected_blocks = range(parity, int(interleaved_n_blocks), 2)
    source_data = raw.get_data(picks=raw.ch_names)
    pieces = [
        source_data[:, block * block_samples : (block + 1) * block_samples]
        for block in selected_blocks
    ]
    condition_data = np.concatenate(pieces, axis=1)
    condition_raw = mne.io.RawArray(
        condition_data,
        raw.info.copy(),
        first_samp=0,
        verbose="ERROR",
    )
    n_condition_blocks = int(interleaved_n_blocks) // 2
    if n_condition_blocks > 1:
        boundary_onsets = [
            float(index) * float(interleaved_block_seconds)
            for index in range(1, n_condition_blocks)
        ]
        condition_raw.set_annotations(
            mne.Annotations(
                onset=boundary_onsets,
                duration=np.zeros(len(boundary_onsets), dtype=float),
                description=["condition_block_boundary"] * len(boundary_onsets),
            )
        )
    return condition_raw


def center_l2_columns(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    centered = data - np.mean(data, axis=0, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=0, keepdims=True), eps)


def gfp_from_data(data: np.ndarray) -> np.ndarray:
    centered = np.asarray(data, dtype=float) - np.mean(data, axis=0, keepdims=True)
    return np.sqrt(np.mean(centered * centered, axis=0))


def extract_recording_gfp_peaks(
    raw: mne.io.BaseRaw,
    min_peak_distance: int,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Use Pycrostates for peak ChData and cache the exact scipy peak indices."""
    pycrostates = require_pycrostates()
    data = np.asarray(raw.get_data(picks=raw.ch_names), dtype=float)
    gfp = gfp_from_data(data)
    peak_indices, _ = find_peaks(gfp, distance=int(min_peak_distance))
    peak_data = pycrostates["extract_gfp_peaks"](
        raw,
        picks=raw.ch_names,
        return_all=False,
        min_peak_distance=int(min_peak_distance),
        reject_by_annotation=False,
        verbose="ERROR",
    )
    if int(peak_data.get_data().shape[1]) != int(peak_indices.size):
        raise RuntimeError(
            "Pycrostates and the cached scipy GFP peak extraction returned different counts."
        )
    return peak_data, peak_indices.astype(np.int64), data[:, peak_indices], gfp


def fit_modkmeans(
    data: Any,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    tol: float,
    random_state: int,
    n_jobs: int,
) -> Any:
    pycrostates = require_pycrostates()
    model = pycrostates["ModKMeans"](
        n_clusters=int(n_clusters),
        n_init=int(n_init),
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(random_state),
    )
    model.fit(data, picks="eeg", n_jobs=int(n_jobs), verbose="ERROR")
    return model




def save_cluster_model(model: Any, path: str | Path, overwrite: bool) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            return path
        path.unlink()
    model.save(path)
    return path


def read_cluster_model(path: str | Path) -> Any:
    return require_pycrostates()["read_cluster"](path)


def _subsample_cluster_data(
    data_unit: np.ndarray,
    labels: np.ndarray,
    max_maps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_maps = data_unit.shape[1]
    if max_maps <= 0 or n_maps <= max_maps:
        return data_unit, labels
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    unique_labels = np.unique(labels)
    per_cluster = max(2, max_maps // max(len(unique_labels), 1))
    for label in unique_labels:
        indices = np.flatnonzero(labels == label)
        take = min(indices.size, per_cluster)
        if take:
            chosen.extend(rng.choice(indices, size=take, replace=False).tolist())
    if len(chosen) < max_maps:
        remaining = np.setdiff1d(np.arange(n_maps), np.asarray(chosen, dtype=int))
        take = min(max_maps - len(chosen), remaining.size)
        if take:
            chosen.extend(rng.choice(remaining, size=take, replace=False).tolist())
    chosen_array = np.asarray(sorted(set(chosen[:max_maps])), dtype=int)
    return data_unit[:, chosen_array], labels[chosen_array]


def _sample_pair_vectors(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_maps = labels.size
    total_pairs = n_maps * (n_maps - 1) // 2
    if total_pairs <= max_pairs:
        row, column = np.triu_indices(n_maps, k=1)
    else:
        rng = np.random.default_rng(seed)
        row = rng.integers(0, n_maps, size=max_pairs * 2)
        column = rng.integers(0, n_maps, size=max_pairs * 2)
        keep = row != column
        first = row[keep]
        second = column[keep]
        row = np.minimum(first, second)
        column = np.maximum(first, second)
        pairs = np.unique(np.column_stack([row, column]), axis=0)
        while pairs.shape[0] < max_pairs:
            additional = max_pairs - pairs.shape[0]
            a = rng.integers(0, n_maps, size=additional * 2)
            b = rng.integers(0, n_maps, size=additional * 2)
            keep = a != b
            candidate = np.column_stack([np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])])
            pairs = np.unique(np.vstack([pairs, candidate]), axis=0)
        row, column = pairs[:max_pairs].T
    distances = distance_matrix[row, column]
    between = labels[row] != labels[column]
    return distances, between


def _davies_bouldin_polarity(
    data_unit: np.ndarray,
    labels: np.ndarray,
    centers_unit: np.ndarray,
) -> float:
    """Davies-Bouldin index with Pycrostates' polarity-invariant distance."""
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return float("nan")
    centers = centers_unit[:, unique_labels]
    scatter = np.empty(unique_labels.size, dtype=float)
    for index, label in enumerate(unique_labels):
        selected = labels == label
        scatter[index] = np.mean(1.0 - np.abs(centers[:, index] @ data_unit[:, selected]))
    center_distance = 1.0 - np.abs(centers.T @ centers)
    np.fill_diagonal(center_distance, np.nan)
    ratios = (scatter[:, None] + scatter[None, :]) / center_distance
    return float(np.nanmean(np.nanmax(ratios, axis=1)))


def _dunn_polarity(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Dunn index with Pycrostates' polarity-invariant distance."""
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return float("nan")
    max_intra = 0.0
    min_inter = np.inf
    for index, first in enumerate(unique_labels):
        first_idx = np.flatnonzero(labels == first)
        if first_idx.size > 1:
            max_intra = max(max_intra, float(np.max(distance_matrix[np.ix_(first_idx, first_idx)])))
        for second in unique_labels[index + 1 :]:
            second_idx = np.flatnonzero(labels == second)
            min_inter = min(
                min_inter,
                float(np.min(distance_matrix[np.ix_(first_idx, second_idx)])),
            )
    if not np.isfinite(min_inter) or max_intra <= 0:
        return float("nan")
    return float(min_inter / max_intra)


def compute_zanesco_metacriterion(
    models_by_k: dict[int, Any],
    data: np.ndarray,
    corr_threshold: float,
    criteria_min_k: int,
    max_maps: int,
    max_pairs: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Evaluate the seven Zanesco metacriterion measures across fitted K models.

    The criterion definitions are Gamma, Silhouette, Davies-Bouldin,
    Point-Biserial, Dunn, Krzanowski-Lai, and Cross-Validation. The three
    The polarity-invariant distance matches Pycrostates' public metric
    definitions. Its metric functions operate on every fitted GFP peak and do
    not expose subsampling, so the distance-based indices use the configured,
    deterministic stratified sample instead; this keeps the analysis feasible
    without changing the clustering or the criterion definitions.
    """
    data = np.asarray(data, dtype=float)
    data_unit = center_l2_columns(data)
    gfp2 = gfp_from_data(data) ** 2
    n_channels, n_maps_total = data.shape
    rows: list[dict[str, Any]] = []

    for k in sorted(models_by_k):
        model = models_by_k[k]
        centers_unit = center_l2_columns(np.asarray(model.cluster_centers_, dtype=float).T)
        correlations = np.abs(centers_unit.T @ data_unit)
        labels = np.argmax(correlations, axis=0).astype(int)
        best_corr = np.max(correlations, axis=0)
        assigned = best_corr >= float(corr_threshold)
        residual_unweighted = float(np.sum(1.0 - best_corr[assigned] ** 2))
        residual_weighted = float(np.sum(gfp2[assigned] * (1.0 - best_corr[assigned] ** 2)))
        assigned_count = int(np.sum(assigned))
        cv_denominator = max(assigned_count * max(n_channels - 1, 1), 1)
        residual_variance = residual_weighted / cv_denominator
        degrees = n_channels - 1 - int(k)
        cv = (
            residual_variance * ((n_channels - 1) / degrees) ** 2
            if degrees > 0
            else float("nan")
        )

        gamma = silhouette = davies_bouldin = point_biserial = dunn = float("nan")
        validation_count = 0
        if assigned_count > max(2, int(k)) and np.unique(labels[assigned]).size > 1:
            validation_data, validation_labels = _subsample_cluster_data(
                data_unit[:, assigned],
                labels[assigned],
                int(max_maps),
                stable_seed(random_seed, "criterion_maps", k),
            )
            validation_count = int(validation_labels.size)
            distance_matrix = 1.0 - np.abs(validation_data.T @ validation_data)
            np.fill_diagonal(distance_matrix, 0.0)
            if all(np.sum(validation_labels == label) > 1 for label in np.unique(validation_labels)):
                silhouette = float(
                    silhouette_score(distance_matrix, validation_labels, metric="precomputed")
                )
            davies_bouldin = _davies_bouldin_polarity(
                validation_data, validation_labels, centers_unit
            )
            dunn = _dunn_polarity(distance_matrix, validation_labels)
            pair_distances, pair_between = _sample_pair_vectors(
                distance_matrix,
                validation_labels,
                int(max_pairs),
                stable_seed(random_seed, "criterion_pairs", k),
            )
            if np.unique(pair_between).size == 2:
                point_biserial = float(
                    pointbiserialr(pair_between.astype(int), pair_distances).statistic
                )
                within_distances = pair_distances[~pair_between]
                between_distances = pair_distances[pair_between]
                if within_distances.size and between_distances.size:
                    u_statistic = mannwhitneyu(
                        between_distances,
                        within_distances,
                        alternative="greater",
                        method="asymptotic",
                    ).statistic
                    gamma = float(
                        2.0
                        * u_statistic
                        / (between_distances.size * within_distances.size)
                        - 1.0
                    )

        gev_fraction = float(model.GEV_)
        rows.append(
            {
                "k": int(k),
                "n_maps_total": int(n_maps_total),
                "n_maps_assigned": assigned_count,
                "assigned_fraction": assigned_count / max(n_maps_total, 1),
                "n_maps_validation": validation_count,
                "GEV_fraction": gev_fraction,
                "GEV_percent": 100.0 * gev_fraction,
                "dispersion": residual_unweighted,
                "weighted_residual": residual_weighted,
                "Gamma": gamma,
                "Silhouette": silhouette,
                "Davies_Bouldin": davies_bouldin,
                "Point_Biserial": point_biserial,
                "Dunn": dunn,
                "Krzanowski_Lai": np.nan,
                "Cross_Validation": cv,
            }
        )

    criteria = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    dispersion_by_k = criteria.set_index("k")["dispersion"].to_dict()
    exponent = 2.0 / max(n_channels, 1)
    for row_index, k in enumerate(criteria["k"].astype(int)):
        if k - 1 not in dispersion_by_k or k + 1 not in dispersion_by_k:
            continue
        diff_k = (k - 1) ** exponent * dispersion_by_k[k - 1] - k**exponent * dispersion_by_k[k]
        diff_next = k**exponent * dispersion_by_k[k] - (k + 1) ** exponent * dispersion_by_k[k + 1]
        if diff_k > 0 and diff_next > 0:
            criteria.loc[row_index, "Krzanowski_Lai"] = float(abs(diff_k / diff_next))
        else:
            criteria.loc[row_index, "Krzanowski_Lai"] = 0.0

    directions = {
        "Gamma": "max",
        "Silhouette": "max",
        "Davies_Bouldin": "min",
        "Point_Biserial": "max",
        "Dunn": "max",
        "Krzanowski_Lai": "max",
        "Cross_Validation": "min",
    }
    vote_rows: list[dict[str, Any]] = []
    candidates = criteria.loc[criteria["k"] >= int(criteria_min_k)].copy()
    for criterion, direction in directions.items():
        finite = candidates.loc[np.isfinite(candidates[criterion]), ["k", criterion]]
        if finite.empty:
            raise RuntimeError(f"No finite values available for metacriterion {criterion}.")
        optimum = finite[criterion].max() if direction == "max" else finite[criterion].min()
        optimal_k = int(finite.loc[np.isclose(finite[criterion], optimum), "k"].min())
        vote_rows.append(
            {
                "criterion": criterion,
                "direction": direction,
                "optimal_k": optimal_k,
                "optimal_value": float(optimum),
            }
        )
    votes = pd.DataFrame(vote_rows)
    selected_k = int(np.median(votes["optimal_k"].to_numpy(dtype=int)))
    criteria["metacriterion_selected"] = criteria["k"].eq(selected_k)
    return criteria, votes, selected_k


def pool_selected_recording_centers(
    recording_index: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray], Any]:
    centers: list[np.ndarray] = []
    source_recording: list[str] = []
    source_subject: list[str] = []
    source_condition: list[str] = []
    source_local_map: list[int] = []
    common_info = None
    common_channels: list[str] | None = None

    for row in recording_index.itertuples(index=False):
        model = read_cluster_model(row.selected_model_file)
        model_centers = np.asarray(model.cluster_centers_, dtype=float)
        channels = list(model.info["ch_names"])
        if common_channels is None:
            common_channels = channels
            common_info = model.info
        elif channels != common_channels:
            raise RuntimeError(f"Channel mismatch in selected model {row.selected_model_file}")
        centers.append(model_centers)
        for local_map in range(model_centers.shape[0]):
            source_recording.append(str(row.recording_id))
            source_subject.append(str(row.subject))
            source_condition.append(str(row.condition))
            source_local_map.append(int(local_map))

    if not centers or common_info is None:
        raise RuntimeError("No successful selected recording models were available for pooling.")
    pooled = np.vstack(centers).T
    metadata = {
        "source_recording": np.asarray(source_recording, dtype="U"),
        "source_subject": np.asarray(source_subject, dtype="U"),
        "source_condition": np.asarray(source_condition, dtype="U"),
        "source_local_map": np.asarray(source_local_map, dtype=np.int16),
        "ch_names": np.asarray(common_channels, dtype="U"),
    }
    return pooled, metadata, common_info


def make_chdata(data: np.ndarray, info: Any) -> Any:
    return require_pycrostates()["ChData"](np.asarray(data, dtype=float), info)


def match_group_maps_to_meta(
    group_centers: np.ndarray,
    meta_maps: np.ndarray,
    meta_labels: list[str],
    *,
    raw_labels: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Match raw maps one-to-one to templates and place them in canonical order.

    Hungarian rows are *raw* map indices and columns are template indices.  Thus
    every selected pair obeys ``canonical[:, template_idx] = raw[:, raw_idx]``
    after the stored polarity multiplier is applied.  No direct ``col_ind`` map
    indexing is valid because template and raw-map index spaces differ.
    """
    raw_maps = center_l2_columns(np.asarray(group_centers, dtype=float).T)
    templates = center_l2_columns(np.asarray(meta_maps, dtype=float))
    if raw_maps.shape != templates.shape:
        raise ValueError(
            f"Group and meta map shapes differ: {raw_maps.shape} vs {templates.shape}"
        )
    n_maps = raw_maps.shape[1]
    if len(meta_labels) != n_maps:
        raise ValueError("One meta label is required for each template column")
    if raw_labels is None:
        raw_labels = [f"raw{index + 1}" for index in range(n_maps)]
    if len(raw_labels) != n_maps:
        raise ValueError("One raw label is required for each raw map column")

    signed_corr = templates.T @ raw_maps  # template index x raw-map index
    raw_indices, template_indices = linear_sum_assignment(-np.abs(signed_corr).T)
    canonical = np.empty_like(raw_maps)
    rows: list[dict[str, Any]] = []
    for raw_index, template_index in zip(raw_indices, template_indices):
        correlation = float(signed_corr[template_index, raw_index])
        polarity = 1 if correlation >= 0 else -1
        canonical[:, template_index] = raw_maps[:, raw_index] * polarity
        rows.append({
            "raw_map_index": int(raw_index),
            "raw_map_label": str(raw_labels[raw_index]),
            "meta_index": int(template_index),
            "meta_label": str(meta_labels[template_index]),
            "signed_spatial_correlation": correlation,
            "absolute_spatial_correlation": abs(correlation),
            "polarity_multiplier": polarity,
            "canonical_output_index": int(template_index),
            # Compatibility aliases used by Stage B's native model reorder.
            "output_index": int(template_index),
            "group_original_index": int(raw_index),
        })
    matching = pd.DataFrame(rows).sort_values("canonical_output_index").reset_index(drop=True)
    return canonical, matching


def save_group_topomap_figure(
    ordered_maps: np.ndarray,
    meta_maps: np.ndarray,
    labels: list[str],
    info: mne.Info,
    path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_maps = ordered_maps.shape[1]
    figure, axes = plt.subplots(2, n_maps, figsize=(2.0 * n_maps, 4.2))
    if n_maps == 1:
        axes = np.asarray(axes)[:, None]
    for index, label in enumerate(labels):
        for row, maps, prefix in ((0, meta_maps, "meta"), (1, ordered_maps, "group")):
            mne.viz.plot_topomap(
                maps[:, index],
                info,
                axes=axes[row, index],
                show=False,
                contours=0,
                sensors=False,
                image_interp="linear",
                extrapolate="local",
                cmap="RdBu_r",
            )
            axes[row, index].set_title(f"{prefix} {label}", fontsize=9)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path




def save_figure(fig, path: str | Path, *, tight: bool = True) -> Path:
    """Save and close a Matplotlib figure for any validation stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {"dpi": 180}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    return path


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    with path.open("w", encoding="utf-8") as stream:
        json.dump(convert(data), stream, indent=2, sort_keys=True)
    return path


def load_recording_jsons(directory: str | Path, pattern: str) -> pd.DataFrame:
    """Build a recording table from atomized per-recording JSON artifacts."""
    files = sorted(Path(directory).glob(pattern))
    records = []
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        record["json_file"] = str(path)
        records.append(record)
    return pd.DataFrame(records)


def normalize_only_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("sub-"):
        return text
    digits = re.sub(r"\D", "", text)
    return f"sub-{digits.zfill(6)}" if digits else text


def filter_only_id(table: pd.DataFrame, only_id: str | None) -> pd.DataFrame:
    normalized = normalize_only_id(only_id)
    if normalized is None:
        return table.copy()
    return table.loc[table["subject"].str.lower() == normalized].copy()


def effective_n_jobs(requested: int, n_tasks: int) -> int:
    """Return a task count that is safe for this platform's joblib backend."""
    if n_tasks <= 0:
        return 1
    # Windows loky monitors workers through WaitForMultipleObjects, which fails
    # when 62 workers create 64 handles. Keep a little headroom below its 63-handle
    # limit; the same cap applies to every validation-stage process pool.
    platform_limit = 60 if os.name == "nt" else int(requested)
    return max(1, min(int(requested), int(n_tasks), platform_limit))

# -----------------------------------------------------------------------------
# Shared continuous FORM characterization and temporal-block geometry
# -----------------------------------------------------------------------------


def _annotation_boundary_samples(
    raw: mne.io.BaseRaw,
    annotation_boundary_regex: str,
) -> np.ndarray:
    pattern = re.compile(annotation_boundary_regex)
    samples: list[int] = []
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        if pattern.search(str(description)):
            sample = int(raw.time_as_index(float(onset), use_rounding=True)[0])
            if 0 < sample < int(raw.n_times):
                samples.append(sample)
    return np.asarray(sorted(set(samples)), dtype=np.int64)


def _select_experimental_boundaries(
    annotation_boundaries: np.ndarray,
    n_times: int,
    block_samples: int,
    expected_blocks: int,
) -> np.ndarray:
    """Select nominal experimental joins when preprocessing adds extra boundaries."""
    candidates = np.asarray(annotation_boundaries, dtype=np.int64)
    needed = int(expected_blocks) - 1
    if candidates.size < needed:
        raise ValueError("Not enough annotation boundaries for experimental-block selection")
    if candidates.size == needed:
        return candidates.copy()

    points = np.r_[0, candidates, int(n_times)].astype(np.int64)
    last = points.size - 1
    n_segments = int(expected_blocks)
    dp = np.full((n_segments + 1, points.size), np.inf, dtype=float)
    previous = np.full((n_segments + 1, points.size), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    for segment in range(1, n_segments + 1):
        j_max = last if segment == n_segments else last - 1
        for j in range(1, j_max + 1):
            for i in range(segment - 1, j):
                if not np.isfinite(dp[segment - 1, i]):
                    continue
                if segment == n_segments and j != last:
                    continue
                length = float(points[j] - points[i])
                cost = dp[segment - 1, i] + (length - float(block_samples)) ** 2
                if cost < dp[segment, j]:
                    dp[segment, j] = cost
                    previous[segment, j] = i

    if not np.isfinite(dp[n_segments, last]):
        raise RuntimeError("Could not select experimental block boundaries")

    selected_points: list[int] = []
    j = last
    for segment in range(n_segments, 0, -1):
        i = int(previous[segment, j])
        if i < 0:
            raise RuntimeError("Experimental-boundary backtracking failed")
        if i != 0:
            selected_points.append(int(points[i]))
        j = i
    return np.asarray(sorted(selected_points), dtype=np.int64)


def detect_condition_blocks(
    raw: mne.io.BaseRaw,
    block_seconds: float,
    expected_blocks: int,
    annotation_boundary_regex: str,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """Return experimental block IDs plus every true temporal discontinuity."""
    n_times = int(raw.n_times)
    sfreq = float(raw.info["sfreq"])
    block_samples = int(round(float(block_seconds) * sfreq))
    annotation_boundaries = _annotation_boundary_samples(raw, annotation_boundary_regex)

    if annotation_boundaries.size >= int(expected_blocks) - 1:
        boundaries = _select_experimental_boundaries(
            annotation_boundaries,
            n_times=n_times,
            block_samples=block_samples,
            expected_blocks=expected_blocks,
        )
        source = (
            "annotations"
            if annotation_boundaries.size == int(expected_blocks) - 1
            else "annotations_selected_from_discontinuities"
        )
    else:
        boundaries = np.asarray(
            [block_samples * index for index in range(1, int(expected_blocks))],
            dtype=np.int64,
        )
        if boundaries.size and boundaries[-1] >= n_times:
            raise RuntimeError(
                f"Recording has {n_times / sfreq:.2f} s, too short to construct "
                f"{expected_blocks} blocks using {block_seconds:.2f} s boundaries."
            )
        source = "fixed_60s_fallback" if block_seconds == 60 else "fixed_duration_fallback"

    samples = np.arange(n_times, dtype=np.int64)
    block_id = np.searchsorted(boundaries, samples, side="right").astype(np.int16)
    if np.unique(block_id).size != int(expected_blocks):
        raise RuntimeError(
            f"Expected {expected_blocks} blocks but constructed {np.unique(block_id).size}."
        )
    discontinuities = np.asarray(
        sorted(set(annotation_boundaries.tolist()) | set(boundaries.tolist())),
        dtype=np.int64,
    )
    return block_id, boundaries, source, discontinuities


def canonical_angles_from_q(
    q: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized polarity-canonical FORM theta/phi representation."""
    q = np.asarray(q, dtype=float)
    if q.shape[0] != 3:
        raise ValueError(f"Expected q with shape (3, samples), got {q.shape}")
    rho = np.linalg.norm(q, axis=0)
    unit = np.divide(q, rho[None, :], out=np.zeros_like(q), where=rho[None, :] > eps)
    theta_raw = (90.0 - np.degrees(np.arctan2(unit[1], unit[0])) + 180.0) % 360.0 - 180.0
    flip = (theta_raw > 90.0) | (theta_raw < -90.0)
    unit[:, flip] *= -1.0
    theta = (90.0 - np.degrees(np.arctan2(unit[1], unit[0])) + 180.0) % 360.0 - 180.0
    theta = np.where(theta > 90.0, theta - 180.0, theta)
    theta = np.where(theta < -90.0, theta + 180.0, theta)
    phi = np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0)))
    theta[rho <= eps] = np.nan
    phi[rho <= eps] = np.nan
    return theta, phi, unit


def _center_unit_maps(data: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    centered = data - np.mean(data, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    valid = norms > eps
    unit_maps = np.divide(
        centered,
        norms[None, :],
        out=np.zeros_like(centered),
        where=norms[None, :] > eps,
    )
    return unit_maps, norms, valid


def _td_form_radial_angular_components(
    q: np.ndarray,
    rho: np.ndarray,
    valid_transition: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Split consecutive FORM displacements into exact radial and angular parts."""
    q = np.asarray(q, dtype=float)
    rho = np.asarray(rho, dtype=float)
    valid_transition = np.asarray(valid_transition, dtype=bool)
    td_rho = np.full(len(rho), np.nan, dtype=float)
    td_psi = np.full(len(rho), np.nan, dtype=float)

    indices = np.flatnonzero(valid_transition)
    if not len(indices):
        return td_rho, td_psi

    rho_now = rho[indices]
    rho_prev = rho[indices - 1]
    td_rho[indices] = np.abs(rho_now - rho_prev)
    td_psi[indices] = 0.0

    nonzero = (rho_now > eps) & (rho_prev > eps)
    if np.any(nonzero):
        now = q[indices[nonzero]] / rho_now[nonzero, None]
        previous = q[indices[nonzero] - 1] / rho_prev[nonzero, None]
        psi_rad = np.arccos(np.clip(np.sum(now * previous, axis=1), -1.0, 1.0))
        td_psi[indices[nonzero]] = (
            2.0
            * np.sqrt(rho_now[nonzero] * rho_prev[nonzero])
            * np.sin(psi_rad / 2.0)
        )
    return td_rho, td_psi


def compute_form_samplewise(
    data: np.ndarray,
    s_orth: np.ndarray,
    block_id: np.ndarray,
    temporal_discontinuity_samples: np.ndarray | None,
    chunk_samples: int,
    eps: float = 1e-12,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute recording-intrinsic samplewise FORM/GFP/TD quantities once."""
    data = np.asarray(data, dtype=float)
    s_orth = np.asarray(s_orth, dtype=float)
    block_id = np.asarray(block_id, dtype=np.int16)
    if data.ndim != 2:
        raise ValueError("data must have shape channels x samples")
    if s_orth.shape != (data.shape[0], 3):
        raise ValueError(f"S_orth has unexpected shape {s_orth.shape}")
    if block_id.size != data.shape[1]:
        raise ValueError("block_id length differs from EEG sample count")

    n_channels, n_samples = data.shape
    unit_maps, norms, valid = _center_unit_maps(data, eps=eps)
    gfp = norms / np.sqrt(float(n_channels))

    q = np.empty((3, n_samples), dtype=np.float64)
    for start in range(0, n_samples, int(chunk_samples)):
        stop = min(start + int(chunk_samples), n_samples)
        q[:, start:stop] = s_orth.T @ unit_maps[:, start:stop]

    rho2 = np.sum(q * q, axis=0)
    rho = np.sqrt(np.maximum(rho2, 0.0))
    theta, phi, q_unit_canonical = canonical_angles_from_q(q)

    dipole_delta_q = np.full((3, n_samples), np.nan, dtype=np.float64)
    dipole_delta_norm = np.full(n_samples, np.nan, dtype=np.float64)
    dipole_delta_norm_antipodal = np.full(n_samples, np.nan, dtype=np.float64)
    dipole_angle_antipodal_deg = np.full(n_samples, np.nan, dtype=np.float64)
    psiD_deg = np.full(n_samples, np.nan, dtype=np.float64)
    td = np.full(n_samples, np.nan, dtype=np.float64)
    td_meed = np.full(n_samples, np.nan, dtype=np.float64)
    td_nonmeed = np.full(n_samples, np.nan, dtype=np.float64)

    for start in range(1, n_samples, int(chunk_samples)):
        stop = min(start + int(chunk_samples), n_samples)
        delta_map = unit_maps[:, start:stop] - unit_maps[:, start - 1 : stop - 1]
        q_now = q[:, start:stop]
        q_prev = q[:, start - 1 : stop - 1]
        delta_q = q_now - q_prev
        delta_meed = s_orth @ delta_q
        delta_nonmeed = delta_map - delta_meed

        dipole_delta_q[:, start:stop] = delta_q
        dipole_delta_norm[start:stop] = np.linalg.norm(delta_q, axis=0)
        dipole_delta_norm_antipodal[start:stop] = np.minimum(
            dipole_delta_norm[start:stop],
            np.linalg.norm(q_now + q_prev, axis=0),
        )
        rho_now = np.linalg.norm(q_now, axis=0)
        rho_prev = np.linalg.norm(q_prev, axis=0)
        valid_angle = (rho_now > eps) & (rho_prev > eps)
        dot = np.zeros(stop - start, dtype=float)
        dot[valid_angle] = np.sum(
            (q_now[:, valid_angle] / rho_now[valid_angle][None, :])
            * (q_prev[:, valid_angle] / rho_prev[valid_angle][None, :]),
            axis=0,
        )
        psiD_deg[start:stop][valid_angle] = np.degrees(
            np.arccos(np.clip(dot[valid_angle], -1.0, 1.0))
        )
        dipole_angle_antipodal_deg[start:stop][valid_angle] = np.degrees(
            np.arccos(np.clip(np.abs(dot[valid_angle]), -1.0, 1.0))
        )
        td[start:stop] = np.linalg.norm(delta_map, axis=0)
        td_meed[start:stop] = np.linalg.norm(delta_meed, axis=0)
        td_nonmeed[start:stop] = np.linalg.norm(delta_nonmeed, axis=0)

    block_boundary = np.r_[True, np.diff(block_id) != 0]
    temporal_discontinuity = block_boundary.copy()
    if temporal_discontinuity_samples is not None:
        discontinuity_samples = np.asarray(temporal_discontinuity_samples, dtype=np.int64)
        discontinuity_samples = discontinuity_samples[
            (discontinuity_samples >= 0) & (discontinuity_samples < n_samples)
        ]
        temporal_discontinuity[discontinuity_samples] = True

    valid_transition = ~temporal_discontinuity
    valid_transition[0] = False
    valid_transition &= valid
    valid_transition[1:] &= valid[:-1]
    valid_transition[1:] &= block_id[1:] == block_id[:-1]
    td_rho, td_psi = _td_form_radial_angular_components(q.T, rho, valid_transition, eps=eps)

    for arr in (td, td_meed, td_nonmeed, td_rho, td_psi, dipole_delta_norm, dipole_delta_norm_antipodal,
                dipole_angle_antipodal_deg, psiD_deg):
        arr[~valid_transition] = np.nan
    dipole_delta_q[:, ~valid_transition] = np.nan

    td2 = td.astype(np.float64) ** 2
    td2_meed = td_meed.astype(np.float64) ** 2
    td2_nonmeed = td_nonmeed.astype(np.float64) ** 2
    td2_rho = td_rho ** 2
    td2_psi = td_psi ** 2
    td_closure = td2 - td2_meed - td2_nonmeed
    td_meed_internal_closure = td2_meed - td2_rho - td2_psi
    td_threeway_closure = td2 - td2_nonmeed - td2_rho - td2_psi
    f_meed = np.divide(
        td2_meed,
        td2,
        out=np.full(n_samples, np.nan, dtype=np.float64),
        where=td2 > eps,
    )

    arrays = {
        "gfp": gfp.astype(np.float32),
        "gfp2": (gfp * gfp).astype(np.float32),
        "form_q": q.T.astype(np.float32),
        "rho": rho.astype(np.float32),
        "rho2": rho2.astype(np.float32),
        "theta_deg": theta.astype(np.float32),
        "phi_deg": phi.astype(np.float32),
        "dipole_delta_q": dipole_delta_q.T.astype(np.float32),
        "dipole_delta_norm": dipole_delta_norm.astype(np.float32),
        "dipole_delta_norm_antipodal": dipole_delta_norm_antipodal.astype(np.float32),
        "dipole_angle_antipodal_deg": dipole_angle_antipodal_deg.astype(np.float32),
        "psiD_deg": psiD_deg.astype(np.float32),
        "td": td.astype(np.float32),
        "td_form": td_meed.astype(np.float32),
        "td_off_form": td_nonmeed.astype(np.float32),
        "td_rho": td_rho.astype(np.float32),
        "td_psi": td_psi.astype(np.float32),
        "td2": td2.astype(np.float32),
        "td2_form": td2_meed.astype(np.float32),
        "td2_off_form": td2_nonmeed.astype(np.float32),
        "td2_rho": td2_rho.astype(np.float32),
        "td2_psi": td2_psi.astype(np.float32),
        "f_form": f_meed.astype(np.float32),
        "td_closure": td_closure.astype(np.float32),
        "td_form_internal_closure": td_meed_internal_closure.astype(np.float32),
        "td_threeway_closure": td_threeway_closure.astype(np.float32),
        "block_id": block_id,
        "block_boundary": block_boundary.astype(bool),
        "temporal_discontinuity": temporal_discontinuity.astype(bool),
        "valid_transition": valid_transition.astype(bool),
        "valid_topography": valid.astype(bool),
    }
    return arrays, unit_maps


def compute_microstate_association(
    unit_maps: np.ndarray,
    maps: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Compute samplewise correlation competition for one fixed map set."""
    unit_maps = np.asarray(unit_maps, dtype=float)
    maps = center_l2_columns(np.asarray(maps, dtype=float))
    corr_signed = unit_maps.T @ maps
    corr_abs = np.abs(corr_signed)
    sorted_corr_abs = np.sort(corr_abs, axis=1)
    max_abs_corr = sorted_corr_abs[:, -1]
    second_abs_corr = sorted_corr_abs[:, -2] if corr_abs.shape[1] > 1 else np.full(len(corr_abs), np.nan)
    margin = max_abs_corr - second_abs_corr
    margin_relative = np.divide(
        margin,
        max_abs_corr,
        out=np.full(len(corr_abs), np.nan, dtype=float),
        where=max_abs_corr > eps,
    )
    sum_abs = np.sum(corr_abs, axis=1, keepdims=True)
    p_abs = np.divide(corr_abs, sum_abs, out=np.zeros_like(corr_abs), where=sum_abs > eps)
    entropy_abs = -np.sum(p_abs * np.log(np.clip(p_abs, eps, None)), axis=1) / np.log(float(corr_abs.shape[1]))
    corr2 = corr_abs ** 2
    sum_r2 = np.sum(corr2, axis=1, keepdims=True)
    p_r2 = np.divide(corr2, sum_r2, out=np.zeros_like(corr2), where=sum_r2 > eps)
    entropy_r2 = -np.sum(p_r2 * np.log(np.clip(p_r2, eps, None)), axis=1) / np.log(float(corr_abs.shape[1]))
    return {
        "corr_signed": corr_signed.astype(np.float32),
        "corr_abs": corr_abs.astype(np.float32),
        "max_abs_corr": max_abs_corr.astype(np.float32),
        "second_abs_corr": second_abs_corr.astype(np.float32),
        "corr_margin": margin.astype(np.float32),
        "corr_margin_relative": margin_relative.astype(np.float32),
        "corr_entropy_abs": entropy_abs.astype(np.float32),
        "corr_entropy_r2": entropy_r2.astype(np.float32),
    }


# ---------------------------------------------------------------------
# Sequence construction, metrics, and endpoint statistics
# ---------------------------------------------------------------------

METHODS = ("raw", "zanesco", "meed")
STATE_LABELS = ("A", "B", "C", "D", "E")
PRIMARY_METRICS = ("GEV_percent", "duration_ms", "occurrence_per_s")
WITHIN_PRIMARY_PUBLISHED_EFFECTS = {
    ("GEV_percent", "A"),
    ("GEV_percent", "C"),
    ("GEV_percent", "D"),
    ("GEV_percent", "E"),
    *(('duration_ms', state) for state in STATE_LABELS),
    ("occurrence_per_s", "C"),
    ("occurrence_per_s", "D"),
    ("occurrence_per_s", "E"),
}
AGE_PRIMARY_PUBLISHED_EFFECTS = {
    ("GEV_percent", "A"),
    ("GEV_percent", "B"),
    ("GEV_percent", "C"),
    ("GEV_percent", "E"),
    *(('duration_ms', state) for state in STATE_LABELS),
    *(('occurrence_per_s', state) for state in STATE_LABELS),
}


def construct_unsmoothed_sequences(
    raw_label: np.ndarray,
    max_abs_corr: np.ndarray,
    rho2: np.ndarray,
    valid_topography: np.ndarray,
    corr_threshold: float,
    rho2_threshold: float,
) -> dict[str, np.ndarray]:
    raw_label = np.asarray(raw_label, dtype=np.int8).copy()
    valid = np.asarray(valid_topography, dtype=bool)
    raw_label[~valid] = -1
    raw = raw_label.copy()
    zanesco = raw_label.copy()
    zanesco[(np.asarray(max_abs_corr) < float(corr_threshold)) | ~valid] = -1
    meed = raw_label.copy()
    meed[(np.asarray(rho2) < float(rho2_threshold)) | ~valid] = -1
    return {"raw": raw, "zanesco": zanesco, "meed": meed}


def _run_boundaries(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    if labels.size == 0:
        empty = np.asarray([], dtype=int)
        return empty, empty, empty
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    stops = np.r_[starts[1:], labels.size]
    values = labels[starts]
    return starts.astype(int), stops.astype(int), values.astype(int)


def smooth_short_segments(
    labels: np.ndarray,
    data: np.ndarray,
    min_segment_samples: int,
) -> np.ndarray:
    """Apply Pycrostates' short-segment rule without repeated full rescans.

    This is a linear run-list translation of Pycrostates' private
    ``_reject_short_segments`` routine. It retains its left-to-right priority,
    its samplewise adjacent-topography correlation comparison, and its handling
    of undefined labels. The private routine rebuilds every run after each
    reassignment, which is impractical for the hour-long raw recordings used
    here.
    """
    from heapq import heapify, heappop
    from pycrostates.utils import _corr_vectors

    sequence = np.asarray(labels, dtype=np.int8)
    sensor_data = np.asarray(data, dtype=float)
    minimum = int(min_segment_samples)
    if sensor_data.shape[1] != sequence.size:
        raise ValueError("data and labels have different sample counts")
    if minimum <= 1 or sequence.size == 0:
        return sequence.copy()

    starts, stops, values = _run_boundaries(sequence)
    n_runs = len(values)
    previous = np.arange(-1, n_runs - 1, dtype=int)
    following = np.arange(1, n_runs + 1, dtype=int)
    following[-1] = -1
    alive = np.ones(n_runs, dtype=bool)
    candidates = [
        (int(starts[index]), int(index))
        for index in range(1, n_runs - 1)
        if values[index] != -1 and stops[index] - starts[index] < minimum
    ]
    heapify(candidates)

    while candidates:
        _, index = heappop(candidates)
        left = int(previous[index])
        right = int(following[index])
        if (
            not alive[index]
            or left < 0
            or right < 0
            or values[index] == -1
            or stops[index] - starts[index] >= minimum
        ):
            continue

        start = int(starts[index])
        stop = int(stops[index])
        while start < stop:
            left_corr = abs(float(_corr_vectors(sensor_data[:, start - 1], sensor_data[:, start])))
            right_corr = abs(float(_corr_vectors(sensor_data[:, stop - 1], sensor_data[:, stop])))
            if abs(right_corr - left_corr) <= 1e-8:
                if stop - start == 1:
                    stops[left] = start + 1
                    start += 1
                else:
                    stops[left] = start + 1
                    starts[right] = stop - 1
                    start += 1
                    stop -= 1
            elif left_corr < right_corr:
                starts[right] = stop - 1
                stop -= 1
            else:
                stops[left] = start + 1
                start += 1

        # The short run has been fully absorbed into its two neighbours.
        alive[index] = False
        following[left] = right
        previous[right] = left
        previous[index] = following[index] = -1
        if values[left] == values[right]:
            stops[left] = stops[right]
            after_right = int(following[right])
            following[left] = after_right
            if after_right >= 0:
                previous[after_right] = left
            alive[right] = False
            previous[right] = following[right] = -1

    output = sequence.copy()
    for index in np.flatnonzero(alive):
        output[starts[index] : stops[index]] = values[index]
    return output


def smooth_short_segments_by_blocks(
    labels: np.ndarray,
    data: np.ndarray,
    block_id: np.ndarray,
    min_segment_samples: int,
) -> np.ndarray:
    """Apply the native short-segment rule independently within each block."""
    sequence = np.asarray(labels, dtype=np.int8)
    block_id = np.asarray(block_id)
    output = sequence.copy()
    for block in np.unique(block_id):
        indices = np.flatnonzero(block_id == block)
        if indices.size == 0:
            continue
        if not np.all(np.diff(indices) == 1):
            raise ValueError(f"Block {block} is not contiguous")
        output[indices] = smooth_short_segments(
            sequence[indices], np.asarray(data)[:, indices], min_segment_samples
        )
    return output


def smooth_short_segments_paper(
    labels: np.ndarray,
    corr_abs: np.ndarray,
    min_segment_samples: int,
) -> np.ndarray:
    """Apply the paper/Cartool-prose short-segment rule for comparison.

    The paper rule marks the initial short labelled runs, never bridges an
    undefined run, and redistributes a marked run by template correlation.
    It intentionally remains separate from the native Pycrostates branch.
    """
    from heapq import heapify, heappop

    original = np.asarray(labels, dtype=np.int8)
    correlations = np.asarray(corr_abs, dtype=float)
    minimum = int(min_segment_samples)
    if minimum <= 1 or original.size == 0:
        return original.copy()
    if correlations.shape[0] != original.size:
        raise ValueError("corr_abs length differs from sequence length")

    starts, stops, values = _run_boundaries(original)
    n_runs = len(values)
    output = original.copy()
    previous = np.arange(-1, n_runs - 1, dtype=int)
    following = np.arange(1, n_runs + 1, dtype=int)
    following[-1] = -1
    alive = np.ones(n_runs, dtype=bool)
    candidates = [
        (int(starts[index]), int(index))
        for index in range(n_runs)
        if values[index] >= 0 and stops[index] - starts[index] < minimum
    ]
    marked = np.zeros(n_runs, dtype=bool)
    marked[[index for _, index in candidates]] = True
    heapify(candidates)

    while candidates:
        _, index = heappop(candidates)
        if not alive[index] or values[index] < 0 or stops[index] - starts[index] >= minimum:
            continue
        left = int(previous[index])
        right = int(following[index])
        left_labeled = left >= 0 and alive[left] and values[left] >= 0 and not marked[left]
        right_labeled = right >= 0 and alive[right] and values[right] >= 0 and not marked[right]
        start, stop = int(starts[index]), int(stops[index])

        if not left_labeled and not right_labeled:
            values[index] = -1
            continue
        if left_labeled and not right_labeled:
            stops[left] = stop
            following[left] = right
            if right >= 0:
                previous[right] = left
        elif right_labeled and not left_labeled:
            starts[right] = start
            previous[right] = left
            if left >= 0:
                following[left] = right
        elif values[left] == values[right]:
            stops[left] = stops[right]
            after_right = int(following[right])
            following[left] = after_right
            if after_right >= 0:
                previous[after_right] = left
            alive[right] = False
            previous[right] = following[right] = -1
        else:
            preference = np.sign(
                correlations[start:stop, values[left]] - correlations[start:stop, values[right]]
            )
            left_count = 0
            while left_count < preference.size and preference[left_count] > 0:
                left_count += 1
            right_count = 0
            while right_count < preference.size - left_count and preference[-1 - right_count] < 0:
                right_count += 1
            split = start + left_count + (preference.size - left_count - right_count) // 2
            stops[left] = split
            starts[right] = split
            following[left] = right
            previous[right] = left

        alive[index] = False
        previous[index] = following[index] = -1

    for index in np.flatnonzero(alive):
        output[starts[index] : stops[index]] = values[index]
    return output


def smooth_short_segments_paper_by_blocks(
    labels: np.ndarray,
    corr_abs: np.ndarray,
    block_id: np.ndarray,
    min_segment_samples: int,
) -> np.ndarray:
    """Apply the paper/Cartool-prose rule independently within each block."""
    sequence = np.asarray(labels, dtype=np.int8)
    block_id = np.asarray(block_id)
    output = sequence.copy()
    for block in np.unique(block_id):
        indices = np.flatnonzero(block_id == block)
        if indices.size == 0 or not np.all(np.diff(indices) == 1):
            continue
        output[indices] = smooth_short_segments_paper(
            sequence[indices], np.asarray(corr_abs)[indices], min_segment_samples
        )
    return output


def build_smoothed_sequence_products(
    unsmoothed: dict[str, np.ndarray],
    corr_abs: np.ndarray,
    block_id: np.ndarray,
    data: np.ndarray,
    min_segment_samples: int,
) -> dict[str, np.ndarray]:
    """Create only the native and paper-equivalent smoothing branches."""
    products: dict[str, np.ndarray] = {}
    for method, sequence in unsmoothed.items():
        products[f"{method}_condition_smoothed"] = smooth_short_segments(
            sequence, data, min_segment_samples
        )
        products[f"{method}_block_smoothed"] = smooth_short_segments_by_blocks(
            sequence, data, block_id, min_segment_samples
        )
        products[f"{method}_paper_condition_smoothed"] = smooth_short_segments_paper(
            sequence, corr_abs, min_segment_samples
        )
        products[f"{method}_paper_block_smoothed"] = smooth_short_segments_paper_by_blocks(
            sequence, corr_abs, block_id, min_segment_samples
        )
    return products


def compute_template_geometry_distances(
    q: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    maps: np.ndarray,
    s_orth: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Compute polarity-invariant dipole angular and theta/phi map distances."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape samples x 3")
    maps = center_l2_columns(np.asarray(maps, dtype=float))
    s_orth = np.asarray(s_orth, dtype=float)
    template_q = s_orth.T @ maps
    template_norm = np.linalg.norm(template_q, axis=0)
    template_u = np.divide(
        template_q,
        template_norm[None, :],
        out=np.zeros_like(template_q),
        where=template_norm[None, :] > eps,
    )
    q_norm = np.linalg.norm(q, axis=1)
    q_u = np.divide(q.T, q_norm[None, :], out=np.zeros((3, len(q))), where=q_norm[None, :] > eps)
    signed_dot = q_u.T @ template_u
    psi = np.degrees(np.arccos(np.clip(np.abs(signed_dot), -1.0, 1.0)))

    theta_distance = np.full_like(psi, np.nan, dtype=float)
    phi_distance = np.full_like(psi, np.nan, dtype=float)
    for j in range(maps.shape[1]):
        th_pos, ph_pos, _ = meed.canonical_theta_phi_from_vector(template_u[:, j])
        th_neg, ph_neg, _ = meed.canonical_theta_phi_from_vector(-template_u[:, j])
        use_neg = signed_dot[:, j] < 0
        theta_target = np.where(use_neg, th_neg, th_pos)
        phi_target = np.where(use_neg, ph_neg, ph_pos)
        theta_distance[:, j] = np.abs((theta_deg - theta_target + 180.0) % 360.0 - 180.0)
        phi_distance[:, j] = np.abs(phi_deg - phi_target)
    invalid = q_norm <= eps
    psi[invalid] = np.nan
    theta_distance[invalid] = np.nan
    phi_distance[invalid] = np.nan
    return {
        "psi_deg": psi.astype(np.float32),
        "theta_distance_deg": theta_distance.astype(np.float32),
        "phi_distance_deg": phi_distance.astype(np.float32),
    }
