"""Real-data MEED utilities, v2.

Narrow scope: load one cleaned EDF at a time, fit subject/group maps with
pycrostates, build MEED representations using the validated `utils.py`, and
compute samplewise correlations plus MEED dipolarity/angular distances.

No smoothing, no threshold-grid cleaning, no sequence metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import inspect
import warnings
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import mne
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

import utils as u

LEGACY_TO_MODERN = getattr(u, "LEGACY_TO_MODERN", {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"})


@dataclass
class TemplateLevel:
    """Map set plus fixed-MEED representation for one level: meta, subject, group."""

    name: str
    B: np.ndarray              # channels x maps, centered/L2 columns
    labels: list[str]
    info: mne.Info
    geom: dict[str, Any]
    meed: dict[str, Any]
    model: Any | None = None

    @property
    def maps_row(self) -> np.ndarray:
        return self.B.T

    @property
    def U(self) -> np.ndarray:
        return np.asarray(self.meed["U"], dtype=float)

    @property
    def D(self) -> np.ndarray:
        return np.asarray(self.meed["D"], dtype=float)

    @property
    def df(self) -> pd.DataFrame:
        out = self.meed["df"].copy()
        out["level"] = self.name
        out["label"] = self.labels
        return out


def standard_names_picker(rawobj, *, set_montage=True):
    old_to_new = {
        "t3": "T7",
        "t4": "T8",
        "t5": "P7",
        "t6": "P8",
    }

    montage = mne.channels.make_standard_montage("standard_1020")
    std = {name.lower(): name for name in montage.ch_names}

    keep = []
    rename = {}
    used = set()
    dropped = []

    for ch in rawobj.ch_names:
        key = ch.strip().lower()
        new_name = old_to_new.get(key, std.get(key))

        if new_name is None:
            dropped.append((ch, "not in standard_1020"))
            continue

        if new_name in used:
            dropped.append((ch, f"duplicate of {new_name}"))
            continue

        keep.append(ch)
        rename[ch] = new_name
        used.add(new_name)

    raw = rawobj.copy().pick(keep)
    raw.rename_channels(rename)

    if set_montage:
        raw.set_montage(montage, on_missing="raise")

    raw.info["description"] = (
        (raw.info.get("description") or "")
        + f"\nstandard_names_picker kept {len(keep)} channels, dropped {len(dropped)}."
    )

    return raw


def find_input_files(data_root: str | Path, pattern: str = "*.*") -> list[Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def load_edf_as_eeg(edf_path: str | Path, *, preload: bool = True, verbose: str | bool = "ERROR") -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=verbose)
    return standard_names_picker(raw)


def sidecar_csv_for_edf(edf_path: str | Path) -> Path | None:
    p = Path(edf_path).with_suffix(".csv")
    return p if p.exists() else None


def read_sidecar_csv(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def align_raw_to_template_channels(raw: mne.io.BaseRaw, template_ch_names: Iterable[str], *, strict: bool = True) -> mne.io.BaseRaw:
    """Align an EEG Raw to the template channel order. No channels are invented."""
    out = standard_names_picker(raw)
    template_ch = [str(ch).strip() for ch in template_ch_names]
    missing = [ch for ch in template_ch if ch not in out.ch_names]
    if missing and strict:
        raise ValueError(f"Raw is missing template EEG channels: {missing}\n \
            Raw channels: {out.ch_names}\nTemplate channels: {template_ch}")
    keep = [ch for ch in template_ch if ch in out.ch_names]
    out.reorder_channels(keep)
    return out


def raw_to_channel_time(raw: mne.io.BaseRaw) -> tuple[np.ndarray, float]:
    return np.asarray(raw.get_data(picks="eeg"), dtype=float), float(raw.info["sfreq"])


def center_l2_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc / np.maximum(np.linalg.norm(Xc, axis=0, keepdims=True), eps)


def gfp_from_channel_time(X: np.ndarray) -> np.ndarray:
    Xc = np.asarray(X, dtype=float) - np.asarray(X, dtype=float).mean(axis=0, keepdims=True)
    return np.sqrt(np.mean(Xc ** 2, axis=0))


def backward_topographic_dissimilarity(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Backward topographic dissimilarity / global map dissimilarity.

    Parameters
    ----------
    X : np.ndarray, shape (n_channels, n_times)
        Raw channel-by-time EEG data, not pre-normalized.
    eps : float
        Small value to avoid division by zero at near-zero GFP.

    Returns
    -------
    td : np.ndarray, shape (n_times,)
        Backward topographic dissimilarity. td[0] is NaN.

    Notes
    -----
    This follows the GFP-normalized map-dissimilarity logic used in the
    GFP/topographic-stability literature:

        1. average-reference each time point
        2. divide each map by its GFP
        3. compute RMS difference between adjacent normalized maps

    This is polarity-sensitive and has theoretical range [0, 2].
    """

    X = np.asarray(X, dtype=float)

    # Average-reference each time point.
    Xc = X - X.mean(axis=0, keepdims=True)

    # GFP per time point.
    gfp = np.sqrt(np.mean(Xc ** 2, axis=0))

    # GFP-normalized maps.
    Z = Xc / np.maximum(gfp[None, :], eps)

    # RMS difference between adjacent normalized maps.
    diff = Z[:, 1:] - Z[:, :-1]

    td = np.full(X.shape[1], np.nan)
    td[1:] = np.sqrt(np.mean(diff ** 2, axis=0))

    return td


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def fit_pycrostates_maps(raw_aligned: mne.io.BaseRaw, *, n_clusters: int = 4, random_state: int = 0, fit_kwargs: dict[str, Any] | None = None) -> tuple[np.ndarray, Any]:
    """Fit pycrostates ModKMeans and return maps as centered/L2 columns."""
    try:
        from pycrostates.cluster import ModKMeans
    except Exception as exc:
        raise ImportError("pycrostates is required for fitting maps.") from exc

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    init_kwargs = {}
    for key in ("max_iter", "tol"):
        if key in fit_kwargs:
            init_kwargs[key] = fit_kwargs.pop(key)
    model = ModKMeans(n_clusters=int(n_clusters), random_state=int(random_state), **init_kwargs)
    raw_eeg = raw_aligned.copy().pick("eeg")
    model.fit(raw_eeg, **_filter_kwargs(model.fit, fit_kwargs))
    maps_row = np.asarray(model.cluster_centers_, dtype=float)
    return u.center_l2_cols(maps_row.T), model


def build_template_level(name: str, B: np.ndarray, labels: Iterable[str], info: mne.Info, *, model: Any | None = None) -> TemplateLevel:
    """Build fixed-MEED representation for a map set."""
    labels = [str(x) for x in labels]
    B = u.center_l2_cols(np.asarray(B, dtype=float))
    geom = u.sensor_geometry(info)
    if hasattr(u, "fixed_meed"):
        meed = u.fixed_meed(B, geom["S_orth"])
    else:
        Q, Bproj, Bmeed, df = u.project_fixed_dipole(B, geom["S_orth"])
        U = np.column_stack([u.unit_from_angles(th, ph) for th, ph in zip(df["theta_deg"], df["phi_deg"])])
        D = np.vstack([df["theta_deg"].to_numpy(float), df["phi_deg"].to_numpy(float)])
        meed = {"Q": Q, "U": U, "D": D, "B_projection": Bproj, "B_meed": Bmeed, "df": df}
    return TemplateLevel(name=name, B=B, labels=labels, info=info.copy(), geom=geom, meed=meed, model=model)


def align_template_maps_to_reference(B: np.ndarray, B_ref: np.ndarray, labels: Iterable[str] | None = None, ref_labels: Iterable[str] | None = None) -> dict[str, Any]:
    """Reorder and polarity-align maps to a reference using absolute correlation."""
    B = u.center_l2_cols(B)
    B_ref = u.center_l2_cols(B_ref)
    C_signed = B_ref.T @ B
    C_abs = np.abs(C_signed)
    rows, cols = linear_sum_assignment(-C_abs)
    ref_order = np.argsort(rows)
    rows, cols = rows[ref_order], cols[ref_order]
    B_aligned = B[:, cols].copy()
    signs = np.sign(np.sum(B_ref[:, rows] * B_aligned, axis=0))
    signs[signs == 0] = 1
    B_aligned *= signs[None, :]
    labels = [str(i) for i in range(B.shape[1])] if labels is None else [str(x) for x in labels]
    ref_labels = [str(i) for i in range(B_ref.shape[1])] if ref_labels is None else [str(x) for x in ref_labels]
    labels_aligned = [ref_labels[i] if i < len(ref_labels) else labels[j] for i, j in zip(rows, cols)]
    return {
        "B_aligned": B_aligned,
        "labels_aligned": labels_aligned,
        "assignment_ref_index": rows,
        "assignment_map_index": cols,
        "corr_signed": C_signed,
        "corr_abs": C_abs,
        "matched_corr_abs": C_abs[rows, cols],
        "signs": signs,
    }


def fit_subject_template_level(raw_aligned: mne.io.BaseRaw, *, n_clusters: int = 4, random_state: int = 0, reference_level: TemplateLevel | None = None, name: str = "subject", fit_kwargs: dict[str, Any] | None = None) -> TemplateLevel:
    B, model = fit_pycrostates_maps(raw_aligned, n_clusters=n_clusters, random_state=random_state, fit_kwargs=fit_kwargs)
    labels = [chr(ord("A") + i) for i in range(B.shape[1])]
    if reference_level is not None:
        aligned = align_template_maps_to_reference(B, reference_level.B, labels, reference_level.labels)
        B, labels = aligned["B_aligned"], aligned["labels_aligned"]
    return build_template_level(name, B, labels, raw_aligned.info, model=model)


def concatenate_raws_in_memory(raws: list[mne.io.BaseRaw]) -> mne.io.RawArray:
    if not raws:
        raise ValueError("No raws supplied.")
    ch0, sf0 = list(raws[0].ch_names), float(raws[0].info["sfreq"])
    arrays = []
    for i, raw in enumerate(raws):
        if list(raw.ch_names) != ch0:
            raise ValueError(f"Raw {i} has different channel order.")
        if not np.isclose(float(raw.info["sfreq"]), sf0):
            raise ValueError(f"Raw {i} has different sampling rate.")
        arrays.append(raw.get_data(picks="eeg"))
    return mne.io.RawArray(np.concatenate(arrays, axis=1), raws[0].info.copy(), verbose="ERROR")


def fit_group_template_level(edf_files: Iterable[str | Path], template_ch_names: Iterable[str], *, n_clusters: int = 4, random_state: int = 0, reference_level: TemplateLevel | None = None, name: str = "group", fit_kwargs: dict[str, Any] | None = None) -> TemplateLevel:
    raws = []
    for p in edf_files:
        raw = load_edf_as_eeg(p, preload=True)
        raws.append(align_raw_to_template_channels(raw, template_ch_names, strict=True))
    group_raw = concatenate_raws_in_memory(raws)
    return fit_subject_template_level(group_raw, n_clusters=n_clusters, random_state=random_state, reference_level=reference_level, name=name, fit_kwargs=fit_kwargs)


def antipodal_angles(U_a: np.ndarray, U_b: np.ndarray) -> np.ndarray:
    A = np.asarray(U_a, dtype=float)
    B = np.asarray(U_b, dtype=float)
    A = A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-12)
    sim = np.clip(np.abs(A.T @ B), -1.0, 1.0)
    return np.degrees(np.arccos(sim))


def compare_template_levels(level_a: TemplateLevel, level_b: TemplateLevel) -> dict[str, Any]:
    C_signed = level_a.B.T @ level_b.B
    C_abs = np.abs(C_signed)
    angles = antipodal_angles(level_a.U, level_b.U)
    rows, cols = linear_sum_assignment(-C_abs)
    matched = pd.DataFrame({
        "a_index": rows,
        "a_label": [level_a.labels[i] for i in rows],
        "b_index": cols,
        "b_label": [level_b.labels[j] for j in cols],
        "abs_corr": C_abs[rows, cols],
        "signed_corr": C_signed[rows, cols],
        "MEED_angle_deg": angles[rows, cols],
    })
    return {"sensor_corr_signed": C_signed, "sensor_corr_abs": C_abs, "MEED_angle_deg": angles, "matched": matched}


def project_data_to_meed(X: np.ndarray, S_orth: np.ndarray) -> dict[str, Any]:
    """Project channel x sample data directly to fixed MEED."""
    Xn = center_l2_columns(X)
    Q = np.asarray(S_orth, dtype=float).T @ Xn
    rho = np.linalg.norm(Q, axis=0)
    U = np.zeros_like(Q)
    theta = np.full(Q.shape[1], np.nan)
    phi = np.full(Q.shape[1], np.nan)
    for i in range(Q.shape[1]):
        th, ph, ui = u.canonical_theta_phi_from_vector(Q[:, i])
        theta[i], phi[i], U[:, i] = th, ph, ui
    return {"X_norm": Xn, "Q": Q, "rho": rho, "projection_R2": rho ** 2, "U": U, "theta_deg": theta, "phi_deg": phi}


def backward_meed_angular_velocity(U_samples: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Backward physical MEED angular velocity.

    Parameters
    ----------
    U_samples : np.ndarray, shape (3, n_times)
        Unit MEED dipole directions for each sample.
        This function treats the dipole as oriented, not polarity-invariant.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    velocity : np.ndarray, shape (n_times,)
        Backward angular velocity in deg/s. velocity[0] is NaN.

    Notes
    -----
    This is polarity-sensitive:

        dpsi_t = arccos(u_t dot u_{t-1})

    not:

        arccos(abs(u_t dot u_{t-1}))

    Therefore opposite dipoles are maximally distant rather than equivalent.
    This is closer to the Zanesco-style adjacent-map dissimilarity logic.
    """

    U = np.asarray(U_samples, dtype=float)

    # Ensure unit directions. This is harmless if U is already normalized.
    U = U / np.maximum(np.linalg.norm(U, axis=0, keepdims=True), 1e-12)

    velocity = np.full(U.shape[1], np.nan)

    if U.shape[1] < 2:
        return velocity

    # Polarity-sensitive adjacent angular displacement.
    sim = np.sum(U[:, 1:] * U[:, :-1], axis=0)
    sim = np.clip(sim, -1.0, 1.0)

    dpsi_deg = np.degrees(np.arccos(sim))

    velocity[1:] = dpsi_deg * float(sfreq)

    return velocity


def sample_template_correlations(X_norm: np.ndarray, level: TemplateLevel, *, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    signed = X_norm.T @ level.B
    absolute = np.abs(signed)
    squared = signed ** 2
    best = np.argmax(absolute, axis=1)
    order = np.argsort(-absolute, axis=1)
    second = order[:, 1] if absolute.shape[1] > 1 else np.full(absolute.shape[0], -1)
    margin = absolute[np.arange(absolute.shape[0]), order[:, 0]] - absolute[np.arange(absolute.shape[0]), order[:, 1]] if absolute.shape[1] > 1 else np.full(absolute.shape[0], np.nan)
    df = pd.DataFrame({
        f"{prefix}_corr_best_idx": best,
        f"{prefix}_corr_best_label": [level.labels[i] for i in best],
        f"{prefix}_corr_best_abs": absolute[np.arange(absolute.shape[0]), best],
        f"{prefix}_corr_best_R2": squared[np.arange(squared.shape[0]), best],
        f"{prefix}_corr_second_idx": second,
        f"{prefix}_corr_margin_abs": margin,
    })
    for j, label in enumerate(level.labels):
        safe = str(label).replace(" ", "_")
        df[f"{prefix}_corr_signed_{safe}"] = signed[:, j]
        df[f"{prefix}_corr_abs_{safe}"] = absolute[:, j]
        df[f"{prefix}_corr_R2_{safe}"] = squared[:, j]
    return df, {"signed": signed, "absolute": absolute, "squared": squared}


def sample_template_angles(U_samples: np.ndarray, level: TemplateLevel, *, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    angles = antipodal_angles(U_samples, level.U)
    best = np.argmin(angles, axis=1)
    order = np.argsort(angles, axis=1)
    second = order[:, 1] if angles.shape[1] > 1 else np.full(angles.shape[0], -1)
    margin = angles[np.arange(angles.shape[0]), order[:, 1]] - angles[np.arange(angles.shape[0]), order[:, 0]] if angles.shape[1] > 1 else np.full(angles.shape[0], np.nan)
    df = pd.DataFrame({
        f"{prefix}_angle_best_idx": best,
        f"{prefix}_angle_best_label": [level.labels[i] for i in best],
        f"{prefix}_angle_min_deg": angles[np.arange(angles.shape[0]), best],
        f"{prefix}_angle_second_idx": second,
        f"{prefix}_angle_margin_deg": margin,
    })
    for j, label in enumerate(level.labels):
        safe = str(label).replace(" ", "_")
        df[f"{prefix}_angle_deg_{safe}"] = angles[:, j]
    return df, {"angles_deg": angles}


def characterize_raw_with_template_levels(raw_aligned: mne.io.BaseRaw, template_levels: dict[str, TemplateLevel], *, data_geometry_level: str = "meta") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute samplewise MEED, correlations, and angular distances for one Raw."""
    if data_geometry_level not in template_levels:
        raise KeyError(f"data_geometry_level={data_geometry_level!r} not in template_levels")
    X, sfreq = raw_to_channel_time(raw_aligned)
    Xn = center_l2_columns(X)
    dip = project_data_to_meed(X, template_levels[data_geometry_level].geom["S_orth"])
    base = pd.DataFrame({
        "sample": np.arange(X.shape[1], dtype=int),
        "time_s": np.arange(X.shape[1], dtype=float) / sfreq,
        "gfp": gfp_from_channel_time(X),
        "backward_topographic_dissimilarity": backward_topographic_dissimilarity(Xn),
        "MEED_theta_deg": dip["theta_deg"],
        "MEED_phi_deg": dip["phi_deg"],
        "MEED_rho": dip["rho"],
        "MEED_projection_R2": dip["projection_R2"],
        "MEED_backward_angular_velocity_deg_s": backward_meed_angular_velocity(dip["U"], sfreq),
    })
    frames = [base]
    matrices: dict[str, Any] = {"X": X, "X_norm": Xn, "data_meed": dip}
    for level_name, level in template_levels.items():
        cdf, cpayload = sample_template_correlations(Xn, level, prefix=level_name)
        adf, apayload = sample_template_angles(dip["U"], level, prefix=level_name)
        frames.extend([cdf, adf])
        matrices[level_name] = {"level": level, "correlations": cpayload, "angles": apayload}
    return pd.concat(frames, axis=1), matrices


def template_level_summary(level: TemplateLevel) -> pd.DataFrame:
    df = level.df.copy()
    keep = ["level", "label", "theta_deg", "phi_deg", "dipolarity", "projection_R2", "R2"]
    return df[[c for c in keep if c in df.columns]]


def summarize_sample_features(sample_df: pd.DataFrame, *, prefix: str = "") -> pd.Series:
    out = {}
    for col in ["gfp", "backward_topographic_dissimilarity", "MEED_projection_R2", "MEED_backward_angular_velocity_deg_s"]:
        if col in sample_df:
            out[f"{prefix}{col}_mean"] = float(np.nanmean(sample_df[col]))
            out[f"{prefix}{col}_median"] = float(np.nanmedian(sample_df[col]))
    for col in sample_df.columns:
        if col.endswith("_corr_best_R2") or col.endswith("_angle_min_deg"):
            out[f"{prefix}{col}_mean"] = float(np.nanmean(sample_df[col]))
            out[f"{prefix}{col}_median"] = float(np.nanmedian(sample_df[col]))
    return pd.Series(out)


def plot_template_meed_coordinates(levels: dict[str, TemplateLevel]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for name, level in levels.items():
        D = level.D
        ax.scatter(D[0], D[1], label=name, s=45)
        for j, lab in enumerate(level.labels):
            ax.text(D[0, j], D[1, j], f"{name}:{lab}", fontsize=8, ha="center", va="bottom")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(r"$\theta$ anatomical azimuth (deg, polarity-canonical)")
    ax.set_ylabel(r"$\phi$ elevation (deg)")
    ax.set_title("Template MEED coordinates")
    ax.legend()
    return fig


def plot_template_comparison_heatmaps(comparison: dict[str, Any], title: str = "") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(comparison["sensor_corr_abs"], vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("absolute map correlation")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    im1 = axes[1].imshow(comparison["MEED_angle_deg"], vmin=0, vmax=90, cmap="magma_r")
    axes[1].set_title("MEED antipodal angle (deg)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    if title:
        fig.suptitle(title)
    return fig


def plot_sample_meed_overview(sample_df: pd.DataFrame, *, start_s: float | None = None, stop_s: float | None = None) -> plt.Figure:
    d = sample_df.copy()
    if start_s is not None:
        d = d[d["time_s"] >= start_s]
    if stop_s is not None:
        d = d[d["time_s"] <= stop_s]
    fig, axes = plt.subplots(5, 1, figsize=(12, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(d["time_s"], d["gfp"], linewidth=0.8)
    axes[0].set_ylabel("GFP")
    axes[1].plot(d["time_s"], d["backward_topographic_dissimilarity"], linewidth=0.8)
    axes[1].set_ylabel("backward TD")
    axes[2].plot(d["time_s"], d["MEED_theta_deg"], linewidth=0.8, label=r"$\theta$")
    axes[2].plot(d["time_s"], d["MEED_phi_deg"], linewidth=0.8, label=r"$\phi$")
    axes[2].set_ylabel("MEED angle")
    axes[2].legend(loc="upper right")
    axes[3].plot(d["time_s"], d["MEED_projection_R2"], linewidth=0.8)
    axes[3].set_ylabel("MEED R2")
    axes[4].plot(d["time_s"], d["MEED_backward_angular_velocity_deg_s"], linewidth=0.8)
    axes[4].set_ylabel("MEED velocity")
    axes[4].set_xlabel("time (s)")
    return fig


# ---------------------------------------------------------------------
# v3 additions: richer section 3/4 real-data characterization
# ---------------------------------------------------------------------

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "bluishgreen": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddishpurple": "#CC79A7",
    "gray": "#999999",
}

OKABE_STATE_COLORS = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluishgreen"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["skyblue"],
    OKABE_ITO["reddishpurple"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
]


def _safe_label(label: Any) -> str:
    return str(label).replace(" ", "_").replace("-", "_").replace("/", "_")


def _level_suffix(prefix: str) -> str:
    aliases = {"subject": "sub"}
    return aliases.get(prefix, prefix)


def parse_recording_info(path: str | Path | None = None, *, sub: str | None = None, ses: str | None = None) -> dict[str, str]:
    """Parse `sub-*` and `ses-*` fields from a filename.

    Explicit `sub` or `ses` override filename parsing. Missing values become
    `"NA"` so the sample table always has stable columns.
    """
    if path is None:
        stem = ""
    else:
        stem = Path(path).stem

    def find_field(prefix: str) -> str:
        match = re.search(rf"{prefix}-([^_]+)", stem)
        return match.group(1) if match else "NA"

    return {
        "sub": str(sub) if sub is not None else find_field("sub"),
        "ses": str(ses) if ses is not None else find_field("ses"),
        "recording": stem if stem else "NA",
    }


def _theta_phi_from_unit_columns(Umat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.empty(Umat.shape[1])
    phi = np.empty(Umat.shape[1])
    for i in range(Umat.shape[1]):
        theta[i], phi[i], _ = u.canonical_theta_phi_from_vector(Umat[:, i])
    return theta, phi


def _template_pair_component_distances(U_samples: np.ndarray, level: TemplateLevel) -> dict[str, np.ndarray]:
    """Return signed/absolute component distances and composite angles.

    For each sample-template pair, the template direction is flipped when needed
    so that the comparison is polarity-invariant before extracting theta/phi
    component differences.
    """
    Us = np.asarray(U_samples, dtype=float)
    Ut = np.asarray(level.U, dtype=float)
    Us = Us / np.maximum(np.linalg.norm(Us, axis=0, keepdims=True), 1e-12)
    Ut = Ut / np.maximum(np.linalg.norm(Ut, axis=0, keepdims=True), 1e-12)

    theta_s, phi_s = _theta_phi_from_unit_columns(Us)
    n_t, n_m = Us.shape[1], Ut.shape[1]

    angle = np.empty((n_t, n_m))
    dtheta_signed = np.empty((n_t, n_m))
    dphi_signed = np.empty((n_t, n_m))
    dtheta_abs = np.empty((n_t, n_m))
    dphi_abs = np.empty((n_t, n_m))

    for j in range(n_m):
        dot = np.sum(Us * Ut[:, [j]], axis=0)
        sign = np.where(dot < 0, -1.0, 1.0)
        Ut_aligned = Ut[:, [j]] * sign[None, :]
        theta_t, phi_t = _theta_phi_from_unit_columns(Ut_aligned)

        dth = u.wrap180(theta_s - theta_t)
        dph = phi_s - phi_t

        dtheta_signed[:, j] = dth
        dphi_signed[:, j] = dph
        dtheta_abs[:, j] = np.abs(dth)
        dphi_abs[:, j] = np.abs(dph)
        angle[:, j] = np.degrees(np.arccos(np.clip(np.abs(dot), -1.0, 1.0)))

    return {
        "angle_deg": angle,
        "dtheta_signed_deg": dtheta_signed,
        "dphi_signed_deg": dphi_signed,
        "dtheta_abs_deg": dtheta_abs,
        "dphi_abs_deg": dphi_abs,
    }


def _topomap_on_axis(ax, values: np.ndarray, info: mne.Info, vlim: float, title: str):
    mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        contours=4,
        sensors=True,
        cmap="RdBu_r",
        vlim=(-vlim, vlim),
        sphere="auto",
        extrapolate="head",
        image_interp="linear",
    )
    ax.set_title(title, fontsize=8)


def plot_template_maps_grid(
    levels: dict[str, TemplateLevel],
    *,
    title: str = "Normalized template maps",
    r2_kind: str = "meed",
    meta_topomap_scale: float = 1.0,
    level_topomap_scale: float = 1.0,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot meta-referenced template maps with non-meta maps positioned by R²."""
    if len(levels) == 0:
        raise ValueError("No template levels provided.")
    if "meta" not in levels:
        raise ValueError("`levels` must include a `meta` TemplateLevel.")
    if r2_kind not in ("meed", "sensor"):
        raise ValueError("`r2_kind` must be either 'meed' or 'sensor'.")

    meta_level = levels["meta"]
    meta_labels = [str(lab) for lab in meta_level.labels]
    x_positions = np.arange(len(meta_labels), dtype=float)
    x_lookup = {lab: x for x, lab in zip(x_positions, meta_labels)}
    vlim = max(float(np.max(np.abs(level.B))) for level in levels.values())
    n_non_meta = max(1, len(levels) - 1)
    r2_col = "MEED_projection_R2" if r2_kind == "meed" else "R2"
    all_r2 = []
    for name, level in levels.items():
        if name == "meta":
            continue
        summary = template_level_summary(level)
        if r2_col in summary.columns:
            vals = summary[r2_col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                all_r2.append(vals)
    min_r2 = float(np.min(np.concatenate(all_r2))) if all_r2 else 0.0
    y_min = min_r2 - 0.15
    y_max = 1.15
    base_size = max(7.5, 1.9 * len(meta_labels), 7.5 + 0.35 * n_non_meta)
    fig_width, fig_height = figsize if figsize is not None else (base_size, base_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_xlim(-0.5, len(meta_labels) - 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_ylabel(r"MEED projection $R^2$" if r2_kind == "meed" else r"Sensor correlation $R^2$")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)

    meta_width = 0.58 * float(meta_topomap_scale)
    meta_height = 0.18 * float(meta_topomap_scale)
    meta_bottom = y_min + 0.02
    for j, lab in enumerate(meta_labels):
        x = x_positions[j]
        inset = ax.inset_axes(
            [x - meta_width / 2.0, meta_bottom, meta_width, meta_height],
            transform=ax.transData,
        )
        _topomap_on_axis(inset, meta_level.B[:, j], meta_level.info, vlim, lab)
        inset.patch.set_edgecolor("black")
        inset.patch.set_linewidth(1.0)

    non_meta_items = [(name, level) for name, level in levels.items() if name != "meta"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(non_meta_items))))
    thumb_width = 0.42 * float(level_topomap_scale)
    thumb_height = 0.14 * float(level_topomap_scale)
    offsets = np.linspace(-0.18, 0.18, max(1, len(non_meta_items)))

    for idx, ((name, level), color, x_offset) in enumerate(zip(non_meta_items, colors, offsets)):
        summary = template_level_summary(level)
        if r2_col not in summary.columns:
            continue

        row_lookup = {str(row["label"]): row for _, row in summary.iterrows()}
        map_lookup = {str(lab): level.B[:, j] for j, lab in enumerate(level.labels)}

        for lab in meta_labels:
            if lab not in row_lookup or lab not in map_lookup:
                continue
            r2 = float(row_lookup[lab].get(r2_col, np.nan))
            if not np.isfinite(r2):
                continue
            x = x_lookup[lab] + x_offset
            y = np.clip(r2, y_min, y_max)
            inset = ax.inset_axes(
                [x - thumb_width / 2.0, y - thumb_height / 2.0, thumb_width, thumb_height],
                transform=ax.transData,
            )
            _topomap_on_axis(inset, map_lookup[lab], level.info, vlim, "")
            inset.patch.set_edgecolor(color)
            inset.patch.set_linewidth(2.0)

    fig.subplots_adjust(bottom=0.12, left=0.10, right=0.98, top=0.92)
    return fig


def template_level_summary(level: TemplateLevel) -> pd.DataFrame:
    """Return map-level MEED descriptors with explicit rho/R2 naming."""
    df = level.df.copy()
    rename = {}
    if "dipolarity" in df.columns:
        rename["dipolarity"] = "rho"
    if "projection_R2" in df.columns:
        rename["projection_R2"] = "MEED_projection_R2"
    df = df.rename(columns=rename)
    df["level"] = level.name
    df["label"] = level.labels

    keep = [
        "level",
        "label",
        "theta_deg",
        "phi_deg",
        "rho",
        "MEED_projection_R2",
        "R2",
    ]
    return df[[c for c in keep if c in df.columns]]


def plot_template_meed_coordinates(
    levels: dict[str, TemplateLevel],
    *,
    annotate: str = "rho_R2",
    title: str = "Template MEED coordinates",
) -> plt.Figure:
    """Scatter MEED template coordinates with rho/R2 annotations."""
    fig, ax = plt.subplots(figsize=(7.5, 5.6))

    for i, (name, level) in enumerate(levels.items()):
        D = level.D
        df = template_level_summary(level)
        color = OKABE_STATE_COLORS[i % len(OKABE_STATE_COLORS)]
        ax.scatter(D[0], D[1], label=name, s=55, color=color)

        for j, lab in enumerate(level.labels):
            row = df.iloc[j]
            if annotate == "rho_R2":
                extra = f"\nρ={row.get('rho', np.nan):.2f}, R²={row.get('MEED_projection_R2', np.nan):.2f}"
            elif annotate == "R2":
                extra = f"\nR²={row.get('MEED_projection_R2', np.nan):.2f}"
            elif annotate == "rho":
                extra = f"\nρ={row.get('rho', np.nan):.2f}"
            else:
                extra = ""
            ax.text(
                D[0, j],
                D[1, j],
                f"{name}:{lab}{extra}",
                fontsize=7,
                ha="center",
                va="bottom",
            )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(r"$\theta$ anatomical azimuth (deg, polarity-canonical)")
    ax.set_ylabel(r"$\phi$ elevation (deg)")
    ax.set_title(title)
    ax.legend()
    return fig


def sample_template_correlations(X_norm: np.ndarray, level: TemplateLevel, *, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Samplewise signed, absolute, and squared correlations to templates.

    Adds compact columns such as `C_Ameta`, `C2_Ameta`, `C_Asub`, `C2_Asub`
    while preserving the previous verbose columns for compatibility.
    """
    signed = X_norm.T @ level.B
    absolute = np.abs(signed)
    squared = signed ** 2

    best = np.argmax(absolute, axis=1)
    order = np.argsort(-absolute, axis=1)
    second = order[:, 1] if absolute.shape[1] > 1 else np.full(absolute.shape[0], -1)
    margin = (
        absolute[np.arange(absolute.shape[0]), order[:, 0]]
        - absolute[np.arange(absolute.shape[0]), order[:, 1]]
        if absolute.shape[1] > 1
        else np.full(absolute.shape[0], np.nan)
    )

    suffix = _level_suffix(prefix)
    df = pd.DataFrame(
        {
            f"{prefix}_corr_best_idx": best,
            f"{prefix}_corr_best_label": [level.labels[i] for i in best],
            f"{prefix}_corr_best_abs": absolute[np.arange(absolute.shape[0]), best],
            f"{prefix}_corr_best_R2": squared[np.arange(squared.shape[0]), best],
            f"{prefix}_corr_second_idx": second,
            f"{prefix}_corr_margin_abs": margin,
        }
    )

    for j, label in enumerate(level.labels):
        safe = _safe_label(label)
        compact = f"{safe}{suffix}"

        # Compact columns requested in the notebook text.
        df[f"C_{compact}"] = absolute[:, j]
        df[f"C2_{compact}"] = squared[:, j]
        df[f"Csign_{compact}"] = signed[:, j]

        # Verbose backward-compatible columns.
        df[f"{prefix}_corr_signed_{safe}"] = signed[:, j]
        df[f"{prefix}_corr_abs_{safe}"] = absolute[:, j]
        df[f"{prefix}_corr_R2_{safe}"] = squared[:, j]

    return df, {"signed": signed, "absolute": absolute, "squared": squared}


def sample_template_angles(U_samples: np.ndarray, level: TemplateLevel, *, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Samplewise MEED component and composite distances to templates.

    Compact columns are:
    `theta_Ameta`, `phi_Ameta`, `angle_Ameta`, and analogous subject/group
    columns. Theta and phi are component distances after polarity alignment;
    `angle_*` is the true antipodal angular distance.
    """
    comp = _template_pair_component_distances(U_samples, level)
    angles = comp["angle_deg"]
    best = np.argmin(angles, axis=1)
    order = np.argsort(angles, axis=1)
    second = order[:, 1] if angles.shape[1] > 1 else np.full(angles.shape[0], -1)
    margin = (
        angles[np.arange(angles.shape[0]), order[:, 1]]
        - angles[np.arange(angles.shape[0]), order[:, 0]]
        if angles.shape[1] > 1
        else np.full(angles.shape[0], np.nan)
    )

    suffix = _level_suffix(prefix)
    df = pd.DataFrame(
        {
            f"{prefix}_angle_best_idx": best,
            f"{prefix}_angle_best_label": [level.labels[i] for i in best],
            f"{prefix}_angle_min_deg": angles[np.arange(angles.shape[0]), best],
            f"{prefix}_angle_second_idx": second,
            f"{prefix}_angle_margin_deg": margin,
        }
    )

    for j, label in enumerate(level.labels):
        safe = _safe_label(label)
        compact = f"{safe}{suffix}"

        # Compact columns requested in the notebook text.
        df[f"theta_{compact}"] = comp["dtheta_abs_deg"][:, j]
        df[f"phi_{compact}"] = comp["dphi_abs_deg"][:, j]
        df[f"angle_{compact}"] = angles[:, j]

        # Signed component deltas for debugging/local interpretation.
        df[f"dtheta_signed_{compact}"] = comp["dtheta_signed_deg"][:, j]
        df[f"dphi_signed_{compact}"] = comp["dphi_signed_deg"][:, j]

        # Verbose backward-compatible columns.
        df[f"{prefix}_angle_deg_{safe}"] = angles[:, j]
        df[f"{prefix}_theta_delta_abs_deg_{safe}"] = comp["dtheta_abs_deg"][:, j]
        df[f"{prefix}_phi_delta_abs_deg_{safe}"] = comp["dphi_abs_deg"][:, j]

    return df, {"angles_deg": angles, **comp}


def backward_meed_component_velocity(theta: np.ndarray, phi: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Backward component velocities for theta and phi in deg/s."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    dtheta = np.full(theta.shape, np.nan)
    dphi = np.full(phi.shape, np.nan)
    if len(theta) > 1:
        dtheta[1:] = u.wrap180(theta[1:] - theta[:-1]) * float(sfreq)
        dphi[1:] = (phi[1:] - phi[:-1]) * float(sfreq)
    return dtheta, dphi


def characterize_raw_with_template_levels(
    raw_aligned: mne.io.BaseRaw,
    template_levels: dict[str, TemplateLevel],
    *,
    data_geometry_level: str = "meta",
    recording_info: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the requested per-sample real-data MEED table.

    Returns columns for subject/session/time, GFP, topographic dissimilarity,
    data MEED theta/phi/rho, all compact correlations to each template level,
    and all compact theta/phi/composite angular distances to each template level.
    """
    if data_geometry_level not in template_levels:
        raise KeyError(f"data_geometry_level={data_geometry_level!r} not in template_levels")

    info_dict = {"sub": "NA", "ses": "NA", "recording": "NA"}
    if recording_info is not None:
        info_dict.update(recording_info)

    X, sfreq = raw_to_channel_time(raw_aligned)
    Xn = center_l2_columns(X)
    dip = project_data_to_meed(X, template_levels[data_geometry_level].geom["S_orth"])
    dtheta_vel, dphi_vel = backward_meed_component_velocity(dip["theta_deg"], dip["phi_deg"], sfreq)
    peak_idx, _ = find_peaks(np.std(X, axis=0), distance=1)
    is_peak = np.zeros(X.shape[1], dtype=bool)
    is_peak[peak_idx] = True

    base = pd.DataFrame(
        {
            "sub": info_dict.get("sub", "NA"),
            "ses": info_dict.get("ses", "NA"),
            "recording": info_dict.get("recording", "NA"),
            "sample": np.arange(X.shape[1], dtype=int),
            "time": np.arange(X.shape[1], dtype=float) / sfreq,
            "time_s": np.arange(X.shape[1], dtype=float) / sfreq,
            "is_peak": is_peak,
            "GFP": gfp_from_channel_time(X),
            "TD": backward_topographic_dissimilarity(Xn),
            "gfp": gfp_from_channel_time(X),
            "backward_topographic_dissimilarity": backward_topographic_dissimilarity(Xn),
            "theta": dip["theta_deg"],
            "phi": dip["phi_deg"],
            "rho": dip["rho"],
            "dipole_x": dip["Q"][0, :],
            "dipole_y": dip["Q"][1, :],
            "dipole_z": dip["Q"][2, :],
            "unit_x": dip["U"][0, :],
            "unit_y": dip["U"][1, :],
            "unit_z": dip["U"][2, :],
            "MEED_theta_deg": dip["theta_deg"],
            "MEED_phi_deg": dip["phi_deg"],
            "MEED_rho": dip["rho"],
            "MEED_projection_R2": dip["projection_R2"],
            "MEED_backward_angular_velocity_deg_s": backward_meed_angular_velocity(dip["U"], sfreq),
            "MEED_backward_theta_velocity_deg_s": dtheta_vel,
            "MEED_backward_phi_velocity_deg_s": dphi_vel,
        }
    )

    frames = [base]
    matrices: dict[str, Any] = {"X": X, "X_norm": Xn, "data_meed": dip}

    for level_name, level in template_levels.items():
        cdf, cpayload = sample_template_correlations(Xn, level, prefix=level_name)
        adf, apayload = sample_template_angles(dip["U"], level, prefix=level_name)
        frames.extend([cdf, adf])
        matrices[level_name] = {"level": level, "correlations": cpayload, "angles": apayload}

    return pd.concat(frames, axis=1), matrices


def _plot_level_state_lines(
    ax,
    d: pd.DataFrame,
    labels: list[str],
    ykind: str,
    level_suffix: str,
    *,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
):
    for j, lab in enumerate(labels):
        safe = _safe_label(lab)
        col = f"{ykind}_{safe}{level_suffix}"
        if col not in d:
            continue
        ax.plot(
            d["time"],
            d[col],
            color=OKABE_STATE_COLORS[j % len(OKABE_STATE_COLORS)],
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            label=f"{level_suffix}:{lab}",
        )


# def plot_subject_timeseries_detail(
#     sample_df: pd.DataFrame,
#     *,
#     labels: list[str],
#     start_s: float | None = None,
#     stop_s: float | None = None,
#     meta_suffix: str = "meta",
#     subject_suffix: str = "sub",
# ) -> plt.Figure:
#     """Long subject-level plot requested for Section 4."""
#     d = sample_df.copy()
#     # if start_s is not None:
#     #     d = d[d["time"] >= start_s]
#     # if stop_s is not None:
#     #     d = d[d["time"] <= stop_s]

#     fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True, constrained_layout=True)

#     t = d["time"].to_numpy()
#     gfp = d["GFP"].to_numpy(dtype=float)
#     if np.nanmax(gfp) > 0:
#         gfp_fill = gfp / np.nanmax(gfp)
#     else:
#         gfp_fill = gfp

#     axes[0].fill_between(t, 0, gfp_fill, color=OKABE_ITO["gray"], alpha=0.35, label="GFP normalized")
#     axes[0].plot(t, d["TD"], color=OKABE_ITO["reddishpurple"], linewidth=1.4, label="TD")
#     axes0twin = axes[0].twinx()
    
#     axes0twin.plot(t, d["MEED_backward_angular_velocity_deg_s"]/1000, color=OKABE_ITO["bluishgreen"], linewidth=1.1, label="MEED angular velocity")
#     axes0twin.set_ylabel("MEED angular velocity (deg/ms)", color=OKABE_ITO["bluishgreen"])
#     axes[0].set_ylabel("GFP/TD")
#     axes[0].legend(ncol=3, fontsize=8, loc="upper right")

#     _plot_level_state_lines(axes[1], d, labels, "C", meta_suffix, linewidth=2.5, alpha=0.60)
#     _plot_level_state_lines(axes[1], d, labels, "C", subject_suffix, linewidth=1.0, alpha=1.00)
#     axes[1].set_ylabel("|corr|")
#     axes[1].set_title("Template correlations: meta thick transparent, subject thin opaque")
#     axes[1].legend(ncol=4, fontsize=7, loc="upper right")

#     axes[2].plot(t, d["theta"], color=OKABE_ITO["blue"], linewidth=1.0, label=r"$\theta$")
#     axes[2].plot(t, d["phi"], color=OKABE_ITO["orange"], linewidth=1.0, label=r"$\phi$")
#     ax_rho = axes[2].twinx()
#     ax_rho.plot(t, d["rho"], color=OKABE_ITO["black"], linewidth=0.8, alpha=0.65, label=r"$\rho$")
#     axes[2].set_ylabel(r"$\theta,\phi$ (deg)")
#     ax_rho.set_ylabel(r"$\rho$")
#     axes[2].legend(loc="upper left", fontsize=8)
#     ax_rho.legend(loc="upper right", fontsize=8)

#     _plot_level_state_lines(axes[3], d, labels, "theta", meta_suffix, linewidth=2.5, alpha=0.60)
#     _plot_level_state_lines(axes[3], d, labels, "theta", subject_suffix, linewidth=1.0, alpha=1.00)
#     axes[3].set_ylabel(r"$|\Delta\theta|$ (deg)")

#     _plot_level_state_lines(axes[4], d, labels, "phi", meta_suffix, linewidth=2.5, alpha=0.60)
#     _plot_level_state_lines(axes[4], d, labels, "phi", subject_suffix, linewidth=1.0, alpha=1.00)
#     axes[4].set_ylabel(r"$|\Delta\phi|$ (deg)")

#     _plot_level_state_lines(axes[5], d, labels, "angle", meta_suffix, linewidth=2.5, alpha=0.60)
#     _plot_level_state_lines(axes[5], d, labels, "angle", subject_suffix, linewidth=1.0, alpha=1.00)
#     axes[5].set_ylabel("composite angle (deg)")
#     axes[5].set_xlabel("time (s)")
    
#     axes[0].set_xlim(start_s if start_s is not None else t[0], stop_s if stop_s is not None else t[-1])

#     return fig

def plot_subject_timeseries_detail(
    sample_df: pd.DataFrame,
    *,
    labels: list[str],
    start_s: float | None = None,
    stop_s: float | None = None,
    meta_suffix: str = "meta",
    subject_suffix: str = "sub",
) -> plt.Figure:
    """Long subject-level plot requested for Section 4."""
    d = sample_df.copy()
    # if start_s is not None:
    #     d = d[d["time"] >= start_s]
    # if stop_s is not None:
    #     d = d[d["time"] <= stop_s]

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(16, 14),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.10, 1.0, 1.0, 1.0, 1.0]},
    )

    t = d["time"].to_numpy()
    gfp = d["GFP"].to_numpy(dtype=float)
    if np.nanmax(gfp) > 0:
        gfp_fill = gfp / np.nanmax(gfp)
    else:
        gfp_fill = gfp

    axes[0].fill_between(t, 0, gfp_fill, color=OKABE_ITO["gray"], alpha=0.35, label="GFP normalized")
    # axes[0].plot(t, d["TD"], color=OKABE_ITO["reddishpurple"], linewidth=1.4, label="TD")
    axes0twin = axes[0].twinx()
    
    axes0twin.plot(t, d["TD"], 
                   color=OKABE_ITO["reddishpurple"], 
                   linewidth=1.4, 
                   label="TD")

    axes0twin.plot(
        t,
        d["MEED_backward_angular_velocity_deg_s"] / 100 * np.pi / 180, # rad/10ms
        color=OKABE_ITO["bluishgreen"],
        linewidth=1.1,
        label="MEED angular velocity",
    )
    axes0twin.set_ylabel("d(ψ)/dt (rad/10ms)", color=OKABE_ITO["bluishgreen"])
    axes[0].set_ylabel("GFP/TD")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")

    _plot_level_state_lines(axes[1], d, labels, "C", meta_suffix, linewidth=2.5, alpha=0.60)
    _plot_level_state_lines(axes[1], d, labels, "C", subject_suffix, linewidth=1.0, alpha=1.00)
    axes[1].set_ylabel("|corr|")
    axes[1].set_title("Template correlations: meta thick transparent, subject thin opaque")
    axes[1].legend(ncol=4, fontsize=7, loc="upper right")

    # Thin theoretical-label strip from argmax over meta correlations
    ax_state = axes[2]
    corr_cols = [f"C_{_safe_label(lab)}{meta_suffix}" for lab in labels]
    available = [c for c in corr_cols if c in d.columns]
    if available:
        corr = d[available].to_numpy(dtype=float)
        corr_for_argmax = np.where(np.isfinite(corr), corr, -np.inf)
        has_valid = np.any(np.isfinite(corr), axis=1)
        winners = np.argmax(corr_for_argmax, axis=1)

        ax_state.set_ylim(0.0, 1.0)
        for j, _ in enumerate(available):
            mask = has_valid & (winners == j)
            ax_state.fill_between(
                t,
                0.0,
                1.0,
                where=mask,
                color=OKABE_STATE_COLORS[j % len(OKABE_STATE_COLORS)],
                alpha=0.95,
                linewidth=0,
                interpolate=True,
            )
    ax_state.set_yticks([])
    ax_state.set_ylabel("")
    ax_state.spines["left"].set_visible(False)
    ax_state.spines["right"].set_visible(False)
    ax_state.spines["top"].set_visible(False)

    axes[3].plot(t, d["theta"], color=OKABE_ITO["blue"], linewidth=1.0, label=r"$\theta$")
    axes[3].plot(t, d["phi"], color=OKABE_ITO["orange"], linewidth=1.0, label=r"$\phi$")
    ax_rho = axes[3].twinx()
    ax_rho.plot(t, d["rho"], color=OKABE_ITO["black"], linewidth=0.8, alpha=0.65, label=r"$\rho$")
    axes[3].set_ylabel(r"$\theta,\phi$ (deg)")
    ax_rho.set_ylabel(r"$\rho$")
    axes[3].legend(loc="upper left", fontsize=8)
    ax_rho.legend(loc="upper right", fontsize=8)

    _plot_level_state_lines(axes[4], d, labels, "theta", meta_suffix, linewidth=2.5, alpha=0.60)
    _plot_level_state_lines(axes[4], d, labels, "theta", subject_suffix, linewidth=1.0, alpha=1.00)
    axes[4].set_ylabel(r"$|\Delta\theta|$ (deg)")

    _plot_level_state_lines(axes[5], d, labels, "phi", meta_suffix, linewidth=2.5, alpha=0.60)
    _plot_level_state_lines(axes[5], d, labels, "phi", subject_suffix, linewidth=1.0, alpha=1.00)
    axes[5].set_ylabel(r"$|\Delta\phi|$ (deg)")

    _plot_level_state_lines(axes[6], d, labels, "angle", meta_suffix, linewidth=2.5, alpha=0.60)
    _plot_level_state_lines(axes[6], d, labels, "angle", subject_suffix, linewidth=1.0, alpha=1.00)
    axes[6].set_ylabel("composite angle (deg)")
    axes[6].set_xlabel("time (s)")

    axes[0].set_xlim(start_s if start_s is not None else t[0], stop_s if stop_s is not None else t[-1])

    return fig


def plot_subject_feature_hexbins(
    sample_df: pd.DataFrame,
    *,
    gridsize: int = 48,
    mincnt: int = 2,
) -> plt.Figure:
    """Hexbin panel for GFP, TD, velocity, and sample rho relations."""
    d = sample_df.copy()

    gfp_uv = d["GFP"].to_numpy(dtype=float) * 1e6
    td = d["TD"].to_numpy(dtype=float)
    rho = d["rho"].to_numpy(dtype=float)
    dpsi = d["MEED_backward_angular_velocity_deg_s"].to_numpy(dtype=float) * (np.pi / 180.0) / 100.0

    fig, axes = plt.subplots(2, 5, figsize=(19.2, 8.2), constrained_layout=True)
    axs = axes.ravel()

    def _hex(ax, x, y, xlabel, ylabel, **kwargs):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return
        hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=mincnt, cmap="viridis", **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(hb, ax=ax, shrink=0.82)

    # TD vs GFP (col 0)
    _hex(axs[0], gfp_uv, td, "GFP (uV)", "TD")
    axs[0].set_title("Zanesco 2020: TD vs GFP")
    
    _hex(axs[5], gfp_uv, td, "GFP (uV)", "TD",
         xscale='log', yscale='log', bins='log')
    axs[5].set_title("Zanesco 2020: TD vs GFP (log-log)")
    
    # rho vs GFP (col 1)
    _hex(axs[1], gfp_uv, rho, "GFP (uV)", r"$\rho_{MEED}$")
    axs[1].set_title(r"$\rho$ vs GFP")

    _hex(axs[6], gfp_uv, rho, "GFP (uV)", r"$\rho_{MEED}$", 
         xscale='log', yscale='log', bins='log')
    axs[6].set_title("rho vs GFP (log-log)")
    
    # rho vs GFP (col 2)
    _hex(axs[2], gfp_uv, rho**2, "GFP (uV)", r"$\rho_{MEED}$")
    axs[2].set_title(r"$\rho^2$ vs GFP")

    _hex(axs[7], gfp_uv, rho**2, "GFP (uV)", r"$\rho_{MEED}$", 
         xscale='log', yscale='log', bins='log')
    axs[7].set_title("rho^2 vs GFP (log-log)")

    # vel vs GFP (col 3)
    _hex(axs[3], gfp_uv, dpsi, "GFP (uV)", r"|d($\psi$)/dt| (rad/10ms)")
    axs[3].set_title("|d(psi)/dt| vs GFP")
    
    _hex(axs[8], gfp_uv, dpsi, "GFP (uV)", r"|d($\psi$)/dt| (rad/10ms)", 
         xscale='log', yscale='log', bins='log')
    axs[8].set_title("|d(psi)/dt| vs GFP (log-log)")
    
    # TD vs vel (col 4)
    _hex(axs[4], td, dpsi, "TD", r"|d($\psi$)/dt| (rad/10ms)")
    axs[4].set_title("TD vs |d(psi)/dt|")
    
    _hex(axs[9], td, dpsi, "TD", r"|d($\psi$)/dt| (rad/10ms)", 
         xscale='log', yscale='log', bins='log')
    axs[9].set_title("TD vs |d(psi)/dt| (log-log)")
    

    return fig


def plot_subject_feature_hexbins_peaks(
    sample_df: pd.DataFrame,
    *,
    gridsize: int = 48,
    mincnt: int = 2,
    peak_min_peak_distance: int = 1,
) -> plt.Figure:
    """Overlay all-sample and GFP-peak hexbins for the core subject features."""
    d = sample_df.copy()

    if "is_peak" in d:
        peak_mask = d["is_peak"].fillna(False).astype(bool).to_numpy()
    else:
        peak_idx, _ = find_peaks(d["GFP"].to_numpy(dtype=float), distance=int(peak_min_peak_distance))
        peak_mask = np.zeros(len(d), dtype=bool)
        peak_mask[peak_idx] = True

    gfp_uv = d["GFP"].to_numpy(dtype=float) * 1e6
    td = d["TD"].to_numpy(dtype=float)
    rho = d["rho"].to_numpy(dtype=float)
    dpsi = d["MEED_backward_angular_velocity_deg_s"].to_numpy(dtype=float) * (np.pi / 180.0) / 100.0

    fig, axes = plt.subplots(2, 5, figsize=(19.2, 8.2), constrained_layout=True)
    axs = axes.ravel()

    def _hex_overlay(ax, x, y, xlabel, ylabel, title, **kwargs):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mall = np.isfinite(x) & np.isfinite(y)
        x_all = x[mall]
        y_all = y[mall]
        if x_all.size == 0:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            return

        ax.hexbin(
            x_all,
            y_all,
            gridsize=gridsize,
            mincnt=mincnt,
            cmap="Greys",
            alpha=1.0,
            linewidths=0.0,
            **kwargs,
        )

        mpeaks = mall & peak_mask
        x_peak = x[mpeaks]
        y_peak = y[mpeaks]
        if x_peak.size > 0:
            hb_peak = ax.hexbin(
                x_peak,
                y_peak,
                gridsize=gridsize,
                mincnt=1,
                cmap="cividis",
                alpha=0.80,
                linewidths=0.0,
                **kwargs,
            )
            fig.colorbar(hb_peak, ax=ax, shrink=0.82)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nall = gray, GFP peaks = cividis")

    _hex_overlay(axs[0], gfp_uv, td, "GFP (uV)", "TD", "Zanesco 2020: TD vs GFP")
    _hex_overlay(axs[5], gfp_uv, td, "GFP (uV)", "TD", "Zanesco 2020: TD vs GFP (log-log)", xscale="log", yscale="log", bins="log")

    _hex_overlay(axs[1], gfp_uv, rho, "GFP (uV)", r"$\rho_{MEED}$", r"$\rho$ vs GFP")
    _hex_overlay(axs[6], gfp_uv, rho, "GFP (uV)", r"$\rho_{MEED}$", "rho vs GFP (log-log)", xscale="log", yscale="log", bins="log")

    _hex_overlay(axs[2], gfp_uv, rho**2, "GFP (uV)", r"$\rho_{MEED}$", r"$\rho^2$ vs GFP")
    _hex_overlay(axs[7], gfp_uv, rho**2, "GFP (uV)", r"$\rho_{MEED}$", "rho^2 vs GFP (log-log)", xscale="log", yscale="log", bins="log")

    _hex_overlay(axs[3], gfp_uv, dpsi, "GFP (uV)", r"|d($\psi$)/dt| (rad/10ms)", "|d(psi)/dt| vs GFP")
    _hex_overlay(axs[8], gfp_uv, dpsi, "GFP (uV)", r"|d($\psi$)/dt| (rad/10ms)", "|d(psi)/dt| vs GFP (log-log)", xscale="log", yscale="log", bins="log")

    _hex_overlay(axs[4], td, dpsi, "TD", r"|d($\psi$)/dt| (rad/10ms)", "TD vs |d(psi)/dt|")
    _hex_overlay(axs[9], td, dpsi, "TD", r"|d($\psi$)/dt| (rad/10ms)", "TD vs |d(psi)/dt| (log-log)", xscale="log", yscale="log", bins="log")

    return fig

def _bivariate_kde_two_groups(
    ax,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    x_neg: np.ndarray,
    y_neg: np.ndarray,
    color: str,
    xlabel: str,
    ylabel: str,
):
    from scipy.stats import gaussian_kde

    xp = np.asarray(x_pos, dtype=float)
    yp = np.asarray(y_pos, dtype=float)
    xn = np.asarray(x_neg, dtype=float)
    yn = np.asarray(y_neg, dtype=float)

    mpos = np.isfinite(xp) & np.isfinite(yp)
    mneg = np.isfinite(xn) & np.isfinite(yn)
    xp, yp = xp[mpos], yp[mpos]
    xn, yn = xn[mneg], yn[mneg]

    if xp.size < 5 or xn.size < 5:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    x_all = np.concatenate([xp, xn])
    y_all = np.concatenate([yp, yn])
    xlo, xhi = np.nanmin(x_all), np.nanmax(x_all)
    ylo, yhi = np.nanmin(y_all), np.nanmax(y_all)
    if not np.isfinite(xlo) or not np.isfinite(xhi) or xlo == xhi:
        xlo, xhi = xlo - 1.0, xhi + 1.0
    if not np.isfinite(ylo) or not np.isfinite(yhi) or ylo == yhi:
        ylo, yhi = ylo - 1.0, yhi + 1.0

    gx = np.linspace(xlo, xhi, 120)
    gy = np.linspace(ylo, yhi, 120)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.vstack([xx.ravel(), yy.ravel()])

    def _safe_kde2(arr_x: np.ndarray, arr_y: np.ndarray):
        try:
            kde = gaussian_kde(np.vstack([arr_x, arr_y]))
            zz = kde(grid).reshape(xx.shape)
            return zz
        except Exception:
            return None

    zz_pos = _safe_kde2(xp, yp)
    zz_neg = _safe_kde2(xn, yn)
    if zz_pos is None or zz_neg is None:
        ax.text(0.5, 0.5, "kde failed", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    def _levels(zz: np.ndarray):
        vals = zz[np.isfinite(zz)]
        if vals.size == 0:
            return None
        q = np.quantile(vals, [0.60, 0.75, 0.88, 0.95])
        q = np.unique(q[q > 0])
        return q if q.size >= 2 else None

    lev_pos = _levels(zz_pos)
    lev_neg = _levels(zz_neg)
    if lev_pos is None or lev_neg is None:
        ax.text(0.5, 0.5, "insufficient variation", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    ax.contour(xx, yy, zz_neg, levels=lev_neg, colors=[OKABE_ITO["gray"]], linewidths=1.2, linestyles="--", alpha=0.95)
    ax.contour(xx, yy, zz_pos, levels=lev_pos, colors=[color], linewidths=1.5, linestyles="-", alpha=0.95)
    ax.plot([], [], color=color, lw=1.5, label="labelled as state")
    ax.plot([], [], color=OKABE_ITO["gray"], lw=1.2, ls="--", label="not labelled")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc="upper right")


def _mean_stat_heatmap(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str, bins: int = 28):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if x.size < 8:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return

    xe = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    ye = np.linspace(np.nanmin(y), np.nanmax(y), bins + 1)
    s, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=z)
    c, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    mean = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)

    im = ax.imshow(
        mean.T,
        origin="lower",
        aspect="auto",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap="magma",
    )
    ax.set_title(title, fontsize=8)
    return im


def _draw_meed_sphere_dipole(fig: plt.Figure, host_ax, vec: np.ndarray, color: str, title: str, view: tuple[float, float]):
    host_ax.axis("off")
    bb = host_ax.get_position()
    ax3 = fig.add_axes([bb.x0, bb.y0, bb.width, bb.height], projection="3d")
    ax3.patch.set_alpha(0.0)

    u0 = np.linspace(0, 2 * np.pi, 28)
    v0 = np.linspace(0, np.pi, 16)
    xs = np.outer(np.cos(u0), np.sin(v0))
    ys = np.outer(np.sin(u0), np.sin(v0))
    zs = np.outer(np.ones_like(u0), np.cos(v0))
    ax3.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, color="0.75", linewidth=0.35, alpha=0.55)

    vv = np.asarray(vec, dtype=float).reshape(3)
    vv = vv / max(np.linalg.norm(vv), 1e-12)
    ax3.quiver(0, 0, 0, vv[0], vv[1], vv[2], color=color, linewidth=2.0, arrow_length_ratio=0.16)

    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-1, 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.set_box_aspect([1, 1, 1])
    ax3.view_init(elev=view[0], azim=view[1])
    ax3.set_title(title, fontsize=8, pad=2)
    return ax3


def plot_subject_uncertainty(
    sample_df: pd.DataFrame,
    *,
    meta_level: TemplateLevel,
    subject_level: TemplateLevel,
    labels: list[str],
    meta_suffix: str = "meta",
    subject_suffix: str = "sub",
) -> plt.Figure:
    """4x8 figure: one 4x2 block per microstate label."""
    n_states = min(4, len(labels))
    fig, axes = plt.subplots(4, 8, figsize=(26, 12), constrained_layout=True)

    # Precompute meta-only best-vs-second correlation margin.
    corr_meta_all = np.column_stack([sample_df[f"C_{_safe_label(lb)}{meta_suffix}"].to_numpy(float) for lb in labels])
    ord_corr = np.argsort(-corr_meta_all, axis=1)
    corr_margin = corr_meta_all[np.arange(corr_meta_all.shape[0]), ord_corr[:, 0]] - corr_meta_all[np.arange(corr_meta_all.shape[0]), ord_corr[:, 1]]

    meta_map_rho = np.asarray(meta_level.df.get("dipolarity", meta_level.df.get("rho", np.full(meta_level.B.shape[1], np.nan))), dtype=float)
    sub_map_rho = np.asarray(subject_level.df.get("dipolarity", subject_level.df.get("rho", np.full(subject_level.B.shape[1], np.nan))), dtype=float)

    for k in range(4):
        c0 = 2 * k
        if k >= n_states:
            for rr in range(4):
                axes[rr, c0].axis("off")
                axes[rr, c0 + 1].axis("off")
            continue

        lab = labels[k]
        safe = _safe_label(lab)
        color = OKABE_STATE_COLORS[k % len(OKABE_STATE_COLORS)]

        cmeta = sample_df[f"C_{safe}{meta_suffix}"].to_numpy(float)
        csub = sample_df[f"C_{safe}{subject_suffix}"].to_numpy(float)
        label_best = sample_df[f"{meta_suffix}_corr_best_label"].astype(str).to_numpy()
        is_lab = label_best == str(lab)

        # Top 2x2: topomap / bivariate kde(corr) / heatmap-margin / topomap
        _topomap_on_axis(axes[0, c0], subject_level.B[:, k], subject_level.info, vlim=float(np.max(np.abs(subject_level.B))), title=f"subject {lab}")
        _topomap_on_axis(axes[1, c0 + 1], meta_level.B[:, k], meta_level.info, vlim=float(np.max(np.abs(meta_level.B))), title=f"meta {lab}")

        _bivariate_kde_two_groups(
            axes[0, c0 + 1],
            cmeta[is_lab],
            csub[is_lab],
            cmeta[~is_lab],
            csub[~is_lab],
            color,
            xlabel=f"|corr| to {lab} (meta)",
            ylabel=f"|corr| to {lab} (sub)",
        )

        im = _mean_stat_heatmap(
            axes[1, c0],
            cmeta,
            csub,
            corr_margin,
            title=f"Delta corr margin map ({lab})",
        )
        if im is not None:
            fig.colorbar(im, ax=axes[1, c0], shrink=0.75)

        axes[1, c0].set_xlabel(f"C_{lab}{meta_suffix}")
        axes[1, c0].set_ylabel(f"C_{lab}{subject_suffix}")

        # Bottom 2x2: MEED sphere+dipole / bivariate kde(angle) / rho heatmap / sphere+dipole
        ang_meta = sample_df[f"angle_{safe}{meta_suffix}"].to_numpy(float)
        ang_sub = sample_df[f"angle_{safe}{subject_suffix}"].to_numpy(float)
        _bivariate_kde_two_groups(
            axes[2, c0 + 1],
            ang_meta[is_lab],
            ang_sub[is_lab],
            ang_meta[~is_lab],
            ang_sub[~is_lab],
            color,
            xlabel=f"angle to {lab} (meta, deg)",
            ylabel=f"angle to {lab} (sub, deg)",
        )

        rho_s = sample_df["rho"].to_numpy(float)
        dm = np.abs(rho_s - meta_map_rho[k])
        ds = np.abs(rho_s - sub_map_rho[k])

        im2 = _mean_stat_heatmap(
            axes[3, c0],
            dm,
            ds,
            rho_s,
            title=f"rho map ({lab})",
        )
        if im2 is not None:
            fig.colorbar(im2, ax=axes[3, c0], shrink=0.75)
        axes[3, c0].set_xlabel(f"|rho-rho_{lab}{meta_suffix}|")
        axes[3, c0].set_ylabel(f"|rho-rho_{lab}{subject_suffix}|")

        # Requested overlapping transparent 3D axes
        _draw_meed_sphere_dipole(fig, axes[2, c0], subject_level.U[:, k], color, f"MEED subject {lab} (top)", view=(90, -90))
        _draw_meed_sphere_dipole(fig, axes[3, c0 + 1], meta_level.U[:, k], color, f"MEED meta {lab} (left)", view=(0, -90))

        axes[3, c0 + 1].axis("off")

    fig.suptitle("Subject uncertainty decomposition by microstate", fontsize=13)
    return fig


def _detect_angle_columns(sample_df: pd.DataFrame) -> tuple[str, str]:
    theta_candidates = ["theta", "MEED_theta_deg", "theta_deg"]
    phi_candidates = ["phi", "MEED_phi_deg", "phi_deg"]
    theta_col = next((c for c in theta_candidates if c in sample_df.columns), None)
    phi_col = next((c for c in phi_candidates if c in sample_df.columns), None)
    if theta_col is None or phi_col is None:
        raise KeyError("Could not detect theta/phi columns in sample_df.")
    return theta_col, phi_col


def _detect_label_column(sample_df: pd.DataFrame, preferred: str | None = None) -> str | None:
    if preferred is not None and preferred in sample_df.columns:
        return preferred
    candidates = [
        "meta_corr_best_label",
        "subject_corr_best_label",
        "sub_corr_best_label",
        "meta_angle_best_label",
        "subject_angle_best_label",
        "sub",
        "label",
    ]
    return next((c for c in candidates if c in sample_df.columns), None)


def _detect_time_column(sample_df: pd.DataFrame) -> str | None:
    for col in ("time", "time_s", "sample"):
        if col in sample_df.columns:
            return col
    return None


def _detect_dipole_xyz_columns(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str, str], str]:
    triplets = [
        ("dipole_x", "dipole_y", "dipole_z"),
        ("Qx", "Qy", "Qz"),
        ("q_x", "q_y", "q_z"),
        ("x", "y", "z"),
    ]
    for triplet in triplets:
        if all(col in sample_df.columns for col in triplet):
            return sample_df.copy(), triplet, "existing"
    raise KeyError(
        "Could not detect native fixed-dipole 3D coordinate columns in sample_df. "
        "Expected columns like `dipole_x`, `dipole_y`, `dipole_z` from the fixed-dipole fit. "
        "Theta/phi-derived hemispheric coordinates are intentionally not used here."
    )


def _save_figure(fig: plt.Figure, path: Path | None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")


def _save_table(df: pd.DataFrame, path: Path | None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def _transition_counts_from_codes(codes: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=int)
    if len(codes) < 2:
        return counts
    for a, b in zip(codes[:-1], codes[1:]):
        if a < 0 or b < 0:
            continue
        counts[a, b] += 1
    return counts


def _transition_probabilities(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)


def _dwell_metrics_from_codes(codes: np.ndarray, state_names: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    n = len(codes)
    for idx, name in enumerate(state_names):
        mask = codes == idx
        occupancy = float(np.mean(mask)) if n > 0 else np.nan
        run_lengths = []
        run = 0
        for code in codes:
            if code == idx:
                run += 1
            elif run > 0:
                run_lengths.append(run)
                run = 0
        if run > 0:
            run_lengths.append(run)
        mean_dwell = float(np.mean(run_lengths)) if run_lengths else 0.0
        self_trans = 0
        total_trans = 0
        for a, b in zip(codes[:-1], codes[1:]):
            if a == idx:
                total_trans += 1
                if b == idx:
                    self_trans += 1
        self_prob = float(self_trans / total_trans) if total_trans > 0 else np.nan
        rows.append(
            {
                f"{prefix}_id": name,
                f"{prefix}_occupancy": occupancy,
                f"{prefix}_mean_dwell": mean_dwell,
                f"{prefix}_self_transition_probability": self_prob,
            }
        )
    return pd.DataFrame(rows)


def _plot_transition_matrix(counts: np.ndarray, labels: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(counts, cmap="magma")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("next")
    ax.set_ylabel("current")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def _plot_topology_graph(counts: np.ndarray, labels: list[str], occupancies: np.ndarray, title: str) -> plt.Figure:
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([np.cos(ang), np.sin(ang)])
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_title(title)
    ax.axis("off")
    max_count = max(int(counts.max()), 1)
    for i in range(n):
        for j in range(n):
            if counts[i, j] <= 0 or i == j:
                continue
            x0, y0 = xy[i]
            x1, y1 = xy[j]
            ax.plot([x0, x1], [y0, y1], color="0.65", linewidth=0.6 + 3.0 * counts[i, j] / max_count, alpha=0.65)
    ax.scatter(xy[:, 0], xy[:, 1], s=500 + 3000 * occupancies, c=np.arange(n), cmap="tab10", edgecolors="black")
    for i, lab in enumerate(labels):
        ax.text(xy[i, 0], xy[i, 1], lab, ha="center", va="center", fontsize=9, color="white")
    return fig


def _cluster_with_gmm(X: np.ndarray, *, n_components_range: range, random_state: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:
        raise ImportError("scikit-learn is required for GMM clustering.") from exc

    best = None
    best_bic = np.inf
    X = np.asarray(X, dtype=float)
    for n_comp in n_components_range:
        if n_comp < 1 or n_comp > len(X):
            continue
        model = GaussianMixture(n_components=int(n_comp), covariance_type="full", random_state=int(random_state))
        model.fit(X)
        bic = model.bic(X)
        if bic < best_bic:
            best_bic = bic
            best = model
    if best is None:
        raise RuntimeError("Could not fit any GMM model.")
    labels = best.predict(X)
    centers = best.means_
    return labels, centers, f"GaussianMixture(n_components={best.n_components})"


def _cluster_with_kmeans(X: np.ndarray, *, n_clusters: int, random_state: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from sklearn.cluster import KMeans
    except Exception as exc:
        raise ImportError("scikit-learn is required for KMeans fallback clustering.") from exc
    model = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=10)
    labels = model.fit_predict(X)
    return labels, model.cluster_centers_, f"KMeans(n_clusters={int(n_clusters)})"


def _cluster_state_space(X: np.ndarray, *, max_components: int = 8, random_state: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    n_obs = len(X)
    if n_obs < 4:
        raise ValueError("Not enough samples for state-space clustering.")
    upper = max(2, min(int(max_components), max(2, n_obs // 10)))
    try:
        return _cluster_with_gmm(X, n_components_range=range(2, upper + 1), random_state=random_state)
    except Exception:
        n_clusters = min(max(2, upper), n_obs)
        return _cluster_with_kmeans(X, n_clusters=n_clusters, random_state=random_state)


def _alpha_from_r2(r2: np.ndarray, *, alpha_min: float = 0.10, alpha_max: float = 0.85) -> np.ndarray:
    r2 = np.asarray(r2, dtype=float)
    out = np.full(r2.shape, alpha_min, dtype=float)
    m = np.isfinite(r2)
    if np.any(m):
        rr = np.clip(r2[m], 0.0, 1.0)
        out[m] = alpha_min + (alpha_max - alpha_min) * rr
    return out


def _3d_scatter_figure(
    X: np.ndarray,
    colors: np.ndarray | None,
    title: str,
    labels: np.ndarray | None = None,
    alpha_values: np.ndarray | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(6.8, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    if colors is None:
        if alpha_values is None:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=8, alpha=0.35, color="0.25")
        else:
            rgba = np.tile(np.array([0.15, 0.15, 0.15, 1.0]), (len(X), 1))
            rgba[:, 3] = np.asarray(alpha_values, dtype=float)
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=8, c=rgba)
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=10, alpha=0.50, cmap="tab10")
    if labels is not None:
        uniq = pd.unique(labels)
        for lab in uniq:
            mask = labels == lab
            if np.sum(mask) == 0:
                continue
            center = np.nanmean(X[mask], axis=0)
            ax.text(center[0], center[1], center[2], str(lab), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig


def _trajectory_3d_figure(X: np.ndarray, title: str, segment_length: int = 250, n_segments: int = 3) -> plt.Figure:
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    n = len(X)
    if n == 0:
        ax.set_title(title)
        return fig
    starts = np.linspace(0, max(0, n - segment_length), num=min(n_segments, max(1, n // max(segment_length, 1))), dtype=int)
    for i, start in enumerate(np.unique(starts)):
        stop = min(n, start + segment_length)
        seg = X[start:stop]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=1.1, alpha=0.8, label=f"seg {i+1}")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=8)
    return fig


def _flow_bins_2d(X: np.ndarray, *, bins: int = 32) -> dict[str, Any]:
    X = np.asarray(X, dtype=float)
    if len(X) < 2:
        raise ValueError("Need at least 2 points for flow estimation.")
    x = X[:, 0]
    y = X[:, 1]
    xe = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    ye = np.linspace(np.nanmin(y), np.nanmax(y), bins + 1)
    touch, _, _ = np.histogram2d(x, y, bins=[xe, ye])

    mids = 0.5 * (X[:-1] + X[1:])
    dX = X[1:] - X[:-1]
    step_modulus = np.sqrt(np.sum(dX ** 2, axis=1))
    ix = np.clip(np.digitize(mids[:, 0], xe) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(mids[:, 1], ye) - 1, 0, bins - 1)
    flow_x = np.zeros((bins, bins), dtype=float)
    flow_y = np.zeros((bins, bins), dtype=float)
    np.add.at(flow_x, (ix, iy), dX[:, 0])
    np.add.at(flow_y, (ix, iy), dX[:, 1])
    modulus = np.sqrt(flow_x ** 2 + flow_y ** 2)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    return {"touch": touch, "flow_x": flow_x, "flow_y": flow_y, "modulus": modulus, "step_modulus": step_modulus, "XX": XX, "YY": YY, "xe": xe, "ye": ye}


def _flow_bins_3d(X: np.ndarray, *, bins: int = 14) -> dict[str, Any]:
    X = np.asarray(X, dtype=float)
    if len(X) < 2:
        raise ValueError("Need at least 2 points for flow estimation.")
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    edges = [np.linspace(mins[k], maxs[k], bins + 1) for k in range(3)]
    mids = 0.5 * (X[:-1] + X[1:])
    dX = X[1:] - X[:-1]
    step_modulus = np.sqrt(np.sum(dX ** 2, axis=1))

    ix = np.clip(np.digitize(mids[:, 0], edges[0]) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(mids[:, 1], edges[1]) - 1, 0, bins - 1)
    iz = np.clip(np.digitize(mids[:, 2], edges[2]) - 1, 0, bins - 1)

    touch = np.zeros((bins, bins, bins), dtype=float)
    np.add.at(touch, (ix, iy, iz), 1.0)
    flow_x = np.zeros_like(touch)
    flow_y = np.zeros_like(touch)
    flow_z = np.zeros_like(touch)
    np.add.at(flow_x, (ix, iy, iz), dX[:, 0])
    np.add.at(flow_y, (ix, iy, iz), dX[:, 1])
    np.add.at(flow_z, (ix, iy, iz), dX[:, 2])
    modulus = np.sqrt(flow_x ** 2 + flow_y ** 2 + flow_z ** 2)

    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    XX, YY, ZZ = np.meshgrid(centers[0], centers[1], centers[2], indexing="ij")
    return {
        "touch": touch,
        "flow_x": flow_x,
        "flow_y": flow_y,
        "flow_z": flow_z,
        "modulus": modulus,
        "step_modulus": step_modulus,
        "XX": XX,
        "YY": YY,
        "ZZ": ZZ,
        "edges": edges,
    }


def _plot_flow_modulus_diagnostics(step_modulus: np.ndarray, aggregated_modulus: np.ndarray, title: str) -> plt.Figure:
    step_modulus = np.asarray(step_modulus, dtype=float)
    aggregated_modulus = np.asarray(aggregated_modulus, dtype=float)
    step_modulus = step_modulus[np.isfinite(step_modulus)]
    aggregated_modulus = aggregated_modulus[np.isfinite(aggregated_modulus) & (aggregated_modulus > 0)]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
    step_log = np.log(step_modulus[step_modulus > 0]) if step_modulus.size else np.array([])
    agg_log = np.log(aggregated_modulus[aggregated_modulus > 0]) if aggregated_modulus.size else np.array([])
    panels = [
        (step_modulus, "Pre-aggregation |F|"),
        (aggregated_modulus, "Post-aggregation |F|"),
        (step_log, "Pre-aggregation log(|F|)"),
        (agg_log, "Post-aggregation log(|F|)"),
    ]
    for ax, (vals, ttl) in zip(axes.ravel(), panels):
        if vals.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(vals, bins=50, color="0.35", alpha=0.85)
        ax.set_title(ttl)
        ax.set_ylabel("count")
    axes[1, 0].set_xlabel("log(|F|)")
    axes[1, 1].set_xlabel("log(|F|)")
    axes[0, 0].set_xlabel("|F|")
    axes[0, 1].set_xlabel("|F|")
    fig.suptitle(title, fontsize=12)
    return fig


def _plot_angular_space_overview(
    d: pd.DataFrame,
    *,
    theta_col: str,
    phi_col: str,
    cluster_codes: np.ndarray,
    label_col: str | None,
    alpha_values: np.ndarray | None,
) -> plt.Figure:
    X = d[[theta_col, phi_col]].to_numpy(float)
    flow = _flow_bins_2d(X, bins=34)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)

    if alpha_values is None:
        axes[0, 0].scatter(d[theta_col], d[phi_col], s=8, alpha=0.30, color="0.25")
    else:
        rgba = np.tile(np.array([0.15, 0.15, 0.15, 1.0]), (len(d), 1))
        rgba[:, 3] = alpha_values
        axes[0, 0].scatter(d[theta_col], d[phi_col], s=8, c=rgba)
    axes[0, 0].set_title("Angular space raw")
    axes[0, 0].set_xlabel(theta_col)
    axes[0, 0].set_ylabel(phi_col)

    if label_col is not None:
        cats = pd.Categorical(d[label_col].astype(str))
        axes[0, 1].scatter(d[theta_col], d[phi_col], c=cats.codes, s=8, alpha=0.45, cmap="tab10")
        axes[0, 1].set_title("Angular space by theoretical label")
    else:
        axes[0, 1].hexbin(d[theta_col], d[phi_col], gridsize=48, mincnt=2, cmap="Greys")
        axes[0, 1].set_title("Angular space touch count")
    axes[0, 1].set_xlabel(theta_col)
    axes[0, 1].set_ylabel(phi_col)

    axes[1, 0].scatter(d[theta_col], d[phi_col], c=cluster_codes, s=8, alpha=0.45, cmap="tab10")
    axes[1, 0].set_title("Angular space by angular space clustering")
    axes[1, 0].set_xlabel(theta_col)
    axes[1, 0].set_ylabel(phi_col)

    hb = axes[1, 1].hexbin(d[theta_col], d[phi_col], gridsize=42, mincnt=1, cmap="Greys")
    mod = flow["modulus"]
    mask = mod > 0
    if np.any(mask):
        log_mod = np.log(mod[mask])
        mu = float(np.mean(log_mod))
        sigma = float(np.std(log_mod))
        vmin = mu - 3.0 * sigma
        vmax = mu + 3.0 * sigma
        color_vals = np.clip(log_mod, vmin, vmax)
        dx = float(np.median(np.diff(flow["xe"]))) if len(flow["xe"]) > 1 else 1.0
        dy = float(np.median(np.diff(flow["ye"]))) if len(flow["ye"]) > 1 else 1.0
        vec_len = 0.42 * min(abs(dx), abs(dy))
        U = vec_len * flow["flow_x"][mask] / np.maximum(mod[mask], 1e-12)
        V = vec_len * flow["flow_y"][mask] / np.maximum(mod[mask], 1e-12)
        q = axes[1, 1].quiver(
            flow["XX"][mask],
            flow["YY"][mask],
            U,
            V,
            color_vals,
            cmap="cividis",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0050,
            headwidth=4.8,
            headlength=6.2,
            headaxislength=5.2,
        )
        q.set_clim(vmin, vmax)
        fig.colorbar(q, ax=axes[1, 1], shrink=0.80, label="log(|F|), post-aggregation")
    fig.colorbar(hb, ax=axes[1, 1], shrink=0.80, label="touch count")
    axes[1, 1].set_title("Angular space touch count + flow field")
    axes[1, 1].set_xlabel(theta_col)
    axes[1, 1].set_ylabel(phi_col)

    return fig


def _plot_cartesian_dipole_space_overview(
    X: np.ndarray,
    *,
    xyz_cols: tuple[str, str, str],
    cluster_codes: np.ndarray,
    label_col_values: np.ndarray | None,
    alpha_values: np.ndarray | None,
) -> plt.Figure:
    flow = _flow_bins_3d(X, bins=14)
    fig = plt.figure(figsize=(15.0, 11.5), constrained_layout=True)
    axs = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]

    if alpha_values is None:
        axs[0].scatter(X[:, 0], X[:, 1], X[:, 2], s=8, alpha=0.35, color="0.25")
    else:
        rgba = np.tile(np.array([0.15, 0.15, 0.15, 1.0]), (len(X), 1))
        rgba[:, 3] = alpha_values
        axs[0].scatter(X[:, 0], X[:, 1], X[:, 2], s=8, c=rgba)
    axs[0].set_title("Cartesian dipole space raw")

    if label_col_values is not None:
        label_codes = pd.Categorical(label_col_values.astype(str)).codes
        axs[1].scatter(X[:, 0], X[:, 1], X[:, 2], c=label_codes, s=10, alpha=0.50, cmap="tab10")
        axs[1].set_title("Cartesian dipole space by theoretical label")
    else:
        axs[1].text2D(0.5, 0.5, "No theoretical label column", ha="center", va="center", transform=axs[1].transAxes)
        axs[1].set_title("Cartesian dipole space label view unavailable")

    axs[2].scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_codes, s=10, alpha=0.50, cmap="tab10")
    axs[2].set_title("Cartesian dipole space by cartesian dipole clustering")

    mod = flow["modulus"]
    positive_mod = mod[mod > 0]
    if positive_mod.size:
        log_mod = np.log(positive_mod)
        threshold = float(np.mean(log_mod))
        mu = float(np.mean(log_mod))
        sigma = float(np.std(log_mod))
        vmin = mu - 3.0 * sigma
        vmax = mu + 3.0 * sigma
        mask_flow = np.zeros_like(mod, dtype=bool)
        mask_flow[mod > 0] = np.log(mod[mod > 0]) > threshold
    else:
        threshold = np.inf
        mask_flow = np.zeros_like(mod, dtype=bool)
    if np.any(mask_flow):
        flow_x = flow["flow_x"][mask_flow]
        flow_y = flow["flow_y"][mask_flow]
        flow_z = flow["flow_z"][mask_flow]
        flow_mod = mod[mask_flow]
        log_flow_mod = np.log(flow_mod)
        color_vals = np.clip(log_flow_mod, vmin, vmax)
        q = axs[3].quiver(
            flow["XX"][mask_flow],
            flow["YY"][mask_flow],
            flow["ZZ"][mask_flow],
            flow_x / np.maximum(flow_mod, 1e-12),
            flow_y / np.maximum(flow_mod, 1e-12),
            flow_z / np.maximum(flow_mod, 1e-12),
            length=0.24,
            normalize=True,
            cmap="cividis",
            lw=1.3,
            arrow_length_ratio=0.38,
            colors=plt.cm.cividis((color_vals - vmin) / max(vmax - vmin, 1e-12)),
        )
        sm = plt.cm.ScalarMappable(cmap="cividis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axs[3], shrink=0.75, label="log(|F|), post-aggregation")
    axs[3].set_title("Cartesian dipole space thresholded flow field")

    for ax in axs:
        ax.set_xlabel(xyz_cols[0])
        ax.set_ylabel(xyz_cols[1])
        ax.set_zlabel(xyz_cols[2])
    return fig


def _plot_cartesian_xy_plane_cuts(
    flow: dict[str, Any],
    *,
    xyz_cols: tuple[str, str, str],
    n_planes: int = 10,
) -> plt.Figure:
    mod = np.asarray(flow["modulus"], dtype=float)
    positive_mod = mod[mod > 0]
    if positive_mod.size:
        log_mod_all = np.log(positive_mod)
        threshold = float(np.mean(log_mod_all))
        mu = float(np.mean(log_mod_all))
        sigma = float(np.std(log_mod_all))
        vmin = mu - 3.0 * sigma
        vmax = mu + 3.0 * sigma
    else:
        threshold = np.inf
        vmin, vmax = -1.0, 1.0

    n_planes = max(1, int(n_planes))
    z_vals = flow["ZZ"][0, 0, :]
    idxs = np.linspace(0, len(z_vals) - 1, num=min(n_planes, len(z_vals)), dtype=int)

    ncols = 5
    nrows = int(np.ceil(len(idxs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    xx = flow["XX"][:, :, 0]
    yy = flow["YY"][:, :, 0]
    for ax, iz in zip(axes, idxs):
        plane_mod = mod[:, :, iz]
        mask = plane_mod > 0
        if np.any(mask):
            log_plane_mod = np.full_like(plane_mod, np.nan, dtype=float)
            log_plane_mod[mask] = np.log(plane_mod[mask])
            mask &= log_plane_mod > threshold
            if np.any(mask):
                u = flow["flow_x"][:, :, iz]
                v = flow["flow_y"][:, :, iz]
                plane_mag = np.maximum(plane_mod, 1e-12)
                dx = float(np.median(np.diff(flow["edges"][0]))) if len(flow["edges"][0]) > 1 else 1.0
                dy = float(np.median(np.diff(flow["edges"][1]))) if len(flow["edges"][1]) > 1 else 1.0
                vec_len = 0.42 * min(abs(dx), abs(dy))
                U = vec_len * u[mask] / plane_mag[mask]
                V = vec_len * v[mask] / plane_mag[mask]
                color_vals = np.clip(log_plane_mod[mask], vmin, vmax)
                q = ax.quiver(
                    xx[mask],
                    yy[mask],
                    U,
                    V,
                    color_vals,
                    cmap="cividis",
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.0050,
                    headwidth=4.8,
                    headlength=6.2,
                    headaxislength=5.2,
                )
                q.set_clim(vmin, vmax)
        ax.set_title(f"{xyz_cols[2]} ≈ {z_vals[iz]:.2f}", fontsize=9)
        ax.set_xlabel(xyz_cols[0])
        ax.set_ylabel(xyz_cols[1])
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[len(idxs):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="cividis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:len(idxs)].tolist(), shrink=0.82, label="log(|F|), post-aggregation")
    fig.suptitle("Cartesian dipole flow field: xy-parallel plane cuts", fontsize=12)
    return fig


def _prepare_consecutive_dynamics(
    d: pd.DataFrame,
    *,
    pos_cols: list[str],
    time_col: str | None,
    group_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_cols = [] if group_cols is None else [c for c in group_cols if c in d.columns]
    sort_cols = group_cols + ([time_col] if time_col is not None and time_col in d.columns else [])
    work = d.sort_values(sort_cols).reset_index(drop=True) if sort_cols else d.reset_index(drop=True).copy()
    X0 = work[pos_cols].to_numpy(float)
    X1 = np.roll(X0, -1, axis=0)
    valid = np.ones(len(work), dtype=bool)
    valid[-1] = False
    for col in group_cols:
        arr = work[col].astype(str).to_numpy()
        valid[:-1] &= arr[:-1] == arr[1:]
    if time_col is not None and time_col in work.columns:
        t = work[time_col].to_numpy(float)
        dt = np.roll(t, -1) - t
        valid[:-1] &= np.isfinite(dt[:-1]) & (dt[:-1] > 0)
        V = np.divide(X1 - X0, dt[:, None], out=np.full_like(X0, np.nan), where=dt[:, None] > 0)
    else:
        V = X1 - X0
    valid &= np.isfinite(X0).all(axis=1) & np.isfinite(V).all(axis=1)
    return work.loc[valid].copy(), X0[valid], V[valid]


def _fit_empirical_knn_field(X: np.ndarray, V: np.ndarray, *, n_neighbors: int = 50) -> dict[str, Any]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:
        raise ImportError("scikit-learn is required for empirical nearest-neighbor vector-field estimation.") from exc
    n_neighbors = max(3, min(int(n_neighbors), len(X)))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    return {"X": np.asarray(X, float), "V": np.asarray(V, float), "nn": nn, "n_neighbors": n_neighbors}


def _empirical_field_eval(model: dict[str, Any], Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 1:
        Q = Q[None, :]
    dist, idx = model["nn"].kneighbors(Q)
    out = np.zeros((len(Q), model["V"].shape[1]), dtype=float)
    for i in range(len(Q)):
        di = dist[i]
        ii = idx[i]
        positive = di[di > 0]
        sigma = float(np.median(positive)) if positive.size else 1.0
        sigma = max(sigma, 1e-12)
        w = np.exp(-(di ** 2) / (2.0 * sigma ** 2))
        if not np.isfinite(w).all() or np.sum(w) <= 0:
            w = np.ones_like(di)
        w = w / np.sum(w)
        out[i] = np.sum(model["V"][ii] * w[:, None], axis=0)
    return out


def _plot_phaseportrait_or_quiver_2d(
    field_model: dict[str, Any],
    X2: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    mesh_dim: int = 25,
) -> plt.Figure:
    xlim = [float(np.nanmin(X2[:, 0])), float(np.nanmax(X2[:, 0]))]
    ylim = [float(np.nanmin(X2[:, 1])), float(np.nanmax(X2[:, 1]))]
    try:
        import phaseportrait

        def empirical_f_2d(x, y):
            val = _empirical_field_eval(field_model, np.array([[x, y]], dtype=float))[0]
            return float(val[0]), float(val[1])

        fig = plt.figure(figsize=(6.6, 5.6))
        portrait = phaseportrait.PhasePortrait2D(
            empirical_f_2d,
            [xlim, ylim],
            MeshDim=mesh_dim,
            Title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        portrait.plot()
        plt.scatter(X2[:, 0], X2[:, 1], s=3, alpha=0.12, color="0.15")
        return fig
    except Exception:
        gx = np.linspace(xlim[0], xlim[1], mesh_dim)
        gy = np.linspace(ylim[0], ylim[1], mesh_dim)
        xx, yy = np.meshgrid(gx, gy)
        Q = np.column_stack([xx.ravel(), yy.ravel()])
        F = _empirical_field_eval(field_model, Q)
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        mod = np.sqrt(np.sum(F ** 2, axis=1))
        q = ax.quiver(xx, yy, F[:, 0].reshape(xx.shape), F[:, 1].reshape(xx.shape), mod.reshape(xx.shape), cmap="cividis")
        fig.colorbar(q, ax=ax, shrink=0.82, label="|F|")
        ax.scatter(X2[:, 0], X2[:, 1], s=3, alpha=0.12, color="0.15")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig


def _plot_phaseportrait_or_quiver_3d(
    field_model: dict[str, Any],
    X3: np.ndarray,
    *,
    title: str,
    xyz_labels: tuple[str, str, str],
    mesh_dim: int = 10,
) -> tuple[plt.Figure, str]:
    try:
        import phaseportrait

        def empirical_f_3d(x, y, z):
            val = _empirical_field_eval(field_model, np.array([[x, y, z]], dtype=float))[0]
            return float(val[0]), float(val[1]), float(val[2])

        fig = plt.figure(figsize=(7.2, 6.0))
        portrait = phaseportrait.PhasePortrait3D(
            empirical_f_3d,
            [
                [float(np.nanmin(X3[:, 0])), float(np.nanmax(X3[:, 0]))],
                [float(np.nanmin(X3[:, 1])), float(np.nanmax(X3[:, 1]))],
                [float(np.nanmin(X3[:, 2])), float(np.nanmax(X3[:, 2]))],
            ],
            MeshDim=mesh_dim,
            Title=title,
            xlabel=xyz_labels[0],
            ylabel=xyz_labels[1],
            zlabel=xyz_labels[2],
        )
        portrait.plot()
        return fig, "phaseportrait3d"
    except Exception:
        mins = np.nanmin(X3, axis=0)
        maxs = np.nanmax(X3, axis=0)
        grids = [np.linspace(mins[k], maxs[k], mesh_dim) for k in range(3)]
        xx, yy, zz = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
        Q = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        F = _empirical_field_eval(field_model, Q)
        mod = np.sqrt(np.sum(F ** 2, axis=1))
        keep = mod > np.nanmedian(mod)
        fig = plt.figure(figsize=(7.2, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        if np.any(keep):
            norm = plt.Normalize(vmin=float(np.nanmin(mod[keep])), vmax=float(np.nanmax(mod[keep])))
            colors = plt.cm.cividis(norm(mod[keep]))
            ax.quiver(Q[keep, 0], Q[keep, 1], Q[keep, 2], F[keep, 0], F[keep, 1], F[keep, 2], length=0.15, normalize=True, colors=colors, linewidth=0.9)
            sm = plt.cm.ScalarMappable(cmap="cividis", norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.8, label="|F|")
        ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=3, alpha=0.08, color="0.15")
        ax.set_title(title)
        ax.set_xlabel(xyz_labels[0])
        ax.set_ylabel(xyz_labels[1])
        ax.set_zlabel(xyz_labels[2])
        return fig, "matplotlib3dquiver"


def _flow_metrics_by_state(assignments: np.ndarray, speeds: np.ndarray, names: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(names):
        mask = assignments == idx
        rows.append(
            {
                f"{prefix}_id": name,
                "n_samples": int(np.sum(mask)),
                "mean_speed": float(np.nanmean(speeds[mask])) if np.any(mask) else np.nan,
                "median_speed": float(np.nanmedian(speeds[mask])) if np.any(mask) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _assign_to_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1)


def _append_report_section(report_path: Path, lines: list[str]):
    if not report_path.exists():
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return
    existing = report_path.read_text(encoding="utf-8")
    report_path.write_text(existing.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _candidate_antipodal_pairs(centers: np.ndarray, transition_counts: np.ndarray, labels: np.ndarray | None = None) -> pd.DataFrame:
    centers = np.asarray(centers, dtype=float)
    norms = np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    unit = centers / norms
    rows = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dot = float(np.clip(np.dot(unit[i], unit[j]), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(dot)))
            rows.append(
                {
                    "cartesian_cluster_1_id": f"C{i}",
                    "cartesian_cluster_2_id": f"C{j}",
                    "dot": dot,
                    "angle_between_lobes": angle_deg,
                    "antipodality_score": abs(dot + 1.0),
                    "within_pair_transition_count": int(transition_counts[i, j] + transition_counts[j, i]),
                }
            )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["antipodality_score", "within_pair_transition_count"], ascending=[True, False]).reset_index(drop=True)


def _infer_attractors_from_pairs(pair_df: pd.DataFrame, n_lobes: int, angle_threshold_deg: float = 150.0) -> tuple[dict[int, int], pd.DataFrame]:
    lobe_to_attr = {i: i for i in range(n_lobes)}
    used = set()
    accepted = []
    next_attr = 0
    for _, row in pair_df.iterrows():
        i = int(str(row["cartesian_cluster_1_id"])[1:])
        j = int(str(row["cartesian_cluster_2_id"])[1:])
        if i in used or j in used:
            continue
        if float(row["angle_between_lobes"]) < angle_threshold_deg:
            continue
        if int(row["within_pair_transition_count"]) <= 0:
            continue
        lobe_to_attr[i] = next_attr
        lobe_to_attr[j] = next_attr
        used.update({i, j})
        accepted.append(row.to_dict())
        next_attr += 1
    for i in range(n_lobes):
        if i not in used:
            lobe_to_attr[i] = next_attr
            next_attr += 1
    return lobe_to_attr, pd.DataFrame(accepted)


def run_angular_state_space_analysis(
    sample_df: pd.DataFrame,
    *,
    outdir: str | Path | None = None,
    theta_col: str | None = None,
    phi_col: str | None = None,
    label_col: str | None = None,
    random_state: int = 0,
    max_components: int = 8,
) -> dict[str, Any]:
    d = sample_df.copy()
    theta_col = theta_col or _detect_angle_columns(d)[0]
    phi_col = phi_col or _detect_angle_columns(d)[1]
    label_col = _detect_label_column(d, preferred=label_col)
    time_col = _detect_time_column(d)

    m = np.isfinite(d[theta_col].to_numpy(float)) & np.isfinite(d[phi_col].to_numpy(float))
    d = d.loc[m].copy()
    X = d[[theta_col, phi_col]].to_numpy(float)
    alpha_values = _alpha_from_r2(d["MEED_projection_R2"].to_numpy(float)) if "MEED_projection_R2" in d.columns else None

    basin_codes, basin_centers, basin_model = _cluster_state_space(X, max_components=max_components, random_state=random_state)
    d["angular_space_cluster_id"] = [f"C{c}" for c in basin_codes]
    basin_names = [f"C{i}" for i in range(len(np.unique(basin_codes)))]
    basin_counts = _transition_counts_from_codes(basin_codes, len(basin_names))
    basin_probs = _transition_probabilities(basin_counts)
    basin_summary = _dwell_metrics_from_codes(basin_codes, basin_names, "angular_space_cluster")

    figs: dict[str, plt.Figure] = {}
    figs["angular_space_overview"] = _plot_angular_space_overview(
        d,
        theta_col=theta_col,
        phi_col=phi_col,
        cluster_codes=basin_codes,
        label_col=label_col,
        alpha_values=alpha_values,
    )
    ang_dyn_df, ang_pos, ang_vel = _prepare_consecutive_dynamics(
        d,
        pos_cols=[theta_col, phi_col],
        time_col=time_col,
        group_cols=["sub", "ses", "recording"],
    )
    ang_field_model = _fit_empirical_knn_field(ang_pos, ang_vel, n_neighbors=50)
    figs["empirical_phaseportrait_theta_phi"] = _plot_phaseportrait_or_quiver_2d(
        ang_field_model,
        ang_pos,
        title="Empirical phase portrait in theta-phi space",
        xlabel=theta_col,
        ylabel=phi_col,
    )

    out_path = Path(outdir) if outdir is not None else None
    if out_path is not None:
        _save_figure(figs["angular_space_overview"], out_path / "angular_space_overview.png")
        _save_figure(figs["empirical_phaseportrait_theta_phi"], out_path / "empirical_phaseportrait_theta_phi.png")
        _save_table(basin_summary, out_path / "angular_space_clustering_summary.csv")
        _save_table(pd.DataFrame(basin_counts, index=basin_names, columns=basin_names).reset_index(names="from_cluster"), out_path / "angular_space_clustering_transition_counts.csv")
        _save_table(pd.DataFrame(basin_probs, index=basin_names, columns=basin_names).reset_index(names="from_cluster"), out_path / "angular_space_clustering_transition_probabilities.csv")
        ang_flow = _flow_bins_2d(X, bins=34)
        _save_figure(
            _plot_flow_modulus_diagnostics(
                ang_flow["step_modulus"],
                ang_flow["modulus"].ravel(),
                "Angular space flow modulus diagnostics",
            ),
            out_path / "angular_space_flow_modulus_diagnostics.png",
        )

    warning = (
        "Raw theta/phi are angular coordinates. Distances in this plot are distorted by "
        "angular circularity, polar singularities, and coordinate wrapping. "
        "Angular plots are diagnostic and descriptive, not the definitive metric space for topology."
    )

    return {
        "data": d,
        "figures": figs,
        "theta_col": theta_col,
        "phi_col": phi_col,
        "label_col": label_col,
        "angular_space_clustering_model": basin_model,
        "angular_space_cluster_centers": basin_centers,
        "angular_space_clustering_summary": basin_summary,
        "angular_space_clustering_transition_counts": pd.DataFrame(basin_counts, index=basin_names, columns=basin_names),
        "angular_space_clustering_transition_probabilities": pd.DataFrame(basin_probs, index=basin_names, columns=basin_names),
        "empirical_theta_phi_field_model": ang_field_model,
        "warning": warning,
    }


def run_dipole_state_space_analysis(
    sample_df: pd.DataFrame,
    *,
    outdir: str | Path | None = None,
    xyz_cols: tuple[str, str, str] | None = None,
    label_col: str | None = None,
    angular_result: dict[str, Any] | None = None,
    random_state: int = 0,
    max_components: int = 8,
) -> dict[str, Any]:
    base_df = sample_df.copy()
    label_col = _detect_label_column(base_df, preferred=label_col)
    if xyz_cols is None:
        d, xyz_cols, xyz_source = _detect_dipole_xyz_columns(base_df)
    else:
        d = base_df.copy()
        xyz_source = "existing"
    time_col = _detect_time_column(d)

    m = np.isfinite(d[list(xyz_cols)].to_numpy(float)).all(axis=1)
    d = d.loc[m].copy()
    X = d[list(xyz_cols)].to_numpy(float)
    alpha_values = _alpha_from_r2(d["MEED_projection_R2"].to_numpy(float)) if "MEED_projection_R2" in d.columns else None
    dyn_df, X_dyn, V_dyn = _prepare_consecutive_dynamics(
        d,
        pos_cols=list(xyz_cols),
        time_col=time_col,
        group_cols=["sub", "ses", "recording"],
    )
    field_model_3d = _fit_empirical_knn_field(X_dyn, V_dyn, n_neighbors=50)

    lobe_codes, lobe_centers, lobe_model = _cluster_state_space(X, max_components=max_components, random_state=random_state)
    lobe_names = [f"C{i}" for i in range(len(np.unique(lobe_codes)))]
    d["cartesian_dipole_cluster_id"] = [lobe_names[c] for c in lobe_codes]

    lobe_counts = _transition_counts_from_codes(lobe_codes, len(lobe_names))
    lobe_probs = _transition_probabilities(lobe_counts)
    pair_df = _candidate_antipodal_pairs(lobe_centers, lobe_counts, labels=d[label_col].to_numpy() if label_col is not None else None)
    lobe_to_attr, accepted_pairs = _infer_attractors_from_pairs(pair_df, len(lobe_names))
    attr_codes = np.asarray([lobe_to_attr[int(code)] for code in lobe_codes], dtype=int)
    attr_unique = sorted(pd.unique(attr_codes))
    attr_reindex = {old: new for new, old in enumerate(attr_unique)}
    attr_codes = np.asarray([attr_reindex[x] for x in attr_codes], dtype=int)
    attr_names = [f"A{i}" for i in range(len(pd.unique(attr_codes)))]
    d["attractor_id"] = [attr_names[c] for c in attr_codes]

    attr_counts = _transition_counts_from_codes(attr_codes, len(attr_names))
    attr_probs = _transition_probabilities(attr_counts)

    lobe_summary = _dwell_metrics_from_codes(lobe_codes, lobe_names, "lobe")
    attr_summary = _dwell_metrics_from_codes(attr_codes, attr_names, "attractor")

    pair_lookup = {(int(str(row["cartesian_cluster_1_id"])[1:]), int(str(row["cartesian_cluster_2_id"])[1:])) for _, row in accepted_pairs.iterrows()}
    pair_lookup |= {(b, a) for a, b in pair_lookup}
    same_lobe = 0
    within_attr = 0
    between_attr = 0
    uncertain = 0
    for a, b, aa, bb in zip(lobe_codes[:-1], lobe_codes[1:], attr_codes[:-1], attr_codes[1:]):
        if a < 0 or b < 0:
            uncertain += 1
        elif a == b:
            same_lobe += 1
        elif aa == bb:
            within_attr += 1
        else:
            between_attr += 1
    transition_type_summary = pd.DataFrame(
        [
            {"transition_type": "same_lobe_persistence", "count": same_lobe},
            {"transition_type": "within_attractor_polarity_switch", "count": within_attr},
            {"transition_type": "between_attractor_configuration_switch", "count": between_attr},
            {"transition_type": "uncertain_or_transition_region", "count": uncertain},
        ]
    )

    if label_col is not None:
        label_lobe_crosstab = pd.crosstab(d[label_col].astype(str), d["cartesian_dipole_cluster_id"].astype(str))
        label_attr_crosstab = pd.crosstab(d[label_col].astype(str), d["attractor_id"].astype(str))
    else:
        label_lobe_crosstab = pd.DataFrame()
        label_attr_crosstab = pd.DataFrame()

    figures: dict[str, plt.Figure] = {}
    figures["cartesian_dipole_space_overview"] = _plot_cartesian_dipole_space_overview(
        X,
        xyz_cols=xyz_cols,
        cluster_codes=lobe_codes,
        label_col_values=d[label_col].to_numpy() if label_col is not None else None,
        alpha_values=alpha_values,
    )

    fig_pairs = plt.figure(figsize=(7.0, 5.8))
    ax_pairs = fig_pairs.add_subplot(111, projection="3d")
    ax_pairs.scatter(lobe_centers[:, 0], lobe_centers[:, 1], lobe_centers[:, 2], c=np.arange(len(lobe_centers)), cmap="tab10", s=90, edgecolors="black")
    for i, center in enumerate(lobe_centers):
        ax_pairs.text(center[0], center[1], center[2], f"L{i}", fontsize=8)
    for _, row in accepted_pairs.iterrows():
        i = int(str(row["cartesian_cluster_1_id"])[1:])
        j = int(str(row["cartesian_cluster_2_id"])[1:])
        seg = lobe_centers[[i, j]]
        ax_pairs.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="black", linewidth=1.5)
    ax_pairs.set_title("Candidate antipodal cartesian-cluster pairs")
    ax_pairs.set_xlabel(xyz_cols[0])
    ax_pairs.set_ylabel(xyz_cols[1])
    ax_pairs.set_zlabel(xyz_cols[2])
    figures["dipole_3d_antipodal_cluster_pairs"] = fig_pairs

    if time_col is not None:
        figures["dipole_3d_trajectory_examples"] = _trajectory_3d_figure(X, "Representative 3D trajectory segments")
    lobe_occupancy = np.asarray([(lobe_codes == i).mean() for i in range(len(lobe_names))], dtype=float)
    attr_occupancy = np.asarray([(attr_codes == i).mean() for i in range(len(attr_names))], dtype=float)
    cart_flow = _flow_bins_3d(X, bins=14)
    figures["cartesian_dipole_flow_modulus_diagnostics"] = _plot_flow_modulus_diagnostics(
        cart_flow["step_modulus"],
        cart_flow["modulus"].ravel(),
        "Cartesian dipole space flow modulus diagnostics",
    )
    figures["cartesian_dipole_xy_plane_cuts"] = _plot_cartesian_xy_plane_cuts(
        cart_flow,
        xyz_cols=xyz_cols,
        n_planes=10,
    )
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        raise ImportError("scikit-learn is required for PCA empirical phase portraits.") from exc
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_dyn)
    dZ = V_dyn @ pca.components_.T
    field_model_pca = _fit_empirical_knn_field(Z, dZ, n_neighbors=50)
    figures["empirical_phaseportrait_cartesian_pca"] = _plot_phaseportrait_or_quiver_2d(
        field_model_pca,
        Z,
        title="Empirical dipole phase portrait in Cartesian PCA space",
        xlabel="PC1",
        ylabel="PC2",
    )
    for (a, b), name in [((0, 1), "xy"), ((0, 2), "xz"), ((1, 2), "yz")]:
        X2 = X_dyn[:, [a, b]]
        V2 = V_dyn[:, [a, b]]
        field_model_2d = _fit_empirical_knn_field(X2, V2, n_neighbors=50)
        figures[f"empirical_phaseportrait_cartesian_{name}"] = _plot_phaseportrait_or_quiver_2d(
            field_model_2d,
            X2,
            title=f"Empirical dipole phase portrait in Cartesian {name.upper()} space",
            xlabel=xyz_cols[a],
            ylabel=xyz_cols[b],
        )
    fig3d, mode3d = _plot_phaseportrait_or_quiver_3d(
        field_model_3d,
        X_dyn,
        title="Empirical dipole phase portrait in Cartesian 3D space",
        xyz_labels=xyz_cols,
    )
    figures["empirical_phaseportrait_cartesian_3d" if mode3d == "phaseportrait3d" else "empirical_cartesian_3d_quiver"] = fig3d
    local_speed = np.sqrt(np.sum(_empirical_field_eval(field_model_3d, X_dyn) ** 2, axis=1))
    dyn_lobe_codes = _assign_to_centers(X_dyn, lobe_centers)
    dyn_attr_codes = np.asarray([attr_reindex[lobe_to_attr[int(code)]] for code in dyn_lobe_codes], dtype=int)
    empirical_vector_field_samples = pd.DataFrame(X_dyn, columns=list(xyz_cols))
    empirical_vector_field_samples[[f"v_{xyz_cols[0]}", f"v_{xyz_cols[1]}", f"v_{xyz_cols[2]}"]] = V_dyn
    empirical_flow_metrics_by_lobe = _flow_metrics_by_state(dyn_lobe_codes, local_speed, lobe_names, "cartesian_dipole_cluster")
    empirical_flow_metrics_by_attractor = _flow_metrics_by_state(dyn_attr_codes, local_speed, attr_names, "attractor")

    out_path = Path(outdir) if outdir is not None else None
    if out_path is not None:
        _save_figure(figures["cartesian_dipole_space_overview"], out_path / "cartesian_dipole_space_overview.png")
        _save_figure(figures["dipole_3d_antipodal_cluster_pairs"], out_path / "dipole_3d_antipodal_cluster_pairs.png")
        if "dipole_3d_trajectory_examples" in figures:
            _save_figure(figures["dipole_3d_trajectory_examples"], out_path / "dipole_3d_trajectory_examples.png")
        _save_figure(figures["cartesian_dipole_flow_modulus_diagnostics"], out_path / "cartesian_dipole_flow_modulus_diagnostics.png")
        _save_figure(figures["cartesian_dipole_xy_plane_cuts"], out_path / "cartesian_dipole_xy_plane_cuts.png")
        _save_figure(figures["empirical_phaseportrait_cartesian_pca"], out_path / "empirical_phaseportrait_cartesian_pca.png")
        _save_figure(figures["empirical_phaseportrait_cartesian_xy"], out_path / "empirical_phaseportrait_cartesian_xy.png")
        _save_figure(figures["empirical_phaseportrait_cartesian_xz"], out_path / "empirical_phaseportrait_cartesian_xz.png")
        _save_figure(figures["empirical_phaseportrait_cartesian_yz"], out_path / "empirical_phaseportrait_cartesian_yz.png")
        if "empirical_phaseportrait_cartesian_3d" in figures:
            _save_figure(figures["empirical_phaseportrait_cartesian_3d"], out_path / "empirical_phaseportrait_cartesian_3d.png")
        if "empirical_cartesian_3d_quiver" in figures:
            _save_figure(figures["empirical_cartesian_3d_quiver"], out_path / "empirical_cartesian_3d_quiver.png")

        _save_table(lobe_summary, out_path / "cartesian_dipole_cluster_dwell_metrics.csv")
        _save_table(attr_summary, out_path / "attractor_dwell_metrics.csv")
        _save_table(pair_df, out_path / "candidate_antipodal_pairs.csv")
        _save_table(accepted_pairs, out_path / "attractor_pair_summary.csv")
        _save_table(pd.DataFrame(lobe_counts, index=lobe_names, columns=lobe_names).reset_index(names="from_cartesian_dipole_cluster"), out_path / "cartesian_dipole_cluster_transition_counts.csv")
        _save_table(pd.DataFrame(lobe_probs, index=lobe_names, columns=lobe_names).reset_index(names="from_cartesian_dipole_cluster"), out_path / "cartesian_dipole_cluster_transition_probabilities.csv")
        _save_table(pd.DataFrame(attr_counts, index=attr_names, columns=attr_names).reset_index(names="from_attractor"), out_path / "attractor_transition_counts.csv")
        _save_table(pd.DataFrame(attr_probs, index=attr_names, columns=attr_names).reset_index(names="from_attractor"), out_path / "attractor_transition_probabilities.csv")
        _save_table(
            pd.DataFrame(
                {
                    "cartesian_dipole_cluster_id": lobe_names,
                    "x": lobe_centers[:, 0],
                    "y": lobe_centers[:, 1],
                    "z": lobe_centers[:, 2],
                    "occupancy": lobe_occupancy,
                }
            ),
            out_path / "cartesian_dipole_cluster_summary.csv",
        )
        _save_table(
            pd.DataFrame(
                {
                    "from_cartesian_dipole_cluster": np.repeat(lobe_names, len(lobe_names)),
                    "to_cartesian_dipole_cluster": lobe_names * len(lobe_names),
                    "weight": lobe_counts.reshape(-1),
                }
            ),
            out_path / "cartesian_dipole_cluster_topology_edges.csv",
        )
        _save_table(
            pd.DataFrame(
                {
                    "from_attractor": np.repeat(attr_names, len(attr_names)),
                    "to_attractor": attr_names * len(attr_names),
                    "weight": attr_counts.reshape(-1),
                }
            ),
            out_path / "topology_edges_attractor_level.csv",
        )
        if label_col is not None:
            _save_table(label_lobe_crosstab.reset_index(), out_path / "label_cartesian_dipole_cluster_crosstab.csv")
            _save_table(label_attr_crosstab.reset_index(), out_path / "label_attractor_crosstab.csv")
        _save_table(empirical_vector_field_samples, out_path / "empirical_vector_field_samples.csv")
        _save_table(empirical_flow_metrics_by_lobe, out_path / "empirical_flow_metrics_by_lobe.csv")
        _save_table(empirical_flow_metrics_by_attractor, out_path / "empirical_flow_metrics_by_attractor.csv")

        angular_warning = angular_result.get("warning") if angular_result is not None else "Angular analysis not supplied."
        report_lines = [
            "# State-space analysis report",
            "",
            "## Inputs",
            f"- dataframe columns used: {', '.join(d.columns.astype(str))}",
            f"- detected angle columns: {', '.join(_detect_angle_columns(d))}",
            f"- detected 3D dipole coordinate columns: {', '.join(xyz_cols)}",
            f"- dipole coordinate source: {xyz_source}",
            f"- detected theoretical label column: {label_col if label_col is not None else 'none'}",
            "",
            "## Angular warning",
            angular_warning,
            "",
            "## Lobe and attractor structure",
            f"- number of detected cartesian dipole clusters: {len(lobe_names)}",
            f"- number of candidate antipodal cartesian-cluster pairs: {len(pair_df)}",
            f"- number of inferred bistable attractors: {len(attr_names)}",
            "",
            "## Dynamics",
            f"- within-attractor polarity-switch rate: {float(within_attr / max(1, len(attr_codes) - 1)):.4f}",
            f"- between-attractor configuration-switch rate: {float(between_attr / max(1, len(attr_codes) - 1)):.4f}",
            "",
            "## Occupancy and dwell",
            lobe_summary.to_markdown(index=False),
            "",
            attr_summary.to_markdown(index=False),
            "",
            "## Candidate antipodal pairs",
            pair_df.head(20).to_markdown(index=False) if len(pair_df) else "none",
            "",
            "## Transition topology summary",
            transition_type_summary.to_markdown(index=False),
            "",
            "## Label comparison",
            label_lobe_crosstab.to_markdown() if len(label_lobe_crosstab) else "No theoretical label column detected.",
            "",
            label_attr_crosstab.to_markdown() if len(label_attr_crosstab) else "No theoretical label column detected.",
            "",
            "## Limitations",
            "- theta/phi visualization is descriptive and suffers from coordinate wrapping and singularity distortions.",
            "- 3D clustering depends on model choice and may split or merge lobes differently across settings.",
            "",
            "## Recommended next checks",
            "- compare cartesian dipole clustering stability across GMM and DBSCAN/HDBSCAN when available",
            "- inspect whether inferred antipodal pairs also share theoretical labels",
            "- compare high-rho subsets versus all samples",
        ]
        report_path = out_path / "state_space_analysis_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        _append_report_section(
            report_path,
            [
                "## Empirical phase portrait estimation",
                "- The phase-portrait function was not analytical.",
                "- It was estimated from observed local trajectory displacements.",
                f"- The Cartesian empirical phase portrait used the already-existing {', '.join(xyz_cols)} dipole coordinates.",
                "- The 2D phase portrait was created by projecting Cartesian positions and velocities into PCA space.",
                "- The theta-phi phase portrait was included only as a diagnostic angular-coordinate view.",
                "- The estimated vector field was used to characterize local flow, lobe stability, polarity switching, and between-attractor transitions.",
            ],
        )

    return {
        "data": d,
        "figures": figures,
        "xyz_cols": xyz_cols,
        "xyz_source": xyz_source,
        "label_col": label_col,
        "cartesian_dipole_clustering_model": lobe_model,
        "cartesian_dipole_cluster_centers": pd.DataFrame(lobe_centers, columns=list(xyz_cols)).assign(cartesian_dipole_cluster_id=lobe_names),
        "empirical_cartesian_field_model": field_model_3d,
        "empirical_cartesian_pca_field_model": field_model_pca,
        "candidate_antipodal_pairs": pair_df,
        "accepted_antipodal_pairs": accepted_pairs,
        "cartesian_dipole_cluster_summary": lobe_summary,
        "attractor_summary": attr_summary,
        "empirical_flow_metrics_by_lobe": empirical_flow_metrics_by_lobe,
        "empirical_flow_metrics_by_attractor": empirical_flow_metrics_by_attractor,
        "cartesian_dipole_cluster_transition_counts": pd.DataFrame(lobe_counts, index=lobe_names, columns=lobe_names),
        "cartesian_dipole_cluster_transition_probabilities": pd.DataFrame(lobe_probs, index=lobe_names, columns=lobe_names),
        "attractor_transition_counts": pd.DataFrame(attr_counts, index=attr_names, columns=attr_names),
        "attractor_transition_probabilities": pd.DataFrame(attr_probs, index=attr_names, columns=attr_names),
        "transition_type_summary": transition_type_summary,
        "label_cartesian_dipole_cluster_crosstab": label_lobe_crosstab,
        "label_attractor_crosstab": label_attr_crosstab,
    }
