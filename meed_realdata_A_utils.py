"""Problem A real-data utilities for MEED.

This file is intentionally analysis-oriented. It builds the first-layer data
products and the figures/tables for Problem A:

1. Zanesco-style GFP/TD replication plus rho and TD decomposition.
2. GFP and MEED quantities in theta-phi space.
3. Non-reducibility of rho2 to GFP or template correlation.

The code assumes the validated method utilities are available as `utils.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, least_squares
from scipy.signal import find_peaks
from scipy.special import expit
from scipy.stats import pearsonr

import utils as u


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


def find_edf_files(data_root: str | Path, max_subjects: int | None = None) -> list[Path]:
    root = Path(data_root)
    files = sorted([p for p in root.glob("*.edf") if p.is_file()])
    return files[:max_subjects] if max_subjects is not None else files


def parse_subject_id(path: str | Path) -> str:
    stem = Path(path).stem
    m = re.search(r"sub-([^_]+)", stem)
    if m:
        return f"sub-{m.group(1)}"
    return stem.replace(" ", "_")


def load_and_align_raw(eeg_path: str | Path, template_ch_names: list[str]) -> mne.io.BaseRaw:
    eeg_path = Path(eeg_path)
    suffix = eeg_path.suffix.lower()
    if suffix == ".edf":
        raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose="ERROR")
    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported file format for {eeg_path}")
    raw = standard_names_picker(raw)
    keep = [ch for ch in template_ch_names if ch in raw.ch_names]
    raw.reorder_channels(keep)
    return raw


def center_l2_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc / np.maximum(np.linalg.norm(Xc, axis=0, keepdims=True), eps)


def gfp_from_channel_time(X: np.ndarray) -> np.ndarray:
    Xc = np.asarray(X, dtype=float) - np.asarray(X, dtype=float).mean(axis=0, keepdims=True)
    return np.sqrt(np.mean(Xc ** 2, axis=0))


def wrap180(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


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


def fit_subject_level(raw: mne.io.BaseRaw, meta_level: TemplateLevel, k: int, random_state: int = 0) -> TemplateLevel:
    from pycrostates.cluster import ModKMeans

    model = ModKMeans(n_clusters=int(k), random_state=int(random_state))
    model.fit(raw.copy().pick("eeg"))
    B = u.center_l2_cols(np.asarray(model.cluster_centers_, dtype=float).T)
    labels = [chr(ord("A") + i) for i in range(B.shape[1])]
    B, labels = align_maps_to_reference(B, meta_level.B, labels, meta_level.labels)
    return build_template_level("subject", B, labels, raw.info, model=model)


def fit_subject_maps_all_k(raw: mne.io.BaseRaw, json_path: str | Path, k_values=range(3, 9), random_state: int = 0):
    levels_by_k = {}
    for k in k_values:
        meta_level, _ = load_meta_level(json_path, k)
        subject_level = fit_subject_level(raw, meta_level, k=k, random_state=random_state)
        levels_by_k[int(k)] = {"meta": meta_level, "subject": subject_level}
    return levels_by_k


# ---------------------------------------------------------------------
# First-layer data products
# ---------------------------------------------------------------------


def unit_and_angles_from_Q(Q: np.ndarray, eps: float = 1e-12):
    rho = np.linalg.norm(Q, axis=0)
    U_raw = Q / np.maximum(rho[None, :], eps)
    U_can = np.zeros_like(U_raw)
    theta = np.full(Q.shape[1], np.nan)
    phi = np.full(Q.shape[1], np.nan)
    for i in range(Q.shape[1]):
        th, ph, ui = u.canonical_theta_phi_from_vector(Q[:, i])
        theta[i] = th
        phi[i] = ph
        U_can[:, i] = ui
    return rho, U_raw, U_can, theta, phi


def backward_td_decomposition(Xn: np.ndarray, S_orth: np.ndarray, eps: float = 1e-12) -> pd.DataFrame:
    """Decompose Zanesco TD into MEED and residual components.

    Xn must be centered and L2-normalized. TD stays on the same 0 to 2
    distance scale as Zanesco DISS. Squared terms are included because the
    MEED projection and residual are orthogonal.
    """
    S = np.asarray(S_orth, dtype=float)
    Q = S.T @ Xn
    dX = Xn[:, 1:] - Xn[:, :-1]
    dQ = Q[:, 1:] - Q[:, :-1]
    dX_meed = S @ dQ
    dX_res = dX - dX_meed

    n = Xn.shape[1]
    TD = np.full(n, np.nan)
    TD_MEED = np.full(n, np.nan)
    TD_res = np.full(n, np.nan)
    TD[1:] = np.linalg.norm(dX, axis=0)
    TD_MEED[1:] = np.linalg.norm(dX_meed, axis=0)
    TD_res[1:] = np.linalg.norm(dX_res, axis=0)

    TD2 = TD ** 2
    TD2_MEED = TD_MEED ** 2
    TD2_res = TD_res ** 2
    F = np.divide(TD2_MEED, TD2, out=np.full(n, np.nan), where=TD2 > eps)
    closure = TD2 - TD2_MEED - TD2_res

    return pd.DataFrame({
        "TD": TD,
        "TD_MEED": TD_MEED,
        "TD_res": TD_res,
        "TD2": TD2,
        "TD2_MEED": TD2_MEED,
        "TD2_res": TD2_res,
        "F_MEED_TD": F,
        "TD_decomposition_closure_error": closure,
    })


def compute_template_distances(U_raw: np.ndarray, theta: np.ndarray, phi: np.ndarray, level: TemplateLevel, prefix: str):
    out = {}
    Ut = level.U / np.maximum(np.linalg.norm(level.U, axis=0, keepdims=True), 1e-12)
    for j, lab in enumerate(level.labels):
        dot = np.sum(U_raw * Ut[:, [j]], axis=0)
        dot = np.clip(dot, -1.0, 1.0)
        out[f"psi_{prefix}_{lab}"] = np.degrees(np.arccos(np.abs(dot)))

        th_pos, ph_pos, _ = u.canonical_theta_phi_from_vector(Ut[:, j])
        th_neg, ph_neg, _ = u.canonical_theta_phi_from_vector(-Ut[:, j])
        theta_t = np.where(dot < 0, th_neg, th_pos)
        phi_t = np.where(dot < 0, ph_neg, ph_pos)
        out[f"theta_{prefix}_{lab}"] = np.abs(wrap180(theta - theta_t))
        out[f"phi_{prefix}_{lab}"] = np.abs(phi - phi_t)
    return pd.DataFrame(out)


def compute_samplewise_characterization(raw: mne.io.BaseRaw, meta_level: TemplateLevel, subject_level: TemplateLevel | None, sub: str) -> pd.DataFrame:
    X = np.asarray(raw.get_data(picks="eeg"), dtype=float)
    sfreq = float(raw.info["sfreq"])
    Xn = center_l2_columns(X)

    S = meta_level.geom["S_orth"]
    Q = np.asarray(S, dtype=float).T @ Xn
    rho, U_raw, U_can, theta, phi = unit_and_angles_from_Q(Q)
    td_parts = backward_td_decomposition(Xn, S)

    psiD = np.full(Xn.shape[1], np.nan)
    dot = np.sum(U_raw[:, 1:] * U_raw[:, :-1], axis=0)
    psiD[1:] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    gfp = gfp_from_channel_time(X)
    local_peaks = np.zeros(X.shape[1], dtype=bool)
    peak_idx, _ = find_peaks(gfp, distance=1)
    local_peaks[peak_idx] = True

    df = pd.DataFrame({
        "sub": sub,
        "sample": np.arange(X.shape[1], dtype=int),
        "time": np.arange(X.shape[1], dtype=float) / sfreq,
        "GFP": gfp,
        "local_GFP_peak": local_peaks,
        "d_x": Q[0],
        "d_y": Q[1],
        "d_z": Q[2],
        "theta": theta,
        "phi": phi,
        "rho": rho,
        "rho2": rho ** 2,
        "psiD": psiD,
    })
    df = pd.concat([df, td_parts], axis=1)

    levels_to_use = [("meta", meta_level)]
    if subject_level is not None:
        levels_to_use.append(("sub", subject_level))

    for prefix, level in levels_to_use:
        corr = np.abs(Xn.T @ level.B)
        for j, lab in enumerate(level.labels):
            df[f"corr_{prefix}_{lab}"] = corr[:, j]
        df = pd.concat([df, compute_template_distances(U_raw, theta, phi, level, prefix)], axis=1)

    meta_corr_cols = [f"corr_meta_{lab}" for lab in meta_level.labels if f"corr_meta_{lab}" in df]
    C = df[meta_corr_cols].to_numpy(float)
    C2 = C ** 2
    p = C2 / np.maximum(C2.sum(axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1) / np.log(max(C.shape[1], 2))
    df["max_meta_corr"] = np.nanmax(C, axis=1)
    df["corr_entropy"] = entropy
    return df


def export_sensor_and_dipole_maps(sub: str, levels_by_k: dict, outdir: str | Path):
    sensor_rows = []
    dipole_rows = []
    for k, levels in levels_by_k.items():
        level = levels["subject"]
        for j, lab in enumerate(level.labels):
            for ch, val in zip(level.info.ch_names, level.B[:, j]):
                sensor_rows.append({"sub": sub, "k": int(k), "label": lab, "channel": ch, "value": float(val)})

            q = level.Q[:, j]
            rho = float(np.linalg.norm(q))
            theta = float(level.D[0, j])
            phi = float(level.D[1, j])
            vals = {
                "d_x": float(q[0]),
                "d_y": float(q[1]),
                "d_z": float(q[2]),
                "theta": theta,
                "phi": phi,
                "rho": rho,
                "rho2": rho ** 2,
            }
            for component, value in vals.items():
                dipole_rows.append({"sub": sub, "k": int(k), "label": lab, "component": component, "value": value})

    sensor_df = pd.DataFrame(sensor_rows)
    dipole_df = pd.DataFrame(dipole_rows)
    outdir = Path(outdir)
    sensor_df.to_csv(outdir / f"{sub}_sensor_maps.csv", index=False)
    dipole_df.to_csv(outdir / f"{sub}_dipole_maps.csv", index=False)
    return sensor_df, dipole_df


def compute_map_fit_r2_table(sub: str, levels_by_k: dict) -> pd.DataFrame:
    rows = []
    for k, levels in sorted(levels_by_k.items()):
        meta_level = levels["meta"]
        subject_level = levels["subject"]
        n_maps = min(meta_level.B.shape[1], subject_level.B.shape[1], len(meta_level.labels), len(subject_level.labels))
        for j in range(n_maps):
            x = np.asarray(meta_level.B[:, j], dtype=float)
            y = np.asarray(subject_level.B[:, j], dtype=float)
            if len(x) == 0 or len(y) == 0 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
                r2 = np.nan
                r = np.nan
            else:
                r = float(np.corrcoef(x, y)[0, 1])
                r2 = float(r ** 2)
            rows.append({
                "sub": sub,
                "k": int(k),
                "label": str(meta_level.labels[j]),
                "sensor_map_r": r,
                "sensor_map_R2": r2,
            })
    return pd.DataFrame(rows)


def export_map_fit_r2_table(sub: str, levels_by_k: dict, outdir: str | Path) -> pd.DataFrame:
    df = compute_map_fit_r2_table(sub, levels_by_k)
    outdir = Path(outdir)
    df.to_csv(outdir / f"{sub}_map_fit_r2.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Item 1 figures and table
# ---------------------------------------------------------------------


def _finite_xy(df, x, y):
    m = np.isfinite(df[x].to_numpy(float)) & np.isfinite(df[y].to_numpy(float))
    return df.loc[m, x].to_numpy(float), df.loc[m, y].to_numpy(float)


def _hexbin_percent(ax, x, y, *, gridsize=55, mincnt=2, cmap="viridis", **kwargs):
    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=mincnt, cmap=cmap, **kwargs)
    counts = np.asarray(hb.get_array(), dtype=float)
    total = counts.sum()
    if total > 0:
        hb.set_array(100.0 * counts / total)
    return hb


def _set_hexbin_average_clim(hexbins):
    arrays = [np.asarray(hb.get_array(), dtype=float) for hb in hexbins if hb is not None and hb.get_array() is not None and len(hb.get_array())]
    if not arrays:
        return
    vmin = float(np.mean([arr.min() for arr in arrays])) + 0.5 * float(np.std([arr.min() for arr in arrays]))
    vmax = float(np.mean([arr.max() for arr in arrays])) - 0.5 * float(np.std([arr.max() for arr in arrays]))
    for hb in hexbins:
        if hb is not None:
            hb.set_clim(vmin=vmin, vmax=vmax)


def _add_linear_regression_overlay(ax, x, y):
    stats = _linear_regression_stats(x, y)
    if stats is None:
        return
    ax.plot(stats["xs"], stats["ys"], "--", color="black", gapcolor="white", dashes=(5, 5), lw=1.5)
    ax.text(
        0.98,
        0.04,
        f"r = {stats['r']:.3f}\nR^2 = {stats['r2']:.3f}\np = {stats['p']:.3e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
    )


def _linear_regression_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None

    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r, p = pearsonr(x, y)

    xs = np.linspace(np.min(x), np.max(x), 100)
    ys = coef[0] + coef[1] * xs
    return {"r": float(r), "r2": float(r2), "p": float(p), "xs": xs, "ys": ys}


def _fit_linear_model(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    out = {
        "n_valid": int(len(x)),
        "intercept": np.nan,
        "slope": np.nan,
        "r": np.nan,
        "R2": np.nan,
        "x_min": np.nan,
        "x_max": np.nan,
    }
    if len(x) == 0:
        return out

    out["x_min"] = float(np.min(x))
    out["x_max"] = float(np.max(x))
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return out

    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    out["intercept"] = float(coef[0])
    out["slope"] = float(coef[1])
    out["R2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    out["r"] = float(pearsonr(x, y)[0])
    return out


def _add_loglog_regression_overlay(ax, x, y):
    stats = _loglog_regression_stats(x, y)
    if stats is None:
        return
    ax.plot(stats["xs"], stats["ys"], "--", color="black", gapcolor="white", lw=1.5)
    ax.text(
        0.98,
        0.04,
        f"r = {stats['r']:.3f}\nR^2 = {stats['r2']:.3f}\np = {stats['p']:.3e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
    )


def _loglog_regression_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None

    lx = np.log10(x)
    ly = np.log10(y)
    X = np.column_stack([np.ones(len(lx)), lx])
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    lyhat = X @ coef
    ss_res = float(np.sum((ly - lyhat) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r, p = pearsonr(lx, ly)

    xs = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 200)
    ys = (10.0 ** coef[0]) * (xs ** coef[1])
    return {"r": float(r), "r2": float(r2), "p": float(p), "xs": xs, "ys": ys}


def _add_linear_regression_overlay_r_only(ax, x, y, *, fontsize: float = 10.0):
    stats = _linear_regression_stats(x, y)
    if stats is None:
        return
    ax.plot(stats["xs"], stats["ys"], "--", color="black", gapcolor="white", dashes=(5, 5), lw=1.5)
    ax.plot([], [], linestyle="none", label=f"r = {stats['r']:.3f}")
    ax.legend(loc="best", fontsize=fontsize, framealpha=0.8, facecolor="white", edgecolor="none", handlelength=0, handletextpad=0.2)


def _add_loglog_regression_overlay_r_only(ax, x, y, *, fontsize: float = 10.0):
    stats = _loglog_regression_stats(x, y)
    if stats is None:
        return
    ax.plot(stats["xs"], stats["ys"], "--", color="black", gapcolor="white", lw=1.5)
    ax.plot([], [], linestyle="none", label=f"r = {stats['r']:.3f}")
    ax.legend(loc="best", fontsize=fontsize, framealpha=0.8, facecolor="white", edgecolor="none", handlelength=0, handletextpad=0.2)


def compute_item1_gfp_link_residuals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    eps = 1e-6
    d["GFP_uV"] = d["GFP"] * 1e6

    for col in [
        "log_GFP_uV",
        "log_TD",
        "log_psiD",
        "log_F_MEED_TD",
        "log_rho2",
        "logGFP_hat_from_TD",
        "logGFP_hat_from_psiD",
        "res_logGFP_psiD_minus_TD",
    ]:
        d[col] = np.nan

    summary = {
        "alpha_TD": np.nan,
        "beta_TD": np.nan,
        "alpha_psiD": np.nan,
        "beta_psiD": np.nan,
        "R2_TD_loglog_GFP": np.nan,
        "R2_psiD_loglog_GFP": np.nan,
        "n_fit": 0,
    }

    gfp_ok = np.isfinite(d["GFP_uV"].to_numpy(float)) & (d["GFP_uV"].to_numpy(float) > 0)
    f_ok = np.isfinite(d["F_MEED_TD"].to_numpy(float))
    rho2_ok = np.isfinite(d["rho2"].to_numpy(float))

    d.loc[gfp_ok, "log_GFP_uV"] = np.log10(d.loc[gfp_ok, "GFP_uV"].to_numpy(float))
    f_clip = np.clip(d.loc[f_ok, "F_MEED_TD"].to_numpy(float), eps, 1.0 - eps)
    d.loc[f_ok, "log_F_MEED_TD"] = np.log10(f_clip)
    d.loc[rho2_ok, "log_rho2"] = np.log10(np.maximum(d.loc[rho2_ok, "rho2"].to_numpy(float), eps))

    m = (
        np.isfinite(d["GFP_uV"].to_numpy(float))
        & np.isfinite(d["TD"].to_numpy(float))
        & np.isfinite(d["psiD"].to_numpy(float))
        & (d["GFP_uV"].to_numpy(float) > 0)
        & (d["TD"].to_numpy(float) > 0)
        & (d["psiD"].to_numpy(float) > 0)
    )
    summary["n_fit"] = int(np.sum(m))
    if summary["n_fit"] < 3:
        return d, summary

    d.loc[m, "log_TD"] = np.log10(d.loc[m, "TD"].to_numpy(float))
    d.loc[m, "log_psiD"] = np.log10(d.loc[m, "psiD"].to_numpy(float))

    def fit_loglog(x: np.ndarray, y: np.ndarray):
        X = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha = float(coef[0])
        beta = float(coef[1])
        yhat = X @ coef
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return alpha, beta, r2

    x = d.loc[m, "log_GFP_uV"].to_numpy(float)
    y_td = d.loc[m, "log_TD"].to_numpy(float)
    y_psi = d.loc[m, "log_psiD"].to_numpy(float)

    alpha_td, beta_td, r2_td = fit_loglog(x, y_td)
    alpha_psi, beta_psi, r2_psi = fit_loglog(x, y_psi)
    summary.update({
        "alpha_TD": alpha_td,
        "beta_TD": beta_td,
        "alpha_psiD": alpha_psi,
        "beta_psiD": beta_psi,
        "R2_TD_loglog_GFP": r2_td,
        "R2_psiD_loglog_GFP": r2_psi,
    })

    td_slope_ok = np.isfinite(beta_td) and (abs(beta_td) > 1e-12)
    psi_slope_ok = np.isfinite(beta_psi) and (abs(beta_psi) > 1e-12)
    if td_slope_ok:
        d.loc[m, "logGFP_hat_from_TD"] = (d.loc[m, "log_TD"] - alpha_td) / beta_td
    if psi_slope_ok:
        d.loc[m, "logGFP_hat_from_psiD"] = (d.loc[m, "log_psiD"] - alpha_psi) / beta_psi

    if td_slope_ok and psi_slope_ok:
        both = m & np.isfinite(d["logGFP_hat_from_TD"].to_numpy(float)) & np.isfinite(d["logGFP_hat_from_psiD"].to_numpy(float))
        d.loc[both, "res_logGFP_psiD_minus_TD"] = d.loc[both, "logGFP_hat_from_psiD"] - d.loc[both, "logGFP_hat_from_TD"]
    return d, summary


def _plot_item1_loglog_panel(ax, d: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, *, title: str | None = None, annotation_mode: str = "full"):
    xx, yy = _finite_xy(d, x, y)
    m = (xx > 0) & (yy > 0)
    if np.sum(m) == 0:
        ax.text(0.5, 0.5, "no valid data", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="0.35")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        return None

    hb = _hexbin_percent(ax, xx[m], yy[m], gridsize=55, mincnt=2, cmap="viridis", xscale="log", yscale="log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)

    if annotation_mode == "full":
        _add_loglog_regression_overlay(ax, xx[m], yy[m])
    elif annotation_mode == "r_only":
        _add_loglog_regression_overlay_r_only(ax, xx[m], yy[m], fontsize=10.5)
    return hb


def _plot_item1_linear_panel(ax, d: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, *, title: str | None = None, add_zero_line: bool = False, annotation_mode: str = "none"):
    if add_zero_line:
        ax.axhline(0.0, color="0.25", lw=1.0, linestyle="--")
    xx, yy = _finite_xy(d, x, y)
    if len(xx) == 0:
        ax.text(0.5, 0.5, "no valid data", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="0.35")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        return None

    hb = _hexbin_percent(ax, xx, yy, gridsize=55, mincnt=2, cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if annotation_mode == "full":
        _add_linear_regression_overlay(ax, xx, yy)
    elif annotation_mode == "r_only":
        _add_linear_regression_overlay_r_only(ax, xx, yy, fontsize=10.5)
    return hb


def plot_item1A_zanesco_panel(df: pd.DataFrame):
    d = df.copy()
    d["GFP_uV"] = d["GFP"] * 1e6
    pairs = [
        ("GFP_uV", "TD", "GFP (uV)", "TD"),
        ("GFP_uV", "rho", "GFP (uV)", "rho"),
        ("GFP_uV", "rho2", "GFP (uV)", "rho2"),
        ("GFP_uV", "psiD", "GFP (uV)", "psiD (deg)"),
        ("TD", "psiD", "TD", "psiD (deg)"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    hexbins = []
    for c, (x, y, xl, yl) in enumerate(pairs):
        xx, yy = _finite_xy(d, x, y)
        hb = _hexbin_percent(axes[0, c], xx, yy, gridsize=55, mincnt=2, cmap="viridis")
        hexbins.append(hb)
        axes[0, c].set_xlabel(xl)
        axes[0, c].set_ylabel(yl)
        axes[0, c].set_title(f"{yl} vs {xl}")

        hb = _plot_item1_loglog_panel(axes[1, c], d, x, y, xl, yl, title="log-log", annotation_mode="full" if (x, y) in {("GFP_uV", "TD"), ("GFP_uV", "psiD"), ("TD", "psiD")} else "none")
        hexbins.append(hb)
    _set_hexbin_average_clim(hexbins)
    for hb, ax in zip(hexbins, axes.ravel()):
        fig.colorbar(hb, ax=ax, shrink=0.80, label="% of samples")
    fig.suptitle("Figure 1A: Zanesco-style GFP/TD relationships plus MEED", fontsize=14)
    return fig


def plot_item1B_peak_overlay(df: pd.DataFrame):
    d = df.copy()
    d["GFP_uV"] = d["GFP"] * 1e6
    peak_col = "is_peak" if "is_peak" in d.columns else "local_GFP_peak"
    peak_mask = d[peak_col].fillna(False).astype(bool).to_numpy()
    pairs = [
        ("GFP_uV", "TD", "GFP (uV)", "TD"),
        ("GFP_uV", "rho", "GFP (uV)", "rho"),
        ("GFP_uV", "rho2", "GFP (uV)", "rho2"),
        ("GFP_uV", "psiD", "GFP (uV)", "psiD (deg)"),
        ("TD", "psiD", "TD", "psiD (deg)"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    hexbins = []
    for c, (x, y, xl, yl) in enumerate(pairs):
        xx, yy = _finite_xy(d, x, y)
        hb = _hexbin_percent(axes[0, c], xx, yy, gridsize=55, mincnt=2, cmap="bone", linewidths=0)
        hexbins.append(hb)
        axes[0, c].scatter(d.loc[peak_mask, x], d.loc[peak_mask, y], s=5, alpha=0.35, color="#fde725")
        axes[0, c].set_xlabel(xl)
        axes[0, c].set_ylabel(yl)
        axes[0, c].set_title(f"{yl} vs {xl}")

        m = (d[x].to_numpy(float) > 0) & (d[y].to_numpy(float) > 0)
        hb = _hexbin_percent(axes[1, c], d.loc[m, x], d.loc[m, y], gridsize=55, mincnt=2, cmap="bone", linewidths=0, xscale="log", yscale="log")
        hexbins.append(hb)
        pm = peak_mask & m
        axes[1, c].scatter(d.loc[pm, x], d.loc[pm, y], s=5, alpha=0.35, color="#fde725")
        axes[1, c].set_xlabel(xl)
        axes[1, c].set_ylabel(yl)
        axes[1, c].set_title("log-log")
    _set_hexbin_average_clim(hexbins)
    fig.suptitle("Figure 1B: all samples as density, local GFP peaks as points", fontsize=14)
    return fig


def plot_item1C_td_decomposition(df: pd.DataFrame):
    d, _ = compute_item1_gfp_link_residuals(df)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    residual_label = "log-GFP residual: psiD - TD"
    specs = [
        (0, 0, "TD", "TD_MEED", "TD", "TD_MEED", False),
        (1, 0, "TD", "TD_res", "TD", "TD_res", False),
        (0, 1, "GFP_uV", "res_logGFP_psiD_minus_TD", "GFP (uV)", residual_label, True),
        (1, 1, "log_GFP_uV", "res_logGFP_psiD_minus_TD", "log GFP (uV)", residual_label, True),
        (0, 2, "F_MEED_TD", "res_logGFP_psiD_minus_TD", "F_MEED_TD", residual_label, True),
        (1, 2, "log_F_MEED_TD", "res_logGFP_psiD_minus_TD", "log F_MEED_TD", residual_label, True),
        (0, 3, "rho2", "res_logGFP_psiD_minus_TD", "rho2", residual_label, True),
        (1, 3, "log_rho2", "res_logGFP_psiD_minus_TD", "log rho2", residual_label, True),
    ]

    hexbins = []
    for r, c, x, y, xl, yl, add_zero_line in specs:
        annotation_mode = "full" if (x, y) == ("log_F_MEED_TD", "res_logGFP_psiD_minus_TD") else "none"
        hb = _plot_item1_linear_panel(axes[r, c], d, x, y, xl, yl, add_zero_line=add_zero_line, annotation_mode=annotation_mode)
        if hb is not None:
            hexbins.append(hb)
        if x == "TD":
            axes[r, c].set_xlim(0, 2)
        if y in {"TD_MEED", "TD_res"}:
            axes[r, c].set_ylim(0, 2)
        if x == "F_MEED_TD":
            axes[r, c].set_xlim(0, 1)
    _set_hexbin_average_clim(hexbins)
    hb_idx = 0
    for r, c, x, y, xl, yl, add_zero_line in specs:
        xx, yy = _finite_xy(d, x, y)
        if len(xx) == 0:
            continue
        fig.colorbar(hexbins[hb_idx], ax=axes[r, c], shrink=0.80, label="% of samples")
        hb_idx += 1

    fvals = d["F_MEED_TD"].to_numpy(float)
    fvals = fvals[np.isfinite(fvals)]
    stats = {
        "median": float(np.nanmedian(fvals)) if len(fvals) else np.nan,
        "mean": float(np.nanmean(fvals)) if len(fvals) else np.nan,
        "std": float(np.nanstd(fvals)) if len(fvals) else np.nan,
        "q1": float(np.nanpercentile(fvals, 25)) if len(fvals) else np.nan,
        "q3": float(np.nanpercentile(fvals, 75)) if len(fvals) else np.nan,
        "p5": float(np.nanpercentile(fvals, 5)) if len(fvals) else np.nan,
        "p95": float(np.nanpercentile(fvals, 95)) if len(fvals) else np.nan,
    }
    axes[0, 0].text(
        0.98,
        0.96,
        (
            f"F median = {stats['median']:.3f}\n"
            f"F mean = {stats['mean']:.3f}\n"
            f"F std = {stats['std']:.3f}\n"
            f"F q1 = {stats['q1']:.3f}\n"
            f"F q3 = {stats['q3']:.3f}\n"
            f"F p5 = {stats['p5']:.3f}\n"
            f"F p95 = {stats['p95']:.3f}"
        ),
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.5},
    )
    fig.suptitle("Figure 1C: TD decomposition and TD-psiD GFP-link residual", fontsize=14)
    return fig


def table1_subject_summary(df: pd.DataFrame, sub: str) -> pd.DataFrame:
    def corr(a, b):
        return df[[a, b]].dropna().corr().iloc[0, 1]

    f = df["F_MEED_TD"].dropna()
    d_res, fit_summary = compute_item1_gfp_link_residuals(df)
    residual = d_res["res_logGFP_psiD_minus_TD"].dropna()
    row = {
        "sub": sub,
        "median_GFP": float(df["GFP"].median()),
        "median_TD": float(df["TD"].median()),
        "median_TD_MEED": float(df["TD_MEED"].median()),
        "median_TD_res": float(df["TD_res"].median()),
        "median_F_MEED_TD": float(df["F_MEED_TD"].median()),
        "IQR_F_MEED_TD": float(f.quantile(0.75) - f.quantile(0.25)),
        "corr_TD_psiD": float(corr("TD", "psiD")),
        "corr_TD_TD_MEED": float(corr("TD", "TD_MEED")),
        "corr_TD_TD_res": float(corr("TD", "TD_res")),
        "median_abs_TD_closure_error": float(np.abs(df["TD_decomposition_closure_error"]).median()),
        "alpha_TD_loglog_GFP": float(fit_summary["alpha_TD"]),
        "beta_TD_loglog_GFP": float(fit_summary["beta_TD"]),
        "alpha_psiD_loglog_GFP": float(fit_summary["alpha_psiD"]),
        "beta_psiD_loglog_GFP": float(fit_summary["beta_psiD"]),
        "R2_TD_loglog_GFP": float(fit_summary["R2_TD_loglog_GFP"]),
        "R2_psiD_loglog_GFP": float(fit_summary["R2_psiD_loglog_GFP"]),
        "median_res_logGFP_psiD_minus_TD": float(residual.median()) if len(residual) else np.nan,
        "IQR_res_logGFP_psiD_minus_TD": float(residual.quantile(0.75) - residual.quantile(0.25)) if len(residual) else np.nan,
    }
    return pd.DataFrame([row])


def _item1_even_spaced_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    idx = np.unique(np.linspace(0, len(df) - 1, n, dtype=int))
    picked = df.iloc[idx].copy()
    if len(picked) >= n:
        return picked.iloc[:n].copy()

    missing = n - len(picked)
    remainder = df.drop(df.index[idx], errors="ignore")
    if len(remainder) == 0:
        return picked.copy()
    extra_idx = np.unique(np.linspace(0, len(remainder) - 1, missing, dtype=int))
    extra = remainder.iloc[extra_idx].copy()
    out = pd.concat([picked, extra], ignore_index=False).sort_values("time")
    return out.iloc[:n].copy()


def select_item1_low_high_f_examples(df: pd.DataFrame, n_examples: int = 5, tail_fraction: float = 0.10) -> pd.DataFrame:
    d = df.copy()
    cols = ["sample", "time", "F_MEED_TD", "GFP", "TD", "psiD", "rho", "rho2"]
    keep_cols = [col for col in cols if col in d.columns]
    d = d[keep_cols].replace([np.inf, -np.inf], np.nan)
    d = d.loc[np.isfinite(d["F_MEED_TD"].to_numpy(float))].copy()
    if len(d) == 0:
        return pd.DataFrame(columns=["group", "rank_in_group", "sample_t1", "time_t1"] + keep_cols)

    max_sample = int(np.nanmax(d["sample"].to_numpy(float))) if len(d) else -1
    d = d.loc[d["sample"].astype(int) < max_sample].copy()
    if len(d) == 0:
        return pd.DataFrame(columns=["group", "rank_in_group", "sample_t1", "time_t1"] + keep_cols)

    tail_n = max(int(np.ceil(len(d) * tail_fraction)), max(20, n_examples))
    tail_n = min(tail_n, len(d))

    low_pool = d.nsmallest(tail_n, "F_MEED_TD").sort_values("time").copy()
    high_pool = d.nlargest(tail_n, "F_MEED_TD").sort_values("time").copy()

    low_sel = _item1_even_spaced_rows(low_pool, n_examples).copy()
    high_sel = _item1_even_spaced_rows(high_pool, n_examples).copy()

    low_sel["group"] = "low_F"
    high_sel["group"] = "high_F"
    low_sel["rank_in_group"] = np.arange(1, len(low_sel) + 1, dtype=int)
    high_sel["rank_in_group"] = np.arange(1, len(high_sel) + 1, dtype=int)
    low_sel["sample_t1"] = low_sel["sample"].astype(int) + 1
    high_sel["sample_t1"] = high_sel["sample"].astype(int) + 1
    low_sel["time_t1"] = low_sel["time"].astype(float) + (1.0 / float(get_fs_from_time_column(d["time"])))
    high_sel["time_t1"] = high_sel["time"].astype(float) + (1.0 / float(get_fs_from_time_column(d["time"])))

    out_cols = ["group", "rank_in_group", "sample_t1", "time_t1"] + keep_cols
    out = pd.concat([low_sel[out_cols], high_sel[out_cols]], ignore_index=True)
    return out


def plot_item1D_low_high_f_examples(raw: mne.io.BaseRaw, sample_df: pd.DataFrame, sub: str | None = None):
    selected = select_item1_low_high_f_examples(sample_df, n_examples=5, tail_fraction=0.10)
    fig, axes = plt.subplots(2, 10, figsize=(16.2, 6.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.07, top=0.88, wspace=0.10, hspace=0.42)

    if sub:
        fig.text(0.012, 0.97, str(sub), ha="left", va="top", fontsize=10)
    fig.text(0.5, 0.935, "Figure 1D: low vs high F_MEED_TD transition pairs (t and t+1)", ha="center", va="top", fontsize=11)

    eeg = np.asarray(raw.get_data(picks="eeg"), dtype=float)
    all_vectors = []
    for row in selected.itertuples(index=False):
        sample_idx = int(row.sample)
        for offset in [0, 1]:
            idx = sample_idx + offset
            if 0 <= idx < eeg.shape[1]:
                all_vectors.append(_normalize_topomap_vector(eeg[:, idx]))
    vmax = float(np.max(np.abs(np.concatenate(all_vectors)))) if all_vectors else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vlim = (-vmax, vmax)

    row_specs = [("low_F", "Low F examples"), ("high_F", "High F examples")]
    for r, (group, row_title) in enumerate(row_specs):
        group_df = selected.loc[selected["group"] == group].copy()
        for pair_idx in range(5):
            ax_t = axes[r, 2 * pair_idx]
            ax_t1 = axes[r, 2 * pair_idx + 1]
            ax_t.set_box_aspect(1)
            ax_t1.set_box_aspect(1)
            if pair_idx == 0:
                ax_t.text(0.0, 1.08, row_title, transform=ax_t.transAxes, ha="left", va="bottom", fontsize=9.5)
            if pair_idx >= len(group_df):
                for ax in [ax_t, ax_t1]:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="0.35")
                    ax.set_axis_off()
                continue

            row = group_df.iloc[pair_idx]
            sample_idx = int(row["sample"])
            sample_t1 = sample_idx + 1
            if not (0 <= sample_idx < eeg.shape[1]) or not (0 <= sample_t1 < eeg.shape[1]):
                for ax in [ax_t, ax_t1]:
                    ax.text(0.5, 0.5, "invalid pair", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="0.35")
                    ax.set_axis_off()
                continue

            for ax, idx, label, time_value in [
                (ax_t, sample_idx, "t", float(row["time"])),
                (ax_t1, sample_t1, "t+1", float(row["time_t1"])),
            ]:
                topo = _normalize_topomap_vector(eeg[:, idx])
                mne.viz.plot_topomap(
                    topo,
                    raw.info,
                    axes=ax,
                    show=False,
                    cmap="RdBu_r",
                    vlim=vlim,
                    contours=0,
                    sensors=True,
                    res=64,
                )
                if label == "t":
                    ax.set_title(
                        f"{label}: s={idx}\nt={time_value:.2f}s\nF={float(row['F_MEED_TD']):.3f}",
                        fontsize=7.6,
                        pad=3.0,
                    )
                else:
                    ax.set_title(
                        f"{label}: s={idx}\nt={time_value:.2f}s",
                        fontsize=7.6,
                        pad=3.0,
                    )

    return fig, selected


def _subject_zscore(series: pd.Series) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    m = np.isfinite(arr)
    if not np.any(m):
        return out
    vals = arr[m]
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    if sd < 1e-12:
        out[m] = 0.0
    else:
        out[m] = (vals - mu) / sd
    return out


def _normalize_topomap_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    arr = arr - float(np.mean(arr))
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        arr = arr / norm
    return arr


def select_exemplary_f_meed_transition_pairs(df: pd.DataFrame, n_examples: int = 5) -> pd.DataFrame:
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.loc[np.isfinite(d["F_MEED_TD"].to_numpy(float))].copy()
    if len(d) == 0:
        return pd.DataFrame(columns=["group", "rank_in_group", "sample", "time", "F_MEED_TD"])

    max_sample = int(np.nanmax(d["sample"].to_numpy(float))) if len(d) else -1
    d = d.loc[d["sample"].astype(int) < max_sample].copy()
    if len(d) == 0:
        return pd.DataFrame(columns=["group", "rank_in_group", "sample", "time", "F_MEED_TD"])

    zf = _subject_zscore(d["F_MEED_TD"])
    d["z_F_MEED_TD"] = zf
    tail_n = min(len(d), max(int(np.ceil(len(d) * 0.10)), max(20, n_examples)))

    low_sel = _item1_even_spaced_rows(d.nsmallest(tail_n, "F_MEED_TD").sort_values("time"), n_examples).copy()
    avg_sel = _item1_even_spaced_rows(d.assign(abs_z=np.abs(d["z_F_MEED_TD"])).nsmallest(tail_n, "abs_z").sort_values("time"), n_examples).copy()
    high_sel = _item1_even_spaced_rows(d.nlargest(tail_n, "F_MEED_TD").sort_values("time"), n_examples).copy()

    groups = [
        ("low_F", low_sel),
        ("avg_F", avg_sel),
        ("high_F", high_sel),
    ]
    rows = []
    for group_name, gdf in groups:
        gdf = gdf.copy().reset_index(drop=True)
        for i, row in gdf.iterrows():
            rows.append({
                "group": group_name,
                "rank_in_group": int(i + 1),
                "sample": int(row["sample"]),
                "time": float(row["time"]),
                "F_MEED_TD": float(row["F_MEED_TD"]),
                "z_F_MEED_TD": float(row["z_F_MEED_TD"]) if np.isfinite(row["z_F_MEED_TD"]) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_exemplary_f_meed_transition_pairs(raw: mne.io.BaseRaw, sample_df: pd.DataFrame, sub: str | None = None):
    selected = select_exemplary_f_meed_transition_pairs(sample_df, n_examples=5)
    fig, axes = plt.subplots(3, 10, figsize=(16.0, 7.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.04, top=0.91, wspace=0.12, hspace=0.42)
    if sub:
        fig.text(0.012, 0.975, str(sub), ha="left", va="top", fontsize=10)
    fig.text(0.5, 0.95, "Exemplary F_MEED transitions: t and t+1", ha="center", va="top", fontsize=11)

    eeg = np.asarray(raw.get_data(picks="eeg"), dtype=float)
    vectors = []
    for row in selected.itertuples(index=False):
        sample_idx = int(row.sample)
        for offset in [0, 1]:
            idx = sample_idx + offset
            if 0 <= idx < eeg.shape[1]:
                vectors.append(_normalize_topomap_vector(eeg[:, idx]))
    vmax = float(np.max(np.abs(np.concatenate(vectors)))) if vectors else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vlim = (-vmax, vmax)

    row_order = [("low_F", "Low F"), ("avg_F", "Average F"), ("high_F", "Very high F")]
    for r, (group_name, row_label) in enumerate(row_order):
        group_df = selected.loc[selected["group"] == group_name].sort_values("rank_in_group").copy()
        for pair_idx in range(5):
            c0 = 2 * pair_idx
            ax_t = axes[r, c0]
            ax_tp1 = axes[r, c0 + 1]
            ax_t.set_box_aspect(1)
            ax_tp1.set_box_aspect(1)
            if pair_idx == 0:
                ax_t.text(0.0, 1.10, row_label, transform=ax_t.transAxes, ha="left", va="bottom", fontsize=9.5)
            if pair_idx >= len(group_df):
                for ax in [ax_t, ax_tp1]:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8.2, color="0.35")
                    ax.set_axis_off()
                continue

            row = group_df.iloc[pair_idx]
            sample_idx = int(row["sample"])
            idx_pair = [sample_idx, sample_idx + 1]
            axes_pair = [ax_t, ax_tp1]
            labels = ["t", "t+1"]
            for ax, idx, lbl in zip(axes_pair, idx_pair, labels):
                if not (0 <= idx < eeg.shape[1]):
                    ax.text(0.5, 0.5, "invalid", transform=ax.transAxes, ha="center", va="center", fontsize=8.2, color="0.35")
                    ax.set_axis_off()
                    continue
                topo = _normalize_topomap_vector(eeg[:, idx])
                mne.viz.plot_topomap(
                    topo,
                    raw.info,
                    axes=ax,
                    show=False,
                    cmap="RdBu_r",
                    vlim=vlim,
                    contours=0,
                    sensors=True,
                    res=64,
                )
                if lbl == "t":
                    ax.set_title(
                        f"{lbl} s={sample_idx}\nF={float(row['F_MEED_TD']):.3f}\nz={float(row['z_F_MEED_TD']):.2f}",
                        fontsize=7.2,
                        pad=2.5,
                    )
                else:
                    ax.set_title(f"{lbl} s={idx}", fontsize=7.2, pad=2.5)

    return fig


def _categorize_standardized_values(z: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    out = np.full(len(z), "avg", dtype=object)
    out[np.isfinite(z) & (z <= -threshold)] = "low"
    out[np.isfinite(z) & (z >= threshold)] = "high"
    return out


def select_exemplary_gfp_rho_maxcorr_cases(df: pd.DataFrame, n_examples: int = 5, threshold: float = 2.0) -> pd.DataFrame:
    d = _prepare_item3_ml_dataframe(df).copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.loc[np.isfinite(d["GFP"].to_numpy(float)) & np.isfinite(d["rho"].to_numpy(float)) & np.isfinite(d["winning_corr"].to_numpy(float))].copy()
    if len(d) == 0:
        return pd.DataFrame(columns=["case_label", "replicate", "sample", "time", "GFP", "rho", "winning_corr", "z_gfp", "z_rho", "z_maxcorr"])

    d["z_gfp"] = _subject_zscore(d["GFP"])
    d["z_rho"] = _subject_zscore(d["rho"])
    d["z_maxcorr"] = _subject_zscore(d["winning_corr"])
    d["gfp_case"] = _categorize_standardized_values(d["z_gfp"].to_numpy(float), threshold=threshold)
    d["rho_case"] = _categorize_standardized_values(d["z_rho"].to_numpy(float), threshold=threshold)
    d["maxcorr_case"] = _categorize_standardized_values(d["z_maxcorr"].to_numpy(float), threshold=threshold)

    rows = []
    for g_case, r_case, m_case in product(["low", "avg", "high"], repeat=3):
        case_df = d.loc[
            (d["gfp_case"] == g_case)
            & (d["rho_case"] == r_case)
            & (d["maxcorr_case"] == m_case)
        ].sort_values("time").copy()
        picked = _item1_even_spaced_rows(case_df, n_examples) if len(case_df) else case_df
        case_label = f"G={g_case} | R={r_case} | M={m_case}"
        for i, row in picked.reset_index(drop=True).iterrows():
            rows.append({
                "case_label": case_label,
                "replicate": int(i + 1),
                "sample": int(row["sample"]),
                "time": float(row["time"]),
                "GFP": float(row["GFP"]),
                "rho": float(row["rho"]),
                "winning_corr": float(row["winning_corr"]),
                "z_gfp": float(row["z_gfp"]),
                "z_rho": float(row["z_rho"]),
                "z_maxcorr": float(row["z_maxcorr"]),
            })
    return pd.DataFrame(rows)


def plot_exemplary_gfp_rho_maxcorr_cases(raw: mne.io.BaseRaw, sample_df: pd.DataFrame, sub: str | None = None):
    selected = select_exemplary_gfp_rho_maxcorr_cases(sample_df, n_examples=5, threshold=2.0)
    case_labels = [f"G={g} | R={r} | M={m}" for g, r, m in product(["low", "avg", "high"], repeat=3)]
    fig, axes = plt.subplots(len(case_labels), 5, figsize=(11.5, 42.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.01, top=0.985, wspace=0.10, hspace=0.42)
    if sub:
        fig.text(0.012, 0.995, str(sub), ha="left", va="top", fontsize=10)
    fig.text(0.5, 0.992, "Exemplary standardized GFP / rho / max-corr cases", ha="center", va="top", fontsize=11)

    eeg = np.asarray(raw.get_data(picks="eeg"), dtype=float)
    vectors = []
    for row in selected.itertuples(index=False):
        idx = int(row.sample)
        if 0 <= idx < eeg.shape[1]:
            vectors.append(eeg[:, idx])
    vmax = float(np.max(np.abs(np.concatenate(vectors)))) if vectors else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vlim = (-vmax, vmax)

    axes = np.asarray(axes)
    for r, case_label in enumerate(case_labels):
        row_df = selected.loc[selected["case_label"] == case_label].sort_values("replicate").copy()
        for c in range(5):
            ax = axes[r, c]
            ax.set_box_aspect(1)
            if c == 0:
                ax.text(-0.02, 0.5, case_label, transform=ax.transAxes, ha="right", va="center", fontsize=7.4)
            if c >= len(row_df):
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color="0.35")
                ax.set_axis_off()
                continue
            row = row_df.iloc[c]
            idx = int(row["sample"])
            if not (0 <= idx < eeg.shape[1]):
                ax.text(0.5, 0.5, "invalid", transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color="0.35")
                ax.set_axis_off()
                continue
            mne.viz.plot_topomap(
                eeg[:, idx],
                raw.info,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                vlim=vlim,
                contours=6,
                sensors=True,
                res=48,
            )
            ax.set_title(
                f"s={idx} t={float(row['time']):.1f}\nzg={float(row['z_gfp']):.1f} zr={float(row['z_rho']):.1f} zm={float(row['z_maxcorr']):.1f}",
                fontsize=6.6,
                pad=2.0,
            )

    return fig


def save_exemplary_subject_plots(sub: str, raw: mne.io.BaseRaw, sample_df: pd.DataFrame, subject_dir: str | Path):
    outdir = Path(subject_dir) / "exemplary_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    save_figure(
        plot_exemplary_f_meed_transition_pairs(raw, sample_df, sub=sub),
        outdir / "figure_exemplary_f_meed_transition_pairs.svg",
        tight=False,
    )
    save_figure(
        plot_exemplary_gfp_rho_maxcorr_cases(raw, sample_df, sub=sub),
        outdir / "figure_exemplary_27_gfp_rho_maxcorr_cases.svg",
        tight=False,
    )
    stale_27_case_path = outdir / "figure_exemplary_gfp_rho_maxcorr_cases.svg"
    if stale_27_case_path.exists():
        stale_27_case_path.unlink()


# ---------------------------------------------------------------------
# Abstract figure components
# ---------------------------------------------------------------------


def _finite_pair_from_arrays(x: np.ndarray, y: np.ndarray, positive: bool = False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if positive:
        m &= (x > 0) & (y > 0)
    return x[m], y[m]


def plot_abstract_subject_component(df: pd.DataFrame, sub: str | None = None):
    d, _ = compute_item1_gfp_link_residuals(df)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.24, top=0.87, wspace=0.34)

    if sub:
        fig.text(0.012, 0.965, str(sub), ha="left", va="top", fontsize=9)

    specs = [
        ("loglog", "GFP_uV", "TD", "GFP (uV)", "TD"),
        ("loglog", "GFP_uV", "psiD", "GFP (uV)", "psiD (deg)"),
        ("loglog", "TD", "psiD", "TD", "psiD (deg)"),
        ("linear", "log_F_MEED_TD", "res_logGFP_psiD_minus_TD", "log F_MEED_TD", "log-GFP residual"),
    ]

    hexbins = []
    for ax, (scale_kind, x, y, xlabel, ylabel) in zip(axes, specs):
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        if scale_kind == "loglog":
            hb = _plot_item1_loglog_panel(ax, d, x, y, xlabel, ylabel, annotation_mode="r_only")
        else:
            hb = _plot_item1_linear_panel(ax, d, x, y, xlabel, ylabel, add_zero_line=True, annotation_mode="r_only")
        hexbins.append(hb)

    _set_hexbin_average_clim(hexbins)

    return fig


def _compute_abstract_group_metrics(sample_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sub, df in sorted(sample_dfs.items()):
        d, _ = compute_item1_gfp_link_residuals(df)
        x = d["log_F_MEED_TD"].to_numpy(float)
        y = d["res_logGFP_psiD_minus_TD"].to_numpy(float)
        fit = _fit_linear_model(x, y)

        m = np.isfinite(x) & np.isfinite(y)
        xv = x[m]
        yv = y[m]
        if len(xv):
            q1 = float(np.nanpercentile(xv, 25))
            q3 = float(np.nanpercentile(xv, 75))
            low_mask = xv <= q1
            high_mask = xv >= q3
            median_residual_low = float(np.nanmedian(yv[low_mask])) if np.any(low_mask) else np.nan
            median_residual_high = float(np.nanmedian(yv[high_mask])) if np.any(high_mask) else np.nan
            median_log_F_low = float(np.nanmedian(xv[low_mask])) if np.any(low_mask) else np.nan
            median_log_F_high = float(np.nanmedian(xv[high_mask])) if np.any(high_mask) else np.nan
        else:
            median_residual_low = np.nan
            median_residual_high = np.nan
            median_log_F_low = np.nan
            median_log_F_high = np.nan

        rows.append({
            "sub": sub,
            "n_valid": int(fit["n_valid"]),
            "slope_residual_vs_log_F_MEED_TD": float(fit["slope"]),
            "intercept_residual_vs_log_F_MEED_TD": float(fit["intercept"]),
            "r_residual_vs_log_F_MEED_TD": float(fit["r"]),
            "R2_residual_vs_log_F_MEED_TD": float(fit["R2"]),
            "median_residual_low_F": median_residual_low,
            "median_residual_high_F": median_residual_high,
            "delta_high_minus_low": float(median_residual_high - median_residual_low) if np.isfinite(median_residual_low) and np.isfinite(median_residual_high) else np.nan,
            "median_log_F_low": median_log_F_low,
            "median_log_F_high": median_log_F_high,
        })

    cols = [
        "sub",
        "n_valid",
        "slope_residual_vs_log_F_MEED_TD",
        "intercept_residual_vs_log_F_MEED_TD",
        "r_residual_vs_log_F_MEED_TD",
        "R2_residual_vs_log_F_MEED_TD",
        "median_residual_low_F",
        "median_residual_high_F",
        "delta_high_minus_low",
        "median_log_F_low",
        "median_log_F_high",
    ]
    return pd.DataFrame(rows, columns=cols)


def _expanded_limits(values, *, include_zero: bool = False, pad_frac: float = 0.08):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if include_zero:
        arr = np.append(arr, 0.0)
    if len(arr) == 0:
        return (-1.0, 1.0)

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if np.isclose(vmin, vmax):
        half = max(abs(vmin) * 0.15, 0.1)
        return (vmin - half, vmax + half)

    pad = (vmax - vmin) * pad_frac
    return (vmin - pad, vmax + pad)


def plot_abstract_group_component(sample_dfs: dict[str, pd.DataFrame]):
    metrics = _compute_abstract_group_metrics(sample_dfs)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.24, top=0.83, wspace=0.50)

    all_x = []
    all_residuals = []
    subject_lines = []
    paired_points = []

    for sub, df in sorted(sample_dfs.items()):
        d, _ = compute_item1_gfp_link_residuals(df)
        x = d["log_F_MEED_TD"].to_numpy(float)
        y = d["res_logGFP_psiD_minus_TD"].to_numpy(float)
        fit = _fit_linear_model(x, y)
        m = np.isfinite(x) & np.isfinite(y)
        xv = x[m]
        yv = y[m]
        if len(xv):
            all_x.append(xv)
            all_residuals.append(yv)
        if np.isfinite(fit["slope"]):
            subject_lines.append({
                "sub": sub,
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "x_min": fit["x_min"],
                "x_max": fit["x_max"],
            })

    paired_y = metrics[["median_residual_low_F", "median_residual_high_F"]].to_numpy(float).ravel()
    pooled_residuals = np.concatenate(all_residuals) if all_residuals else np.array([], dtype=float)
    y_limits = _expanded_limits(np.concatenate([pooled_residuals, paired_y[np.isfinite(paired_y)]]) if len(pooled_residuals) or np.any(np.isfinite(paired_y)) else np.array([]), include_zero=True)

    ax = axes[0]
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("log F_MEED_TD", fontsize=8.5)
    ax.set_ylabel("log-GFP residual", fontsize=8.5)
    ax.set_title("Subject slopes", fontsize=9)
    ax.axhline(0.0, color="0.25", lw=1.0, linestyle="--")
    ax.set_ylim(*y_limits)

    if all_x and subject_lines:
        pooled_x = np.concatenate(all_x)
        x_limits = _expanded_limits(pooled_x)
        ax.set_xlim(*x_limits)
        x_grid = np.linspace(float(np.min(pooled_x)), float(np.max(pooled_x)), 200)
        y_grid_stack = []
        for entry in subject_lines:
            x_mask = (x_grid >= entry["x_min"]) & (x_grid <= entry["x_max"])
            if not np.any(x_mask):
                continue
            y_line = np.full_like(x_grid, np.nan, dtype=float)
            y_line[x_mask] = entry["intercept"] + entry["slope"] * x_grid[x_mask]
            y_grid_stack.append(y_line)
            ax.plot(x_grid[x_mask], y_line[x_mask], color="tab:blue", lw=0.8, alpha=0.35)

        if y_grid_stack:
            line_stack = np.vstack(y_grid_stack)
            median_line = np.full(line_stack.shape[1], np.nan, dtype=float)
            valid_cols = np.any(np.isfinite(line_stack), axis=0)
            median_line[valid_cols] = np.nanmedian(line_stack[:, valid_cols], axis=0)
            ax.plot(x_grid, median_line, color="black", lw=2.0)
    else:
        ax.text(0.5, 0.5, "no valid subject fits", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.35")

    n_subjects = int(len(metrics))
    negative_slopes = int(np.sum(metrics["slope_residual_vs_log_F_MEED_TD"].to_numpy(float) < 0))
    median_slope = float(np.nanmedian(metrics["slope_residual_vs_log_F_MEED_TD"].to_numpy(float))) if n_subjects else np.nan
    ax.text(
        0.03,
        0.97,
        f"n = {n_subjects}\n{negative_slopes}/{n_subjects} negative\nmedian slope = {median_slope:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.8},
    )

    ax = axes[1]
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=7.5)
    ax.set_title("Low vs high MEED fraction", fontsize=9)
    ax.set_ylabel("median log-GFP residual", fontsize=8.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["low F", "high F"])
    ax.axhline(0.0, color="0.25", lw=1.0, linestyle="--")
    ax.set_ylim(*y_limits)

    for row in metrics.itertuples(index=False):
        if not np.isfinite(row.median_residual_low_F) or not np.isfinite(row.median_residual_high_F):
            continue
        ys = [float(row.median_residual_low_F), float(row.median_residual_high_F)]
        paired_points.extend(ys)
        ax.plot([0, 1], ys, color="tab:blue", lw=0.8, alpha=0.35)
        ax.scatter([0, 1], ys, color="tab:blue", s=10, alpha=0.55)

    if not paired_points:
        ax.text(0.5, 0.5, "no paired summaries", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.35")

    return fig


def save_abstract_group_outputs(processed_outputs, outdir: str | Path):
    outdir = Path(outdir)
    group_dir = outdir / "group_abstract_figure"
    group_dir.mkdir(parents=True, exist_ok=True)

    records = processed_outputs.to_dict("records") if isinstance(processed_outputs, pd.DataFrame) else list(processed_outputs)
    sample_dfs = {}
    for row in records:
        if "sub" not in row or "analysis_csv" not in row or not row["analysis_csv"]:
            continue
        csv_path = Path(row["analysis_csv"])
        if not csv_path.exists():
            continue
        sample_dfs[str(row["sub"])] = pd.read_csv(csv_path)

    if not sample_dfs:
        return None

    metrics = _compute_abstract_group_metrics(sample_dfs)
    save_table(metrics, group_dir / "group_abstract_item1_metrics.csv")
    save_figure(plot_abstract_group_component(sample_dfs), group_dir / "figure_abstract_group_component.svg", tight=False)
    stale_png = group_dir / "figure_abstract_group_component.png"
    if stale_png.exists():
        stale_png.unlink()


def plot_group_map_fit_r2_summary(map_fit_df: pd.DataFrame):
    d = map_fit_df.copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["sensor_map_R2", "k", "label"])
    ks = sorted(d["k"].astype(int).unique())
    if not ks:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "no fit R^2 data", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="0.35")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, len(ks), figsize=(4.8 * len(ks), 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, ks):
        dk = d.loc[d["k"].astype(int) == int(k)].copy()
        labels = sorted(dk["label"].astype(str).unique())
        data = [dk.loc[dk["label"].astype(str) == lab, "sensor_map_R2"].dropna().to_numpy(float) for lab in labels]
        valid_data = [arr if len(arr) else np.array([np.nan]) for arr in data]
        ax.boxplot(valid_data, labels=labels, showfliers=False)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("label")
        ax.set_ylabel("sensor-map R^2")
        ax.set_title(f"k={k}")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Group map-fit R^2 by label", fontsize=14)
    return fig


def save_group_map_fit_r2_outputs(processed_outputs, outdir: str | Path):
    outdir = Path(outdir)
    group_dir = outdir / "group_map_fit"
    group_dir.mkdir(parents=True, exist_ok=True)

    records = processed_outputs.to_dict("records") if isinstance(processed_outputs, pd.DataFrame) else list(processed_outputs)
    dfs = []
    for row in records:
        map_fit_csv = row.get("map_fit_csv")
        if not map_fit_csv:
            continue
        csv_path = Path(map_fit_csv)
        if not csv_path.exists():
            continue
        dfs.append(pd.read_csv(csv_path))

    if not dfs:
        return None

    group_df = pd.concat(dfs, ignore_index=True)
    save_table(group_df, group_dir / "group_map_fit_r2.csv")
    save_figure(plot_group_map_fit_r2_summary(group_df), group_dir / "figure_group_map_fit_r2.png")
    return group_df


# ---------------------------------------------------------------------
# Item 2 figures and table
# ---------------------------------------------------------------------


def overlay_template_centers(ax, meta_level: TemplateLevel, subject_level: TemplateLevel | None = None, k: int | None = None):
    for level, marker, color, prefix in [(meta_level, "o", "black", "meta"), (subject_level, "^", "tab:red", "sub")]:
        if level is None:
            continue
        ax.scatter(level.D[0], level.D[1], marker=marker, s=45, color=color, edgecolor="white", linewidth=0.5, zorder=5)
        for j, lab in enumerate(level.labels):
            label = f"{lab}_{k}" if k is not None else str(lab)
            label = f"{prefix}:{label}"
            ax.text(
                level.D[0, j],
                level.D[1, j],
                label,
                fontsize=7,
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1.5},
                zorder=6,
            )


def plot_item2A_template_geometry(levels_by_k: dict):
    ks = sorted(levels_by_k)
    ncols = 3
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.5 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax, k in zip(axes, ks):
        levels = levels_by_k[k]
        overlay_template_centers(ax, levels["meta"], levels["subject"], k=k)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.set_title(f"k={k}")
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("phi (deg)")
    for ax in axes[len(ks):]:
        ax.axis("off")
    fig.suptitle("Figure 2A: template geometry across model orders", fontsize=14)
    return fig


def plot_theta_phi_stat_grid(df: pd.DataFrame, meta_level: TemplateLevel, subject_level: TemplateLevel, specs: list[tuple[str, str, callable]], title: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax, (col, ttl, reducer) in zip(axes, specs):
        d = df[["theta", "phi", col]].dropna()
        hb = ax.hexbin(d["theta"], d["phi"], C=d[col], gridsize=45, reduce_C_function=reducer, mincnt=2, cmap="viridis")
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("phi (deg)")
        ax.set_title(ttl)
        overlay_template_centers(ax, meta_level, subject_level, k=None)
        fig.colorbar(hb, ax=ax, shrink=0.85)
    fig.suptitle(title, fontsize=14)
    return fig


def plot_item2B_gfp_theta_phi(df: pd.DataFrame, meta_level: TemplateLevel, subject_level: TemplateLevel):
    specs = [
        ("GFP", "mean GFP", np.nanmean),
        ("GFP", "median GFP", np.nanmedian),
        ("GFP", "std GFP", np.nanstd),
        ("GFP", "IQR GFP", lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
    ]
    return plot_theta_phi_stat_grid(df, meta_level, subject_level, specs, "Figure 2B: GFP distribution in MEED space")


def plot_item2C_meed_theta_phi(df: pd.DataFrame, meta_level: TemplateLevel, subject_level: TemplateLevel):
    specs = [
        ("rho2", "mean rho2", np.nanmean),
        ("TD", "mean TD", np.nanmean),
        ("TD_MEED", "mean TD_MEED", np.nanmean),
        ("F_MEED_TD", "mean F_MEED_TD", np.nanmean),
    ]
    return plot_theta_phi_stat_grid(df, meta_level, subject_level, specs, "Figure 2C: MEED field-conformity in MEED space")


def table2_template_bin_summary(df: pd.DataFrame, meta_level: TemplateLevel, subject_level: TemplateLevel, sub: str) -> pd.DataFrame:
    rows = []
    for level_name, level in [("meta", meta_level), ("sub", subject_level)]:
        labels = level.labels
        psi_cols = [f"psi_{level_name}_{lab}" for lab in labels]
        D = df[psi_cols].to_numpy(float)
        nearest = np.nanargmin(D, axis=1)
        for j, lab in enumerate(labels):
            m = nearest == j
            rows.append({
                "sub": sub,
                "template_level": level_name,
                "label": lab,
                "theta": float(level.D[0, j]),
                "phi": float(level.D[1, j]),
                "rho2": float(np.linalg.norm(level.Q[:, j]) ** 2),
                "median_GFP_nearest": float(df.loc[m, "GFP"].median()),
                "median_TD_nearest": float(df.loc[m, "TD"].median()),
                "median_F_MEED_TD_nearest": float(df.loc[m, "F_MEED_TD"].median()),
                "median_distance_to_template": float(df.loc[m, psi_cols[j]].median()),
                "n_nearest_samples": int(np.sum(m)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Item 3: ML-style predictability and uncertainty analysis
# ---------------------------------------------------------------------


def _item3_ml_core_target_specs():
    return [
        ("winning_corr", "Winning corr"),
    ]


def _item3_ml_main_predictor_specs():
    return [
        ("GFP", ["GFP"]),
        ("rho", ["rho"]),
        ("GFP+rho", ["GFP", "rho"]),
    ]


def _item3_ml_exploratory_predictor_specs():
    return [
        ("GFP+rho+d_xyz", ["GFP", "rho", "d_x", "d_y", "d_z"]),
    ]


def _item3_ml_model_specs():
    return [
        ("linear", "OLS"),
    ]


def _item3_ml_metrics_columns():
    return [
        "analysis_scope",
        "sub",
        "held_out_sub",
        "target",
        "target_label",
        "predictor_set",
        "model_family",
        "model_label",
        "n_fit",
        "n_eval",
        "r",
        "R2",
        "RMSE",
        "fit_success",
    ]


def _item3_ml_target_summary_columns():
    return [
        "sub",
        "target",
        "target_label",
        "n_valid",
        "mean",
        "median",
        "std",
        "p25",
        "p75",
        "min",
        "max",
    ]


ITEM3_ML_MAX_ROWS_PER_SUBJECT = 1500
ITEM3_SOFTMAX_LAMBDAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def _clip01(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 0.0, 1.0)


def _softmax_probs(scores: np.ndarray, lam: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = lam * (scores - np.nanmax(scores, axis=1, keepdims=True))
    e = np.exp(shifted)
    denom = np.maximum(np.nansum(e, axis=1, keepdims=True), 1e-12)
    return e / denom


def _lambda_to_tag(lam: float) -> str:
    return f"{float(lam):g}".replace(".", "p")


def _tag_to_lambda(tag: str) -> float:
    return float(str(tag).replace("p", "."))


def _softmax_target_specs(lambdas=ITEM3_SOFTMAX_LAMBDAS):
    specs = []
    for lam in lambdas:
        tag = _lambda_to_tag(lam)
        specs.append((f"softmax_confidence_lambda_{tag}", f"Softmax confidence lambda={float(lam):g}"))
        specs.append((f"softmax_entropy_lambda_{tag}", f"Softmax entropy lambda={float(lam):g}"))
    return specs


def _prepare_item3_ml_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["GFP_uV"] = d["GFP"] * 1e6

    meta_corr_cols = sorted(col for col in d.columns if col.startswith("corr_meta_"))
    psi_meta_cols = sorted(col for col in d.columns if col.startswith("psi_meta_"))

    n = len(d)
    d["winning_corr"] = np.nan
    d["second_max_meta_corr"] = np.nan
    d["corr_margin"] = np.nan
    d["relative_corr_margin"] = np.nan
    d["corr_entropy"] = np.nan
    d["dipole_angle_gap"] = np.nan

    if meta_corr_cols:
        C = np.abs(d[meta_corr_cols].to_numpy(float))
        C_sorted = np.sort(C, axis=1)[:, ::-1]
        winning = C_sorted[:, 0]
        second = C_sorted[:, 1] if C.shape[1] > 1 else np.full(n, np.nan)
        margin = winning - second
        relative_margin = np.divide(
            margin,
            np.maximum(winning, 1e-12),
            out=np.full(n, np.nan),
            where=np.isfinite(winning),
        )
        C2 = C ** 2
        p = C2 / np.maximum(C2.sum(axis=1, keepdims=True), 1e-12)
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1) / np.log(max(C.shape[1], 2))

        d["winning_corr"] = _clip01(winning)
        d["max_meta_corr"] = d["winning_corr"]
        d["second_max_meta_corr"] = _clip01(second)
        d["corr_margin"] = _clip01(margin)
        d["relative_corr_margin"] = _clip01(relative_margin)
        d["corr_entropy"] = _clip01(entropy)

    if psi_meta_cols:
        P = d[psi_meta_cols].to_numpy(float)
        P_sorted = np.sort(P, axis=1)
        if P.shape[1] > 1:
            gap = (P_sorted[:, 1] - P_sorted[:, 0]) / 180.0
            d["dipole_angle_gap"] = _clip01(gap)

    return d


def _add_item3_softmax_targets(df: pd.DataFrame, lambdas=ITEM3_SOFTMAX_LAMBDAS):
    d = df.copy()
    meta_corr_cols = sorted(col for col in d.columns if col.startswith("corr_meta_"))
    target_specs = []
    if not meta_corr_cols:
        return d, target_specs

    C = np.abs(d[meta_corr_cols].to_numpy(float))
    k = C.shape[1]

    for lam in lambdas:
        tag = _lambda_to_tag(lam)
        probs = _softmax_probs(C, float(lam))

        conf_col = f"softmax_confidence_lambda_{tag}"
        ent_col = f"softmax_entropy_lambda_{tag}"

        d[conf_col] = _clip01(np.nanmax(probs, axis=1))
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1) / np.log(max(k, 2))
        d[ent_col] = _clip01(entropy)
        target_specs.append((conf_col, f"Softmax confidence lambda={float(lam):g}"))
        target_specs.append((ent_col, f"Softmax entropy lambda={float(lam):g}"))

    return d, target_specs


def _parse_softmax_target_name(target: str) -> tuple[str, float] | None:
    target = str(target)
    if target.startswith("softmax_confidence_lambda_"):
        tag = target.replace("softmax_confidence_lambda_", "")
        return "confidence", _tag_to_lambda(tag)
    if target.startswith("softmax_entropy_lambda_"):
        tag = target.replace("softmax_entropy_lambda_", "")
        return "entropy", _tag_to_lambda(tag)
    return None


def _item3_target_summary(df: pd.DataFrame, sub: str) -> pd.DataFrame:
    d = _prepare_item3_ml_dataframe(df)
    rows = []
    for target, label in _item3_ml_core_target_specs():
        vals = d[target].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            rows.append({
                "sub": sub,
                "target": target,
                "target_label": label,
                "n_valid": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "min": np.nan,
                "max": np.nan,
            })
            continue
        rows.append({
            "sub": sub,
            "target": target,
            "target_label": label,
            "n_valid": int(len(vals)),
            "mean": float(np.nanmean(vals)),
            "median": float(np.nanmedian(vals)),
            "std": float(np.nanstd(vals)),
            "p25": float(np.nanpercentile(vals, 25)),
            "p75": float(np.nanpercentile(vals, 75)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        })
    return pd.DataFrame(rows, columns=_item3_ml_target_summary_columns())


def _item3_ml_analysis_subset(df: pd.DataFrame, max_rows: int = ITEM3_ML_MAX_ROWS_PER_SUBJECT) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    idx = _item3_even_subsample_idx(len(df), max_rows)
    return df.iloc[idx].copy()


def _item3_even_subsample_idx(n_rows: int, max_rows: int | None):
    if max_rows is None or n_rows <= max_rows:
        return np.arange(n_rows, dtype=int)
    return np.unique(np.linspace(0, n_rows - 1, int(max_rows), dtype=int))


def _item3_standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    sigma[~np.isfinite(sigma) | (sigma < 1e-12)] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def _item3_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    out = {"n_eval": int(len(y_true)), "r": np.nan, "R2": np.nan, "RMSE": np.nan}
    if len(y_true) == 0:
        return out

    out["RMSE"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if len(y_true) >= 2 and (not np.allclose(y_true, y_true[0])) and (not np.allclose(y_pred, y_pred[0])):
        out["r"] = float(pearsonr(y_true, y_pred)[0])
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    out["R2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return out


def _item3_fit_linear_model(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    X1 = np.column_stack([np.ones(len(X_fit)), X_fit])
    beta, *_ = np.linalg.lstsq(X1, y_fit, rcond=None)
    return {
        "success": True,
        "intercept": float(beta[0]),
        "coef": np.asarray(beta[1:], dtype=float),
    }


def _item3_fit_sigmoid_model(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    p = X_fit.shape[1]
    mean_y = float(np.clip(np.nanmean(y_fit), 1e-4, 1.0 - 1e-4))
    init = np.zeros(p + 1, dtype=float)
    init[0] = np.log(mean_y / (1.0 - mean_y))

    def residuals(params):
        return expit(params[0] + X_fit @ params[1:]) - y_fit

    try:
        res = least_squares(residuals, init, max_nfev=150)
        return {
            "success": bool(res.success),
            "intercept": float(res.x[0]),
            "coef": np.asarray(res.x[1:], dtype=float),
        }
    except Exception:
        return {
            "success": False,
            "intercept": np.nan,
            "coef": np.full(p, np.nan),
        }


def _item3_predict_from_fit(model_family: str, fit: dict, X_eval: np.ndarray) -> np.ndarray:
    intercept = float(fit["intercept"])
    coef = np.asarray(fit["coef"], dtype=float)
    linear_part = intercept + X_eval @ coef
    if model_family == "sigmoid":
        return expit(linear_part)
    return linear_part


def _item3_fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str,
    predictors: list[str],
    model_family: str,
    max_fit_rows: int = 1500,
):
    cols = [target] + predictors
    d_train = train_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    d_test = test_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    min_rows = max(len(predictors) + 3, 10)
    if len(d_train) < min_rows or len(d_test) < 5:
        return {
            "n_fit": int(len(d_train)),
            "n_eval": int(len(d_test)),
            "r": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "fit_success": False,
            "y_true": np.array([], dtype=float),
            "y_pred": np.array([], dtype=float),
        }

    X_train = d_train[predictors].to_numpy(float)
    y_train = d_train[target].to_numpy(float)
    X_test = d_test[predictors].to_numpy(float)
    y_test = d_test[target].to_numpy(float)
    X_train_z, X_test_z = _item3_standardize_train_test(X_train, X_test)

    fit_idx = _item3_even_subsample_idx(len(X_train_z), max_fit_rows)
    X_fit = X_train_z[fit_idx]
    y_fit = y_train[fit_idx]

    if model_family == "sigmoid":
        fit = _item3_fit_sigmoid_model(X_fit, y_fit)
    else:
        fit = _item3_fit_linear_model(X_fit, y_fit)

    if not fit["success"] or not np.all(np.isfinite(fit["coef"])) or not np.isfinite(fit["intercept"]):
        return {
            "n_fit": int(len(X_fit)),
            "n_eval": int(len(y_test)),
            "r": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "fit_success": False,
            "y_true": y_test,
            "y_pred": np.full_like(y_test, np.nan, dtype=float),
        }

    y_pred = _item3_predict_from_fit(model_family, fit, X_test_z)
    metrics = _item3_regression_metrics(y_test, y_pred)
    return {
        "n_fit": int(len(X_fit)),
        "n_eval": int(metrics["n_eval"]),
        "r": metrics["r"],
        "R2": metrics["R2"],
        "RMSE": metrics["RMSE"],
        "fit_success": bool(fit["success"]),
        "y_true": y_test,
        "y_pred": y_pred,
    }


def _empty_item3_metrics_df():
    return pd.DataFrame(columns=_item3_ml_metrics_columns())


def _run_item3_within_subject_models(
    sample_dfs: dict[str, pd.DataFrame],
    *,
    target_specs,
    predictor_specs,
    analysis_scope: str,
):
    rows = []
    for sub, df in sorted(sample_dfs.items()):
        d = _item3_ml_analysis_subset(_prepare_item3_ml_dataframe(df))
        for target, label in target_specs:
            for predictor_set, predictors in predictor_specs:
                for model_family, model_label in _item3_ml_model_specs():
                    res = _item3_fit_and_score(
                        d,
                        d,
                        target=target,
                        predictors=predictors,
                        model_family=model_family,
                    )
                    rows.append({
                        "analysis_scope": analysis_scope,
                        "sub": sub,
                        "held_out_sub": None,
                        "target": target,
                        "target_label": label,
                        "predictor_set": predictor_set,
                        "model_family": model_family,
                        "model_label": model_label,
                        "n_fit": int(res["n_fit"]),
                        "n_eval": int(res["n_eval"]),
                        "r": res["r"],
                        "R2": res["R2"],
                        "RMSE": res["RMSE"],
                        "fit_success": bool(res["fit_success"]),
                    })
    return pd.DataFrame(rows, columns=_item3_ml_metrics_columns()) if rows else _empty_item3_metrics_df()


def _run_item3_loso_models(
    sample_dfs: dict[str, pd.DataFrame],
    *,
    target_specs,
    predictor_specs,
    analysis_scope: str,
):
    subs = sorted(sample_dfs)
    if len(subs) < 2:
        return _empty_item3_metrics_df()

    prepared = {sub: _prepare_item3_ml_dataframe(df) for sub, df in sample_dfs.items()}
    prepared = {sub: _item3_ml_analysis_subset(df) for sub, df in prepared.items()}
    fold_rows = []
    pooled_predictions = {}
    for held_out_sub in subs:
        train_parts = [prepared[sub] for sub in subs if sub != held_out_sub]
        if not train_parts:
            continue
        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = prepared[held_out_sub]

        for target, label in target_specs:
            for predictor_set, predictors in predictor_specs:
                for model_family, model_label in _item3_ml_model_specs():
                    res = _item3_fit_and_score(
                        train_df,
                        test_df,
                        target=target,
                        predictors=predictors,
                        model_family=model_family,
                    )
                    fold_rows.append({
                        "analysis_scope": f"{analysis_scope}_fold",
                        "sub": None,
                        "held_out_sub": held_out_sub,
                        "target": target,
                        "target_label": label,
                        "predictor_set": predictor_set,
                        "model_family": model_family,
                        "model_label": model_label,
                        "n_fit": int(res["n_fit"]),
                        "n_eval": int(res["n_eval"]),
                        "r": res["r"],
                        "R2": res["R2"],
                        "RMSE": res["RMSE"],
                        "fit_success": bool(res["fit_success"]),
                    })
                    key = (target, label, predictor_set, model_family, model_label)
                    pooled_predictions.setdefault(key, {"y_true": [], "y_pred": []})
                    if len(res["y_true"]) and len(res["y_pred"]):
                        pooled_predictions[key]["y_true"].append(np.asarray(res["y_true"], dtype=float))
                        pooled_predictions[key]["y_pred"].append(np.asarray(res["y_pred"], dtype=float))

    pooled_rows = []
    for (target, label, predictor_set, model_family, model_label), vals in pooled_predictions.items():
        if vals["y_true"] and vals["y_pred"]:
            y_true = np.concatenate(vals["y_true"])
            y_pred = np.concatenate(vals["y_pred"])
            metrics = _item3_regression_metrics(y_true, y_pred)
            n_eval = metrics["n_eval"]
        else:
            metrics = {"r": np.nan, "R2": np.nan, "RMSE": np.nan, "n_eval": 0}
            n_eval = 0
        pooled_rows.append({
            "analysis_scope": f"{analysis_scope}_pooled",
            "sub": None,
            "held_out_sub": "ALL",
            "target": target,
            "target_label": label,
            "predictor_set": predictor_set,
            "model_family": model_family,
            "model_label": model_label,
            "n_fit": np.nan,
            "n_eval": int(n_eval),
            "r": metrics["r"],
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "fit_success": True,
        })

    rows = fold_rows + pooled_rows
    return pd.DataFrame(rows, columns=_item3_ml_metrics_columns()) if rows else _empty_item3_metrics_df()


def _run_item3_softmax_lambda_sweep_models(sample_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    prepared = {}
    target_specs = _softmax_target_specs(ITEM3_SOFTMAX_LAMBDAS)

    for sub, df in sorted(sample_dfs.items()):
        d = _item3_ml_analysis_subset(_prepare_item3_ml_dataframe(df))
        d, _ = _add_item3_softmax_targets(d, lambdas=ITEM3_SOFTMAX_LAMBDAS)
        prepared[sub] = d

    out = _run_item3_loso_models(
        prepared,
        target_specs=target_specs,
        predictor_specs=_item3_ml_main_predictor_specs(),
        analysis_scope="loso_softmax_lambda_sweep",
    )

    if out.empty:
        return out

    parsed = out["target"].map(_parse_softmax_target_name)
    out["softmax_quantity"] = parsed.map(lambda x: x[0] if x is not None else None)
    out["lambda"] = parsed.map(lambda x: x[1] if x is not None else np.nan)
    return out


def _choose_item3_exemplar_confidence_lambda(softmax_loso_df: pd.DataFrame) -> float:
    d = softmax_loso_df.copy()
    d = d.loc[
        d["analysis_scope"].astype(str).str.endswith("_fold")
        & (d["model_family"] == "linear")
        & (d["predictor_set"] == "GFP+rho")
        & (d["softmax_quantity"] == "confidence")
        & np.isfinite(d["lambda"])
        & np.isfinite(d["R2"])
    ].copy()
    if d.empty:
        return float(ITEM3_SOFTMAX_LAMBDAS[0])

    summary = d.groupby("lambda", as_index=False).agg(mean_R2=("R2", "mean"))
    summary = summary.sort_values(["mean_R2", "lambda"], ascending=[False, True]).reset_index(drop=True)
    return float(summary.loc[0, "lambda"])


def _item3_prediction_rows(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str,
    predictors: list[str],
    model_family: str,
    obs_col: str,
    pred_col: str,
) -> pd.DataFrame:
    carry_cols = [col for col in ["sub", "sample", "time"] if col in test_df.columns]
    train_cols = [target] + predictors
    test_cols = carry_cols + train_cols

    d_train = train_df[train_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d_test = test_df[test_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    min_rows = max(len(predictors) + 3, 10)
    if len(d_train) < min_rows or len(d_test) < 5:
        return pd.DataFrame(columns=carry_cols + [obs_col, pred_col])

    X_train = d_train[predictors].to_numpy(float)
    y_train = d_train[target].to_numpy(float)
    X_test = d_test[predictors].to_numpy(float)
    y_test = d_test[target].to_numpy(float)
    X_train_z, X_test_z = _item3_standardize_train_test(X_train, X_test)

    fit_idx = _item3_even_subsample_idx(len(X_train_z), 1500)
    X_fit = X_train_z[fit_idx]
    y_fit = y_train[fit_idx]

    if model_family == "sigmoid":
        fit = _item3_fit_sigmoid_model(X_fit, y_fit)
    else:
        fit = _item3_fit_linear_model(X_fit, y_fit)

    if not fit["success"] or not np.all(np.isfinite(fit["coef"])) or not np.isfinite(fit["intercept"]):
        return pd.DataFrame(columns=carry_cols + [obs_col, pred_col])

    y_pred = _item3_predict_from_fit(model_family, fit, X_test_z)
    out = d_test[carry_cols].copy()
    out[obs_col] = y_test
    out[pred_col] = y_pred
    return out


def _run_item3_loso_sample_predictions(
    sample_dfs: dict[str, pd.DataFrame],
    *,
    confidence_lambda: float,
    predictor_set: str = "GFP+rho",
    model_family: str = "linear",
) -> pd.DataFrame:
    predictor_map = {name: predictors for name, predictors in _item3_ml_main_predictor_specs()}
    if predictor_set not in predictor_map:
        raise KeyError(f"Unknown predictor_set={predictor_set!r}")

    predictors = predictor_map[predictor_set]
    confidence_target = f"softmax_confidence_lambda_{_lambda_to_tag(confidence_lambda)}"
    prepared = {}
    for sub, df in sorted(sample_dfs.items()):
        d = _item3_ml_analysis_subset(_prepare_item3_ml_dataframe(df))
        d, _ = _add_item3_softmax_targets(d, lambdas=(confidence_lambda,))
        prepared[sub] = d

    rows = []
    subs = sorted(prepared)
    for held_out_sub in subs:
        train_parts = [prepared[sub] for sub in subs if sub != held_out_sub]
        if not train_parts:
            continue
        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = prepared[held_out_sub]

        winning_df = _item3_prediction_rows(
            train_df,
            test_df,
            target="winning_corr",
            predictors=predictors,
            model_family=model_family,
            obs_col="winning_corr_obs",
            pred_col="winning_corr_pred",
        )
        confidence_df = _item3_prediction_rows(
            train_df,
            test_df,
            target=confidence_target,
            predictors=predictors,
            model_family=model_family,
            obs_col="softmax_confidence_obs",
            pred_col="softmax_confidence_pred",
        )
        if winning_df.empty or confidence_df.empty:
            continue

        merged = winning_df.merge(confidence_df, on=["sub", "sample", "time"], how="inner")
        if merged.empty:
            continue

        merged["winning_corr_abs_error"] = np.abs(
            merged["winning_corr_obs"].to_numpy(float) - merged["winning_corr_pred"].to_numpy(float)
        )
        merged["softmax_confidence_abs_error"] = np.abs(
            merged["softmax_confidence_obs"].to_numpy(float) - merged["softmax_confidence_pred"].to_numpy(float)
        )
        merged["predictor_set"] = predictor_set
        merged["model_family"] = model_family
        merged["held_out_sub"] = held_out_sub
        merged["softmax_lambda"] = float(confidence_lambda)
        rows.append(merged)

    if not rows:
        cols = [
            "sub",
            "sample",
            "time",
            "winning_corr_obs",
            "winning_corr_pred",
            "winning_corr_abs_error",
            "softmax_confidence_obs",
            "softmax_confidence_pred",
            "softmax_confidence_abs_error",
            "predictor_set",
            "model_family",
            "held_out_sub",
            "softmax_lambda",
        ]
        return pd.DataFrame(columns=cols)

    return pd.concat(rows, ignore_index=True)


def _item3_zscore(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    m = np.isfinite(arr)
    if not np.any(m):
        return out
    vals = arr[m]
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    if sd < 1e-12:
        out[m] = 0.0
    else:
        out[m] = (vals - mu) / sd
    return out


def select_group_item3_exemplar_samples(
    sample_predictions_df: pd.DataFrame,
    *,
    processed_outputs=None,
    n_per_category: int = 5,
) -> pd.DataFrame:
    d = sample_predictions_df.copy()
    if d.empty:
        cols = [
            "category",
            "panel_index",
            "sub",
            "sample",
            "time",
            "winning_corr_obs",
            "winning_corr_pred",
            "winning_corr_abs_error",
            "softmax_confidence_obs",
            "softmax_confidence_pred",
            "softmax_confidence_abs_error",
            "joint_error",
            "eeg_path",
        ]
        return pd.DataFrame(columns=cols)

    if processed_outputs is not None:
        records = processed_outputs.to_dict("records") if isinstance(processed_outputs, pd.DataFrame) else list(processed_outputs)
        eeg_map = {str(row["sub"]): str(row["eeg_path"]) for row in records if row.get("sub") and row.get("eeg_path")}
        d["eeg_path"] = d["sub"].astype(str).map(eeg_map)
    elif "eeg_path" not in d.columns:
        d["eeg_path"] = None

    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "winning_corr_obs",
            "winning_corr_pred",
            "winning_corr_abs_error",
            "softmax_confidence_obs",
            "softmax_confidence_pred",
            "softmax_confidence_abs_error",
            "sub",
            "sample",
            "time",
        ]
    ).copy()
    if d.empty:
        return d

    d["joint_error"] = _item3_zscore(d["winning_corr_abs_error"]) + _item3_zscore(d["softmax_confidence_abs_error"])

    low_corr_q = float(d["winning_corr_obs"].quantile(0.25))
    high_corr_q = float(d["winning_corr_obs"].quantile(0.75))
    low_conf_q = float(d["softmax_confidence_obs"].quantile(0.25))
    high_conf_q = float(d["softmax_confidence_obs"].quantile(0.75))

    categories = [
        ("well predicted low max corr / uncertain", (d["winning_corr_obs"] <= low_corr_q) & (d["softmax_confidence_obs"] <= low_conf_q), True),
        ("badly predicted low max corr / uncertain", (d["winning_corr_obs"] <= low_corr_q) & (d["softmax_confidence_obs"] <= low_conf_q), False),
        ("well predicted high max corr / certain", (d["winning_corr_obs"] >= high_corr_q) & (d["softmax_confidence_obs"] >= high_conf_q), True),
        ("badly predicted high max corr / certain", (d["winning_corr_obs"] >= high_corr_q) & (d["softmax_confidence_obs"] >= high_conf_q), False),
    ]

    used_subjects = set()
    selected_rows = []
    for category_name, mask, ascending in categories:
        pool = d.loc[mask].copy()
        if pool.empty:
            continue
        pool = pool.sort_values(
            ["joint_error", "winning_corr_abs_error", "softmax_confidence_abs_error", "time"],
            ascending=[ascending, ascending, ascending, True],
        )
        chosen = 0
        for row in pool.itertuples(index=False):
            sub = str(row.sub)
            if sub in used_subjects:
                continue
            used_subjects.add(sub)
            out = pd.Series(row._asdict()).to_dict()
            out["category"] = category_name
            out["panel_index"] = int(chosen + 1)
            selected_rows.append(out)
            chosen += 1
            if chosen >= n_per_category:
                break

    if not selected_rows:
        return pd.DataFrame(columns=list(d.columns) + ["category", "panel_index"])

    out_df = pd.DataFrame(selected_rows)
    ordered_categories = [name for name, _, _ in categories]
    out_df["category"] = pd.Categorical(out_df["category"], categories=ordered_categories, ordered=True)
    out_df = out_df.sort_values(["category", "panel_index"]).reset_index(drop=True)
    return out_df


def _load_raw_for_topomap_examples(eeg_path: str | Path) -> mne.io.BaseRaw:
    eeg_path = Path(eeg_path)
    suffix = eeg_path.suffix.lower()
    if suffix == ".edf":
        raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose="ERROR")
    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported file format for {eeg_path}")
    return standard_names_picker(raw)


def plot_group_item3D_group_exemplar_topomaps(selected_df: pd.DataFrame):
    row_order = [
        "well predicted low max corr / uncertain",
        "badly predicted low max corr / uncertain",
        "well predicted high max corr / certain",
        "badly predicted high max corr / certain",
    ]
    fig, axes = plt.subplots(4, 5, figsize=(14.5, 10.2), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.04, top=0.92, wspace=0.30, hspace=0.48)
    fig.suptitle("Group exemplar topomaps from cached LOSO predictions", fontsize=11)

    if selected_df.empty:
        for ax in np.asarray(axes).ravel():
            ax.text(0.5, 0.5, "no exemplar selections", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="0.35")
            ax.set_axis_off()
        return fig

    raw_cache = {}
    vector_cache = {}
    vmax = 0.0
    for row in selected_df.itertuples(index=False):
        if not getattr(row, "eeg_path", None):
            continue
        sub = str(row.sub)
        if sub not in raw_cache:
            raw_cache[sub] = _load_raw_for_topomap_examples(row.eeg_path)
        eeg = np.asarray(raw_cache[sub].get_data(picks="eeg"), dtype=float)
        sample_idx = int(row.sample)
        if 0 <= sample_idx < eeg.shape[1]:
            vec = eeg[:, sample_idx]
            vector_cache[(sub, sample_idx)] = vec
            vmax = max(vmax, float(np.max(np.abs(vec))))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vlim = (-vmax, vmax)

    for r, category_name in enumerate(row_order):
        row_df = selected_df.loc[selected_df["category"].astype(str) == category_name].sort_values("panel_index").copy()
        for c in range(5):
            ax = axes[r, c]
            ax.set_box_aspect(1)
            if c == 0:
                ax.text(0.0, 1.10, category_name, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
            if c >= len(row_df):
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="0.35")
                ax.set_axis_off()
                continue

            row = row_df.iloc[c]
            key = (str(row["sub"]), int(row["sample"]))
            if key not in vector_cache or str(row["sub"]) not in raw_cache:
                ax.text(0.5, 0.5, "missing raw", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="0.35")
                ax.set_axis_off()
                continue

            mne.viz.plot_topomap(
                vector_cache[key],
                raw_cache[str(row["sub"])].info,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                vlim=vlim,
                contours=0,
                sensors=True,
                res=64,
            )
            ax.set_title(
                (
                    f"{row['sub']}\n"
                    f"s={int(row['sample'])} t={float(row['time']):.2f}s\n"
                    f"wc={float(row['winning_corr_obs']):.3f} c={float(row['softmax_confidence_obs']):.3f}\n"
                    f"err={float(row['joint_error']):.2f}"
                ),
                fontsize=7.4,
                pad=3.0,
            )

    return fig


def _run_item3_exploratory_models(sample_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []

    prepared_softmax = {}
    softmax_target_specs = None
    for sub, df in sorted(sample_dfs.items()):
        d = _item3_ml_analysis_subset(_prepare_item3_ml_dataframe(df))
        d, specs = _add_item3_softmax_targets(d)
        prepared_softmax[sub] = d
        if softmax_target_specs is None:
            softmax_target_specs = specs

    if softmax_target_specs:
        softmax_within = _run_item3_within_subject_models(
            prepared_softmax,
            target_specs=softmax_target_specs,
            predictor_specs=[("GFP+rho", ["GFP", "rho"])],
            analysis_scope="within_subject_softmax",
        )
        softmax_loso = _run_item3_loso_models(
            prepared_softmax,
            target_specs=softmax_target_specs,
            predictor_specs=[("GFP+rho", ["GFP", "rho"])],
            analysis_scope="loso_softmax",
        )
        for frame in [softmax_within, softmax_loso]:
            if not frame.empty:
                frame = frame.copy()
                frame["exploratory_kind"] = "softmax_target"
                frames.append(frame)

    exploratory_within = _run_item3_within_subject_models(
        sample_dfs,
        target_specs=_item3_ml_core_target_specs(),
        predictor_specs=_item3_ml_exploratory_predictor_specs(),
        analysis_scope="within_subject_exploratory_predictors",
    )
    exploratory_loso = _run_item3_loso_models(
        sample_dfs,
        target_specs=_item3_ml_core_target_specs(),
        predictor_specs=_item3_ml_exploratory_predictor_specs(),
        analysis_scope="loso_exploratory_predictors",
    )
    for frame in [exploratory_within, exploratory_loso]:
        if not frame.empty:
            frame = frame.copy()
            frame = frame.loc[frame["model_family"] == "linear"].copy()
            frame["exploratory_kind"] = "exploratory_predictors"
            frames.append(frame)

    if not frames:
        cols = _item3_ml_metrics_columns() + ["exploratory_kind"]
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, ignore_index=True)
    if "exploratory_kind" not in out.columns:
        out["exploratory_kind"] = "unspecified"
    return out


def _plot_winning_corr_boxplot(ax, df: pd.DataFrame, *, title: str):
    predictor_order = ["GFP", "rho", "GFP+rho"]
    data = []

    for predictor_set in predictor_order:
        vals = df.loc[
            (df["target"] == "winning_corr")
            & (df["predictor_set"] == predictor_set)
            & (df["model_family"] == "linear"),
            "R2",
        ].dropna().to_numpy(float)
        data.append(vals if len(vals) else np.array([np.nan]))

    if any(np.isfinite(arr).any() for arr in data):
        ax.boxplot(data, labels=predictor_order, showfliers=False)
    else:
        ax.text(0.5, 0.5, "no valid fits", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="0.35")

    ax.axhline(0.0, color="0.3", lw=1.0, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("R^2", fontsize=9)
    ax.set_xlabel("Predictor", fontsize=9)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)


def plot_group_item3B_winning_corr_model_performance(within_df: pd.DataFrame, loso_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.24, top=0.84, wspace=0.38)

    within_linear = within_df.loc[
        (within_df["model_family"] == "linear")
        & (within_df["target"] == "winning_corr")
    ].copy()

    loso_fold = loso_df.loc[
        loso_df["analysis_scope"].astype(str).str.endswith("_fold")
        & (loso_df["model_family"] == "linear")
        & (loso_df["target"] == "winning_corr")
    ].copy()

    _plot_winning_corr_boxplot(axes[0], within_linear, title="Within-subject")
    _plot_winning_corr_boxplot(axes[1], loso_fold, title="Leave-one-subject-out")

    y_vals = []
    for frame in [within_linear, loso_fold]:
        vals = frame["R2"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if len(vals):
            y_vals.append(vals)
    if y_vals:
        yy = np.concatenate(y_vals)
        ymin = min(-0.05, float(np.nanpercentile(yy, 2)) - 0.03)
        ymax = max(0.05, float(np.nanpercentile(yy, 98)) + 0.05)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    fig.suptitle("Prediction of winning-template correlation", fontsize=11)
    return fig


def plot_group_item3C_softmax_lambda_sweep(softmax_loso_df: pd.DataFrame):
    d = softmax_loso_df.copy()
    d = d.loc[
        d["analysis_scope"].astype(str).str.endswith("_fold")
        & (d["model_family"] == "linear")
        & np.isfinite(d["lambda"])
        & d["softmax_quantity"].isin(["confidence", "entropy"])
    ].copy()

    predictor_order = ["GFP", "rho", "GFP+rho"]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.22, top=0.84)

    if d.empty:
        ax.text(0.5, 0.5, "no valid softmax LOSO results", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="0.35")
        ax.set_axis_off()
        return fig

    summary = (
        d.groupby(["lambda", "predictor_set", "softmax_quantity"], as_index=False)
         .agg(mean_R2=("R2", "mean"), std_R2=("R2", "std"), n=("R2", "count"))
    )

    line_handles = []
    line_labels = []

    for predictor_set in predictor_order:
        q_conf = summary.loc[
            (summary["predictor_set"] == predictor_set)
            & (summary["softmax_quantity"] == "confidence")
        ].sort_values("lambda")

        if len(q_conf):
            line, = ax.plot(
                q_conf["lambda"],
                q_conf["mean_R2"],
                marker="o",
                lw=1.8,
                linestyle="-",
                label=f"{predictor_set} confidence",
            )
            ax.fill_between(
                q_conf["lambda"].to_numpy(float),
                (q_conf["mean_R2"] - q_conf["std_R2"]).to_numpy(float),
                (q_conf["mean_R2"] + q_conf["std_R2"]).to_numpy(float),
                color=line.get_color(),
                alpha=0.13,
                linewidth=0,
            )
            line_handles.append(line)
            line_labels.append(predictor_set)

        q_ent = summary.loc[
            (summary["predictor_set"] == predictor_set)
            & (summary["softmax_quantity"] == "entropy")
        ].sort_values("lambda")

        if len(q_ent):
            line_ent, = ax.plot(
                q_ent["lambda"],
                q_ent["mean_R2"],
                marker="s",
                lw=1.8,
                linestyle="--",
                label=f"{predictor_set} entropy",
            )
            ax.fill_between(
                q_ent["lambda"].to_numpy(float),
                (q_ent["mean_R2"] - q_ent["std_R2"]).to_numpy(float),
                (q_ent["mean_R2"] + q_ent["std_R2"]).to_numpy(float),
                color=line_ent.get_color(),
                alpha=0.10,
                linewidth=0,
            )

    ax.axhline(0.0, color="0.3", lw=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Softmax sharpness lambda", fontsize=9)
    ax.set_ylabel("LOSO OLS R^2", fontsize=9)
    ax.set_title("Prediction of soft assignment structure", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", axis="both", alpha=0.25)

    if line_handles:
        leg1 = ax.legend(
            line_handles,
            line_labels,
            title="Predictor",
            loc="upper right",
            fontsize=8,
            title_fontsize=8,
            framealpha=0.85,
        )
        ax.add_artist(leg1)

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-", marker="o", label="confidence"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", marker="s", label="entropy"),
    ]
    ax.legend(
        style_handles,
        ["confidence", "entropy"],
        title="Target",
        loc="lower left",
        fontsize=8,
        title_fontsize=8,
        framealpha=0.85,
    )

    return fig


def plot_group_item3A_target_relationships(sample_dfs: dict[str, pd.DataFrame]):
    pooled = []
    for sub, df in sorted(sample_dfs.items()):
        d = _item3_ml_analysis_subset(_prepare_item3_ml_dataframe(df))
        d["sub"] = sub
        pooled.append(d)
    pooled_df = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()

    target_specs = _item3_ml_core_target_specs()
    predictor_specs = [("GFP_uV", "GFP (uV)"), ("rho", "rho")]
    fig, axes = plt.subplots(2, len(target_specs), figsize=(18.5, 7.2), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.90, wspace=0.34, hspace=0.28)

    hexbins = []
    for r, (predictor, xlabel) in enumerate(predictor_specs):
        for c, (target, title) in enumerate(target_specs):
            ax = axes[r, c]
            ax.set_box_aspect(1)
            ax.tick_params(labelsize=8)
            if pooled_df.empty:
                ax.text(0.5, 0.5, "no valid data", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.35")
                continue
            x, y = _finite_xy(pooled_df, predictor, target)
            hb = _hexbin_percent(ax, x, y, gridsize=55, mincnt=2, cmap="viridis")
            hexbins.append(hb)
            _add_linear_regression_overlay_r_only(ax, x, y, fontsize=9.5)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(title, fontsize=9)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if target in {"winning_corr", "corr_margin", "relative_corr_margin", "corr_entropy", "dipole_angle_gap"}:
                ax.set_ylim(-0.02, 1.02)
    _set_hexbin_average_clim(hexbins)
    return fig


def _item3_combo_order():
    ordered = []
    for model_family, model_label in _item3_ml_model_specs():
        for predictor_set, _ in _item3_ml_main_predictor_specs():
            ordered.append((predictor_set, model_family, f"{predictor_set}\n{model_label}"))
    return ordered


def _plot_item3_metric_panel(ax, df: pd.DataFrame, *, target: str, metric: str, pooled_df: pd.DataFrame | None = None):
    combo_order = _item3_combo_order()
    positions = np.arange(1, len(combo_order) + 1)
    data = []
    labels = []
    pooled_vals = []
    for predictor_set, model_family, combo_label in combo_order:
        vals = df.loc[
            (df["target"] == target)
            & (df["predictor_set"] == predictor_set)
            & (df["model_family"] == model_family),
            metric,
        ].dropna().to_numpy(float)
        data.append(vals if len(vals) else np.array([np.nan]))
        labels.append(combo_label)
        if pooled_df is not None:
            pooled_match = pooled_df.loc[
                (pooled_df["target"] == target)
                & (pooled_df["predictor_set"] == predictor_set)
                & (pooled_df["model_family"] == model_family),
                metric,
            ].dropna().to_numpy(float)
            pooled_vals.append(float(pooled_match[0]) if len(pooled_match) else np.nan)

    if any(np.isfinite(arr).any() for arr in data):
        ax.boxplot(data, positions=positions, widths=0.72, showfliers=False)
    else:
        ax.text(0.5, 0.5, "no valid fits", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.35")
    if pooled_df is not None:
        for x0, y0 in zip(positions, pooled_vals):
            if np.isfinite(y0):
                ax.scatter(x0, y0, marker="D", s=24, color="black", zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", rotation=25)
    if metric == "r":
        ax.axhline(0.0, color="0.3", lw=1.0, linestyle="--")
    elif metric == "R2":
        ax.axhline(0.0, color="0.3", lw=1.0, linestyle="--")


def plot_group_item3B_within_subject_model_performance(within_df: pd.DataFrame):
    fig, axes = plt.subplots(2, len(_item3_ml_core_target_specs()), figsize=(18.5, 7.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.20, top=0.90, wspace=0.28, hspace=0.32)
    metric_rows = [("r", "Pearson r"), ("R2", "R^2")]
    for r, (metric, ylabel) in enumerate(metric_rows):
        for c, (target, title) in enumerate(_item3_ml_core_target_specs()):
            ax = axes[r, c]
            _plot_item3_metric_panel(ax, within_df, target=target, metric=metric)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)
    return fig


def plot_group_item3C_loso_model_performance(loso_df: pd.DataFrame):
    fold_df = loso_df.loc[loso_df["analysis_scope"].astype(str).str.endswith("_fold")].copy()
    pooled_df = loso_df.loc[loso_df["analysis_scope"].astype(str).str.endswith("_pooled")].copy()
    fig, axes = plt.subplots(2, len(_item3_ml_core_target_specs()), figsize=(18.5, 7.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.20, top=0.90, wspace=0.28, hspace=0.32)
    metric_rows = [("r", "Pearson r"), ("R2", "R^2")]
    for r, (metric, ylabel) in enumerate(metric_rows):
        for c, (target, title) in enumerate(_item3_ml_core_target_specs()):
            ax = axes[r, c]
            _plot_item3_metric_panel(ax, fold_df, target=target, metric=metric, pooled_df=pooled_df)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)
    return fig


def _item3_uncertainty_note_text() -> str:
    return """# Item 3 uncertainty note

## Overview

This Item 3 redesign treats predictability as a regression problem. The main question is whether samplewise `GFP`, `rho`, or `GFP + rho` can predict how strongly one meta-map wins, and how ambiguous that winner is relative to the alternatives.

## Why `winning_corr` is useful but not a pure uncertainty measure

`winning_corr` is the largest meta-correlation at a sample. A higher value usually means the current sample more closely resembles at least one available meta-map. That makes it confidence-like. It is still not a full uncertainty measure, because it says nothing about how close the runner-up map is. A sample can have a high winning correlation and still be ambiguous if the second-best correlation is almost the same.

## Why margin and relative margin matter

`corr_margin = max_meta_corr - second_max_meta_corr` is a separability measure. It asks whether the winner is clearly ahead of the runner-up. `relative_corr_margin` rescales the same idea by the winning correlation, which helps distinguish between:

- a large absolute gap at high overall correlation
- a similar absolute gap at mediocre overall correlation

These are still deterministic functions of one observed correlation vector. They measure separation, not instability under perturbation.

## Why entropy is different

`corr_entropy` is computed from normalized squared meta-correlations. This turns the full vector of correlations into a probability-like distribution over candidate maps. High entropy means the evidence is spread across several templates; low entropy means it is concentrated. Entropy is therefore a distributional ambiguity measure, whereas margin only compares the top two entries.

## Why dipole-angle gap is geometry-aware

`dipole_angle_gap` uses the smallest and second-smallest `psi_meta_*` values, normalized by `180`. This measures how much angular room exists between the best and second-best meta-map directions. It is useful because ambiguity may remain geometrically structured even when correlations are similar.

## Why softmax-derived metrics are exploratory

Softmax confidence and softmax entropy depend on a temperature or gain coefficient. A larger coefficient exaggerates rank differences; a smaller coefficient smooths them. Because there is no single coefficient justified by the current deterministic data alone, these metrics are useful exploratory summaries rather than first-class primary endpoints.

## What cannot be identified from the current data alone

True rank instability is not identifiable from a single deterministic correlation vector. To claim instability, we need an explicit perturbation model, such as:

- bootstrap resampling across channels or time
- measurement-noise injection
- re-referencing perturbations
- template uncertainty propagation

Without one of those, we can quantify ambiguity, concentration, and geometric separation, but not the probability that the winning label would change under realistic perturbations.

## Interpretation of the modeling results

The main models here ask whether `GFP`, `rho`, or both together explain these confidence-like or ambiguity-like targets. If `rho` outperforms `GFP` for `winning_corr`, that supports the idea that dipole conformity carries information about label confidence that amplitude alone does not capture. If `GFP + rho` improves over either alone, then amplitude and conformity are complementary.

The exploratory `GFP + rho + d_x + d_y + d_z` model tests a narrower hypothesis: directional dipole coordinates may explain residual uncertainty structure beyond amplitude and conformity. That result should be treated as secondary because those coordinates are more geometry-specific and may be less stable across datasets.
"""


def save_group_item3_ml_outputs(processed_outputs, outdir: str | Path):
    outdir = Path(outdir)
    group_dir = outdir / "group_item3_ml"
    group_dir.mkdir(parents=True, exist_ok=True)

    records = processed_outputs.to_dict("records") if isinstance(processed_outputs, pd.DataFrame) else list(processed_outputs)
    sample_dfs = {}
    for row in records:
        if "sub" not in row or "analysis_csv" not in row or not row["analysis_csv"]:
            continue
        csv_path = Path(row["analysis_csv"])
        if not csv_path.exists():
            continue
        sample_dfs[str(row["sub"])] = pd.read_csv(csv_path)

    if not sample_dfs:
        return None

    within_path = group_dir / "table_3_within_subject_model_comparison.csv"
    loso_path = group_dir / "table_3_loso_model_comparison.csv"
    softmax_path = group_dir / "table_3_softmax_lambda_sweep.csv"
    sample_pred_path = group_dir / "table_3_sample_predictions_loso.csv"
    exemplar_path = group_dir / "table_3D_group_exemplar_samples.csv"

    if within_path.exists() and loso_path.exists() and softmax_path.exists():
        within_df = pd.read_csv(within_path)
        loso_df = pd.read_csv(loso_path)
        softmax_loso_df = pd.read_csv(softmax_path)
    else:
        within_df = _run_item3_within_subject_models(
            sample_dfs,
            target_specs=[("winning_corr", "Winning corr")],
            predictor_specs=_item3_ml_main_predictor_specs(),
            analysis_scope="within_subject",
        )
        loso_df = _run_item3_loso_models(
            sample_dfs,
            target_specs=[("winning_corr", "Winning corr")],
            predictor_specs=_item3_ml_main_predictor_specs(),
            analysis_scope="loso",
        )
        softmax_loso_df = _run_item3_softmax_lambda_sweep_models(sample_dfs)
        save_table(within_df, within_path)
        save_table(loso_df, loso_path)
        save_table(softmax_loso_df, softmax_path)

    if sample_pred_path.exists():
        sample_predictions_df = pd.read_csv(sample_pred_path)
        if "eeg_path" not in sample_predictions_df.columns:
            eeg_map = {str(row["sub"]): str(row["eeg_path"]) for row in records if row.get("sub") and row.get("eeg_path")}
            sample_predictions_df["eeg_path"] = sample_predictions_df["sub"].astype(str).map(eeg_map)
            save_table(sample_predictions_df, sample_pred_path)
    else:
        confidence_lambda = _choose_item3_exemplar_confidence_lambda(softmax_loso_df)
        sample_predictions_df = _run_item3_loso_sample_predictions(
            sample_dfs,
            confidence_lambda=confidence_lambda,
            predictor_set="GFP+rho",
            model_family="linear",
        )
        eeg_map = {str(row["sub"]): str(row["eeg_path"]) for row in records if row.get("sub") and row.get("eeg_path")}
        if not sample_predictions_df.empty:
            sample_predictions_df["eeg_path"] = sample_predictions_df["sub"].astype(str).map(eeg_map)
        save_table(sample_predictions_df, sample_pred_path)

    if exemplar_path.exists():
        exemplar_df = pd.read_csv(exemplar_path)
        if "eeg_path" not in exemplar_df.columns:
            eeg_map = {str(row["sub"]): str(row["eeg_path"]) for row in records if row.get("sub") and row.get("eeg_path")}
            exemplar_df["eeg_path"] = exemplar_df["sub"].astype(str).map(eeg_map)
            save_table(exemplar_df, exemplar_path)
    else:
        exemplar_df = select_group_item3_exemplar_samples(sample_predictions_df, processed_outputs=processed_outputs, n_per_category=5)
        save_table(exemplar_df, exemplar_path)

    save_figure(
        plot_group_item3B_winning_corr_model_performance(within_df, loso_df),
        group_dir / "figure_3B_winning_corr_model_performance.svg",
    )
    save_figure(
        plot_group_item3C_softmax_lambda_sweep(softmax_loso_df),
        group_dir / "figure_3C_softmax_lambda_sweep.svg",
    )
    save_figure(
        plot_group_item3D_group_exemplar_topomaps(exemplar_df),
        group_dir / "figure_3D_group_exemplar_topomaps.svg",
        tight=False,
    )
    for stale_name in [
        "figure_3A_target_relationships.svg",
        "figure_3B_within_subject_model_performance.svg",
        "figure_3C_loso_model_performance.svg",
        "figure_3A_target_relationships.png",
        "figure_3B_within_subject_model_performance.png",
        "figure_3C_loso_model_performance.png",
        "figure_3B_winning_corr_model_performance.png",
        "figure_3C_softmax_lambda_sweep.png",
        "table_3_uncertainty_exploratory.csv",
        "item3_uncertainty_note.md",
    ]:
        stale_path = group_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    return {
        "within": within_df,
        "loso": loso_df,
        "softmax_loso": softmax_loso_df,
        "sample_predictions": sample_predictions_df,
        "exemplars": exemplar_df,
    }


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------


def save_figure(fig, path: str | Path, *, tight: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": 180}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **save_kwargs)
    plt.close(fig)


def save_table(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_item_outputs(
    sub: str,
    raw: mne.io.BaseRaw,
    sample_df: pd.DataFrame,
    levels_by_k: dict,
    subject_dir: str | Path,
    base_k: int = 4,
    write_exemplary_plots: bool = False,
):
    subject_dir = Path(subject_dir)
    item1 = subject_dir / "item1_zanesco_td"
    item2 = subject_dir / "item2_theta_phi"
    item3 = subject_dir / "item3_nonredundancy"
    for p in [item1, item2, item3]:
        p.mkdir(parents=True, exist_ok=True)

    if base_k not in levels_by_k:
        raise KeyError(f"Requested base_k={base_k} not found in levels_by_k; available keys: {sorted(levels_by_k)}")

    meta_base = levels_by_k[base_k]["meta"]
    sub_base = levels_by_k[base_k].get("subject")

    save_figure(plot_item1A_zanesco_panel(sample_df), item1 / "figure_1A_zanesco_replication_panel.png")
    save_figure(plot_item1B_peak_overlay(sample_df), item1 / "figure_1B_gfp_peak_overlay_panel.png")
    save_figure(plot_item1C_td_decomposition(sample_df), item1 / "figure_1C_td_decomposition_panel.png")
    save_figure(plot_abstract_subject_component(sample_df, sub=sub), item1 / "figure_abstract_subject_component.svg", tight=False)
    save_table(table1_subject_summary(sample_df, sub), item1 / "table_1_subject_summary.csv")
    fig_item1d, table_item1d = plot_item1D_low_high_f_examples(raw, sample_df, sub=sub)
    save_figure(fig_item1d, item1 / "figure_1D_low_high_F_examples.svg", tight=False)
    save_table(table_item1d, item1 / "table_1D_low_high_F_examples.csv")
    stale_abstract_png = item1 / "figure_abstract_subject_component.png"
    if stale_abstract_png.exists():
        stale_abstract_png.unlink()

    if sub_base is not None:
        save_figure(plot_item2A_template_geometry(levels_by_k), item2 / "figure_2A_template_geometry_across_k.png")
        save_figure(plot_item2B_gfp_theta_phi(sample_df, meta_base, sub_base), item2 / "figure_2B_gfp_theta_phi.png")
        save_figure(plot_item2C_meed_theta_phi(sample_df, meta_base, sub_base), item2 / "figure_2C_meed_theta_phi.png")
        save_table(table2_template_bin_summary(sample_df, meta_base, sub_base, sub), item2 / "table_2_template_bin_summary.csv")

    save_table(_item3_target_summary(sample_df, sub), item3 / "table_3_target_summary.csv")
    for stale_name in [
        "figure_3A_nonredundancy_panel.png",
        "figure_3B_predictor_boxplots.png",
        "figure_3C_filtering_comparison.png",
        "table_3_predictive_models.csv",
    ]:
        stale_path = item3 / stale_name
        if stale_path.exists():
            stale_path.unlink()

    if write_exemplary_plots:
        save_exemplary_subject_plots(sub, raw, sample_df, subject_dir)
