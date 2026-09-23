"""Stage-D-specific descriptive, plotting, spatial, and rho-ceiling helpers.

This module owns helpers used only by ``form_validation_d_descriptive.py``.
Shared loading, clustering, FORM, sequence, and evaluation mechanics remain in
``form_validation_abc_utils.py``.
"""

from __future__ import annotations

import json
import warnings

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, MultipleLocator, NullFormatter, NullLocator
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import Normalize
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import least_squares, nnls
from scipy import stats
from scipy.stats import pearsonr

try:
    from scipy.special import sph_harm_y

    def _spherical_harmonic(l, m, polar, azimuth):
        return sph_harm_y(l, m, polar, azimuth)
except ImportError:
    from scipy.special import sph_harm

    def _spherical_harmonic(l, m, polar, azimuth):
        return sph_harm(m, l, azimuth, polar)

import form_validation_abc_utils as mvu
import form_method_utils as u

# Shared primitives used by the Stage-D-only helpers below. They remain owned by
# form_validation_abc_utils because A/B/C and D use them as well.
TemplateLevel = mvu.TemplateLevel
standard_names_picker = mvu.standard_names_picker
get_fs_from_time_column = mvu.get_fs_from_time_column
load_meta_level = mvu.load_meta_level
build_template_level = mvu.build_template_level
read_cluster_model = mvu.read_cluster_model
stable_seed = mvu.stable_seed
fit_modkmeans = mvu.fit_modkmeans
make_chdata = mvu.make_chdata
match_group_maps_to_meta = mvu.match_group_maps_to_meta
compute_microstate_association = mvu.compute_microstate_association
STAGE_C_ASSOCIATION_SCHEMA = "instantaneous-pooled-association-and-gev-v5"
_center_unit_maps = mvu._center_unit_maps
_td_form_radial_angular_components = mvu._td_form_radial_angular_components


# ---------------------------------------------------------------------
# Recording and cohort descriptive outputs
# ---------------------------------------------------------------------

def sample_cache_to_dataframe(
    sample_cache_file: str | Path,
    *,
    subject: str,
    condition: str,
    recording_id: str,
) -> pd.DataFrame:
    """Load canonical samples; retain legacy MEED column keys for cached diagnostics."""
    with np.load(sample_cache_file, allow_pickle=False) as raw_cache:
        c = mvu.canonical_cache_names({key: raw_cache[key] for key in raw_cache.files})
        meta_labels = c["meta_labels"].astype(str).tolist()
        q = c["form_q"].astype(float)
        td = c["td"].astype(float)
        td_meed = c["td_form"].astype(float)
        td_res = c["td_off_form"].astype(float)
        td2 = c["td2"].astype(float)
        td2_meed = c["td2_form"].astype(float)
        td2_res = c["td2_off_form"].astype(float)

        new_component_fields = {
            "td_rho",
            "td_psi",
            "td2_rho",
            "td2_psi",
            "td_form_internal_closure",
            "td_threeway_closure",
        }
        if new_component_fields.issubset(c):
            td_rho = c["td_rho"].astype(float)
            td_psi = c["td_psi"].astype(float)
            td2_rho = c["td2_rho"].astype(float)
            td2_psi = c["td2_psi"].astype(float)
            td_meed_internal_closure = c["td_form_internal_closure"].astype(float)
            td_threeway_closure = c["td_threeway_closure"].astype(float)
        else:
            # Existing Stage-A caches predate the radial/angular FORM split.
            # Their q, rho, TD, and discontinuity arrays are sufficient to
            # reconstruct it without reopening the EEG or recomputing FORM.
            if "valid_transition" not in c:
                raise RuntimeError("Stage-A cache predates the endpoint-valid transition contract; rerun Stage A")
            valid_transition = c["valid_transition"].astype(bool)
            td_rho, td_psi = _td_form_radial_angular_components(
                q,
                c["rho"].astype(float),
                valid_transition,
            )
            td2_rho = td_rho ** 2
            td2_psi = td_psi ** 2
            td_meed_internal_closure = td2_meed - td2_rho - td2_psi
            td_threeway_closure = td2 - td2_res - td2_rho - td2_psi

        f_res = np.divide(
            td2_res, td2, out=np.full(len(td2), np.nan), where=td2 > 1e-12
        )
        f_rho = np.divide(
            td2_rho, td2, out=np.full(len(td2), np.nan), where=td2 > 1e-12
        )
        f_psi = np.divide(
            td2_psi, td2, out=np.full(len(td2), np.nan), where=td2 > 1e-12
        )
        f_meed = c["f_form"].astype(float)
        rho = c["rho"].astype(float)
        if "valid_transition" not in c:
            raise RuntimeError("Stage-A cache predates the endpoint-valid transition contract; rerun Stage A")
        valid_transition = c["valid_transition"].astype(bool)
        rho_pair = np.full(len(rho), np.nan)
        rho_pair2 = np.full(len(rho), np.nan)
        transition_index = np.flatnonzero(valid_transition)
        rho_pair2[transition_index] = rho[transition_index] * rho[transition_index - 1]
        rho_pair[transition_index] = np.sqrt(np.maximum(rho_pair2[transition_index], 0.0))

        df = pd.DataFrame({
            "sub": str(subject),
            "subject": str(subject),
            "condition": str(condition),
            "recording_id": str(recording_id),
            "sample": c["sample_index"].astype(np.int64),
            "time": c["time_s"].astype(float),
            "GFP": c["gfp"].astype(float),
            "local_GFP_peak": c["local_gfp_peak"].astype(bool),
            "d_x": q[:, 0],
            "d_y": q[:, 1],
            "d_z": q[:, 2],
            "theta": c["theta_deg"].astype(float),
            "phi": c["phi_deg"].astype(float),
            "rho": rho,
            "rho2": c["rho2"].astype(float),
            "rho_pair": rho_pair,
            "rho_pair2": rho_pair2,
            "psiD": c["psiD_deg"].astype(float),
            "dipole_angle_antipodal_deg": c["dipole_angle_antipodal_deg"].astype(float),
            "dipole_delta_norm": c["dipole_delta_norm"].astype(float),
            "dipole_delta_norm_antipodal": c["dipole_delta_norm_antipodal"].astype(float),
            "TD": td,
            "TD_MEED": td_meed,
            "TD_res": td_res,
            "TD_rho": td_rho,
            "TD_psi": td_psi,
            "TD2": td2,
            "TD2_MEED": td2_meed,
            "TD2_res": td2_res,
            "TD2_rho": td2_rho,
            "TD2_psi": td2_psi,
            "F_MEED_TD": f_meed,
            "F_res": f_res,
            "F_rho": f_rho,
            "F_psi": f_psi,
            "TD_decomposition_closure_error": c["td_closure"].astype(float),
            "TD_MEED_internal_closure_error": td_meed_internal_closure,
            "TD_threeway_closure_error": td_threeway_closure,
            "fraction_closure_error": f_res + f_rho + f_psi - 1.0,
            "F_MEED_component_error": f_meed - f_rho - f_psi,
        })
        psi_meta = c["psi_meta_deg"].astype(float)
        theta_meta = c["theta_meta_distance_deg"].astype(float)
        phi_meta = c["phi_meta_distance_deg"].astype(float)
        for j, label in enumerate(meta_labels):
            df[f"psi_meta_{label}"] = psi_meta[:, j]
            df[f"theta_meta_{label}"] = theta_meta[:, j]
            df[f"phi_meta_{label}"] = phi_meta[:, j]
    # Assignment-strength analyses use the canonical pooled K=5 dictionary.
    # Stage-A meta geometry remains for retained map diagnostics.
    association_file = Path(sample_cache_file).parents[2] / "c_backfit" / "associations" / f"{recording_id}_group_A-E.npz"
    association_metadata = association_file.parents[1] / "metadata" / f"{recording_id}_group_A-E.json"
    if not association_file.is_file() or not association_metadata.is_file():
        raise FileNotFoundError(f"Stage-C pooled association and metadata are required for {recording_id}")
    metadata = json.loads(association_metadata.read_text(encoding="utf-8"))
    if metadata.get("cache_schema") != STAGE_C_ASSOCIATION_SCHEMA:
        raise RuntimeError(f"Stage-C association for {recording_id} has stale schema; rerun Stage C")
    with np.load(association_file, allow_pickle=False) as association:
        corr = association["corr_abs"].astype(float)
        labels = association["labels"].astype(str).tolist()
        valid_topography = association["valid_topography"].astype(bool)
        if corr.shape[0] != len(df) or len(valid_topography) != len(df) or tuple(labels) != ("A", "B", "C", "D", "E"):
            raise RuntimeError(f"Invalid Stage-C association contract for {recording_id}")
        df["valid_topography"] = valid_topography
        for index, label in enumerate(labels):
            df[f"corr_group_{label}"] = corr[:, index]
        df["max_group_corr"] = association["max_abs_corr"].astype(float)
        df["second_max_group_corr"] = association["second_abs_corr"].astype(float)
        df["corr_margin"] = association["corr_margin"].astype(float)
    return df


def add_selected_map_columns(
    df: pd.DataFrame,
    data: np.ndarray,
    selected_level: TemplateLevel,
) -> pd.DataFrame:
    """Add D-map-source correlations to one recording."""
    d = df.copy()
    unit_maps, _, _ = _center_unit_maps(np.asarray(data, dtype=float))
    association = compute_microstate_association(unit_maps, selected_level.B)
    for index, label in enumerate(selected_level.labels):
        d[f"corr_selected_{label}"] = association["corr_abs"][:, index]
    return d


def _subject_sample_dfs_from_records(records) -> dict[str, pd.DataFrame]:
    grouped: dict[str, list[pd.DataFrame]] = {}
    for row in records:
        cache_path = row.get("sample_cache_file")
        if not cache_path or not Path(cache_path).exists():
            continue
        subject = str(row.get("subject") or row.get("sub"))
        condition = str(row.get("condition", ""))
        recording_id = str(row.get("recording_id", subject))
        df = sample_cache_to_dataframe(
            cache_path,
            subject=subject,
            condition=condition,
            recording_id=recording_id,
        )
        grouped.setdefault(subject, []).append(df)
    return {
        subject: pd.concat(parts, ignore_index=True)
        for subject, parts in grouped.items()
        if parts
    }


def _item3_subject_dfs_from_records(records) -> dict[str, pd.DataFrame]:
    """Load, derive, and bound one subject before reading the next subject."""
    grouped: dict[str, list[dict]] = {}
    for row in records:
        subject = str(row.get("subject") or row.get("sub"))
        grouped.setdefault(subject, []).append(row)

    prepared: dict[str, pd.DataFrame] = {}
    for subject, subject_records in sorted(grouped.items()):
        parts = []
        for row in subject_records:
            cache_path = row.get("sample_cache_file")
            if not cache_path or not Path(cache_path).exists():
                continue
            parts.append(sample_cache_to_dataframe(
                cache_path,
                subject=subject,
                condition=str(row.get("condition", "")),
                recording_id=str(row.get("recording_id", subject)),
            ))
        if not parts:
            continue
        subject_df = pd.concat(parts, ignore_index=True)
        prepared[subject] = _item3_ml_analysis_subset(
            _prepare_item3_ml_dataframe(subject_df)
        )
    return prepared

# ---------------------------------------------------------------------
# Cached first-layer exports
# ---------------------------------------------------------------------


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


def _format_p_value(p: float) -> str:
    """Use a compact, publication-style p-value label."""
    if not np.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3g}"

def _add_linear_regression_overlay(ax, x, y):
    stats = _linear_regression_stats(x, y)
    if stats is None:
        return
    ax.plot(stats["xs"], stats["ys"], "--", color="black", gapcolor="white", dashes=(5, 5), lw=1.5)
    ax.text(
        0.98,
        0.04,
        f"r = {stats['r']:.3f}\nR^2 = {stats['r2']:.3f}\n{_format_p_value(stats['p'])}",
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
        f"r = {stats['r']:.3f}\nR^2 = {stats['r2']:.3f}\n{_format_p_value(stats['p'])}",
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


def _item1_components() -> list[tuple[str, str]]:
    return [
        ("TD", "TD"),
        ("TD_MEED", "TD_MEED"),
        ("TD_res", "TD_res"),
        ("TD_rho", "TD_rho"),
        ("TD_psi", "TD_psi"),
    ]


def compute_item1_gfp_component_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Fit each TD-compatible component against GFP on its own valid samples."""
    gfp_uv = df["GFP"].to_numpy(float) * 1e6
    rows = []
    for component, column in _item1_components():
        values = df[column].to_numpy(float)
        valid = np.isfinite(gfp_uv) & np.isfinite(values) & (gfp_uv > 0) & (values > 0)
        x = np.log10(gfp_uv[valid])
        y = np.log10(values[valid])
        row = {
            "component": component,
            "n_valid": int(len(x)),
            "alpha": np.nan,
            "beta": np.nan,
            "r": np.nan,
            "p_raw": np.nan,
            "R2": np.nan,
        }
        if len(x) >= 3:
            design = np.column_stack([np.ones(len(x)), x])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ coefficients
            total = float(np.sum((y - np.mean(y)) ** 2))
            row.update({
                "alpha": float(coefficients[0]),
                "beta": float(coefficients[1]),
                "r": float(pearsonr(x, y)[0]) if np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan,
                "p_raw": float(pearsonr(x, y)[1]) if np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan,
                "R2": float(1.0 - np.sum((y - fitted) ** 2) / total) if total > 0 else np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def plot_td_gfp_example(df: pd.DataFrame):
    """F08: one common-lattice GFP-to-TD-component row from a cached recording."""
    components = [(r"$TD$", "TD", "TD"), (r"$TD_{\mathrm{FORM}}$", "TD_MEED", "TD_MEED"),
                  (r"$TD_{\mathrm{res}}$", "TD_res", "TD_res"), (r"$TD_{\psi}$", "TD_psi", "TD_psi"),
                  (r"$TD_{\rho}$", "TD_rho", "TD_rho")]
    gfp = df["GFP"].to_numpy(float) * 1e6
    finite_y = np.concatenate([df[column].to_numpy(float) for _, column, _ in components])
    valid_x = gfp[np.isfinite(gfp) & (gfp > 0)]; valid_y = finite_y[np.isfinite(finite_y) & (finite_y > 0)]
    extent = (np.log10(valid_x.min()), np.log10(valid_x.max()), np.log10(valid_y.min()), np.log10(valid_y.max()))
    fig, axes = plt.subplots(1, 5, figsize=(u.PAPER_WIDTH_IN, 1.8), layout="constrained")
    fits = compute_item1_gfp_component_fits(df).set_index("component")
    hexbins = []
    shared_axes = [0, 1, 2, 3]
    shared_ylim = (1e-3, 3.0)
    shared_extent = (extent[0], extent[1], np.log10(shared_ylim[0]), np.log10(shared_ylim[1]))
    for index, (label, column, fit_component) in enumerate(components):
        axis = axes[index]; values = df[column].to_numpy(float)
        valid = np.isfinite(gfp) & np.isfinite(values) & (gfp > 0) & (values > 0)
        panel_extent = shared_extent if index in shared_axes else extent
        hexbin = axis.hexbin(gfp[valid], values[valid], xscale="log", yscale="log", extent=panel_extent,
                             gridsize=34, mincnt=2, cmap="viridis", linewidths=0, bins=None)
        hexbins.append(hexbin)
        axis.set_xlabel("GFP (µV)", fontsize=10); axis.set_ylabel("")
        axis.set_title(label, fontsize=10)
        axis.set_xlim(10 ** extent[0], 10 ** extent[1])
        if index in shared_axes:
            axis.set_ylim(*shared_ylim)
            if index != shared_axes[0]:
                axis.tick_params(labelleft=False)
        fit = fits.loc[fit_component]
        if np.isfinite(fit.beta):
            x = np.logspace(np.log10(gfp[valid].min()), np.log10(gfp[valid].max()), 100)
            axis.plot(x, 10 ** fit.alpha * x ** fit.beta, "--", color="black", lw=.8)
        u.set_publication_spines(axis, categorical=False); axis.tick_params(labelsize=10)
    colorbar = fig.colorbar(hexbins[0], ax=axes, shrink=.65, pad=.01)
    colorbar.set_label("samples/bin", fontsize=10, labelpad=2)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e3:g}"))
    colorbar.ax.set_title(r"$\times10^{3}$", fontsize=10, pad=4)
    colorbar.ax.tick_params(labelsize=10)
    return fig, fits.reset_index(), {"common_log_extent": shared_extent, "shared_y_limits": shared_ylim,
        "shared_y_components": [components[i][0] for i in shared_axes], "rho_y_independent": True}


def write_td_gfp_group_outputs(records, outdir: str | Path) -> pd.DataFrame:
    """Write TD/GFP fits with descriptive empirical intervals and corrections."""
    rows = []
    for record in records:
        path = record.get("sample_cache_file")
        if not path or not Path(path).exists(): continue
        frame = sample_cache_to_dataframe(path, subject=str(record["subject"]), condition=str(record["condition"]), recording_id=str(record["recording_id"]))
        fits = compute_item1_gfp_component_fits(frame)
        for _, fit in fits.iterrows():
            component = str(fit.component)
            values = frame[component].to_numpy(float)
            rows.append({"subject": str(record["subject"]), "condition": str(record["condition"]), "recording_id": str(record["recording_id"]),
                         "component": component, "mean_td": float(np.nanmean(values)), **fit.to_dict()})
    output = Path(outdir); output.mkdir(parents=True, exist_ok=True)
    fits = pd.DataFrame(rows)
    fits["p_bonferroni_5"] = np.minimum(fits["p_raw"] * 5.0, 1.0)
    fits["significant_nominal"] = fits["p_raw"] < 0.05
    fits["significant_bonferroni_5"] = fits["p_bonferroni_5"] < 0.05
    fits.to_csv(output / "td_gfp_participant_fits.csv", index=False)
    summary = fits.groupby(["condition", "component"], as_index=False).agg(
        n_participants=("subject", "nunique"), n_recordings=("recording_id", "nunique"),
        mean_td=("mean_td", "mean"), sd_td=("mean_td", "std"),
        mean_beta=("beta", "mean"), sd_beta=("beta", "std"), mean_r2=("R2", "mean"),
        p5_beta=("beta", lambda values: np.nanpercentile(values, 5)),
        p95_beta=("beta", lambda values: np.nanpercentile(values, 95)),
        p5_r2=("R2", lambda values: np.nanpercentile(values, 5)),
        p95_r2=("R2", lambda values: np.nanpercentile(values, 95)),
        nominal_significant_fits=("significant_nominal", "sum"),
        bonferroni_significant_fits=("significant_bonferroni_5", "sum"),
    )
    summary.to_csv(output / "td_gfp_component_summary.csv", index=False)
    summary.to_html(output / "table_td_gfp_group.html", index=False, float_format=lambda value: f"{value:.4g}")
    (output / "td_gfp_statistics_metadata.json").write_text(json.dumps({"row_unit": "recording participant-condition-component", "interval": "central 90% empirical interval (participant p5-p95), not a confidence interval", "multiple_testing": "nominal and Bonferroni correction across five TD measures within participant-condition", "fit": "full finite-positive log10 GFP_uV versus TD component regression"}, indent=2) + "\n")
    return fits

def compute_item1_gfp_link_residuals(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare GFP implied by each TD component with GFP implied by total TD."""
    d = df.copy()
    d["GFP_uV"] = d["GFP"] * 1e6
    fits = compute_item1_gfp_component_fits(d).set_index("component")
    epsilon = 1e-6

    gfp = d["GFP_uV"].to_numpy(float)
    gfp_valid = np.isfinite(gfp) & (gfp > 0)
    d["log_GFP_uV"] = np.nan
    d.loc[gfp_valid, "log_GFP_uV"] = np.log10(gfp[gfp_valid])
    for column in ("F_MEED_TD", "rho_pair2"):
        values = d[column].to_numpy(float)
        valid = np.isfinite(values)
        d[f"log_{column}"] = np.nan
        d.loc[valid, f"log_{column}"] = np.log10(np.maximum(values[valid], epsilon))

    for component, column in _item1_components():
        values = d[column].to_numpy(float)
        valid = np.isfinite(values) & (values > 0)
        log_column = f"log_{component}"
        d[log_column] = np.nan
        d.loc[valid, log_column] = np.log10(values[valid])

        fit = fits.loc[component]
        predicted_column = f"logGFP_hat_from_{component}"
        d[predicted_column] = np.nan
        if np.isfinite(fit["beta"]) and abs(fit["beta"]) > 1e-12:
            d.loc[valid, predicted_column] = (
                d.loc[valid, log_column] - float(fit["alpha"])
            ) / float(fit["beta"])

    td_hat = d["logGFP_hat_from_TD"].to_numpy(float)
    for component, _ in _item1_components()[1:]:
        component_hat = d[f"logGFP_hat_from_{component}"].to_numpy(float)
        residual = np.full(len(d), np.nan)
        valid = np.isfinite(component_hat) & np.isfinite(td_hat)
        residual[valid] = component_hat[valid] - td_hat[valid]
        d[f"res_logGFP_{component}_minus_TD"] = residual
    return d, fits.reset_index()


def _set_decade_log_ticks(ax) -> None:
    """Label logarithmic axes only at integer powers of ten."""
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        axis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
        axis.set_minor_formatter(NullFormatter())


def _set_log10_value_ticks(ax, *, x: bool = True, y: bool = True) -> None:
    """Display stored log10 coordinates as integer-decade labels."""
    formatter = FuncFormatter(
        lambda value, _: rf"$10^{{{int(round(value))}}}$"
        if np.isclose(value, round(value))
        else ""
    )
    for enabled, axis in ((x, ax.xaxis), (y, ax.yaxis)):
        if enabled:
            axis.set_major_locator(MultipleLocator(1))
            axis.set_major_formatter(formatter)
            axis.set_minor_locator(NullLocator())


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
    _set_decade_log_ticks(ax)
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
        if x.startswith("log_"):
            _set_log10_value_ticks(ax, x=True, y=False)
        if y.startswith("log_") or y.startswith("res_log"):
            _set_log10_value_ticks(ax, x=False, y=True)
        if title is not None:
            ax.set_title(title)
        return None

    hb = _hexbin_percent(ax, xx, yy, gridsize=55, mincnt=2, cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x.startswith("log_"):
        _set_log10_value_ticks(ax, x=True, y=False)
    if y.startswith("log_") or y.startswith("res_log"):
        _set_log10_value_ticks(ax, x=False, y=True)
    if title is not None:
        ax.set_title(title)

    if annotation_mode == "full":
        _add_linear_regression_overlay(ax, xx, yy)
    elif annotation_mode == "r_only":
        _add_linear_regression_overlay_r_only(ax, xx, yy, fontsize=10.5)
    return hb

def plot_item1A_zanesco_panel(df: pd.DataFrame):
    """Retained Figure 1A: GFP versus all TD quantities, linear and log-log."""
    d = df.copy()
    d["GFP_uV"] = d["GFP"] * 1e6
    pairs = [("GFP_uV", column, "GFP (uV)", label.replace("MEED", "FORM")) for label, column in _item1_components()]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    hexbins = []
    for c, (x, y, xlabel, ylabel) in enumerate(pairs):
        hb = _plot_item1_linear_panel(axes[0, c], d, x, y, xlabel, ylabel, title=f"{ylabel} vs GFP", annotation_mode="full")
        if hb is not None:
            hexbins.append(hb)
        hb = _plot_item1_loglog_panel(axes[1, c], d, x, y, xlabel, ylabel, title="log-log", annotation_mode="full")
        if hb is not None:
            hexbins.append(hb)
    for ax in axes.ravel():
        if ax.collections:
            fig.colorbar(ax.collections[0], ax=ax, shrink=.80, label="% of samples")
    fig.suptitle("Figure 1A: GFP versus components of topographic dissimilarity", fontsize=14)
    return fig




def _add_td_component_loglog_overlay(ax, x: np.ndarray, y: np.ndarray):
    """Dashed fit and compact, publication-formatted log-log statistics."""
    fit = _loglog_regression_stats(x, y)
    if fit is None:
        return
    ax.plot(fit["xs"], fit["ys"], "--", color="black", gapcolor="white", lw=1.7, zorder=3)
    p_label = "p < 0.001" if fit["p"] < .001 else f"p = {fit['p']:.3f}"
    ax.text(0.97, 0.05, f"r = {fit['r']:.3f}\n$R^2$ = {fit['r2']:.3f}\n{p_label}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
            bbox={"facecolor": "white", "alpha": .78, "edgecolor": "none", "pad": 2.5})


def _component_hexbin_extent(df: pd.DataFrame, x_column: str, components: list[str]) -> tuple[float, float, float, float]:
    """One common log-space lattice extent so component hexbins have equal area."""
    x = df[x_column].to_numpy(float)
    y = np.concatenate([df[column].to_numpy(float) for column in components])
    x = x[np.isfinite(x) & (x > 0)]
    y = y[np.isfinite(y) & (y > 0)]
    if not len(x) or not len(y):
        raise ValueError("component hexbin grid requires positive finite values")
    lx, ly = np.log10(x), np.log10(y)
    padx = max(.02, .02 * np.ptp(lx)); pady = max(.02, .02 * np.ptp(ly))
    return float(lx.min() - padx), float(lx.max() + padx), float(ly.min() - pady), float(ly.max() + pady)


def _plot_component_hexbin_grid(df: pd.DataFrame, x_column: str, xlabel: str):
    """Four shared-axis component diagnostics with a common-area log hexbin lattice."""
    specs = [("TD_res", r"TD$_{res}$"), ("TD_MEED", r"TD$_{FORM}$"),
             ("TD_psi", r"TD$_{psi}$"), ("TD_rho", r"TD$_{rho}$")]
    extent = _component_hexbin_extent(df, x_column, [column for column, _ in specs])
    norm = Normalize(vmin=.05, vmax=1.0, clip=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2), sharex=True, sharey=True, constrained_layout=True)
    hexbins = []
    for ax, (column, label) in zip(axes.flat, specs):
        x, y = _finite_xy(df, x_column, column)
        valid = (x > 0) & (y > 0)
        if not np.any(valid):
            ax.text(.5, .5, "no valid data", transform=ax.transAxes, ha="center", va="center")
            continue
        hb = _hexbin_percent(ax, x[valid], y[valid], gridsize=58, mincnt=2, cmap="viridis", norm=norm,
                             extent=extent, xscale="log", yscale="log", linewidths=0)
        hexbins.append(hb)
        _add_td_component_loglog_overlay(ax, x[valid], y[valid])
        ax.set(xscale="log", yscale="log", xlim=(10 ** extent[0], 10 ** extent[1]),
               ylim=(10 ** extent[2], 10 ** extent[3]), xlabel=xlabel, ylabel=label)
        _set_decade_log_ticks(ax)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=9)
    if len(hexbins) >= 2:
        colorbar = fig.colorbar(hexbins[0], ax=list(axes[0]), shrink=.82, label="% of samples")
        colorbar.set_ticks([.05, .1, .2, .5, 1])
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_recording_td_component_hexbin(df: pd.DataFrame):
    """Figure 1E: total TD versus its four components."""
    return _plot_component_hexbin_grid(df, "TD", "TD")


def plot_recording_gfp_component_hexbin(df: pd.DataFrame):
    """Figure 1F: GFP versus the four TD components, matching Figure 1E."""
    d = df.copy()
    d["GFP_uV"] = d["GFP"] * 1e6
    return _plot_component_hexbin_grid(d, "GFP_uV", "GFP (µV)")


def plot_item1B_peak_overlay(df: pd.DataFrame):
    d = df.copy()
    d["GFP_uV"] = d["GFP"] * 1e6
    peak_col = "is_peak" if "is_peak" in d.columns else "local_GFP_peak"
    peak_mask = d[peak_col].fillna(False).astype(bool).to_numpy()
    pairs = [("GFP_uV", column, "GFP (uV)", label.replace("MEED", "FORM")) for label, column in _item1_components()]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    hexbins = []
    for c, (x, y, xl, yl) in enumerate(pairs):
        xx, yy = _finite_xy(d, x, y)
        hb = _hexbin_percent(axes[0, c], xx, yy, gridsize=55, mincnt=2, cmap="bone", linewidths=0)
        hexbins.append(hb)
        axes[0, c].scatter(d.loc[peak_mask, x], d.loc[peak_mask, y], s=5, alpha=0.35, color="#fde725")
        axes[0, c].set_xlabel(xl)
        axes[0, c].set_ylabel(yl)
        axes[0, c].set_title(f"{yl} vs GFP")
        _add_linear_regression_overlay(axes[0, c], xx, yy)

        m = np.isfinite(d[x].to_numpy(float)) & np.isfinite(d[y].to_numpy(float)) & (d[x].to_numpy(float) > 0) & (d[y].to_numpy(float) > 0)
        hb = _hexbin_percent(axes[1, c], d.loc[m, x], d.loc[m, y], gridsize=55, mincnt=2, cmap="bone", linewidths=0, xscale="log", yscale="log")
        hexbins.append(hb)
        pm = peak_mask & m
        axes[1, c].scatter(d.loc[pm, x], d.loc[pm, y], s=5, alpha=0.35, color="#fde725")
        axes[1, c].set_xlabel(xl)
        axes[1, c].set_ylabel(yl)
        axes[1, c].set_title("log-log")
        _add_loglog_regression_overlay(axes[1, c], d.loc[m, x].to_numpy(float), d.loc[m, y].to_numpy(float))
    _set_hexbin_average_clim(hexbins)
    for ax in axes.ravel():
        if ax.collections:
            fig.colorbar(ax.collections[0], ax=ax, shrink=0.80, label="% of samples")
    fig.suptitle("Figure 1B: GFP components with local GFP-peak overlays", fontsize=14)
    return fig


def plot_item1C_td_decomposition(df: pd.DataFrame):
    """Show where each component's GFP-implied value diverges from total TD."""
    d, _ = compute_item1_gfp_link_residuals(df)
    panel_side_inches = 10.0 / 2.54
    figure_side_inches = 4 * panel_side_inches
    fig, axes = plt.subplots(
        4, 4, figsize=(figure_side_inches, figure_side_inches), constrained_layout=True
    )
    fig.patch.set_alpha(0)
    components = ["TD_MEED", "TD_psi", "TD_rho", "TD_res"]
    hexbins = []
    for row, component in enumerate(components):
        residual = f"res_logGFP_{component}_minus_TD"
        specs = [
            ("loglog", "TD", component, "TD", component),
            ("linear", "log_GFP_uV", residual, "GFP (uV)", f"implied GFP: {component} - TD"),
            ("linear", "log_F_MEED_TD", residual, "F_MEED_TD", f"implied GFP: {component} - TD"),
            ("linear", "log_rho_pair2", residual, r"$\rho_{pair}^2$", f"implied GFP: {component} - TD"),
        ]
        for axis, (kind, x, y, xlabel, ylabel) in zip(axes[row], specs):
            if kind == "loglog":
                hb = _plot_item1_loglog_panel(axis, d, x, y, xlabel, ylabel, annotation_mode="full")
            else:
                hb = _plot_item1_linear_panel(
                    axis, d, x, y, xlabel, ylabel, add_zero_line=True, annotation_mode="full"
                )
            if hb is not None:
                hexbins.append(hb)
    _set_hexbin_average_clim(hexbins)
    for axis in axes[:, -1]:
        if axis.collections:
            fig.colorbar(axis.collections[0], ax=axis, shrink=0.78, label="% of samples")
    for axis in fig.axes:
        axis.patch.set_alpha(0)
        axis.tick_params(labelsize=14)
        axis.xaxis.label.set_size(14)
        axis.yaxis.label.set_size(14)
        axis.title.set_size(14)
        for text in axis.texts:
            text.set_fontsize(14)
    fig.suptitle("Figure 1C: component-specific GFP-link residuals versus TD", fontsize=14)
    return fig


def table1_subject_summary(df: pd.DataFrame, sub: str) -> pd.DataFrame:
    residual_df, fit_table = compute_item1_gfp_link_residuals(df)
    fits = fit_table.set_index("component")
    row = {
        "sub": sub,
        "median_GFP": float(df["GFP"].median()),
        "median_TD": float(df["TD"].median()),
        "median_TD_MEED": float(df["TD_MEED"].median()),
        "median_TD_res": float(df["TD_res"].median()),
        "median_TD_rho": float(df["TD_rho"].median()),
        "median_TD_psi": float(df["TD_psi"].median()),
        "median_F_MEED_TD": float(df["F_MEED_TD"].median()),
        "median_F_res": float(df["F_res"].median()),
        "median_F_rho": float(df["F_rho"].median()),
        "median_F_psi": float(df["F_psi"].median()),
        "median_abs_TD_closure_error": float(np.abs(df["TD_decomposition_closure_error"]).median()),
        "median_abs_TD_MEED_internal_closure_error": float(np.abs(df["TD_MEED_internal_closure_error"]).median()),
        "max_abs_TD_MEED_internal_closure_error": float(np.abs(df["TD_MEED_internal_closure_error"]).max()),
        "median_abs_TD_threeway_closure_error": float(np.abs(df["TD_threeway_closure_error"]).median()),
        "max_abs_TD_threeway_closure_error": float(np.abs(df["TD_threeway_closure_error"]).max()),
        "median_abs_fraction_closure_error": float(np.abs(df["fraction_closure_error"]).median()),
        "median_abs_F_MEED_component_error": float(np.abs(df["F_MEED_component_error"]).median()),
    }
    for component, _ in _item1_components():
        fit = fits.loc[component]
        row.update({
            f"n_valid_{component}_loglog_GFP": int(fit["n_valid"]),
            f"alpha_{component}_loglog_GFP": float(fit["alpha"]),
            f"beta_{component}_loglog_GFP": float(fit["beta"]),
            f"r_{component}_loglog_GFP": float(fit["r"]),
            f"R2_{component}_loglog_GFP": float(fit["R2"]),
        })
    for component in ("TD_MEED", "TD_psi", "TD_rho", "TD_res"):
        residual = residual_df[f"res_logGFP_{component}_minus_TD"]
        residual_fit = _fit_linear_model(
            residual_df["log_F_MEED_TD"].to_numpy(float), residual.to_numpy(float)
        )
        row.update({
            f"median_res_logGFP_{component}_minus_TD": float(residual.median()),
            f"IQR_res_logGFP_{component}_minus_TD": float(
                residual.quantile(0.75) - residual.quantile(0.25)
            ),
            f"slope_res_logGFP_{component}_minus_TD_vs_log_F_MEED": float(residual_fit["slope"]),
            f"r_res_logGFP_{component}_minus_TD_vs_log_F_MEED": float(residual_fit["r"]),
            f"R2_res_logGFP_{component}_minus_TD_vs_log_F_MEED": float(residual_fit["R2"]),
        })
    return pd.DataFrame([row])


def table1_decomposition_qc(df: pd.DataFrame, sub: str) -> pd.DataFrame:
    """Keep exact decomposition checks separate from the empirical GFP analysis."""
    checks = {
        "TD_two_way": df["TD_decomposition_closure_error"],
        "TD_MEED_internal": df["TD_MEED_internal_closure_error"],
        "TD_threeway": df["TD_threeway_closure_error"],
        "fraction": df["fraction_closure_error"],
        "MEED_fraction": df["F_MEED_component_error"],
    }
    row = {"sub": sub}
    for name, values in checks.items():
        absolute = np.abs(values.to_numpy(float))
        absolute = absolute[np.isfinite(absolute)]
        row[f"median_abs_{name}_closure_error"] = float(np.median(absolute)) if len(absolute) else np.nan
        row[f"max_abs_{name}_closure_error"] = float(np.max(absolute)) if len(absolute) else np.nan
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
    fig.text(0.5, 0.935, "Figure 1D: low vs high FORM transition fractions (t and t+1)", ha="center", va="top", fontsize=11)

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
                    extrapolate="local",
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
    fig.text(0.5, 0.95, "Exemplary FORM transitions: t and t+1", ha="center", va="top", fontsize=11)

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
                    extrapolate="local",
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
                extrapolate="local",
            )
            ax.set_title(
                f"s={idx} t={float(row['time']):.1f}\nzg={float(row['z_gfp']):.1f} zr={float(row['z_rho']):.1f} zm={float(row['z_maxcorr']):.1f}",
                fontsize=6.6,
                pad=2.0,
            )

    return fig


def save_exemplary_subject_plots(sub: str, raw: mne.io.BaseRaw, sample_df: pd.DataFrame, subject_dir: str | Path):
    outdir = Path(subject_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_figure(
        plot_exemplary_f_meed_transition_pairs(raw, sample_df, sub=sub),
        outdir / "figure_S3_exemplary_f_meed_transition_pairs.svg",
        tight=False,
    )
    save_figure(
        plot_exemplary_gfp_rho_maxcorr_cases(raw, sample_df, sub=sub),
        outdir / "figure_S4_exemplary_gfp_rho_winning_corr_cases.svg",
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
    fig, axes = plt.subplots(1, 5, figsize=(21.5, 3.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.24, top=0.87, wspace=0.34)

    if sub:
        fig.text(0.012, 0.965, str(sub), ha="left", va="top", fontsize=9)

    specs = [
        ("loglog", "GFP_uV", "TD", "GFP (uV)", "TD"),
        ("loglog", "GFP_uV", "TD_MEED", "GFP (uV)", "TD_FORM"),
        ("loglog", "GFP_uV", "TD_psi", "GFP (uV)", "TD_psi"),
        ("linear", "log_F_MEED_TD", "res_logGFP_TD_psi_minus_TD", "FORM transition fraction", "implied GFP: TD_psi - TD"),
        ("linear", "log_F_MEED_TD", "res_logGFP_TD_MEED_minus_TD", "FORM transition fraction", "implied GFP: TD_FORM - TD"),
    ]

    hexbins = []
    for ax, (kind, x, y, xlabel, ylabel) in zip(axes, specs):
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        if kind == "loglog":
            hb = _plot_item1_loglog_panel(ax, d, x, y, xlabel, ylabel, annotation_mode="r_only")
        else:
            hb = _plot_item1_linear_panel(
                ax, d, x, y, xlabel, ylabel, add_zero_line=True, annotation_mode="r_only"
            )
        if hb is not None:
            hexbins.append(hb)

    _set_hexbin_average_clim(hexbins)

    return fig


def _map_meed_rho2(maps: np.ndarray, labels: list[str], meta_level: TemplateLevel) -> np.ndarray:
    level = build_template_level("map_fit", maps, labels, meta_level.info)
    return np.sum(level.Q * level.Q, axis=0)


def _validate_condition_to_group_matching(
    raw_maps: np.ndarray,
    canonical_maps: np.ndarray,
    matching: pd.DataFrame,
    pooled_maps: np.ndarray,
    pooled_labels: list[str],
) -> None:
    """Prove raw→pooled-A-E placement and canonical identity rematch to pooled maps."""
    required = {"raw_map_index", "pooled_target_index", "polarity_multiplier", "canonical_output_index"}
    if required.difference(matching.columns):
        raise ValueError("Condition-to-group matching is missing required permutation columns")
    matching = matching.sort_values("canonical_output_index").reset_index(drop=True)
    expected = np.arange(raw_maps.shape[1])
    if not np.array_equal(matching["canonical_output_index"].to_numpy(int), expected):
        raise AssertionError("Condition matching does not cover canonical pooled A-E exactly once")
    for row in matching.itertuples(index=False):
        expected_map = raw_maps[:, row.raw_map_index] * row.polarity_multiplier
        if not np.allclose(canonical_maps[:, row.canonical_output_index], expected_map, atol=1e-6, rtol=1e-6):
            raise AssertionError("Canonical condition map is not the documented raw map times its polarity")
    rematched, rematching = match_group_maps_to_meta(
        canonical_maps.T, pooled_maps, pooled_labels, raw_labels=list(pooled_labels)
    )
    identity = rematching.sort_values("canonical_output_index")["raw_map_index"].to_numpy(int)
    if not np.array_equal(identity, expected) or not np.allclose(rematched, mvu.center_l2_columns(canonical_maps), atol=1e-6, rtol=1e-6):
        raise AssertionError("Rematching canonical condition maps to pooled A-E must return identity [0,1,2,3,4]")


def plot_condition_to_group_matching_diagnostic(
    condition_maps: dict[str, np.ndarray],
    matching: pd.DataFrame,
    pooled_maps: np.ndarray,
    pooled_labels: list[str],
    info: mne.Info,
):
    """Debug-only hierarchy audit: EC final A-E | pooled A-E | EO final A-E."""
    figure, axes = plt.subplots(len(pooled_labels), 3, figsize=(u.PAPER_WIDTH_IN, 8.0), layout="constrained")
    for row, label in enumerate(pooled_labels):
        class_color = u.STATE_COLORS[label]
        for column, condition in ((0, "EC"), (2, "EO")):
            match = matching.loc[(matching.condition == condition) & (matching.pooled_target_label == label)].iloc[0]
            axis = axes[row, column]
            mne.viz.plot_topomap(condition_maps[condition][:, row], info, axes=axis, show=False,
                                 sensors=False, contours=0, cmap="RdBu_r", sphere="auto")
            axis.set_title(f"{condition} {label} ← {match.raw_map_label}", color=class_color, fontsize=10)
        middle = axes[row, 1]
        mne.viz.plot_topomap(pooled_maps[:, row], info, axes=middle, show=False,
                             sensors=False, contours=0, cmap="RdBu_r", sphere="auto")
        middle.set_title(f"Pooled {label}", color=class_color, fontsize=10)
        for left, right in ((axes[row, 0], middle), (middle, axes[row, 2])):
            figure.add_artist(ConnectionPatch((1, .5), (0, .5), coordsA=left.transAxes, coordsB=right.transAxes,
                                              axesA=left, axesB=right, color=class_color, linewidth=.8, alpha=.8))
    return figure


def plot_condition_group_raw_maps(raw_maps: dict[str, np.ndarray], info: mne.Info):
    """Debug-only raw EC/EO maps before any pooled-label inheritance."""
    figure, axes = plt.subplots(5, 2, figsize=(4.0, 8.0), layout="constrained")
    for row in range(5):
        for column, condition in enumerate(("EC", "EO")):
            axis = axes[row, column]
            mne.viz.plot_topomap(raw_maps[condition][:, row], info, axes=axis, show=False,
                                 sensors=False, contours=0, cmap="RdBu_r", sphere="auto")
            axis.set_title(f"{condition.lower()}{row + 1}", fontsize=10)
    return figure


def _absolute_spatial_correlation(first_maps: np.ndarray, second_maps: np.ndarray) -> np.ndarray:
    return np.abs(mvu.center_l2_columns(first_maps).T @ mvu.center_l2_columns(second_maps))


def compute_group_map_meed_fit_table(
    records,
    meta_level: TemplateLevel,
    group_maps_file: str | Path,
    *,
    selected_k: int,
    n_init: int,
    max_iter: int,
    tol: float,
    random_seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, np.ndarray, list[str]]:
    """Fit EC/EO groups independently and inherit A-E labels from pooled maps only."""
    with np.load(group_maps_file, allow_pickle=False) as cache:
        pooled_maps = cache["group_maps"].astype(float)
        pooled_labels = cache["labels"].astype(str).tolist()
    if tuple(pooled_labels) != u.CANONICAL_LABELS:
        raise RuntimeError(f"Pooled group maps must already be canonical A-E, got {pooled_labels}")

    individual_rows = []
    maps_by_condition: dict[str, list[np.ndarray]] = {"EC": [], "EO": []}
    for record in records:
        condition = str(record["condition"])
        if condition not in maps_by_condition:
            continue
        model_file = Path(record["model_dir"]) / f"k-{int(selected_k):02d}_modkmeans.fif"
        if not model_file.exists():
            continue
        model = read_cluster_model(model_file)
        cached_centers = np.asarray(model.cluster_centers_, dtype=float)
        # Subject-condition maps inherit pooled A-E identity; they never target Koenig directly.
        maps, _ = match_group_maps_to_meta(cached_centers, pooled_maps, pooled_labels)
        maps_by_condition[condition].append(cached_centers.T)
        for label, rho2 in zip(pooled_labels, _map_meed_rho2(maps, pooled_labels, meta_level)):
            individual_rows.append({"map_source": "subject_condition", "recording_id": str(record["recording_id"]),
                                    "condition": condition, "label": label, "meed_rho2": float(rho2)})

    condition_rows, condition_maps, raw_condition_maps, matching_tables = [], {}, {}, []
    for condition, maps_list in maps_by_condition.items():
        if not maps_list:
            continue
        model = fit_modkmeans(make_chdata(np.column_stack(maps_list), meta_level.info), n_clusters=int(selected_k),
                              n_init=int(n_init), max_iter=int(max_iter), tol=float(tol),
                              random_state=stable_seed(random_seed, "condition_group", condition), n_jobs=int(n_jobs))
        raw_maps = np.asarray(model.cluster_centers_, dtype=float).T
        raw_labels = [f"{condition.lower()}{index + 1}" for index in range(raw_maps.shape[1])]
        _, matching = match_group_maps_to_meta(raw_maps.T, pooled_maps, pooled_labels, raw_labels=raw_labels)
        matching = matching.rename(columns={"meta_index": "pooled_target_index", "meta_label": "pooled_target_label"})
        matching.insert(0, "condition", condition)
        maps = np.empty_like(raw_maps)
        for row in matching.itertuples(index=False):
            maps[:, row.pooled_target_index] = raw_maps[:, row.raw_map_index] * row.polarity_multiplier
        _validate_condition_to_group_matching(raw_maps, maps, matching, pooled_maps, pooled_labels)
        condition_maps[condition], raw_condition_maps[condition] = maps, raw_maps
        matching_tables.append(matching)
        for label, rho2 in zip(pooled_labels, _map_meed_rho2(maps, pooled_labels, meta_level)):
            condition_rows.append({"map_source": "condition_group", "recording_id": "condition_group",
                                   "condition": condition, "label": label, "meed_rho2": float(rho2)})

    if set(condition_maps) != {"EC", "EO"}:
        raise RuntimeError("Condition-group matching requires independently fitted EC and EO maps")
    matching_table = pd.concat(matching_tables, ignore_index=True)
    print("pooled vs Koenig |r|:\n", _absolute_spatial_correlation(pooled_maps, meta_level.B))
    print("EC raw vs pooled A-E |r|:\n", _absolute_spatial_correlation(pooled_maps, raw_condition_maps["EC"]))
    print("EO raw vs pooled A-E |r|:\n", _absolute_spatial_correlation(pooled_maps, raw_condition_maps["EO"]))
    group_rows = [{"map_source": "pooled_group", "recording_id": "pooled_group", "condition": "all",
                   "label": label, "meed_rho2": float(rho2)}
                  for label, rho2 in zip(pooled_labels, _map_meed_rho2(pooled_maps, pooled_labels, meta_level))]
    return (pd.DataFrame(individual_rows + condition_rows + group_rows), condition_maps, raw_condition_maps,
            matching_table, pooled_maps, pooled_labels)


def plot_group_map_fit_r2_summary(map_fit_df: pd.DataFrame, *, condition_maps: dict[str, np.ndarray] | None = None,
                                  info: mne.Info | None = None):
    """F06: EC/EO in-axis topography pairs over fixed-scale FORM-conformity boxes."""
    d = map_fit_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["meed_rho2", "label"])
    individual = d.loc[d.map_source.eq("subject_condition")].copy()
    condition_group = d.loc[d.map_source.eq("condition_group")].copy()
    pooled_group = d.loc[d.map_source.eq("pooled_group")].copy()
    labels = [label for label in u.CANONICAL_LABELS if label in set(d.label)]
    has_maps = condition_maps is not None and info is not None and all(condition in condition_maps for condition in ("EC", "EO"))
    fig, axis = plt.subplots(figsize=(u.PAPER_WIDTH_IN, 3.5), layout="constrained")
    positions = {}
    for index, label in enumerate(labels):
        for condition, offset in (("EC", -0.19), ("EO", 0.19)):
            position = index + offset; positions[label, condition] = position
            values = individual.loc[individual.label.eq(label) & individual.condition.eq(condition), "meed_rho2"].to_numpy(float)
            if not len(values): continue
            artists = axis.boxplot([values], positions=[position], widths=.32, patch_artist=True,
                                  showfliers=False, manage_ticks=False,
                                  medianprops={"color": "black", "linewidth": .8})
            artists["boxes"][0].set(facecolor=u.STATE_COLORS[label], edgecolor="0.25",
                                    hatch=u.CONDITION_HATCH[condition], linewidth=.7)
    for _, row in condition_group.iterrows():
        axis.scatter(positions[row.label, row.condition], row.meed_rho2, marker="D", s=34,
                     facecolors="white", edgecolors=u.STATE_COLORS[row.label], linewidths=1.1, zorder=5)
    for _, row in pooled_group.iterrows():
        axis.scatter(labels.index(row.label), row.meed_rho2, marker="D", s=34,
                     color=u.STATE_COLORS[row.label], edgecolors="black", linewidths=.6, zorder=6)
    axis.set(xticks=range(len(labels)), xticklabels=labels, ylabel=r"FORM conformity $\rho^2$", xlim=(-.55, len(labels)-.45), ylim=(.625, 1.025))
    for tick in axis.get_xticklabels():
        tick.set_color(u.STATE_COLORS[tick.get_text()]); tick.set_fontweight("bold")
    if has_maps:
        values = np.concatenate([np.asarray(condition_maps[c], float).ravel() for c in ("EC", "EO")])
        vmax = float(np.nanmax(np.abs(values)))
        # Compact EC/EO pairs live inside the y=0.625..1.025 data panel.
        for index, label in enumerate(labels):
            x = (index + .5) / len(labels)
            for col, condition in enumerate(("EC", "EO")):
                map_axis = axis.inset_axes([x - .075 + col * .075, .57, .07, .20], transform=axis.transAxes)
                mne.viz.plot_topomap(condition_maps[condition][:, index], info, axes=map_axis, show=False,
                                     sensors=False, contours=0, cmap="RdBu_r", vlim=(-vmax, vmax), sphere="auto")
                for spine in map_axis.spines.values():
                    spine.set_edgecolor(u.STATE_COLORS[label]); spine.set_linewidth(.7)
    u.set_publication_spines(axis, categorical=True)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.25", hatch="///", label="EC subject maps"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.25", label="EO subject maps"),
               Line2D([], [], marker="D", markerfacecolor="white", markeredgecolor="0.2", linestyle="", label="condition group"),
               Line2D([], [], marker="D", color="0.2", linestyle="", label="pooled group")]
    axis.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=10, columnspacing=1.0)
    return fig


def save_group_map_fit_r2_outputs(
    processed_outputs,
    outdir: str | Path,
    *,
    meta_json: str | Path,
    group_maps_file: str | Path,
    selected_k: int,
    n_init: int,
    max_iter: int,
    tol: float,
    random_seed: int,
    n_jobs: int,
):
    outdir = Path(outdir)
    group_dir = outdir
    group_dir.mkdir(parents=True, exist_ok=True)
    records = processed_outputs.to_dict("records") if isinstance(processed_outputs, pd.DataFrame) else list(processed_outputs)
    meta_level, _ = load_meta_level(meta_json, int(selected_k))
    group_df, condition_maps, raw_condition_maps, condition_matching, pooled_maps, pooled_labels = compute_group_map_meed_fit_table(
        records,
        meta_level,
        group_maps_file,
        selected_k=selected_k,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_seed=random_seed,
        n_jobs=n_jobs,
    )
    save_table(group_df, group_dir / "group_map_fit_r2.csv")
    np.savez_compressed(
        group_dir / "condition_group_maps_raw.npz",
        EC_raw_maps=raw_condition_maps["EC"].astype(np.float32), EO_raw_maps=raw_condition_maps["EO"].astype(np.float32),
        EC_raw_labels=np.asarray([f"ec{index}" for index in range(1, raw_condition_maps["EC"].shape[1] + 1)], dtype="U"),
        EO_raw_labels=np.asarray([f"eo{index}" for index in range(1, raw_condition_maps["EO"].shape[1] + 1)], dtype="U"),
        ch_names=np.asarray(meta_level.info.ch_names, dtype="U"),
    )
    required_matching_columns = ["condition", "raw_map_index", "raw_map_label", "pooled_target_index", "pooled_target_label",
                                 "signed_spatial_correlation", "absolute_spatial_correlation", "polarity_multiplier",
                                 "canonical_output_index"]
    condition_matching.loc[:, required_matching_columns].to_csv(group_dir / "condition_to_group_matching.csv", index=False)
    np.savez_compressed(
        group_dir / "condition_group_maps_A-E.npz",
        EC=condition_maps["EC"].astype(np.float32), EO=condition_maps["EO"].astype(np.float32),
        EC_maps_A_E=condition_maps["EC"].astype(np.float32), EO_maps_A_E=condition_maps["EO"].astype(np.float32),
        labels=np.asarray(meta_level.labels, dtype="U"), ch_names=np.asarray(meta_level.info.ch_names, dtype="U"),
    )
    with np.load(group_dir / "condition_group_maps_raw.npz", allow_pickle=False) as raw_cache, \
         np.load(group_dir / "condition_group_maps_A-E.npz", allow_pickle=False) as canonical_cache:
        saved_matching = pd.read_csv(group_dir / "condition_to_group_matching.csv")
        saved_condition_maps = {condition: canonical_cache[condition].astype(float) for condition in ("EC", "EO")}
        for condition in ("EC", "EO"):
            _validate_condition_to_group_matching(raw_cache[f"{condition}_raw_maps"].astype(float),
                                                  saved_condition_maps[condition],
                                                  saved_matching.loc[saved_matching.condition.eq(condition)],
                                                  pooled_maps, pooled_labels)
    diagnostic = plot_condition_to_group_matching_diagnostic(saved_condition_maps, saved_matching, pooled_maps, pooled_labels, meta_level.info)
    diagnostic.savefig(group_dir / "figure_condition_to_group_matching_diagnostic.svg", format="svg", bbox_inches=None)
    plt.close(diagnostic)
    raw_diagnostic = plot_condition_group_raw_maps(raw_condition_maps, meta_level.info)
    raw_diagnostic.savefig(group_dir / "figure_condition_group_raw_maps.svg", format="svg", bbox_inches=None)
    plt.close(raw_diagnostic)
    save_figure(plot_group_map_fit_r2_summary(group_df, condition_maps=saved_condition_maps, info=meta_level.info), group_dir / "figure_S2_group_map_fit_r2.png")
    figure = plot_group_map_fit_r2_summary(group_df, condition_maps=saved_condition_maps, info=meta_level.info)
    figure.savefig(group_dir / "figure_lemon_conformity.svg", format="svg", bbox_inches=None)
    u.assert_svg_width(group_dir / "figure_lemon_conformity.svg")
    plt.close(figure)
    (group_dir / "figure_lemon_conformity_provenance.json").write_text(json.dumps({
        "metric": "rho_squared", "subject_maps": "fixed K=5 matched per recording",
        "condition_group_maps": "independently EC/EO-fitted raw maps matched and reordered to pooled canonical A-E",
        "raw_maps": "condition_group_maps_raw.npz; pre-matching raw order ec1..ec5 / eo1..eo5",
        "matching": "condition_to_group_matching.csv; raw→pooled Hungarian pairs and polarity",
        "pooled_group_maps": "Stage-B selected-K pool"
    }, indent=2) + "\n")
    return group_df


# ---------------------------------------------------------------------
# Item 3: ML-style predictability and uncertainty analysis
# ---------------------------------------------------------------------


def _item3_ml_core_target_specs():
    return [
        ("winning_corr", "Winning corr"),
    ]


def _item3_ml_main_predictor_specs():
    return [
        ("GFP", ["GFP_model"]),
        ("rho", ["rho"]),
        ("GFP+rho", ["GFP_model", "rho"]),
    ]


def _item3_ml_exploratory_predictor_specs():
    return [
        ("GFP+rho+d_xyz", ["GFP_model", "rho", "d_x", "d_y", "d_z"]),
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
ITEM3_USE_LOG_GFP = False
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
    if "valid_topography" not in df:
        raise ValueError("Item 3 requires the Stage-C valid_topography mask")
    d = df.loc[df["valid_topography"].astype(bool)].copy()
    d["GFP_uV"] = d["GFP"] * 1e6
    d["GFP_model"] = d["GFP_uV"]
    if ITEM3_USE_LOG_GFP:
        valid_gfp = np.isfinite(d["GFP_model"]) & (d["GFP_model"] > 0)
        d.loc[valid_gfp, "GFP_model"] = np.log10(d.loc[valid_gfp, "GFP_model"])
        d.loc[~valid_gfp, "GFP_model"] = np.nan

    corr_prefix = "corr_selected_" if any(col.startswith("corr_selected_") for col in d) else "corr_group_"
    group_corr_cols = sorted(col for col in d.columns if col.startswith(corr_prefix))
    psi_meta_cols = sorted(col for col in d.columns if col.startswith("psi_meta_"))

    n = len(d)
    d["winning_corr"] = np.nan
    d["second_max_group_corr"] = np.nan
    d["corr_margin"] = np.nan
    d["relative_corr_margin"] = np.nan
    d["corr_entropy"] = np.nan
    d["dipole_angle_gap"] = np.nan

    if group_corr_cols:
        C = np.abs(d[group_corr_cols].to_numpy(float))
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
        d["max_group_corr"] = d["winning_corr"]
        d["second_max_group_corr"] = _clip01(second)
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
    corr_prefix = "corr_selected_" if any(col.startswith("corr_selected_") for col in d) else "corr_group_"
    group_corr_cols = sorted(col for col in d.columns if col.startswith(corr_prefix))
    target_specs = []
    if not group_corr_cols:
        return d, target_specs

    C = np.abs(d[group_corr_cols].to_numpy(float))
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


def plot_item3_microstate_likeness(df: pd.DataFrame, sub: str | None = None):
    """Show GFP and FORM dipolarity as separate explanations of map likeness."""
    d = _prepare_item3_ml_dataframe(df)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["GFP_model", "rho", "winning_corr"]
    )
    d = _item3_ml_analysis_subset(d)
    if d.empty:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "no valid microstate-likeness samples", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return fig

    y = d["winning_corr"].to_numpy(float)
    gfp = d[["GFP_model"]].to_numpy(float)
    rho = d[["rho"]].to_numpy(float)
    gfp_z, _ = _item3_standardize_train_test(gfp, gfp)
    rho_z, _ = _item3_standardize_train_test(rho, rho)
    gfp_prediction = _item3_predict_from_fit("linear", _item3_fit_linear_model(gfp_z, y), gfp_z)
    rho_prediction = _item3_predict_from_fit("linear", _item3_fit_linear_model(rho_z, y), rho_z)
    gfp_metrics = _item3_regression_metrics(y, gfp_prediction)
    rho_metrics = _item3_regression_metrics(y, rho_prediction)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    title = "FORM conformity and microstate-likeness"
    if sub is not None:
        title = f"{sub}: {title}"
    fig.suptitle(title, fontsize=14)

    x_label = "log10 GFP (µV)" if ITEM3_USE_LOG_GFP else "GFP (µV)"
    color_points = axes[0].scatter(
        d["GFP_model"], y, c=d["rho"], cmap="viridis", vmin=0.0, vmax=1.0,
        s=14, alpha=0.75, linewidths=0, rasterized=True,
    )
    order = np.argsort(d["GFP_model"].to_numpy(float))
    axes[0].plot(d["GFP_model"].to_numpy(float)[order], gfp_prediction[order], color="black", lw=1.7)
    axes[0].set(
        xlabel=x_label, ylabel="winning map correlation",
        title=f"GFP only: R² = {gfp_metrics['R2']:.2f}", ylim=(0.0, 1.02),
    )
    colorbar = fig.colorbar(color_points, ax=axes[0], pad=0.02)
    colorbar.set_label("FORM conformity \u03c1")

    axes[1].scatter(d["rho"], y, color="#009e73", s=14, alpha=0.65, linewidths=0, rasterized=True)
    order = np.argsort(d["rho"].to_numpy(float))
    axes[1].plot(d["rho"].to_numpy(float)[order], rho_prediction[order], color="black", lw=1.7)
    axes[1].plot([0.0, 1.0], [0.0, 1.0], color="0.25", ls="--", lw=1.4, label=r"$r_{max}=\rho$")
    axes[1].set(
        xlabel="FORM conformity \u03c1", ylabel="winning map correlation",
        title=f"ρ only: R² = {rho_metrics['R2']:.2f}", xlim=(0.0, 1.0), ylim=(0.0, 1.02),
    )
    axes[1].legend(frameon=False, loc="lower right")
    return fig

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
    max_fit_rows: int | None = None,
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


def _item3_loso_linear_fit_and_score(
    prepared: dict[str, pd.DataFrame],
    held_out_sub: str,
    *,
    target: str,
    predictors: list[str],
    max_fit_rows: int | None = None,
):
    """Fit the existing LOSO OLS rule without forming a cohort dataframe."""
    cols = [target] + predictors
    train_parts = []
    for sub in sorted(prepared):
        if sub == held_out_sub:
            continue
        d = prepared[sub][cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d):
            train_parts.append((
                d[predictors].to_numpy(float),
                d[target].to_numpy(float),
            ))

    test = prepared[held_out_sub][cols].replace([np.inf, -np.inf], np.nan).dropna()
    X_test = test[predictors].to_numpy(float)
    y_test = test[target].to_numpy(float)
    n_train = int(sum(len(y) for _, y in train_parts))
    min_rows = max(len(predictors) + 3, 10)
    if n_train < min_rows or len(y_test) < 5:
        return {
            "n_fit": n_train,
            "n_eval": int(len(y_test)),
            "r": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "fit_success": False,
            "y_true": np.array([], dtype=float),
            "y_pred": np.array([], dtype=float),
        }

    sum_x = sum((X.sum(axis=0) for X, _ in train_parts), start=np.zeros(len(predictors)))
    sum_x2 = sum((np.square(X).sum(axis=0) for X, _ in train_parts), start=np.zeros(len(predictors)))
    mean_x = sum_x / n_train
    sd_x = np.sqrt(np.maximum(sum_x2 / n_train - np.square(mean_x), 0.0))
    sd_x[~np.isfinite(sd_x) | (sd_x < 1e-12)] = 1.0

    selected = _item3_even_subsample_idx(n_train, max_fit_rows)
    X_fit_parts = []
    y_fit_parts = []
    offset = 0
    for X, y in train_parts:
        local = selected[(selected >= offset) & (selected < offset + len(y))] - offset
        if len(local):
            X_fit_parts.append(X[local])
            y_fit_parts.append(y[local])
        offset += len(y)
    X_fit = np.concatenate(X_fit_parts, axis=0)
    y_fit = np.concatenate(y_fit_parts)
    fit = _item3_fit_linear_model((X_fit - mean_x) / sd_x, y_fit)
    if not np.all(np.isfinite(fit["coef"])) or not np.isfinite(fit["intercept"]):
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

    y_pred = _item3_predict_from_fit("linear", fit, (X_test - mean_x) / sd_x)
    metrics = _item3_regression_metrics(y_test, y_pred)
    return {
        "n_fit": int(len(X_fit)),
        "n_eval": int(metrics["n_eval"]),
        "r": metrics["r"],
        "R2": metrics["R2"],
        "RMSE": metrics["RMSE"],
        "fit_success": True,
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
        for target, label in target_specs:
            for predictor_set, predictors in predictor_specs:
                for model_family, model_label in _item3_ml_model_specs():
                    if model_family != "linear":
                        raise ValueError(f"Unsupported Item 3 LOSO model family: {model_family}")
                    res = _item3_loso_linear_fit_and_score(
                        prepared,
                        held_out_sub,
                        target=target,
                        predictors=predictors,
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
    carry_cols = [col for col in ["sub", "condition", "recording_id", "sample", "time"] if col in test_df.columns]
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

    X_fit = X_train_z
    y_fit = y_train

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

        merge_cols = [col for col in ["sub", "condition", "recording_id", "sample", "time"] if col in winning_df.columns and col in confidence_df.columns]
        merged = winning_df.merge(confidence_df, on=merge_cols, how="inner")
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
            "condition",
            "recording_id",
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
            "condition",
            "recording_id",
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
        eeg_map = {str(row.get("recording_id", row.get("sub"))): str(row["eeg_path"]) for row in records if row.get("eeg_path")}
        key_col = "recording_id" if "recording_id" in d.columns else "sub"
        d["eeg_path"] = d[key_col].astype(str).map(eeg_map)
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
            "condition",
            "recording_id",
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
        recording_key = str(getattr(row, "recording_id", row.sub))
        if recording_key not in raw_cache:
            raw_cache[recording_key] = _load_raw_for_topomap_examples(row.eeg_path)
        eeg = np.asarray(raw_cache[recording_key].get_data(picks="eeg"), dtype=float)
        sample_idx = int(row.sample)
        if 0 <= sample_idx < eeg.shape[1]:
            vec = eeg[:, sample_idx]
            vector_cache[(recording_key, sample_idx)] = vec
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
            recording_key = str(row["recording_id"]) if "recording_id" in row.index else str(row["sub"])
            key = (recording_key, int(row["sample"]))
            if key not in vector_cache or recording_key not in raw_cache:
                ax.text(0.5, 0.5, "missing raw", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color="0.35")
                ax.set_axis_off()
                continue

            mne.viz.plot_topomap(
                vector_cache[key],
                raw_cache[recording_key].info,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                vlim=vlim,
                contours=0,
                sensors=True,
                res=64,
                extrapolate="local",
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
            predictor_specs=[("GFP+rho", ["GFP_model", "rho"])],
            analysis_scope="within_subject_softmax",
        )
        softmax_loso = _run_item3_loso_models(
            prepared_softmax,
            target_specs=softmax_target_specs,
            predictor_specs=[("GFP+rho", ["GFP_model", "rho"])],
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


def plot_group_item3B_winning_corr_model_performance(
    within_df: pd.DataFrame,
    loso_df: pd.DataFrame,
):
    """Paired within-subject and LOSO R² distributions for winning-correlation OLS."""
    predictor_order = ["GFP", "rho", "GFP+rho"]
    predictor_labels = ["GFP", r"$\rho$", r"GFP+$\rho$"]
    palette = sns.color_palette(n_colors=len(predictor_order))

    within = within_df.loc[
        (within_df["analysis_scope"] == "within_subject")
        & (within_df["model_family"] == "linear")
        & (within_df["target"] == "winning_corr")
        & (within_df["predictor_set"].isin(predictor_order))
    ].copy()
    within["subject_id"] = within["sub"].astype(str)

    loso = loso_df.loc[
        (loso_df["analysis_scope"] == "loso_fold")
        & (loso_df["model_family"] == "linear")
        & (loso_df["target"] == "winning_corr")
        & (loso_df["predictor_set"].isin(predictor_order))
    ].copy()
    loso["subject_id"] = loso["held_out_sub"].astype(str)

    def draw_panel(ax, data: pd.DataFrame, title: str):
        if data.empty:
            ax.text(0.5, 0.5, "no valid fits", transform=ax.transAxes,
                    ha="center", va="center", color="0.35")
            ax.set_axis_off()
            return
        sns.lineplot(
            data=data,
            x="predictor_set",
            y="R2",
            units="subject_id",
            estimator=None,
            sort=False,
            color="0.45",
            alpha=0.13,
            linewidth=0.7,
            marker=None,
            ax=ax,
            zorder=1,
        )
        sns.boxplot(
            data=data,
            x="predictor_set",
            y="R2",
            order=predictor_order,
            palette=palette,
            width=0.50,
            whis=1.5,
            showfliers=False,
            linewidth=1.25,
            saturation=0.85,
            ax=ax,
            zorder=3,
        )
        ax.set(
            title=title,
            xlabel="Predictor set",
            ylabel=r"$R^2$",
            xticks=range(len(predictor_order)),
            xticklabels=predictor_labels,
        )
        ax.grid(True, axis="y", alpha=0.28)
        ax.grid(False, axis="x")
        sns.despine(ax=ax, bottom=True, left=False)

    with sns.axes_style("whitegrid"), sns.plotting_context("talk"):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
        fig.patch.set_visible(False)
        for ax in axes:
            ax.patch.set_visible(False)
        draw_panel(axes[0], within, "Within-subject")
        draw_panel(axes[1], loso, "LOSO")
        fig.tight_layout()
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
    _set_decade_log_ticks(ax)
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




def save_group_item3_ml_outputs(processed_outputs, outdir: str | Path, progress=None):
    outdir = Path(outdir)
    group_dir = outdir
    group_dir.mkdir(parents=True, exist_ok=True)

    def report(message: str):
        if progress is not None:
            progress(message)

    records = processed_outputs.to_dict("records") if isinstance(processed_outputs, pd.DataFrame) else list(processed_outputs)

    within_path = group_dir / "table_3_within_subject_model_comparison.csv"
    loso_path = group_dir / "table_3_loso_model_comparison.csv"
    softmax_path = group_dir / "table_3_softmax_lambda_sweep.csv"
    sample_pred_path = group_dir / "table_3_sample_predictions_loso.csv"
    exemplar_path = group_dir / "table_3D_group_exemplar_samples.csv"

    need_model_tables = not (within_path.exists() and loso_path.exists() and softmax_path.exists())
    need_sample_predictions = not sample_pred_path.exists()
    sample_dfs = None
    if need_model_tables or need_sample_predictions:
        report("loading subject-wise cached samples")
        sample_dfs = _item3_subject_dfs_from_records(records)
        if not sample_dfs:
            return None

    if not need_model_tables:
        report("reusing saved model-comparison tables")
        within_df = pd.read_csv(within_path)
        loso_df = pd.read_csv(loso_path)
        softmax_loso_df = pd.read_csv(softmax_path)
    else:
        report("fitting within-subject, leave-one-subject-out, and softmax models")
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
        report("reusing saved leave-one-subject-out predictions")
        sample_predictions_df = pd.read_csv(sample_pred_path)
        if "eeg_path" not in sample_predictions_df.columns:
            eeg_map = {str(row.get("recording_id", row.get("sub"))): str(row["eeg_path"]) for row in records if row.get("eeg_path")}
            key_col = "recording_id" if "recording_id" in sample_predictions_df.columns else "sub"
            sample_predictions_df["eeg_path"] = sample_predictions_df[key_col].astype(str).map(eeg_map)
            save_table(sample_predictions_df, sample_pred_path)
    else:
        report("deriving leave-one-subject-out sample predictions")
        confidence_lambda = _choose_item3_exemplar_confidence_lambda(softmax_loso_df)
        sample_predictions_df = _run_item3_loso_sample_predictions(
            sample_dfs,
            confidence_lambda=confidence_lambda,
            predictor_set="GFP+rho",
            model_family="linear",
        )
        eeg_map = {str(row.get("recording_id", row.get("sub"))): str(row["eeg_path"]) for row in records if row.get("eeg_path")}
        if not sample_predictions_df.empty:
            key_col = "recording_id" if "recording_id" in sample_predictions_df.columns else "sub"
            sample_predictions_df["eeg_path"] = sample_predictions_df[key_col].astype(str).map(eeg_map)
        save_table(sample_predictions_df, sample_pred_path)

    if exemplar_path.exists():
        report("reusing saved exemplar samples")
        exemplar_df = pd.read_csv(exemplar_path)
        if "eeg_path" not in exemplar_df.columns:
            eeg_map = {str(row.get("recording_id", row.get("sub"))): str(row["eeg_path"]) for row in records if row.get("eeg_path")}
            key_col = "recording_id" if "recording_id" in exemplar_df.columns else "sub"
            exemplar_df["eeg_path"] = exemplar_df[key_col].astype(str).map(eeg_map)
            save_table(exemplar_df, exemplar_path)
    else:
        report("selecting exemplar samples")
        exemplar_df = select_group_item3_exemplar_samples(sample_predictions_df, processed_outputs=processed_outputs, n_per_category=5)
        save_table(exemplar_df, exemplar_path)

    report("writing model-performance and exemplar figures")
    save_figure(
        plot_group_item3B_winning_corr_model_performance(within_df, loso_df),
        group_dir / "figure_3B_winning_template_prediction_performance.svg",
    )
    save_figure(
        plot_group_item3C_softmax_lambda_sweep(softmax_loso_df),
        group_dir / "figure_3C_softmax_confidence_sweep.svg",
    )
    save_figure(
        plot_group_item3D_group_exemplar_topomaps(exemplar_df),
        group_dir / "figure_3D_winning_template_exemplar_topomaps.svg",
        tight=False,
    )
    for stale_name in [
        "figure_3A_target_relationships.svg",
        "figure_3B_within_subject_model_performance.svg",
        "figure_3C_loso_model_performance.svg",
        "figure_3A_target_relationships.png",
        "figure_3B_within_subject_model_performance.png",
        "figure_3C_loso_model_performance.png",
        "figure_3B_winning_template_prediction_performance.png",
        "figure_3C_softmax_confidence_sweep.png",
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
    save_kwargs = {"dpi": 600}
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
    recording_dir: str | Path,
    *,
    write_exemplary_plots: bool = False,
):
    """Write all retained recording-level D leaves directly into one folder."""
    recording_dir = Path(recording_dir)
    recording_dir.mkdir(parents=True, exist_ok=True)

    save_figure(plot_item1A_zanesco_panel(sample_df), recording_dir / "figure_1A_gfp_td_relationships.png")
    save_figure(plot_item1B_peak_overlay(sample_df), recording_dir / "figure_1B_gfp_peak_td_relationships.png")
    save_figure(
        plot_item1C_td_decomposition(sample_df),
        recording_dir / "figure_1C_td_component_decomposition.svg",
        tight=False,
    )
    (recording_dir / "item1c_layout_v10.txt").write_text(
        "Figure 1C: 4 by 4 grid of 10 cm square panels; 14 pt text; transparent figure and axes patches; colorbars on the last column only; log10-value ticks shown as decades.\\n",
        encoding="utf-8",
    )
    save_figure(plot_abstract_subject_component(sample_df, sub=sub), recording_dir / "figure_abstract_subject_td_component.svg", tight=False)
    save_table(table1_subject_summary(sample_df, sub), recording_dir / "table_1_subject_summary.csv")
    save_figure(plot_recording_td_component_hexbin(sample_df), recording_dir / "figure_1E_td_component_loglog_hexbin.svg")
    save_figure(plot_recording_gfp_component_hexbin(sample_df), recording_dir / "figure_1F_gfp_component_loglog_hexbin.svg")
    (recording_dir / "item1_component_hexbin_layout_v2.txt").write_text(
        "Figures 1E/1F: shared log-space hexbin lattice; linear 0.05-1% color scale.\n", encoding="utf-8"
    )
    save_table(table1_decomposition_qc(sample_df, sub), recording_dir / "table_1_decomposition_qc.csv")
    fig_item1d, table_item1d = plot_item1D_low_high_f_examples(raw, sample_df, sub=sub)
    save_figure(fig_item1d, recording_dir / "figure_1D_low_high_f_examples.svg", tight=False)
    save_table(table_item1d, recording_dir / "table_1D_low_high_f_examples.csv")

    save_figure(
        plot_item3_microstate_likeness(sample_df, sub=sub),
        recording_dir / "figure_3A_microstate_likeness.png",
    )
    save_table(_item3_target_summary(sample_df, sub), recording_dir / "table_3_target_summary.csv")

    for stale_name in [
        "figure_abstract_subject_td_component.png",
        "figure_3A_nonredundancy_panel.png",
        "figure_3B_predictor_boxplots.png",
        "figure_3C_filtering_comparison.png",
        "table_3_predictive_models.csv",
    ]:
        stale_path = recording_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    if write_exemplary_plots:
        save_exemplary_subject_plots(sub, raw, sample_df, recording_dir)


# ---------------------------------------------------------------------
# Discrete spherical-harmonic decomposition
# ---------------------------------------------------------------------

def real_degree_block(sensor_directions: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate the real degree-l spherical-harmonic candidate columns."""
    S = np.asarray(sensor_directions, dtype=float)
    if S.ndim != 2 or S.shape[1] != 3:
        raise ValueError("sensor_directions must have shape (channels, 3)")
    norms = np.linalg.norm(S, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("sensor_directions cannot contain a zero vector")
    S = S / norms[:, None]
    polar = np.arccos(np.clip(S[:, 2], -1.0, 1.0))
    azimuth = np.mod(np.arctan2(S[:, 1], S[:, 0]), 2.0 * np.pi)
    columns = []
    for m in range(-int(degree), int(degree) + 1):
        y = _spherical_harmonic(int(degree), abs(m), polar, azimuth)
        if m < 0:
            columns.append(np.sqrt(2.0) * (-1) ** m * y.imag)
        elif m > 0:
            columns.append(np.sqrt(2.0) * (-1) ** m * y.real)
        else:
            columns.append(y.real)
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError(f"degree {degree} contains non-finite harmonic values")
    return result


def build_discrete_orthogonal_harmonic_blocks(
    sensor_directions: np.ndarray,
    s_orth: np.ndarray,
    *,
    lmax: int = 8,
    rank_tolerance: float = 1e-8,
) -> dict[int, np.ndarray]:
    """Construct Q1...Qlmax, explicitly retaining Q1 as the FORM basis."""
    S = np.asarray(sensor_directions, dtype=float)
    q1 = np.asarray(s_orth, dtype=float)
    if q1.shape != (S.shape[0], 3):
        raise ValueError("s_orth must have shape (channels, 3)")
    if np.linalg.norm(q1.T @ q1 - np.eye(3)) > 1e-8:
        raise ValueError("s_orth is not orthonormal")

    blocks = {1: q1}
    for degree in range(2, int(lmax) + 1):
        candidate = real_degree_block(S, degree)
        lower = np.column_stack([blocks[l] for l in sorted(blocks)])
        lower_basis, _ = np.linalg.qr(lower, mode="reduced")
        residual = candidate - lower_basis @ (lower_basis.T @ candidate)
        left, singular, _ = np.linalg.svd(residual, full_matrices=False)
        scale = max(1.0, float(np.linalg.norm(candidate, ord=2)))
        rank = int(np.sum(singular > float(rank_tolerance) * scale))
        blocks[degree] = left[:, :rank] if rank else np.empty((S.shape[0], 0), dtype=float)
    return blocks


def block_energies(values: np.ndarray, blocks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Return E_l(v)=||Q_l.T v||² for every value column."""
    values = np.asarray(values, dtype=float)
    return {
        degree: np.sum(np.square(block.T @ values), axis=0)
        for degree, block in blocks.items()
    }


def _add_remainder(energies: dict, total: np.ndarray) -> dict:
    result = dict(energies)
    result["remainder"] = np.asarray(total, dtype=float) - sum(result.values())
    return result


def spatial_frequency_decomposition(
    unit_maps: np.ndarray,
    s_orth: np.ndarray,
    sensor_directions: np.ndarray,
    rho2: np.ndarray,
    td2: np.ndarray,
    td2_meed: np.ndarray,
    td2_residual: np.ndarray,
    td2_rho: np.ndarray,
    td2_psi: np.ndarray,
    valid_transition: np.ndarray,
    *,
    lmax: int = 8,
) -> dict:
    """Resolve maps and additive TD components into degrees 1...lmax.

    Map fractions use unit map energy. Every transition-derived spectrum uses
    the same TD² denominator, so TD_FORM, TD_rho, TD_psi, and TD_res are
    directly additive components of the transition spectrum.
    """
    X = np.asarray(unit_maps, dtype=float)
    arrays = [rho2, td2, td2_meed, td2_residual, td2_rho, td2_psi, valid_transition]
    rho2, td2, td2_meed, td2_residual, td2_rho, td2_psi, valid_transition = [
        np.asarray(value, dtype=float if index < 6 else bool)
        for index, value in enumerate(arrays)
    ]
    if X.ndim != 2 or any(len(a) != X.shape[1] for a in arrays):
        raise ValueError("maps and samplewise arrays have incompatible shapes")

    blocks = build_discrete_orthogonal_harmonic_blocks(sensor_directions, s_orth, lmax=lmax)
    degrees = tuple(range(1, int(lmax) + 1))
    map_energy = _add_remainder(block_energies(X, blocks), np.sum(np.square(X), axis=0))
    map_total = np.sum(np.square(X), axis=0)

    transition_indices = np.flatnonzero(valid_transition & np.isfinite(td2) & (td2 > 0.0))
    delta = X[:, transition_indices] - X[:, transition_indices - 1]
    transition_total = td2[transition_indices]
    transition_energy = _add_remainder(block_energies(delta, blocks), transition_total)

    # Orthogonal sensor-space residual. Its entries are the TD_res contribution
    # on the common TD² scale, not fractions re-normalized within TD_res.
    delta_meed = s_orth @ (s_orth.T @ delta)
    delta_residual = delta - delta_meed
    residual_energy = block_energies(delta_residual, blocks)
    residual_energy[1] = np.zeros_like(transition_total)
    residual_energy = _add_remainder(residual_energy, td2_residual[transition_indices])

    # Existing FORM, radial, and angular TD terms are all first-order vectors.
    def first_order_component(values: np.ndarray) -> dict:
        energies = {degree: np.zeros_like(transition_total) for degree in degrees}
        energies[1] = values[transition_indices]
        return _add_remainder(energies, values[transition_indices])

    meed_energy = first_order_component(td2_meed)
    rho_energy = first_order_component(td2_rho)
    psi_energy = first_order_component(td2_psi)

    within_error = max(
        np.linalg.norm(block.T @ block - np.eye(block.shape[1]))
        for block in blocks.values()
    )
    between_error = max(
        np.linalg.norm(blocks[left].T @ blocks[right])
        for left in blocks for right in blocks if left > right
    )
    validation = {
        "max_abs_map_e1_minus_rho2": float(np.nanmax(np.abs(map_energy[1] - rho2))),
        "max_abs_transition_e1_minus_td2_meed": float(np.nanmax(np.abs(
            transition_energy[1] - td2_meed[transition_indices]
        ))),
        "max_abs_total_closure": float(np.nanmax(np.abs(
            transition_total - sum(transition_energy.values())
        ))),
        "max_abs_residual_closure": float(np.nanmax(np.abs(
            td2_residual[transition_indices] - sum(residual_energy.values())
        ))),
        "max_abs_meed_rho_psi_closure": float(np.nanmax(np.abs(
            td2_meed[transition_indices] - td2_rho[transition_indices] - td2_psi[transition_indices]
        ))),
        "max_abs_transition_component_closure": float(np.nanmax(np.abs(
            transition_total - sum(residual_energy.values()) - sum(rho_energy.values()) - sum(psi_energy.values())
        ))),
        "max_within_block_orthogonality_error": float(within_error),
        "max_between_block_orthogonality_error": float(between_error),
        "block_ranks": ";".join(f"l={degree}:{block.shape[1]}" for degree, block in blocks.items()),
    }
    return {
        "blocks": blocks,
        "degrees": degrees,
        "map_energy": map_energy,
        "map_total": map_total,
        "transition_energy": transition_energy,
        "transition_total": transition_total,
        "residual_energy": residual_energy,
        "meed_energy": meed_energy,
        "rho_energy": rho_energy,
        "psi_energy": psi_energy,
        "transition_indices": transition_indices,
        "validation": validation,
    }


def summarize_spatial_frequency(decomposition: dict, *, recording_id: str) -> pd.DataFrame:
    """Return six compact spectra with explicit denominators and degree order."""
    rows = []
    degree_order = (*decomposition["degrees"], "remainder")
    specs = [
        ("map", decomposition["map_energy"], decomposition["map_total"], "unit_map_energy"),
        ("transition", decomposition["transition_energy"], decomposition["transition_total"], "TD2"),
        ("transition_residual", decomposition["residual_energy"], decomposition["transition_total"], "TD2"),
        ("transition_meed", decomposition["meed_energy"], decomposition["transition_total"], "TD2"),
        ("transition_meed_rho", decomposition["rho_energy"], decomposition["transition_total"], "TD2"),
        ("transition_meed_psi", decomposition["psi_energy"], decomposition["transition_total"], "TD2"),
    ]
    for spectrum_kind, energies, denominator, normalization in specs:
        for order, degree in enumerate(degree_order, start=1):
            energy = energies[degree]
            fraction = np.divide(energy, denominator, out=np.full_like(energy, np.nan), where=denominator > 1e-12)
            finite_fraction = fraction[np.isfinite(fraction)]
            finite_energy = energy[np.isfinite(energy)]
            rows.append({
                "recording_id": recording_id, "spectrum_kind": spectrum_kind,
                "degree": str(degree), "degree_order": order, "normalization": normalization,
                "n_valid": int(len(finite_fraction)), "mean_energy": float(np.mean(finite_energy)),
                "median_energy": float(np.median(finite_energy)), "mean_fraction": float(np.mean(finite_fraction)),
                "median_fraction": float(np.median(finite_fraction)),
                "p25_fraction": float(np.percentile(finite_fraction, 25)),
                "p75_fraction": float(np.percentile(finite_fraction, 75)),
            })
    return pd.DataFrame(rows)


def plot_spatial_frequency_summary(summary: pd.DataFrame):
    """Six aligned spectra, with physically matched colours across decompositions."""
    colours = {
        "map": "#4c78a8", "td": "#e45756", "meed": "#54a24b",
        "nonmeed": "#9d9d9d", "rho": "#b279a2", "psi": "#f2cf5b",
    }
    labels = {"map": "map energy", "td": "TD²", "meed": "FORM", "nonmeed": "off-FORM", "rho": "TD_rho", "psi": "TD_psi"}
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2), constrained_layout=True)

    def spectrum(kind):
        return summary.loc[summary["spectrum_kind"].eq(kind)].sort_values("degree_order").reset_index(drop=True)

    def single(ax, kind, title, colour, ylabel):
        d = spectrum(kind); x = d["degree_order"].to_numpy() - 1; y = d["median_fraction"].to_numpy(float)
        lo, hi = d["p25_fraction"].to_numpy(float), d["p75_fraction"].to_numpy(float)
        ax.bar(x, y, color=colour, width=.72)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="none", color="0.2", capsize=2, lw=.8)
        ax.set(xticks=x, xticklabels=d["degree"], ylim=(0, 1), xlabel="spatial degree", ylabel=ylabel, title=title)
        ax.spines[["top", "right"]].set_visible(False)

    def paired(ax, left_kind, right_kind, title, left_label, right_label, left_colour, right_colour, ylabel, split_map=False):
        left, right = spectrum(left_kind), spectrum(right_kind)
        x = left["degree_order"].to_numpy() - 1
        ly, ry = left["median_fraction"].to_numpy(float), right["median_fraction"].to_numpy(float)
        if split_map:
            is_l1 = left["degree"].astype(str).eq("1").to_numpy()
            ly, ry = np.where(is_l1, ly, 0.0), np.where(is_l1, 0.0, ly)
            llo = np.where(is_l1, left["p25_fraction"].to_numpy(float), 0.0); lhi = np.where(is_l1, left["p75_fraction"].to_numpy(float), 0.0)
            rlo = np.where(is_l1, 0.0, left["p25_fraction"].to_numpy(float)); rhi = np.where(is_l1, 0.0, left["p75_fraction"].to_numpy(float))
        else:
            llo, lhi = left["p25_fraction"].to_numpy(float), left["p75_fraction"].to_numpy(float)
            rlo, rhi = right["p25_fraction"].to_numpy(float), right["p75_fraction"].to_numpy(float)
        width = .36
        ax.bar(x - width / 2, ly, width, color=left_colour, label=left_label)
        ax.bar(x + width / 2, ry, width, color=right_colour, label=right_label)
        ax.errorbar(x - width / 2, ly, yerr=np.vstack([ly - llo, lhi - ly]), fmt="none", color="0.2", capsize=2, lw=.75)
        ax.errorbar(x + width / 2, ry, yerr=np.vstack([ry - rlo, rhi - ry]), fmt="none", color="0.2", capsize=2, lw=.75)
        ax.set(xticks=x, xticklabels=left["degree"], ylim=(0, 1), xlabel="spatial degree", ylabel=ylabel, title=title)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    single(axes[0, 0], "map", "Map energy by spatial frequency", colours["map"], "share of map energy")
    paired(axes[1, 0], "map", "map", "Map energy: FORM versus off-FORM", labels["meed"], labels["nonmeed"], colours["meed"], colours["nonmeed"], "share of map energy", split_map=True)
    single(axes[0, 1], "transition", "TD² by spatial frequency", colours["td"], "share of TD²")
    paired(axes[1, 1], "transition_meed", "transition_residual", "TD²: FORM versus residual", labels["meed"], labels["nonmeed"], colours["meed"], colours["nonmeed"], "share of TD²")
    single(axes[0, 2], "transition_meed", "TD_FORM² by spatial frequency", colours["meed"], "share of TD²")
    paired(axes[1, 2], "transition_meed_rho", "transition_meed_psi", "TD_FORM²: radial versus angular", labels["rho"], labels["psi"], colours["rho"], colours["psi"], "share of TD²")
    return fig


# ---------------------------------------------------------------------
# Finite-template rho-ceiling model
# ---------------------------------------------------------------------


def evaluate_template_ceiling(
    q: np.ndarray,
    rho: np.ndarray,
    unit_maps: np.ndarray,
    template_u: np.ndarray,
    template_sensor_maps: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict:
    """Evaluate k-specific sensor maxima and the exact projected-FORM control."""
    q = np.asarray(q, dtype=float)
    rho = np.asarray(rho, dtype=float)
    X = np.asarray(unit_maps, dtype=float)
    U = np.asarray(template_u, dtype=float)
    B = np.asarray(template_sensor_maps, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (samples, 3)")
    if X.shape[1] != len(q) or len(rho) != len(q):
        raise ValueError("q, rho, and unit_maps have incompatible sample counts")
    if U.shape[0] != 3 or B.shape != (X.shape[0], U.shape[1]):
        raise ValueError("template FORM directions and sensor maps have incompatible shapes")
    r_max_meed = np.max(np.abs(q @ U), axis=1)
    r_max_sensor = np.max(np.abs(X.T @ B), axis=1)
    valid = np.isfinite(rho) & np.isfinite(r_max_sensor) & (rho > eps) & (rho <= 1.0 + eps)
    cosine = np.divide(r_max_meed, rho, out=np.full_like(rho, np.nan), where=rho > eps)
    psi_min_deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    retained_fraction = np.divide(r_max_sensor, rho, out=np.full_like(rho, np.nan), where=valid)
    return {
        "rho": rho,
        "r_max_sensor": r_max_sensor,
        "r_max_meed": r_max_meed,
        "psi_min_deg": psi_min_deg,
        "retained_fraction": retained_fraction,
        "valid": valid,
        "projected_identity_error": r_max_meed - rho * np.cos(np.radians(psi_min_deg)),
        "projected_bound_violation": r_max_meed - rho,
        "max_abs_q_norm_minus_rho": float(np.nanmax(np.abs(np.linalg.norm(q, axis=1) - rho))),
    }


def _beta1_valid_arrays(rho: np.ndarray, r_max_sensor: np.ndarray, *, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho, dtype=float)
    r_max_sensor = np.asarray(r_max_sensor, dtype=float)
    valid = np.isfinite(rho) & np.isfinite(r_max_sensor) & (rho > eps) & (rho <= 1.0 + eps)
    return np.clip(rho[valid], eps, 1.0), r_max_sensor[valid]


def fit_beta1_mean_ratio(rho: np.ndarray, r_max_sensor: np.ndarray) -> dict:
    """Primary M2 mean ceiling fraction, with raw correlation-scale residuals."""
    rho, r_max_sensor = _beta1_valid_arrays(rho, r_max_sensor)
    if len(rho) < 2:
        raise ValueError("beta1 fitting requires at least two valid samples")
    sum_rho2 = float(np.dot(rho, rho))
    sum_rho_rmax = float(np.dot(rho, r_max_sensor))
    beta1 = float(np.mean(r_max_sensor / rho))
    residual = r_max_sensor - beta1 * rho
    residual_mean = float(np.mean(residual))
    residual_sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else np.nan
    residual_mad = float(np.median(np.abs(residual - np.median(residual))))
    residual_rmse = float(np.sqrt(np.mean(residual ** 2)))
    relation = float(pearsonr(rho, residual)[0]) if np.ptp(rho) > 0 and np.ptp(residual) > 0 else np.nan
    return {
        "model": "M2", "n_samples": int(len(rho)), "beta1": float(beta1),
        "sum_rho": float(np.sum(rho)), "sum_rmax_sensor": float(np.sum(r_max_sensor)),
        "sum_rho2": sum_rho2, "sum_rho_rmax_sensor": sum_rho_rmax,
        "sum_rmax_sensor2": float(np.dot(r_max_sensor, r_max_sensor)),
        "residual_mean": residual_mean, "residual_rmse": residual_rmse,
        "residual_sd": residual_sd, "residual_mad": residual_mad,
        "residual_skewness": float(stats.skew(residual, bias=False)) if len(residual) > 2 else np.nan,
        "residual_excess_kurtosis": float(stats.kurtosis(residual, fisher=True, bias=False)) if len(residual) > 3 else np.nan,
        "residual_rho_pearson_r": relation,
        "residual": residual, "rho": rho, "r_max_sensor": r_max_sensor,
    }




def _nnls_two_columns(x0: np.ndarray, x1: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Exact two-column NNLS via its finite active-set candidates."""
    g00, g01, g11 = float(x0 @ x0), float(x0 @ x1), float(x1 @ x1)
    h0, h1 = float(x0 @ y), float(x1 @ y)
    candidates = [(0.0, 0.0), (max(h0 / g00, 0.0) if g00 else 0.0, 0.0),
                  (0.0, max(h1 / g11, 0.0) if g11 else 0.0)]
    gram = np.array([[g00, g01], [g01, g11]])
    try:
        both = np.linalg.solve(gram, np.array([h0, h1]))
        if np.all(both >= 0): candidates.append((float(both[0]), float(both[1])))
    except np.linalg.LinAlgError:
        pass
    return min(candidates, key=lambda b: b[0] * (g00 * b[0] + 2 * g01 * b[1]) + b[1] * g11 * b[1] - 2 * (h0 * b[0] + h1 * b[1]))
def fit_beta1_error_model(rho: np.ndarray, r_max_sensor: np.ndarray, model: str) -> dict:
    """Fit M1, primary M2, or M3 residual-variance sensitivity on one mask."""
    rho, r_max_sensor = _beta1_valid_arrays(rho, r_max_sensor)
    if len(rho) < 2:
        raise ValueError("beta1 fitting requires at least two valid samples")
    ols_beta = float(np.dot(rho, r_max_sensor) / np.dot(rho, rho))
    sigma_psi2 = sigma_perp2 = np.nan
    variance = np.ones_like(rho)
    if model == "M1":
        beta1 = ols_beta
        native = r_max_sensor - beta1 * rho
    elif model == "M2":
        beta1 = float(np.mean(r_max_sensor / rho))
        native = r_max_sensor / rho - beta1
    elif model == "M3":
        # M3 deliberately retains the M2 mean trajectory; it only decomposes
        # the raw correlation-scale residual variance.
        beta1 = float(np.mean(r_max_sensor / rho))
        m2_residual = r_max_sensor - beta1 * rho
        design = np.column_stack([rho ** 2, np.maximum(1.0 - rho ** 2, 0.0)])
        sigma_psi2, sigma_perp2 = _nnls_two_columns(design[:, 0], design[:, 1], m2_residual ** 2)
        variance = sigma_psi2 * rho ** 2 + sigma_perp2 * np.maximum(1.0 - rho ** 2, 0.0)
        variance = np.maximum(variance, max(np.finfo(float).eps, float(np.max(variance)) * 1e-12))
        native = (r_max_sensor - beta1 * rho) / np.sqrt(variance)
    else:
        raise ValueError(f"unknown beta1 error model {model!r}")
    raw = r_max_sensor - beta1 * rho
    mad = lambda x: float(np.median(np.abs(x - np.median(x))))
    return {
        "model": model, "n_samples": int(len(rho)), "beta1": beta1, "rho_min": float(np.min(rho)),
        "raw_scale_rmse": float(np.sqrt(np.mean(raw ** 2))), "raw_residual_mad": mad(raw),
        "model_native_residual_mad": mad(native), "model_native_residual_sd": float(np.std(native, ddof=1)),
        "model_native_residual_skewness": float(stats.skew(native, bias=False)) if len(native) > 2 else np.nan,
        "model_native_residual_excess_kurtosis": float(stats.kurtosis(native, fisher=True, bias=False)) if len(native) > 3 else np.nan,
        "sigma_psi2": sigma_psi2, "sigma_perp2": sigma_perp2,
        "variance_design_rank": int(np.linalg.matrix_rank(np.column_stack([rho ** 2, 1.0 - rho ** 2]))) if model == "M3" else np.nan,
        "variance_design_condition": float(np.linalg.cond(np.column_stack([rho ** 2, 1.0 - rho ** 2]))) if model == "M3" else np.nan,
        "variance_boundary_solution": bool((sigma_psi2 == 0.0) or (sigma_perp2 == 0.0)) if model == "M3" else False,
        "residual_rho_pearson_r": float(pearsonr(rho, native)[0]) if np.ptp(rho) and np.ptp(native) else np.nan,
        "rho": rho, "r_max_sensor": r_max_sensor, "raw_residual": raw,
        "native_residual": native, "variance": variance,
    }

def beta1_model_comparison_rows(samples: pd.DataFrame) -> pd.DataFrame:
    """Return exact subject-by-k fits for all declared error formulations."""
    required = {"subject", "k", "rho", "r_max_sensor"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"missing beta1 comparison columns: {sorted(missing)}")
    rows = []
    for (subject, k), data in samples.groupby(["subject", "k"], sort=True):
        rho, response = data.rho.to_numpy(float), data.r_max_sensor.to_numpy(float)
        for model in ("M1", "M2", "M3"):
            fit = fit_beta1_error_model(rho, response, model)
            rows.append({key: value for key, value in fit.items()
                         if key not in {"rho", "r_max_sensor", "raw_residual", "native_residual", "variance"}}
                        | {"subject": str(subject), "k": int(k)})
    return pd.DataFrame(rows)
def _deterministic_rows(length: int, maximum: int) -> np.ndarray:
    return np.arange(length) if length <= maximum else np.linspace(0, length - 1, maximum, dtype=int)


def _candidate_map_rho(level) -> np.ndarray:
    """Dipolarity of the actual candidate maps, not their unit FORM direction."""
    return np.linalg.norm(np.asarray(level.meed["Q"], dtype=float), axis=0)


def write_beta1_recording_outputs(
    record_dir: Path, levels_by_k: dict[int, object], cached_q: np.ndarray, cached_rho: np.ndarray,
    unit_maps: np.ndarray, sample_index: np.ndarray, time_s: np.ndarray, max_rows: int, *, subject: str, condition: str,
) -> list[dict]:
    """Write primary M2 condition summaries for every group template bank."""
    recording_id = record_dir.name
    fit_rows, sample_rows, qc_rows, contributions, native_arrays = [], [], [], [], {}
    for k, level in sorted(levels_by_k.items()):
        evaluated = evaluate_template_ceiling(cached_q, cached_rho, unit_maps, level.U, level.B)
        valid = evaluated["valid"] & (evaluated["rho"] > 1e-10)
        fit = fit_beta1_mean_ratio(evaluated["rho"][valid], evaluated["r_max_sensor"][valid])
        native_arrays[f"rho_{k}"] = fit["rho"].astype(np.float32)
        native_arrays[f"r_max_sensor_{k}"] = fit["r_max_sensor"].astype(np.float32)
        candidate_rho = _candidate_map_rho(level)
        shared = {"recording_id": recording_id, "subject": subject, "condition": condition, "template_source": "group", "k": int(k)}
        fit_row = {
            **shared, **{key: value for key, value in fit.items() if key not in {"residual", "rho", "r_max_sensor"}},
            "candidate_map_rho_min": float(np.min(candidate_rho)),
            "candidate_map_rho_median": float(np.median(candidate_rho)),
            "candidate_map_rho_max": float(np.max(candidate_rho)),
            "projected_identity_max_abs_error": float(np.nanmax(np.abs(evaluated["projected_identity_error"][valid]))),
            "projected_bound_max_violation": float(np.nanmax(evaluated["projected_bound_violation"][valid])),
            "q_rho_max_abs_error": evaluated["max_abs_q_norm_minus_rho"],
        }
        fit_rows.append(fit_row)
        keep = _deterministic_rows(len(fit["rho"]), max_rows)
        sample_rows.append(pd.DataFrame({
            **shared, "sample_index": sample_index[valid][keep], "time_s": time_s[valid][keep],
            "rho": fit["rho"][keep], "r_max_sensor": fit["r_max_sensor"][keep],
            "residual": fit["residual"][keep],
        }))
        qc_rows.append({**shared, "candidate_map_rho_min": float(np.min(candidate_rho)),
                        "candidate_map_rho_median": float(np.median(candidate_rho)),
                        "candidate_map_rho_max": float(np.max(candidate_rho)),
                        "projected_identity_max_abs_error": fit_row["projected_identity_max_abs_error"],
                        "projected_bound_max_violation": fit_row["projected_bound_max_violation"]})
        # The CSV retains max_rows for recording figures.  Cohort aggregation needs
        # only a deterministic, subject-balanced diagnostic subset.
        cohort_rows = sample_rows[-1].iloc[_deterministic_rows(len(sample_rows[-1]), 60)].copy()
        contributions.append({"fit": fit_row, "samples": cohort_rows, "candidate_qc": qc_rows[-1], "native_samples_file": str(record_dir / "beta1_native_samples.npz")})
    fits = pd.DataFrame(fit_rows)
    samples = pd.concat(sample_rows, ignore_index=True)
    fits.to_csv(record_dir / "beta1_condition_fits.csv", index=False)
    samples.to_csv(record_dir / "beta1_plot_samples.csv", index=False)
    native_file = record_dir / "beta1_native_samples.npz"
    np.savez_compressed(native_file, **native_arrays)
    pd.DataFrame(qc_rows).to_csv(record_dir / "beta1_candidate_map_qc.csv", index=False)
    write_beta1_recording_figures(record_dir, samples, fits)
    return contributions


def write_beta1_recording_figures(
    record_dir: str | Path, samples: pd.DataFrame | None = None, fits: pd.DataFrame | None = None
) -> None:
    """Redraw recording Figure 4 leaves from saved beta1 tables only."""
    record_dir = Path(record_dir)
    if fits is None:
        fits = pd.read_csv(record_dir / "beta1_condition_fits.csv")
    if samples is None:
        samples = pd.read_csv(record_dir / "beta1_plot_samples.csv")

    for model in ("M1", "M2", "M3"):
        figure = plot_beta1_recording_model_scaling(samples, model, n_rows=2)
        figure.savefig(record_dir / f"figure_4A_rho_scaling_{model}.svg", bbox_inches="tight")
        if model == "M1":
            figure.savefig(record_dir / "figure_4A_rho_scaling.svg", bbox_inches="tight")
        plt.close(figure)


def load_beta1_recording_outputs(record_dir: str | Path) -> list[dict]:
    """Load M2 recording summaries, upgrading legacy summaries from native samples."""
    record_dir = Path(record_dir)
    fits = pd.read_csv(record_dir / "beta1_condition_fits.csv")
    samples = pd.read_csv(record_dir / "beta1_plot_samples.csv")
    qc = pd.read_csv(record_dir / "beta1_candidate_map_qc.csv")
    if "model" not in fits or not fits["model"].eq("M2").all():
        # Old generic tables were M1. Rebuild only these summaries from the
        # full native arrays, without reopening EEG or changing M1/M2/M3 fits.
        warnings.warn(f"{record_dir.name}: updating legacy beta summaries to M2 from cached native samples.",
                      stacklevel=2)
        updated = []
        with np.load(record_dir / "beta1_native_samples.npz", allow_pickle=False) as native:
            for row in fits.to_dict(orient="records"):
                k = int(row["k"])
                fit = fit_beta1_mean_ratio(native[f"rho_{k}"], native[f"r_max_sensor_{k}"])
                row.update({key: value for key, value in fit.items()
                            if key not in {"residual", "rho", "r_max_sensor"}})
                updated.append(row)
        fits = pd.DataFrame(updated)
        beta = samples["k"].map(fits.set_index("k")["beta1"])
        samples["residual"] = samples["r_max_sensor"] - beta * samples["rho"]
        fits.to_csv(record_dir / "beta1_condition_fits.csv", index=False)
        samples.to_csv(record_dir / "beta1_plot_samples.csv", index=False)
    contributions = []
    for row in fits.to_dict(orient="records"):
        per_k = samples.loc[samples.k.eq(row["k"])].copy()
        per_k = per_k.iloc[_deterministic_rows(len(per_k), 60)].copy()
        contributions.append({"fit": row, "samples": per_k,
                              "candidate_qc": qc.loc[qc.k.eq(row["k"])].iloc[0].to_dict(), "native_samples_file": str(record_dir / "beta1_native_samples.npz")})
    return contributions




def beta1_native_residual_render_samples(contributions: list[dict], comparison: pd.DataFrame, maximum: int = 300) -> pd.DataFrame:
    """Reconstruct model-native residuals, capping only display rows per subject-k."""
    lookup = comparison.set_index(["subject", "k", "model"])
    rows = []
    seen = set()
    for contribution in contributions:
        fit = contribution["fit"]; subject, k = str(fit["subject"]), int(fit["k"])
        key = (subject, k, contribution["native_samples_file"])
        if key in seen: continue
        seen.add(key)
        with np.load(contribution["native_samples_file"], allow_pickle=False) as native:
            rho, response = native[f"rho_{k}"].astype(float), native[f"r_max_sensor_{k}"].astype(float)
        keep = _deterministic_rows(len(rho), maximum)
        for model in ("M1", "M2", "M3"):
            params = lookup.loc[(subject, k, model)]
            if model == "M1": residual = response - params.beta1 * rho
            elif model == "M2": residual = response / rho - params.beta1
            else:
                variance = params.sigma_psi2 * rho ** 2 + params.sigma_perp2 * np.maximum(1 - rho ** 2, 0)
                residual = (response - params.beta1 * rho) / np.sqrt(np.maximum(variance, np.finfo(float).eps))
            rows.append(pd.DataFrame({"subject": subject, "k": k, "model": model, "native_residual": residual[keep]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["subject", "k", "model", "native_residual"])
def beta1_model_comparison_from_native(contributions: list[dict]) -> pd.DataFrame:
    """Fit every subject-by-k model exactly, retaining only one subject-k at once."""
    grouped: dict[tuple[str, int], list[dict]] = {}
    for contribution in contributions:
        fit = contribution["fit"]
        grouped.setdefault((str(fit["subject"]), int(fit["k"])), []).append(contribution)
    rows = []
    for (subject, k), items in sorted(grouped.items()):
        rho_parts, response_parts = [], []
        for item in items:
            with np.load(item["native_samples_file"], allow_pickle=False) as native:
                rho_parts.append(native[f"rho_{k}"].astype(float))
                response_parts.append(native[f"r_max_sensor_{k}"].astype(float))
        rho, response = np.concatenate(rho_parts), np.concatenate(response_parts)
        for model in ("M1", "M2", "M3"):
            fit = fit_beta1_error_model(rho, response, model)
            rows.append({key: value for key, value in fit.items()
                         if key not in {"rho", "r_max_sensor", "raw_residual", "native_residual", "variance"}}
                        | {"subject": subject, "k": k})
    return pd.DataFrame(rows)
def _holm(pvalues: list[float]) -> np.ndarray:
    if not pvalues:
        return np.empty(0, dtype=float)
    p = np.asarray(pvalues, dtype=float)
    order, adjusted, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def _page_test(wide: pd.DataFrame) -> tuple[float, float]:
    if wide.shape[0] < 2 or wide.shape[1] < 3:
        return np.nan, np.nan
    # Verified separately with a deterministic increasing toy matrix in tests.
    result = stats.page_trend_test(wide.to_numpy(float))
    return float(result.statistic), float(result.pvalue)


def beta1_group_tests(subject_fits: pd.DataFrame, condition_fits: pd.DataFrame) -> pd.DataFrame:
    """Subject-unit trend, residual-scale, and paired within-subject condition tests."""
    rows: list[dict] = []
    ks = sorted(subject_fits.k.unique())
    wide = subject_fits.pivot(index="subject", columns="k", values="beta1").reindex(columns=ks).dropna()
    statistic, pvalue = _page_test(wide)
    rows.append({"endpoint": "beta1", "analysis": "condition-collapsed subject fit", "test": "Page ordered alternative",
                 "direction": "beta1 increases as k increases", "statistic": statistic, "raw_p": pvalue,
                 "holm_p": np.nan, "n_subjects": len(wide), "k_left": np.nan, "k_right": np.nan,
                 "median_paired_difference": np.nan})
    posthoc = []
    if np.isfinite(pvalue) and pvalue < .05:
        for left, right in zip(ks[:-1], ks[1:]):
            pair = wide[[left, right]].dropna()
            difference = pair[right] - pair[left]
            test = stats.wilcoxon(difference, alternative="greater", method="auto") if len(pair) else None
            posthoc.append({"endpoint": "beta1", "analysis": "condition-collapsed subject fit", "test": "adjacent one-sided Wilcoxon",
                            "direction": f"beta1_{right} > beta1_{left}", "statistic": float(test.statistic) if test else np.nan,
                            "raw_p": float(test.pvalue) if test else np.nan, "holm_p": np.nan, "n_subjects": len(pair),
                            "k_left": left, "k_right": right, "median_paired_difference": float(np.median(difference)) if len(pair) else np.nan})
        for row, adjusted in zip(posthoc, _holm([row["raw_p"] for row in posthoc])):
            row["holm_p"] = adjusted
    rows.extend(posthoc)
    mad_wide = subject_fits.pivot(index="subject", columns="k", values="residual_mad").reindex(columns=ks).dropna()
    if mad_wide.shape[0] >= 2 and len(ks) >= 3:
        test = stats.friedmanchisquare(*[mad_wide[k].to_numpy(float) for k in ks])
        f_stat, f_p = float(test.statistic), float(test.pvalue)
    else:
        f_stat = f_p = np.nan
    rows.append({"endpoint": "residual_mad", "analysis": "condition-collapsed subject fit", "test": "Friedman repeated measures",
                 "direction": "two-sided any k effect", "statistic": f_stat, "raw_p": f_p, "holm_p": np.nan,
                 "n_subjects": len(mad_wide), "k_left": np.nan, "k_right": np.nan, "median_paired_difference": np.nan})
    condition_rows = []
    for k in ks:
        pair = condition_fits.loc[condition_fits.k.eq(k)].pivot(index="subject", columns="condition", values="beta1")
        if {"EC", "EO"}.issubset(pair.columns):
            difference = pair["EO"] - pair["EC"]
            test = stats.wilcoxon(difference, alternative="two-sided", method="auto") if len(difference) else None
            condition_rows.append({"endpoint": "beta1", "analysis": "condition within subject", "test": "paired Wilcoxon EC versus EO",
                                   "direction": "two-sided", "statistic": float(test.statistic) if test else np.nan,
                                   "raw_p": float(test.pvalue) if test else np.nan, "holm_p": np.nan,
                                   "n_subjects": len(difference), "k_left": k, "k_right": k,
                                   "median_paired_difference": float(np.median(difference)) if len(difference) else np.nan})
    for row, adjusted in zip(condition_rows, _holm([row["raw_p"] for row in condition_rows])):
        row["holm_p"] = adjusted
    return pd.DataFrame(rows + condition_rows)






def plot_beta1_model_residual_diagnostics(residuals: pd.DataFrame, model: str):
    """Figure 4C: per-k empirical nested quantiles and compact Gaussian Q–Q plots."""
    data = residuals.loc[residuals.model.eq(model)]
    ks = sorted(data.k.unique())
    fig, axes = plt.subplots(len(ks), 2, figsize=(9, max(3.0, 2.25 * len(ks))), sharex="col", sharey="col", constrained_layout=True)
    axes = np.atleast_2d(axes)
    probabilities = np.linspace(.01, .99, 101)
    for row, k in enumerate(ks):
        values = data.loc[data.k.eq(k), "native_residual"].dropna().to_numpy(float)
        colour = plt.cm.viridis((row + .5) / max(len(ks), 1))
        if len(values) < 3:
            continue
        empirical = np.quantile(values, probabilities)
        normal = stats.norm.ppf(probabilities, loc=np.mean(values), scale=np.std(values, ddof=1))
        ax, qq = axes[row]
        ax.plot(probabilities, empirical, color=colour, lw=2.4)
        for lo, hi, alpha in ((.25, .75, .34), (.10, .90, .20), (.05, .95, .12)):
            qlo, qhi = np.quantile(values, [lo, hi])
            ax.fill_between([lo, hi], [qlo, qlo], [qhi, qhi], color=colour, alpha=alpha, linewidth=0)
        qq.plot(normal, empirical, color=colour, lw=1.5)
        lo, hi = min(normal.min(), empirical.min()), max(normal.max(), empirical.max())
        qq.plot([lo, hi], [lo, hi], color="0.25", ls="--", lw=.8)
        ax.set_ylabel(f"k={k}\nresidual")
    axes[-1, 0].set(xlabel="empirical cumulative probability", title="nested empirical quantiles")
    axes[-1, 1].set(xlabel="Gaussian theoretical quantile", title="Q–Q")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    return fig


def plot_beta1_model_scaling(
    rho: np.ndarray,
    response: np.ndarray,
    model: str,
    k: int,
    colour,
    ax=None,
):
    """Figure 4A single-k raw-scale fit with nested empirical residual envelopes."""
    fit = fit_beta1_error_model(rho, response, model)
    grid = np.linspace(0, 1, 300)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    else:
        fig = ax.figure

    for q_low, q_high in ((0.05, 0.95), (0.10, 0.90), (0.25, 0.75)):
        lo, hi = np.quantile(fit["native_residual"], [q_low, q_high])
        if model == "M1":
            low = fit["beta1"] * grid + lo
            high = fit["beta1"] * grid + hi
        elif model == "M2":
            low = grid * (fit["beta1"] + lo)
            high = grid * (fit["beta1"] + hi)
        else:
            variance = (
                fit["sigma_psi2"] * grid ** 2
                + fit["sigma_perp2"] * np.maximum(1 - grid ** 2, 0)
            )
            scale = np.sqrt(variance)
            low = fit["beta1"] * grid + lo * scale
            high = fit["beta1"] * grid + hi * scale
        ax.fill_between(grid, low, high, color=colour, alpha=.25, linewidth=0)

    ax.scatter(
        fit["rho"], fit["r_max_sensor"], s=2, alpha=.10,
        color="0.25", linewidths=0, rasterized=True,
    )
    ax.plot(
        grid, fit["beta1"] * grid, color=colour, gapcolor="white",
        lw=1.8, ls="--", dashes=(5, 4), zorder=4,
        label=rf"$k={k}$, $\hat{{\beta}}_1={fit['beta1']:.3f}$",
    )
    ax.plot(grid, grid, color="0.25", ls="--", lw=.8, label=r"ceiling $r=\rho$")
    ax.set(
        xlim=(0, 1), ylim=(0, 1), xticks=(0, .5, 1), yticks=(0, .5, 1),
        xlabel=r"sample dipolarity $\rho$",
        ylabel=r"winning sensor correlation $r_{max}$",
        title=rf"$k={k}$",
    )
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _plot_beta1_model_scaling_grid(
    samples: pd.DataFrame,
    model: str,
    *,
    n_rows: int = 2,
    n_cols: int | None = None,
):
    """Figure 4A wrapper: one square panel per k with a shared viridis mapping."""
    required = {"rho", "r_max_sensor", "k"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"missing Figure 4A sample columns: {sorted(missing)}")
    ks = sorted(int(k) for k in samples.k.unique())
    if not ks:
        raise ValueError("no k values available for Figure 4A")
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    if n_cols is None:
        n_cols = int(np.ceil(len(ks) / n_rows))
    if n_cols < 1 or n_rows * n_cols < len(ks):
        raise ValueError("Figure 4A layout has fewer axes than k values")

    width = u.PAPER_WIDTH_IN
    height = 3.2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    colours = dict(zip(ks, plt.cm.viridis(np.linspace(.1, .9, len(ks)))))

    for ax, k in zip(axes, ks):
        d = samples.loc[samples.k.eq(k)]
        plot_beta1_model_scaling(
            d["rho"].to_numpy(float), d["r_max_sensor"].to_numpy(float),
            model, k, colours[k], ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        row, col = divmod(ks.index(k), n_cols)
        ax.set_box_aspect(1)
        ax.set_xlabel(r"$\rho$" if row == n_rows - 1 else "")
        ax.set_ylabel(r"$r^{*}$" if col == 0 else "")
    for ax in axes[len(ks):]:
        ax.set_visible(False)
    return fig


def plot_beta1_group_model_scaling(
    samples: pd.DataFrame,
    model: str,
    n_rows: int = 2,
    n_cols: int | None = None,
):
    """Group Figure 4A: one raw-scale fit panel per k."""
    return _plot_beta1_model_scaling_grid(samples, model, n_rows=n_rows, n_cols=n_cols)


def plot_beta1_recording_model_scaling(
    samples: pd.DataFrame,
    model: str,
    n_rows: int = 2,
    n_cols: int | None = None,
):
    """Recording Figure 4A: one raw-scale fit panel per k."""
    return _plot_beta1_model_scaling_grid(samples, model, n_rows=n_rows, n_cols=n_cols)


def plot_beta1_model_boxplots(comparison: pd.DataFrame, tests: pd.DataFrame, model: str):
    """Figure 4B: beta and the declared residual MAD for one model."""
    data = comparison.loc[comparison.model.eq(model)].copy()
    ks = sorted(data.k.unique())
    palette = dict(zip(ks, plt.cm.viridis(np.linspace(.1, .9, len(ks)))))
    n_panels = 3 if model == "M3" else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(u.PAPER_WIDTH_IN, 3.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    mad_column = "raw_residual_mad" if model == "M2" else "model_native_residual_mad"
    mad_label = "Raw correlation-scale residual MAD" if model == "M2" else "Model-native residual MAD"
    for ax, value, label in zip(axes, ["beta1", mad_column],
                                [r"Ceiling fraction $\hat{\beta}_1$", mad_label]):
        for _, subject in data.groupby("subject", sort=False):
            ax.plot([ks.index(k) for k in subject.k], subject[value], color="0.35", alpha=.06, lw=.5, zorder=1)
        sns.boxplot(data=data, x="k", y=value, hue="k", order=ks, hue_order=ks, palette=palette,
                    dodge=False, showfliers=False, width=.58, ax=ax, zorder=2, legend=False)
        ax.set(xlabel="k", ylabel=label)
        ax.set_xticks(range(len(ks)), [str(k) if int(k) % 2 else "" for k in ks])
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        if value == "beta1" and ks:
            first = data.loc[data.k.eq(ks[0]), value].dropna().to_numpy(float)
            medians = data.groupby("k")[value].median().reindex(ks)
            step = abs(float(medians.iloc[1] - medians.iloc[0])) if len(medians) > 1 else 0.0
            q1, q3 = np.quantile(first, [.25, .75])
            fence = q1 - 1.5 * (q3 - q1)
            whisker = first[first >= fence]
            whisker_distance = float(medians.iloc[0] - np.min(whisker)) if len(whisker) else 0.0
            ax.set_ylim(bottom=min(float(medians.iloc[0] - step), float(medians.iloc[0] - 1.5 * whisker_distance)))
    if model == "M3":
        error_data = data.melt(id_vars=["k"], value_vars=["sigma_psi2", "sigma_perp2"],
                               var_name="error", value_name="variance")
        sns.boxplot(data=error_data, x="k", y="variance", hue="error", order=ks,
                    palette={"sigma_psi2": "#56b4e9", "sigma_perp2": "#e69f00"},
                    showfliers=False, width=.58, ax=axes[2])
        axes[2].set(xlabel="k", ylabel="Error variance")
        axes[2].set_xticks(range(len(ks)), [str(k) if int(k) % 2 else "" for k in ks])
        axes[2].legend(labels=[r"$\sigma_{\psi}^{2}$", r"$\sigma_{\perp}^{2}$"], frameon=False, fontsize=10)
        axes[2].spines[["top", "right", "bottom"]].set_visible(False)
    axes[0].axhline(1, color="0.25", ls="--", lw=.9)
    return fig
def beta1_model_comparison_tests(comparison: pd.DataFrame) -> pd.DataFrame:
    """Run subject-unit Page and adjacent Holm tests per model."""
    rows = []
    for model, data in comparison.groupby("model", sort=True):
        ks = sorted(data.k.unique())
        wide = data.pivot(index="subject", columns="k", values="beta1").reindex(columns=ks).dropna()
        statistic, pvalue = _page_test(wide)
        rows.append({"model": model, "endpoint": "beta1", "test": "Page ordered alternative",
                     "direction": "beta1 increases as k increases", "statistic": statistic, "raw_p": pvalue,
                     "holm_p": np.nan, "n_subjects": len(wide), "k_left": np.nan, "k_right": np.nan})
        adjacent = []
        if np.isfinite(pvalue) and pvalue < .05:
            for left, right in zip(ks[:-1], ks[1:]):
                pair = wide[[left, right]].dropna()
                result = stats.wilcoxon(pair[right] - pair[left], alternative="greater", method="auto") if len(pair) else None
                adjacent.append({"model": model, "endpoint": "beta1", "test": "adjacent one-sided Wilcoxon",
                                 "direction": f"beta1_{right} > beta1_{left}", "statistic": float(result.statistic) if result else np.nan,
                                 "raw_p": float(result.pvalue) if result else np.nan, "holm_p": np.nan,
                                 "n_subjects": len(pair), "k_left": left, "k_right": right})
        for row, adjusted in zip(adjacent, _holm([row["raw_p"] for row in adjacent])):
            row["holm_p"] = adjusted
        rows.extend(adjacent)
    return pd.DataFrame(rows)
def _combine_subject_beta1(detailed: pd.DataFrame, contributions: list[dict]) -> pd.DataFrame:
    """Pool EC/EO native samples for exact M2 summaries, not plotting subsets."""
    native_paths = {
        (str(c["fit"]["recording_id"]), int(c["fit"]["k"])): c["native_samples_file"]
        for c in contributions
    }
    rows = []
    for (subject, source, k), d in detailed.groupby(["subject", "template_source", "k"], sort=True):
        rho_parts, response_parts = [], []
        for recording_id in d.recording_id:
            with np.load(native_paths[(str(recording_id), int(k))], allow_pickle=False) as native:
                rho_parts.append(native[f"rho_{int(k)}"].astype(float))
                response_parts.append(native[f"r_max_sensor_{int(k)}"].astype(float))
        fit = fit_beta1_mean_ratio(np.concatenate(rho_parts), np.concatenate(response_parts))
        rows.append({key: value for key, value in fit.items()
                     if key not in {"residual", "rho", "r_max_sensor"}}
                    | {"subject": subject, "template_source": source, "k": int(k),
                       "n_conditions": int(d.condition.nunique())})
    return pd.DataFrame(rows)


def _subject_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().to_numpy(float)
    if len(values) < 2:
        return np.nan, np.nan
    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    half = stats.t.ppf(.975, len(values) - 1) * sem
    return float(np.mean(values) - half), float(np.mean(values) + half)


def beta1_group_summary(subject_fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, k), d in subject_fits.groupby(["template_source", "k"], sort=True):
        low, high = _subject_ci(d.beta1)
        rows.append({"template_source": source, "k": int(k), "n_subjects": int(d.subject.nunique()),
                     "mean_beta1": float(d.beta1.mean()), "median_beta1": float(d.beta1.median()),
                     "beta1_sd": float(d.beta1.std(ddof=1)), "beta1_ci_low": low, "beta1_ci_high": high,
                     "mean_residual_mad": float(d.residual_mad.mean()), "median_residual_mad": float(d.residual_mad.median()),
                     "mean_residual_sd": float(d.residual_sd.mean()), "mean_residual_rmse": float(d.residual_rmse.mean())})
    return pd.DataFrame(rows)


def plot_recording_beta1_scaling(samples: pd.DataFrame, fits: pd.DataFrame):
    ks = sorted(fits.k.unique())
    representative = sorted(set([ks[0], ks[len(ks) // 2], ks[-1]]))
    fig, ax = plt.subplots(figsize=(6.6, 5.1), constrained_layout=True)
    colours = plt.cm.viridis(np.linspace(.15, .85, len(representative)))
    grid = np.linspace(0, 1, 200)
    for k, colour in zip(representative, colours):
        d = samples.loc[samples.k.eq(k)]
        beta = float(fits.loc[fits.k.eq(k), "beta1"].iloc[0])
        ax.scatter(d.rho, d.r_max_sensor, s=3, alpha=.10, color=colour, linewidths=0, rasterized=True)
        ax.plot(grid, beta * grid, color=colour, lw=1.6, label=rf"$k={k}$, $\beta_1={beta:.3f}$")
    ax.plot(grid, grid, "--", color="0.25", lw=.9, label=r"ceiling $r=\rho$")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r"sample dipolarity $\rho$", ylabel=r"winning sensor correlation $r_{max}$")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_beta1_group_figure(subject_fits: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ks = sorted(subject_fits.k.unique())
    positions = {k: index for index, k in enumerate(ks)}
    for _, d in subject_fits.groupby("subject", sort=False):
        ax.plot(d.k.map(positions), d.beta1, color="0.35", lw=.55, alpha=.20, zorder=1)
    sns.boxplot(data=subject_fits, x="k", y="beta1", order=ks, color="#56b4e9", showfliers=False, width=.55, ax=ax, zorder=2)
    ordered = summary.set_index("k").loc[ks]
    ax.plot(range(len(ks)), ordered.mean_beta1, color="#0072b2", marker="o", lw=1.8, zorder=4, label="subject mean")
    ax.errorbar(range(len(ks)), ordered.mean_beta1, yerr=[ordered.mean_beta1 - ordered.beta1_ci_low, ordered.beta1_ci_high - ordered.mean_beta1], fmt="none", color="#0072b2", lw=1.1, capsize=3, zorder=4)
    ax.axhline(1, color="0.25", ls="--", lw=.9)
    page = tests.loc[tests.test.eq("Page ordered alternative")].iloc[0]
    p_text = "not available" if not np.isfinite(page.raw_p) else f"L={page.statistic:.1f}, p={page.raw_p:.3g}, n={int(page.n_subjects)}"
    ax.text(.01, .98, f"Page increasing trend: {p_text}", transform=ax.transAxes, va="top", fontsize=9)
    ax.set(xlabel="template count $k$", ylabel=r"ceiling fraction $\hat{\beta}_1$", ylim=(0, 1.05))
    ax.spines[["top", "right", "bottom"]].set_visible(False); ax.legend(frameon=False)
    return fig


def plot_beta1_residual_diagnostics(subject_fits: pd.DataFrame, samples: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    ks = sorted(subject_fits.k.unique())
    positions = {k: index for index, k in enumerate(ks)}
    for _, d in subject_fits.groupby("subject", sort=False):
        axes[0].plot(d.k.map(positions), d.residual_mad, color="0.35", lw=.5, alpha=.18)
    sns.boxplot(data=subject_fits, x="k", y="residual_mad", order=ks, color="#e69f00", showfliers=False, ax=axes[0])
    axes[0].set(xlabel="template count $k$", ylabel="residual MAD")
    # Equal per-subject contribution was already imposed in the deterministic recording samples.
    balanced = []
    for (_, k), d in samples.groupby(["subject", "k"], sort=False):
        balanced.append(d.iloc[_deterministic_rows(len(d), 120)])
    balanced = pd.concat(balanced, ignore_index=True) if balanced else samples.iloc[0:0]
    residual = balanced.residual.to_numpy(float)
    sns.histplot(residual, bins=45, stat="density", color="#56b4e9", alpha=.4, ax=axes[1])
    if len(residual) > 1 and np.std(residual) > 0:
        x = np.linspace(np.min(residual), np.max(residual), 200)
        axes[1].plot(x, stats.norm.pdf(x, np.mean(residual), np.std(residual, ddof=1)), color="#0072b2", lw=1.4)
        stats.probplot(residual, dist="norm", plot=axes[2])
    axes[1].set(xlabel="residual", ylabel="density", title="subject-balanced residuals")
    axes[2].set(title="Gaussian Q-Q")
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    return fig


def subject_balanced_beta1_plot_samples(samples: pd.DataFrame, *, max_rows_per_subject_k: int = 120) -> pd.DataFrame:
    """Deterministically cap rendering rows without changing any fitted statistic."""
    parts = []
    for _, d in samples.groupby(["subject", "template_source", "k"], sort=True):
        parts.append(d.iloc[_deterministic_rows(len(d), max_rows_per_subject_k)])
    return pd.concat(parts, ignore_index=True) if parts else samples.iloc[0:0].copy()


def finalize_beta1_group_outputs(contributions: list[dict]) -> dict[str, pd.DataFrame]:
    detailed = pd.DataFrame([c["fit"] for c in contributions])
    samples = pd.concat([c["samples"] for c in contributions], ignore_index=True)
    candidate_qc = pd.DataFrame([c["candidate_qc"] for c in contributions])
    subject = _combine_subject_beta1(detailed, contributions)
    render_samples = subject_balanced_beta1_plot_samples(samples)
    render_samples = render_samples.drop(columns="residual", errors="ignore").merge(
        subject[["subject", "template_source", "k", "beta1"]],
        on=["subject", "template_source", "k"], how="left", validate="many_to_one",
    )
    render_samples["residual"] = render_samples.r_max_sensor - render_samples.beta1 * render_samples.rho
    summary = beta1_group_summary(subject)
    tests = beta1_group_tests(subject, detailed)
    condition_summary = detailed.groupby(["template_source", "condition", "k"], as_index=False).agg(
        n_subjects=("subject", "nunique"), mean_beta1=("beta1", "mean"), median_beta1=("beta1", "median"),
        mean_residual_mad=("residual_mad", "mean"), mean_residual_sd=("residual_sd", "mean"), mean_residual_rmse=("residual_rmse", "mean"))
    candidate_summary = candidate_qc.groupby(["template_source", "k"], as_index=False).agg(
        candidate_map_rho_min=("candidate_map_rho_min", "min"), candidate_map_rho_median=("candidate_map_rho_median", "median"),
        candidate_map_rho_max=("candidate_map_rho_max", "max"), projected_identity_max_abs_error=("projected_identity_max_abs_error", "max"),
        projected_bound_max_violation=("projected_bound_max_violation", "max"))
    return {"detailed": detailed, "samples": samples, "render_samples": render_samples,
            "candidate_qc": candidate_qc, "candidate_summary": candidate_summary,
            "subject": subject, "summary": summary, "tests": tests, "condition_summary": condition_summary}

def load_group_template_sets(cfg: dict, base: Path) -> dict[int, tuple[np.ndarray, list[str]]]:
    """Load unordered Stage-B model families once for the D finite-k branch."""
    output_root = base / Path(cfg["output_root"])
    sets = {}
    for k in cfg["d_descriptive"]["group_k_values"]:
        model_file = output_root / "b_group_level" / "models" / f"k-{int(k):02d}_modkmeans.fif"
        if not model_file.exists():
            raise FileNotFoundError(f"Group model K={k} is unavailable: {model_file}")
        model = mvu.read_cluster_model(model_file)
        sets[int(k)] = (np.asarray(model.cluster_centers_, dtype=float).T, [str(label) for label in model.cluster_names])
    return sets


def write_spatial_group_summary(outdir: Path) -> None:
    """Retained spatial-frequency group summary, independent of finite-k inference."""
    group_dir = outdir / "group"
    parts = [pd.read_csv(path) for path in outdir.glob("recordings/*/spatial_frequency_summary.csv")]
    if not parts:
        return
    subjects = pd.concat(parts, ignore_index=True)
    subjects.to_csv(group_dir / "group_spatial_frequency_recordings.csv", index=False)
    summary = subjects.groupby(["spectrum_kind", "degree"], as_index=False).agg(
        n_recordings=("recording_id", "nunique"), mean_fraction=("mean_fraction", "mean"),
        median_fraction=("median_fraction", "median"),
        p25_fraction=("median_fraction", lambda x: x.quantile(.25)),
        p75_fraction=("median_fraction", lambda x: x.quantile(.75)),
    )
    summary.to_csv(group_dir / "group_spatial_frequency_summary.csv", index=False)
    specs = [("map", "map"), ("transition", "TD"), ("transition_residual", "TD_res"), ("transition_meed", "TD_FORM"), ("transition_meed_rho", "TD_rho"), ("transition_meed_psi", "TD_psi")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, (kind, title) in zip(axes.flat, specs):
        data = subjects.loc[subjects["spectrum_kind"].eq(kind)].sort_values("degree_order")
        order = data[["degree", "degree_order"]].drop_duplicates().sort_values("degree_order")
        x, labels = order["degree_order"].to_numpy() - 1, order["degree"].tolist()
        for _, recording in data.groupby("recording_id"):
            ax.plot(x, recording.set_index("degree_order").loc[order["degree_order"], "median_fraction"], color="0.7", lw=.6)
        group = summary.loc[summary["spectrum_kind"].eq(kind)].set_index("degree").loc[labels]
        ax.plot(x, group["median_fraction"], color="#0072b2", lw=2)
        ax.fill_between(x, group["p25_fraction"], group["p75_fraction"], color="#0072b2", alpha=.2)
        ax.set(xticks=x, xticklabels=labels, title=title, ylim=(0, 1), xlabel="spatial degree", ylabel="share of TD2" if kind != "map" else "share of map energy")
    fig.savefig(group_dir / "figure_S1_group_spatial_frequency.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
