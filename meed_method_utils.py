from __future__ import annotations

import json
import math
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import Rbf
from scipy.optimize import differential_evolution, minimize
try:
    from scipy.special import sph_harm_y
except ImportError:
    from scipy.special import sph_harm as sph_harm_y

import warnings

warnings.filterwarnings("ignore")

LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

#%% microstate loading

MAP_LABEL_REMAP = {
    4: {
        "Metamap_4.1": "C",
        "Metamap_4.2": "B",
        "Metamap_4.3": "A",
        "Metamap_4.4": "D",
    }
}

# MANUAL_REORDER_STR = {
#     4: "CBAD",
#     5: "DCBAE",
#     6: "DBCFEA",
#     7: "DFAEGBC",
#     8: "GACEFHBD",
# }

MANUAL_REORDER_STR = {
    4: "BACD",
    5: "ECDAB",
    6: "BAFECD",
    7: "AFECDGB",
    8: "FECAGBHD",
}


def remap_labels(labels, k):
    mapping = MAP_LABEL_REMAP.get(int(k), {})
    mapping_lc = {str(src).lower(): dst for src, dst in mapping.items()}
    return [mapping_lc.get(str(lb).lower(), lb) for lb in labels]


def center_l2_cols(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    return X / np.maximum(np.linalg.norm(X, axis=0, keepdims=True), eps)


def infer_labels_from_k4(entry, ref_entry):
    raw = np.asarray(entry["Maps"], dtype=float).T
    B = center_l2_cols(raw)

    ref_raw = np.asarray(ref_entry["Maps"], dtype=float).T
    ref_B = center_l2_cols(ref_raw)

    ref_labels = remap_labels(ref_entry.get("Labels", []), 4)

    corr = np.abs(ref_B.T @ B)
    best_ref = np.argmax(corr, axis=0)

    inferred_labels = [ref_labels[i] for i in best_ref]
    inferred_corr = [corr[i, j] for j, i in enumerate(best_ref)]

    return inferred_labels, inferred_corr, corr


def labels_from_manual_order(k):
    order_str = MANUAL_REORDER_STR.get(int(k))
    if order_str is None:
        return None
    return list(order_str)


def reorder_entry(entry, labels):
    labels = list(labels)

    def sort_key(i):
        return (str(labels[i]), i)

    order = np.array(sorted(range(len(labels)), key=sort_key), dtype=int)

    entry = copy.deepcopy(entry)
    entry["Maps"] = [entry["Maps"][i] for i in order]
    entry["Labels"] = [labels[i] for i in order]

    if "ColorMap" in entry and len(entry["ColorMap"]) == len(order):
        entry["ColorMap"] = [entry["ColorMap"][i] for i in order]

    entry["Order"] = [labels[i] for i in order]

    return entry, [labels[i] for i in order], order


def reorder_entry_by_manual_order(entry, manual_order_str):
    manual_labels = list(manual_order_str)
    if len(manual_labels) != len(entry["Maps"]):
        raise ValueError(
            f"Manual reorder string has length {len(manual_labels)}, "
            f"but entry has {len(entry['Maps'])} maps."
        )
    return reorder_entry(entry, manual_labels)


def load_microstates(json_path="metamaps_export.json", k=4, reorder_str=MANUAL_REORDER_STR, warn=True):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matches = [x for x in data if len(x["Maps"]) == k]
    if not matches:
        available = sorted({len(x["Maps"]) for x in data})
        raise ValueError(f"No map set with k={k}. Available map counts: {available}")

    ref_matches = [x for x in data if len(x["Maps"]) == 4]
    if not ref_matches:
        raise ValueError("Cannot infer A/B/C/D labels because no K=4 reference set exists.")

    ref_entry = ref_matches[0]
    reordered_all_sets = []
    checks = {}
    selected_entry = None
    selected_labels = None

    for x in data:
        kk = len(x["Maps"])

        if kk not in reorder_str:
            raise ValueError(f"No manual reorder string provided for K={kk}.")

        manual_labels = list(reorder_str[kk])

        if kk == 4:
            inferred_labels = remap_labels(x.get("Labels", [f"Map_{i+1}" for i in range(kk)]), 4)
            inferred_corr = [1.0] * kk
            corr = None
        else:
            inferred_labels, inferred_corr, corr = infer_labels_from_k4(x, ref_entry)

        checks[kk] = {
            "manual_unordered": manual_labels,
            "inferred_unordered": inferred_labels,
            "inferred_corr": inferred_corr,
            "corr_to_k4": corr,
        }

        manual_known = [lb for lb in manual_labels if lb in {"A", "B", "C", "D"}]
        inferred_known = [lb for lb in inferred_labels if lb in {"A", "B", "C", "D"}]

        if warn and kk != 4:
            if sorted(manual_known) != sorted(inferred_known):
                warnings.warn(
                    f"K={kk}: inferred canonical labels {inferred_known} do not match "
                    f"manual canonical labels {manual_known}. Plotting still uses manual order."
                )

        x_reordered, labels_reordered, _ = reorder_entry_by_manual_order(x, reorder_str[kk])
        reordered_all_sets.append(x_reordered)

        if kk == k and selected_entry is None:
            selected_entry = x_reordered
            selected_labels = labels_reordered

    entry = selected_entry

    raw = np.asarray(entry["Maps"], dtype=float).T
    B = center_l2_cols(raw)

    original_ch_names = [c["labels"] for c in entry["chanlocs"]]
    ch_names = [LEGACY_TO_MODERN.get(ch, ch) for ch in original_ch_names]

    info = mne.create_info(ch_names=ch_names, sfreq=1.0, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        on_missing="raise",
    )

    return {
        "all_sets": reordered_all_sets,
        "entry": entry,
        "raw": raw,
        "B": B,
        "labels": selected_labels,
        "checks": checks,
        "original_ch_names": original_ch_names,
        "ch_names": ch_names,
        "info": info,
    }
    
#%% microstate plotting


def plot_raw_and_normalized(raw, normed, labels, info):
    M = raw.shape[1]
    fig, axes = plt.subplots(2, M, figsize=(2.6 * M, 5.2), constrained_layout=True)
    if M == 1:
        axes = axes[:, None]

    vmax_raw = np.max(np.abs(raw))
    vmax_norm = np.max(np.abs(normed))
    for j, label in enumerate(labels):
        mne.viz.plot_topomap(
            raw[:, j], info, axes=axes[0, j], show=False, contours=4, sensors=True,
            cmap="RdBu_r", vlim=(-vmax_raw, vmax_raw), sphere="auto",
            extrapolate="head", image_interp="linear",
        )
        axes[0, j].set_title(f"raw\n{label}", fontsize=8)

        mne.viz.plot_topomap(
            normed[:, j], info, axes=axes[1, j], show=False, contours=4, sensors=True,
            cmap="RdBu_r", vlim=(-vmax_norm, vmax_norm), sphere="auto",
            extrapolate="head", image_interp="linear",
        )
        axes[1, j].set_title(f"centered + L2\n{label}", fontsize=8)
    return fig


def make_info_from_entry(entry):
    original_ch_names = [c["labels"] for c in entry["chanlocs"]]
    ch_names = [LEGACY_TO_MODERN.get(ch, ch) for ch in original_ch_names]

    info = mne.create_info(ch_names=ch_names, sfreq=1.0, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        on_missing="raise",
    )
    return info


def plot_reordered_metamaps_all_k(ds, normalized=False):
    all_sets = ds["all_sets"]

    n_solutions = len(all_sets)
    max_k = max(len(sol["Maps"]) for sol in all_sets)

    fig, axes = plt.subplots(
        n_solutions,
        max_k,
        figsize=(2.4 * max_k, 2.2 * n_solutions),
        constrained_layout=True,
    )

    if n_solutions == 1:
        axes = axes[None, :]
    if max_k == 1:
        axes = axes[:, None]

    maps_for_vlim = []
    for sol in all_sets:
        raw = np.asarray(sol["Maps"], dtype=float).T
        if normalized:
            raw = center_l2_cols(raw)
        maps_for_vlim.append(raw)

    vmax = max(np.max(np.abs(x)) for x in maps_for_vlim)

    last_im = None

    for s, sol in enumerate(all_sets):
        k = len(sol["Maps"])
        labels = sol.get("Labels", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:k])

        info = make_info_from_entry(sol)

        raw = np.asarray(sol["Maps"], dtype=float).T
        if normalized:
            raw = center_l2_cols(raw)

        for m in range(max_k):
            ax = axes[s, m]
            ax.set_axis_off()

            if m >= k:
                continue

            last_im, _ = mne.viz.plot_topomap(
                raw[:, m],
                info,
                axes=ax,
                show=False,
                contours=4,
                sensors=True,
                cmap="RdBu_r",
                vlim=(-vmax, vmax),
                sphere="auto",
                extrapolate="head",
                image_interp="linear",
            )

            ax.set_title(str(labels[m]), fontsize=10)

            if m == 0:
                ax.set_ylabel(f"K={k}", fontsize=11)

    fig.suptitle(
        "Reordered meta-microstates, manual order",
        fontsize=14,
    )

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.65, label="a.u.")

    return fig
    
#%% non-orthogonal base characterization


def plot_matrix(M, row_labels, col_labels, title="", cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(col_labels) + 3), max(4, 0.25 * len(row_labels) + 2)))
    im = ax.imshow(M, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.75)
    return fig


def plot_gram_heatmap(G, labels):
    fig, ax = plt.subplots(figsize=(5.3, 4.6))
    im = ax.imshow(G, vmin=-1, vmax=1, cmap="RdBu_r")
    short = [pretty_label(x) for x in labels]
    ax.set_xticks(np.arange(len(labels)), short, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), short)
    ax.set_title("Gram matrix: pairwise topographic correlations")
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.75)
    return fig


def plot_eigenspectrum(eig_df, metrics, comparison_spaces=None):
    """Plot original Gram eigenspectrum, optionally with fixed/full comparisons.

    Parameters
    ----------
    eig_df : DataFrame
        Output of gram_diagnostics for the original space.
    metrics : Series or dict
        Metrics for the original space.
    comparison_spaces : dict or None
        Optional mapping {name: B_space} where each B_space is Nch x M.
        For each entry, the function recomputes Gram eigenspectrum and overlays it.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.semilogy(eig_df["rank_index"], eig_df["eigenvalue"], marker="o", linewidth=2.0, label="original eigs")
    ax.axvline(metrics["effective_rank_entropy"], linestyle="--", linewidth=1.1, label=f"effective rank = {metrics['effective_rank_entropy']:.2f}")
    ax.axvline(metrics["participation_ratio"], linestyle=":", linewidth=1.3, label=f"participation ratio = {metrics['participation_ratio']:.2f}")

    summary_lines = [
        f"original: r_eff={float(metrics['effective_rank_entropy']):.2f}, r_pr={float(metrics['participation_ratio']):.2f}"
    ]

    if comparison_spaces:
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(3, len(comparison_spaces) + 1)))
        for i, (name, Bcmp) in enumerate(comparison_spaces.items(), start=1):
            _, m_cmp, e_cmp = gram_diagnostics(center_l2_cols(Bcmp))
            c = colors[i]
            ax.semilogy(
                e_cmp["rank_index"],
                e_cmp["eigenvalue"],
                marker="o",
                linewidth=1.5,
                alpha=0.9,
                color=c,
                label=f"{name} eigs",
            )
            ax.axvline(float(m_cmp["effective_rank_entropy"]), linestyle="--", linewidth=0.9, alpha=0.55, color=c)
            ax.axvline(float(m_cmp["participation_ratio"]), linestyle=":", linewidth=1.0, alpha=0.55, color=c)
            summary_lines.append(
                f"{name}: r_eff={float(m_cmp['effective_rank_entropy']):.2f}, r_pr={float(m_cmp['participation_ratio']):.2f}"
            )

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Gram eigenvalue (log scale)")

    cond_full = metrics.get("condition_number_full", metrics.get("condition_number", np.nan))
    cond_nz = metrics.get("condition_number_nonzero_spectrum", np.nan)
    if np.isinf(cond_full):
        title = f"Gram eigenspectrum (full cond = inf; nonzero cond = {cond_nz:.2e})"
    else:
        title = f"Gram eigenspectrum (condition number = {cond_full:.2e})"
    ax.set_title(title)
    ax.set_xticks(eig_df["rank_index"])
    ax.text(
        1.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    ax.legend()
    return fig


def plot_dimension_by_m(dim_df):
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(dim_df["M"], dim_df["effective_rank_entropy"], marker="o", label="effective rank")
    ax.plot(dim_df["M"], dim_df["participation_ratio"], marker="o", label="participation ratio")
    ax.set_xlabel("number of maps M")
    ax.set_ylabel("effective dimension")
    ax.set_title("Effective dimension as map count increases")
    ax.legend()
    return fig



def gram_diagnostics(B, tol=None):
    """Diagnostics for a column-wise map matrix.

    B is Nch x M. Columns are expected to be centered and normalized if the
    Gram matrix is meant to be interpreted as topographic correlation.

    Important: after an intentional reduction, rank deficiency is expected.
    For example, four theta-phi fixed-dipole templates live in a 3D linear
    embedding, so their M x M Gram matrix has rank at most 3 and the full
    condition number is infinite. This is not a failure and is not caused by
    retaining rho. It is the linear embedding of a 2D angular manifold.
    """
    B = np.asarray(B, dtype=float)
    G = B.T @ B
    eig = np.sort(np.linalg.eigvalsh(G))[::-1]

    if tol is None:
        tol = np.finfo(float).eps * max(G.shape) * max(float(eig[0]), 1.0)

    nonzero = eig > tol
    rank = int(np.sum(nonzero))
    n_tiny = int(np.sum(~nonzero))

    p = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)
    H = -np.sum(p[p > 0] * np.log(p[p > 0]))

    condition_full = float(eig[0] / eig[-1]) if eig[-1] > tol else float("inf")
    if rank >= 2:
        condition_nonzero = float(eig[0] / eig[nonzero][-1])
    elif rank == 1:
        condition_nonzero = 1.0
    else:
        condition_nonzero = float("nan")

    offdiag = G[np.triu_indices_from(G, 1)]
    metrics = pd.Series(
        {
            "n_maps": B.shape[1],
            "n_channels": B.shape[0],
            "rank": rank,
            "n_tiny_eigenvalues": n_tiny,
            "condition_number_full": condition_full,
            "condition_number_nonzero_spectrum": condition_nonzero,
            "effective_rank_entropy": float(np.exp(H)) if eig.sum() > 0 else float("nan"),
            "participation_ratio": float((eig.sum() ** 2) / np.sum(eig ** 2)) if np.sum(eig ** 2) > 0 else float("nan"),
            "mean_abs_offdiag_corr": float(np.mean(np.abs(offdiag))) if len(offdiag) else 0.0,
            "max_abs_offdiag_corr": float(np.max(np.abs(offdiag))) if len(offdiag) else 0.0,
        }
    )

    # Backward-compatible alias. Full condition is the literal basis condition;
    # it can be infinite after dimensionality reduction by construction.
    metrics["condition_number"] = metrics["condition_number_full"]

    eig_df = pd.DataFrame(
        {
            "rank_index": np.arange(1, len(eig) + 1),
            "eigenvalue": eig,
            "is_nonzero": nonzero,
            "fraction": eig / eig.sum() if eig.sum() > 0 else np.nan,
            "cumulative_fraction": np.cumsum(eig / eig.sum()) if eig.sum() > 0 else np.nan,
        }
    )
    return G, metrics, eig_df


def effective_dimension_by_m(all_sets):
    rows = []
    for entry in all_sets:
        maps = np.asarray(entry["Maps"], dtype=float).T
        B = center_l2_cols(maps)
        _, metrics, _ = gram_diagnostics(B)
        rows.append(
            {
                "M": B.shape[1],
                "effective_rank_entropy": metrics["effective_rank_entropy"],
                "participation_ratio": metrics["participation_ratio"],
                "condition_number": metrics["condition_number"],
            }
        )
    return pd.DataFrame(rows).sort_values("M").reset_index(drop=True)


def effective_dimension_by_m_models(all_sets, fast_full=True, seed=1):
    """Effective-rank / participation-ratio curves for original, fixed, full.

    Parameters
    ----------
    all_sets : list
        List of map-set entries (each with Maps and chanlocs).
    fast_full : bool
        Use fast full-dipole fitting for each K.
    seed : int
        Base seed for deterministic full-dipole fits.
    """
    rows = []
    for entry in all_sets:
        k = len(entry["Maps"])
        raw = np.asarray(entry["Maps"], dtype=float).T
        B = center_l2_cols(raw)

        original_ch_names = [c["labels"] for c in entry["chanlocs"]]
        ch_names = [LEGACY_TO_MODERN.get(ch, ch) for ch in original_ch_names]
        info = mne.create_info(ch_names=ch_names, sfreq=1.0, ch_types="eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="raise")

        geom = sensor_geometry(info)
        _, _, Bfixed, _ = project_fixed_dipole(B, geom["S_orth"])

        spaces = {"original": B, "fixed_MEED_theta_phi": Bfixed}
        try:
            Bfull, _ = fit_full_dipole(B, geom["S"], fast=fast_full, seed=seed + k)
            spaces["full_dipole"] = Bfull
        except Exception:
            pass

        for rep, Brep in spaces.items():
            _, metrics, _ = gram_diagnostics(center_l2_cols(Brep))
            rows.append(
                {
                    "M": k,
                    "representation": rep,
                    "effective_rank_entropy": float(metrics["effective_rank_entropy"]),
                    "participation_ratio": float(metrics["participation_ratio"]),
                }
            )

    return pd.DataFrame(rows).sort_values(["M", "representation"]).reset_index(drop=True)


def plot_dimension_by_m_models(dim_models_df):
    """Plot effective-rank and participation-ratio curves by representation."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True, sharex=True)
    reps = list(dim_models_df["representation"].drop_duplicates())

    for rep in reps:
        d = dim_models_df[dim_models_df["representation"] == rep].sort_values("M")
        axes[0].plot(d["M"], d["effective_rank_entropy"], marker="o", label=rep)
        axes[1].plot(d["M"], d["participation_ratio"], marker="o", label=rep)

    axes[0].set_title("effective rank vs K")
    axes[0].set_xlabel("number of maps K")
    axes[0].set_ylabel("effective rank (entropy)")

    axes[1].set_title("participation ratio vs K")
    axes[1].set_xlabel("number of maps K")
    axes[1].set_ylabel("participation ratio")

    axes[0].legend()
    axes[1].legend()
    return fig

#%% microstate-equivalent electric dipole: fitting, visualization and reconstruction quality


def center_l2(v):
    v = np.asarray(v, dtype=float)
    v = v - np.mean(v)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def pretty_label(label):
    s = str(label)
    return s.replace("MetaMap_", "").replace("Metamap_", "")


def anat_azimuth_raw(x, y):
    old = np.degrees(np.arctan2(y, x))
    return float(wrap180(90.0 - old))


def anat_azimuth(x, y):
    theta = anat_azimuth_raw(x, y)
    if theta > 90.0:
        theta -= 180.0
    elif theta < -90.0:
        theta += 180.0
    return theta


def elevation(v):
    v = np.asarray(v, dtype=float)
    r = np.linalg.norm(v)
    if r == 0:
        return float("nan")
    return float(np.degrees(np.arcsin(np.clip(v[2] / r, -1, 1))))


def canonical_theta_phi_from_vector(v, eps=1e-12):
    """Return polarity-canonical anatomical angles and direction.

    Canonicalization uses microstate polarity equivalence (v and -v are
    equivalent): azimuth is always folded to [-90, +90], and elevation is
    adjusted consistently by flipping the direction when needed.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= eps:
        return np.nan, np.nan, np.zeros_like(v)

    u = v / n
    theta_raw = anat_azimuth_raw(u[0], u[1])
    if theta_raw > 90.0 or theta_raw < -90.0:
        u = -u
    theta = anat_azimuth(u[0], u[1])
    phi = elevation(u)
    return float(theta), float(phi), u


def unit_from_angles(theta_deg, phi_deg):
    """Return unit vector from anatomical azimuth theta and elevation phi."""
    old_az = np.deg2rad(90.0 - theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.array([
        np.cos(phi) * np.cos(old_az),
        np.cos(phi) * np.sin(old_az),
        np.sin(phi),
    ])


def inv_sqrtm(M, eps=1e-12):
    vals, vecs = np.linalg.eigh(M)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T, vals


def r2_score(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def polarity_sign(y, yhat):
    """Return +1 or -1 to maximize map correlation under polarity equivalence."""
    y0 = center_l2(y)
    h0 = center_l2(yhat)
    return 1.0 if float(np.dot(y0, h0)) >= 0 else -1.0


def align_polarity(y, yhat):
    """Align prediction polarity to target map polarity."""
    return polarity_sign(y, yhat) * np.asarray(yhat, dtype=float)


def r2_score_polarity_invariant(y, yhat):
    """R2 using the better orientation between yhat and -yhat."""
    yhat_aligned = align_polarity(y, yhat)
    return r2_score(y, yhat_aligned)


def fit_sphere_fallback(pos):
    A = np.column_stack([2 * pos, np.ones(pos.shape[0])])
    b = np.sum(pos ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    radius = np.sqrt(sol[3] + np.sum(center ** 2))
    return center, float(radius)


def sensor_geometry(info, center_mode="fit", project_to_sphere=True):
    pos = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=float)

    if center_mode == "zero":
        origin = np.zeros(3)
        radius = float(np.mean(np.linalg.norm(pos, axis=1)))
    else:
        try:
            radius, origin, _ = mne.bem.fit_sphere_to_headshape(info, units="m", verbose=False)
            origin = np.asarray(origin, dtype=float)
            radius = float(radius)
        except Exception:
            origin, radius = fit_sphere_fallback(pos)

    S = (pos - origin[None, :]) / radius
    if project_to_sphere:
        S = S / np.linalg.norm(S, axis=1, keepdims=True)

    Sc = S - S.mean(axis=0, keepdims=True)
    M = Sc.T @ Sc
    M_inv_sqrt, M_eig = inv_sqrtm(M)
    S_orth = Sc @ M_inv_sqrt

    return {
        "pos_m": pos,
        "origin_m": origin,
        "radius_m": radius,
        "S": S,
        "Sc": Sc,
        "M": M,
        "M_eig": M_eig,
        "M_condition": float(M_eig.max() / M_eig.min()),
        "S_orth": S_orth,
        "S_orth_residual_fro": float(np.linalg.norm(S_orth.T @ S_orth - np.eye(3), ord="fro")),
    }


def project_fixed_dipole(B, S_orth):
    """Project maps into the geometry-corrected fixed-dipole subspace.

    Returns
    -------
    Q : array, shape (3, M)
        Linear projection coordinates. Their norm is a diagnostic, not a retained coordinate.
    Bproj : array, shape (Nch, M)
        Orthogonal projection retaining projection strength.
    Bunit : array, shape (Nch, M)
        Unit-norm theta-phi templates. This is the actual 2D reduced template family.
    df : DataFrame
        Contains theta, phi, dipolarity, projection R2, and theta-phi-only R2.
    """
    Q = S_orth.T @ B
    Bproj = S_orth @ Q

    rows = []
    U = np.zeros_like(Q)
    Bunit_cols = []
    for m in range(B.shape[1]):
        q = Q[:, m]
        dipolarity = np.linalg.norm(q)
        theta, phi, u = canonical_theta_phi_from_vector(q)
        U[:, m] = u
        Bunit_m = S_orth @ u
        Bunit_m = align_polarity(B[:, m], Bunit_m)
        Bunit_cols.append(Bunit_m)
        rows.append(
            {
                "map_index": m,
                "q_x": q[0],
                "q_y": q[1],
                "q_z": q[2],
                "dipolarity": dipolarity,
                "projection_R2": dipolarity ** 2,
                "R2": r2_score_polarity_invariant(B[:, m], Bunit_m),
                "theta_deg": theta,
                "phi_deg": phi,
            }
        )

    Bunit = np.column_stack(Bunit_cols)
    return Q, Bproj, Bunit, pd.DataFrame(rows)


def fit_full_dipole(B, S, fast=True, seed=1):
    def basis(r):
        d = S - r[None, :]
        dist = np.linalg.norm(d, axis=1)
        A = d / (dist[:, None] ** 3)
        return np.column_stack([np.ones(S.shape[0]), A])

    def fit_at_r(b, r):
        A = basis(r)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = center_l2(A @ coef)
        corr = float(np.dot(center_l2(b), pred))
        return coef, pred, corr, r2_score(b, pred)

    def objective(r, b):
        if np.linalg.norm(r) >= 0.95:
            return 1e3 + 1e3 * (np.linalg.norm(r) - 0.95) ** 2
        _, _, corr, _ = fit_at_r(b, r)
        return 1.0 - corr

    rows = []
    Bhat = []
    bounds = [(-0.85, 0.85), (-0.85, 0.85), (-0.20, 0.85)]

    for m in range(B.shape[1]):
        b = B[:, m]

        if fast:
            starts = [np.zeros(3), [0.2, 0, 0], [-0.2, 0, 0], [0, 0.2, 0], [0, -0.2, 0], [0, 0, 0.2]]
            best = None
            for st in starts:
                res = minimize(lambda r: objective(r, b), np.asarray(st), method="Nelder-Mead", options={"maxiter": 700})
                if best is None or res.fun < best.fun:
                    best = res
            loc = best.x
        else:
            de = differential_evolution(
                lambda r: objective(r, b),
                bounds=bounds,
                seed=seed,
                maxiter=70,
                popsize=8,
                polish=False,
                workers=1,
            )
            res = minimize(
                lambda r: objective(r, b),
                de.x,
                method="Nelder-Mead",
                options={"maxiter": 900, "xatol": 1e-8, "fatol": 1e-8},
            )
            loc = res.x if np.linalg.norm(res.x) < 0.95 else de.x

        coef, pred, corr, r2 = fit_at_r(b, loc)
        mom = coef[1:4]
        mom = safe_unit(mom)
        loc_theta, loc_phi, _ = canonical_theta_phi_from_vector(loc)
        mom_theta, mom_phi, _ = canonical_theta_phi_from_vector(mom)
        Bhat.append(pred)
        rows.append(
            {
                "map_index": m,
                "loc_x": loc[0],
                "loc_y": loc[1],
                "loc_z": loc[2],
                "loc_radius": np.linalg.norm(loc),
                "loc_azimuth_anat_deg": loc_theta,
                "loc_elevation_deg": loc_phi,
                "moment_x": mom[0],
                "moment_y": mom[1],
                "moment_z": mom[2],
                "moment_azimuth_anat_deg": mom_theta,
                "moment_elevation_deg": mom_phi,
                "corr": corr,
                "R2": r2,
            }
        )

    return np.column_stack(Bhat), pd.DataFrame(rows)


def transformed_space_diagnostics(spaces, intrinsic_dims=None):
    """Recompute map-space diagnostics for several representations.

    `spaces` maps names to Nch x M matrices. The function reports literal
    linear-embedding diagnostics. If a representation is intentionally reduced,
    full condition number may be infinite because rank < M.

    `intrinsic_dims` can declare the intended model dimension, e.g.
    {"fixed_theta_phi": 2}. This is reported separately from linear rank.
    """
    intrinsic_dims = intrinsic_dims or {}
    rows = []
    for name, B in spaces.items():
        _, metrics, _ = gram_diagnostics(center_l2_cols(B))
        for metric, value in metrics.items():
            rows.append({"space": name, "metric": metric, "value": value})
        if name in intrinsic_dims:
            rows.append({"space": name, "metric": "declared_intrinsic_dimension", "value": intrinsic_dims[name]})
    return pd.DataFrame(rows)


def real_sph_basis(S, lmax=4):
    Sn = S / np.linalg.norm(S, axis=1, keepdims=True)
    x, y, z = Sn[:, 0], Sn[:, 1], Sn[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    cols, degs = [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            if m < 0:
                col = np.sqrt(2) * (-1) ** m * Y.imag
            elif m == 0:
                col = Y.real
            else:
                col = np.sqrt(2) * (-1) ** m * Y.real
            cols.append(col)
            degs.append(l)
    return np.column_stack(cols), np.array(degs)


#%% sensitivity and null simulations


def fixed_dipole_stability(B, labels, S_orth, S):
    Q = S_orth.T @ B
    U = Q / np.linalg.norm(Q, axis=0, keepdims=True)

    rows = []
    for m, lab in enumerate(labels):
        angles = []
        for drop in range(B.shape[0]):
            keep = np.ones(B.shape[0], dtype=bool)
            keep[drop] = False
            Sc = S[keep] - S[keep].mean(axis=0, keepdims=True)
            Minvsqrt, _ = inv_sqrtm(Sc.T @ Sc)
            Sdrop = Sc @ Minvsqrt
            q = Sdrop.T @ B[keep, m]
            q = q / np.linalg.norm(q)
            angle = np.degrees(np.arccos(np.clip(np.dot(U[:, m], q), -1, 1)))
            angles.append(angle)
        rows.append(
            {
                "label": lab,
                "jackknife_mean_angle_deg": float(np.mean(angles)),
                "jackknife_max_angle_deg": float(np.max(angles)),
                "jackknife_sd_angle_deg": float(np.std(angles)),
            }
        )
    return pd.DataFrame(rows)


def fixed_dipole_null_simulations(S_orth, S, n=250, seed=1):
    rng = np.random.default_rng(seed)
    Y, degs = real_sph_basis(S, 4)
    rows = []

    for kind in ["iid_random", "low_order_l1_l2", "high_order_l3_l4"]:
        for _ in range(n):
            if kind == "iid_random":
                v = rng.normal(size=S.shape[0])
            else:
                coeff = np.zeros(Y.shape[1])
                idx = np.where((degs == 1) | (degs == 2))[0] if kind == "low_order_l1_l2" else np.where((degs == 3) | (degs == 4))[0]
                coeff[idx] = rng.normal(size=len(idx))
                v = Y @ coeff + 0.05 * rng.normal(size=S.shape[0])

            b = center_l2(v)
            q = S_orth.T @ b
            rows.append(
                {
                    "kind": kind,
                    "projection_R2": float(np.sum(q ** 2)),
                    "dipolarity": float(np.linalg.norm(q)),
                }
            )
    return pd.DataFrame(rows)


def plot_geometry(geometry):
    residual = geometry["S_orth"].T @ geometry["S_orth"] - np.eye(3)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    im0 = axes[0].imshow(geometry["M"], cmap="viridis")
    axes[0].set_title(f"$M_S=S_c^T S_c$ (cond={geometry['M_condition']:.2f})")
    fig.colorbar(im0, ax=axes[0], shrink=0.75)
    im1 = axes[1].imshow(residual, cmap="RdBu_r")
    axes[1].set_title(f"$S_{{orth}}^T S_{{orth}}-I$ (Fro={geometry['S_orth_residual_fro']:.2e})")
    fig.colorbar(im1, ax=axes[1], shrink=0.75)
    return fig


#%% usage with real data

#%% generic

def pair_topomap_rgba(original, reduced, info, vlim):
    fig, axes = plt.subplots(1, 2, figsize=(1.75, 0.86), dpi=120)
    fig.patch.set_alpha(0)
    for ax, vals, title in zip(axes, [original, reduced], ["orig", "red"]):
        ax.set_facecolor((1, 1, 1, 0))
        mne.viz.plot_topomap(
            vals, info, axes=ax, show=False, contours=2, sensors=False,
            cmap="RdBu_r", vlim=(-vlim, vlim), sphere="auto",
            extrapolate="head", image_interp="linear",
        )
        ax.set_title(title, fontsize=5)
        ax.axis("off")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def add_inset(ax, xy, img, zoom=0.42):
    ab = AnnotationBbox(
        OffsetImage(img, zoom=zoom), xy, frameon=True, pad=0.04,
        bboxprops=dict(edgecolor="black", linewidth=0.6),
    )
    ax.add_artist(ab)


def padded(v, frac=0.30):
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    span = hi - lo
    if span == 0:
        span = 1
    return lo - frac * span, hi + frac * span


def plot_pca_space(B, Bhat, labels, scores, info):
    vlim = float(max(np.max(np.abs(B)), np.max(np.abs(Bhat))))
    x = scores[:, 0]
    y = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(x)
    fig, ax = plt.subplots(figsize=(7.0, 5.7))
    ax.scatter(x, y, s=20)
    for i, lab in enumerate(labels):
        add_inset(ax, (x[i], y[i]), pair_topomap_rgba(B[:, i], Bhat[:, i], info, vlim))
        ax.text(x[i], y[i], pretty_label(lab), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA-2 space with original and reduced topomaps")
    ax.set_xlim(*padded(x))
    ax.set_ylim(*padded(y))
    return fig


def plot_fixed_space(B, Bhat_unit, labels, fixed_df, info):
    vlim = float(max(np.max(np.abs(B)), np.max(np.abs(Bhat_unit))))
    x = fixed_df["theta_deg"].values
    y = fixed_df["phi_deg"].values
    fig, ax = plt.subplots(figsize=(7.0, 5.7))
    ax.scatter(x, y, s=20)
    for i, lab in enumerate(labels):
        add_inset(ax, (x[i], y[i]), pair_topomap_rgba(B[:, i], Bhat_unit[:, i], info, vlim))
        ax.text(x[i], y[i], pretty_label(lab), fontsize=7, ha="center", va="bottom")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(r"$\theta$ anatomical azimuth (deg, polarity-canonical: -90 left, 0 front, +90 right)")
    ax.set_ylabel(r"$\phi$ elevation (deg)")
    ax.set_title(r"Fixed-dipole angular space: $D=[\theta,\phi]$")
    ax.set_xlim(*padded(x))
    ax.set_ylim(*padded(y))
    return fig


def plot_full_dipole_space(B, Bfull, labels, full_df, info):
    """Space characterization for full dipole: location and moment angles."""
    if Bfull is None or full_df is None:
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        ax.axis("off")
        ax.text(0.02, 0.65, "Full dipole not available.", fontsize=12)
        return fig

    vlim = float(max(np.max(np.abs(B)), np.max(np.abs(Bfull))))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), constrained_layout=True)

    panels = [
        ("loc_azimuth_anat_deg", "loc_elevation_deg", "Full dipole location space"),
        ("moment_azimuth_anat_deg", "moment_elevation_deg", "Full dipole moment space"),
    ]

    for ax, (xcol, ycol, title) in zip(axes, panels):
        x = full_df[xcol].values
        y = full_df[ycol].values
        ax.scatter(x, y, s=20)
        for i, lab in enumerate(labels):
            bfull_i = align_polarity(B[:, i], Bfull[:, i])
            add_inset(ax, (x[i], y[i]), pair_topomap_rgba(B[:, i], bfull_i, info, vlim))
            ax.text(x[i], y[i], pretty_label(lab), fontsize=7, ha="center", va="bottom")
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel(r"$\theta$ anatomical azimuth (deg, polarity-canonical)")
        ax.set_ylabel(r"$\phi$ elevation (deg)")
        ax.set_title(title)
        ax.set_xlim(*padded(x))
        ax.set_ylim(*padded(y))
    return fig


def plot_reconstruction_gallery(B, Bfixed_unit, Bpca, Bfull, labels, info):
    rows = [("original", B), ("theta-phi fixed dipole", Bfixed_unit), ("PCA-2", Bpca)]
    if Bfull is not None:
        rows.append(("full dipole", Bfull))
    M = B.shape[1]
    vlim = float(max(np.max(np.abs(mat)) for _, mat in rows))

    fig, axes = plt.subplots(len(rows), M, figsize=(2.55 * M, 2.35 * len(rows)), constrained_layout=True)
    if M == 1:
        axes = axes[:, None]

    for r, (name, mat) in enumerate(rows):
        for j, lab in enumerate(labels):
            ax = axes[r, j]
            vals = mat[:, j]
            if name != "original":
                vals = align_polarity(B[:, j], vals)
            mne.viz.plot_topomap(
                vals, info, axes=ax, show=False, contours=4, sensors=True,
                cmap="RdBu_r", vlim=(-vlim, vlim), sphere="auto",
                extrapolate="head", image_interp="linear",
            )
            ax.set_title(f"{name}\n{lab}", fontsize=8)
    return fig


def plot_r2_comparison(labels, pca_df, fixed_df, full_df=None):
    idx = np.arange(len(labels))
    width = 0.27 if full_df is not None else 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(idx - width, pca_df["R2"], width, label="PCA-2")
    ax.bar(idx, fixed_df["R2"], width, label="fixed dipole theta-phi")

    if full_df is not None:
        ax.bar(idx + width, full_df["R2"], width, label="full dipole")

    ax.set_xticks(idx, [pretty_label(x) for x in labels])
    ax.set_ylabel(r"$R^2$")
    ax.set_title("Reconstruction quality")
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    mins = [pca_df["R2"].min(), fixed_df["R2"].min()]
    if full_df is not None:
        mins.append(full_df["R2"].min())
    ymin_zoom = max(0.0, min(mins) - 0.05)

    axins = inset_axes(ax, width="45%", height="45%", loc="lower left", borderpad=1.5)
    axins.bar(idx - width, pca_df["R2"], width)
    axins.bar(idx, fixed_df["R2"], width)
    if full_df is not None:
        axins.bar(idx + width, full_df["R2"], width)
    axins.set_ylim(ymin_zoom, 1.0)
    axins.set_xlim(-0.7, len(labels) - 0.3)
    axins.set_xticks(idx)
    axins.set_xticklabels([pretty_label(x) for x in labels], rotation=45, fontsize=7)
    axins.tick_params(axis="y", labelsize=7)
    return fig


def draw_plane(ax):
    t = np.linspace(0, 2 * np.pi, 180)
    ax.plot(np.cos(t), np.sin(t), 0 * t, color="black", linewidth=0.9)
    nose = np.array([[-0.08, 1.0, 0], [0, 1.13, 0], [0.08, 1.0, 0]])
    ax.plot(nose[:, 0], nose[:, 1], nose[:, 2], color="black", linewidth=0.9)


def sphere_values(S, values):
    theta = np.linspace(0, 2 * np.pi, 68)
    phi = np.linspace(0, np.pi / 2, 34)
    TH, PH = np.meshgrid(theta, phi)
    XS = np.sin(PH) * np.cos(TH)
    YS = np.sin(PH) * np.sin(TH)
    ZS = np.cos(PH)
    rbf = Rbf(S[:, 0], S[:, 1], S[:, 2], values, function="multiquadric", smooth=0.01)
    V = np.clip(rbf(XS, YS, ZS), -0.65, 0.65)
    return XS, YS, ZS, V


def plot_dipole_sphere(B, labels, fixed_df, S, elev=60, azim=140):
    M = B.shape[1]
    cols = min(M, 2)
    rows = int(math.ceil(M / cols))
    fig = plt.figure(figsize=(4.2 * cols, 4.0 * rows))
    norm = Normalize(vmin=-0.65, vmax=0.65)
    cmap = plt.get_cmap("RdBu_r")

    for i in range(M):
        row = fixed_df.iloc[i]
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        XS, YS, ZS, V = sphere_values(S, B[:, i])
        fc = cmap(norm(V))
        fc[..., -1] = 0.30
        ax.plot_surface(XS, YS, ZS, facecolors=fc, linewidth=0.15, edgecolor=(0, 0, 0, 0.10), antialiased=True, shade=False)
        draw_plane(ax)
        ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=B[:, i], cmap="RdBu_r", vmin=-0.65, vmax=0.65, s=18, edgecolor="black", linewidth=0.2, depthshade=False)
        direction = unit_from_angles(row["theta_deg"], row["phi_deg"])
        ax.scatter([0], [0], [0], s=40, c="black", depthshade=False)
        ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], length=0.55, normalize=True, linewidth=2.2, color="black", arrow_length_ratio=0.20)
        ax.set_title(f"{labels[i]}\ntheta={row['theta_deg']:.1f}, phi={row['phi_deg']:.1f}", fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.18, 1.18)
        ax.set_zlim(-0.18, 1.05)
        ax.set_box_aspect([1, 1, 0.65])
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_full_dipole_sphere(B, labels, full_df, S, elev=60, azim=140):
    """Plot each full-dipole fit with its moment arrow at the fitted location."""
    M = B.shape[1]
    cols = min(M, 2)
    rows = int(math.ceil(M / cols))
    fig = plt.figure(figsize=(4.2 * cols, 4.0 * rows))
    norm = Normalize(vmin=-0.65, vmax=0.65)
    cmap = plt.get_cmap("RdBu_r")

    for i in range(M):
        row = full_df.iloc[i]
        location = np.array([row["loc_x"], row["loc_y"], row["loc_z"]], dtype=float)
        moment = np.array(
            [row["moment_x"], row["moment_y"], row["moment_z"]], dtype=float,
        )
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        XS, YS, ZS, V = sphere_values(S, B[:, i])
        fc = cmap(norm(V))
        fc[..., -1] = 0.30
        ax.plot_surface(XS, YS, ZS, facecolors=fc, linewidth=0.15, edgecolor=(0, 0, 0, 0.10), antialiased=True, shade=False)
        draw_plane(ax)
        ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=B[:, i], cmap="RdBu_r", vmin=-0.65, vmax=0.65, s=18, edgecolor="black", linewidth=0.2, depthshade=False)
        ax.scatter(*location, s=40, c="black", depthshade=False)
        ax.quiver(*location, *moment, length=0.55, normalize=True, linewidth=2.2, color="black", arrow_length_ratio=0.20)
        ax.set_title(
            f"{labels[i]}\nlocation=({location[0]:.2f}, {location[1]:.2f}, {location[2]:.2f})",
            fontsize=8,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.18, 1.18)
        ax.set_zlim(-0.18, 1.05)
        ax.set_box_aspect([1, 1, 0.65])
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_jackknife_stability(jack_df):
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    labels = (
        jack_df["label"]
        .astype(str)
        .str.replace("MetaMap_", "", regex=False)
        .str.replace("Metamap_", "", regex=False)
    )
    ax.bar(labels, jack_df["jackknife_max_angle_deg"])
    ax.set_ylabel("max angular deviation (deg)")
    ax.set_title("Leave-one-electrode-out fixed-dipole stability")
    return fig


def plot_null_simulations(null_df):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    kinds = ["iid_random", "low_order_l1_l2", "high_order_l3_l4"]
    ax.boxplot([null_df.loc[null_df["kind"] == k, "projection_R2"] for k in kinds], labels=kinds, showfliers=False)
    ax.set_ylabel("fixed-dipole projection R2")
    ax.set_title("Null simulations: arbitrary versus smooth maps")
    ax.tick_params(axis="x", labelrotation=20)
    return fig


def metric_table_wide(metrics_long):
    return metrics_long.pivot(index="metric", columns="space", values="value")

# ---------------------------------------------------------------------
# v5 additions: MEED-only workflow, fixed versus full dipole comparison,
# per-map local angular derivative maps, all-M spider plots, and full/fixed
# robustness diagnostics.
# ---------------------------------------------------------------------

def safe_unit_columns(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    return X / np.maximum(np.linalg.norm(X, axis=0, keepdims=True), eps)


def antipodal_angle(u, v, degrees=True):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u = u / np.maximum(np.linalg.norm(u), 1e-12)
    v = v / np.maximum(np.linalg.norm(v), 1e-12)
    a = np.arccos(np.clip(abs(float(np.dot(u, v))), -1.0, 1.0))
    return float(np.degrees(a) if degrees else a)


def fixed_meed(B, S_orth):
    """Fixed-location microstate-equivalent electric dipole representation.

    Returns exact angular MEED coordinates D_MEED with shape 2 x M.
    `dipolarity` and `projection_R2` are diagnostics only, not retained
    coordinates.
    """
    Q, Bproj, Bunit, df = project_fixed_dipole(B, S_orth)
    rho = df["dipolarity"].values
    U = Q / np.maximum(rho[None, :], 1e-12)
    D = np.vstack([df["theta_deg"].values, df["phi_deg"].values])
    return {
        "Q": Q,
        "U": U,
        "D": D,
        "B_projection": Bproj,
        "B_meed": Bunit,
        "df": df,
    }


def tangent_basis_from_direction(u0):
    """Local orthonormal tangent basis at direction u0.

    First column is local theta direction, second column is local phi direction.
    """
    u0 = np.asarray(u0, dtype=float)
    u0 = u0 / np.maximum(np.linalg.norm(u0), 1e-12)
    theta0 = anat_azimuth(u0[0], u0[1])
    phi0 = elevation(u0)

    old_az = np.deg2rad(90.0 - theta0)
    phi = np.deg2rad(phi0)

    e_theta = np.array([
        -np.cos(phi) * np.sin(old_az),
        np.cos(phi) * np.cos(old_az),
        0.0,
    ])
    e_phi = np.array([
        -np.sin(phi) * np.cos(old_az),
        -np.sin(phi) * np.sin(old_az),
        np.cos(phi),
    ])

    e_theta = e_theta / np.maximum(np.linalg.norm(e_theta), 1e-12)
    e_phi = e_phi - e_theta * np.dot(e_theta, e_phi)
    e_phi = e_phi / np.maximum(np.linalg.norm(e_phi), 1e-12)
    return np.column_stack([e_theta, e_phi]), theta0, phi0


def local_meed_derivative_maps(S_orth, U):
    """Compute local dV/dtheta and dV/dphi maps for each MEED direction.

    The returned maps are topographic effects of small angular changes around
    each microstate separately. They are not global basis maps.
    """
    U = safe_unit_columns(U)
    rows = []
    theta_maps = []
    phi_maps = []
    ref_maps = []

    for m in range(U.shape[1]):
        T, theta0, phi0 = tangent_basis_from_direction(U[:, m])
        dtheta = center_l2(S_orth @ T[:, 0])
        dphi = center_l2(S_orth @ T[:, 1])
        ref = center_l2(S_orth @ U[:, m])
        theta_maps.append(dtheta)
        phi_maps.append(dphi)
        ref_maps.append(ref)
        rows.append({"map_index": m, "theta_deg": theta0, "phi_deg": phi0})

    return {
        "theta_maps": np.column_stack(theta_maps),
        "phi_maps": np.column_stack(phi_maps),
        "reference_maps": np.column_stack(ref_maps),
        "df": pd.DataFrame(rows),
    }


def plot_local_meed_derivative_maps(B, labels, derivatives, info):
    """For each microstate, show original, MEED, dV/dtheta, dV/dphi."""
    theta_maps = derivatives["theta_maps"]
    phi_maps = derivatives["phi_maps"]
    ref_maps = derivatives["reference_maps"]
    M = B.shape[1]
    fig, axes = plt.subplots(4, M, figsize=(2.55 * M, 9.0), constrained_layout=True)
    if M == 1:
        axes = axes[:, None]

    rows = [
        ("original", B),
        ("MEED template", ref_maps),
        (r"local $dV/d\theta$", theta_maps),
        (r"local $dV/d\phi$", phi_maps),
    ]
    vmax = max(1e-12, float(max(np.max(np.abs(mat)) for _, mat in rows)))

    for r, (name, mat) in enumerate(rows):
        for j, lab in enumerate(labels):
            ax = axes[r, j]
            mne.viz.plot_topomap(
                mat[:, j], info, axes=ax, show=False, contours=4, sensors=True,
                cmap="RdBu_r", vlim=(-vmax, vmax), sphere="auto",
                extrapolate="head", image_interp="linear",
            )
            ax.set_title(f"{name}\n{pretty_label(lab)}", fontsize=8)
    fig.suptitle("Per-microstate local effects of theta and phi", fontsize=12)
    return fig


def plot_meed_coordinate_heatmap(D, labels):
    fig, ax = plt.subplots(figsize=(max(5.5, 0.65 * len(labels) + 3.0), 2.8))
    im = ax.imshow(D, aspect="auto", cmap="RdBu_r")
    ax.set_yticks([0, 1], [r"$\theta$", r"$\phi$"])
    ax.set_xticks(np.arange(len(labels)), [pretty_label(x) for x in labels], rotation=45, ha="right")
    ax.set_title(r"$D_{MEED}\in\mathbb{R}^{2\times M}$: azimuth and elevation")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def plot_fixed_full_reconstruction_gallery(B, Bfixed, Bfull, labels, info):
    rows = [("original", B), ("fixed MEED", Bfixed)]
    if Bfull is not None:
        rows.append(("full dipole", Bfull))
    M = B.shape[1]
    fig, axes = plt.subplots(len(rows), M, figsize=(2.55 * M, 2.55 * len(rows)), constrained_layout=True)
    if M == 1:
        axes = axes[:, None]
    vmax = max(1e-12, float(max(np.max(np.abs(mat)) for _, mat in rows)))
    for r, (name, mat) in enumerate(rows):
        for j, lab in enumerate(labels):
            ax = axes[r, j]
            vals = mat[:, j]
            if name != "original":
                vals = align_polarity(B[:, j], vals)
            mne.viz.plot_topomap(
                vals, info, axes=ax, show=False, contours=4, sensors=True,
                cmap="RdBu_r", vlim=(-vmax, vmax), sphere="auto",
                extrapolate="head", image_interp="linear",
            )
            ax.set_title(f"{name}\n{pretty_label(lab)}", fontsize=8)
    return fig


def dipole_models_for_k(json_path, k, fit_full=True, fast_full=True, seed=1):
    ds = load_microstates(json_path, k)
    B = ds["B"]
    labels = remap_labels(ds["labels"], k)
    info = ds["info"]
    geom = sensor_geometry(info)
    meed = fixed_meed(B, geom["S_orth"])
    fixed_df = meed["df"].copy()
    fixed_df["label"] = labels
    fixed_df["M"] = k
    fixed_df["model"] = "fixed_MEED"

    if fit_full:
        Bfull, full_df = fit_full_dipole(B, geom["S"], fast=fast_full, seed=seed)
        full_df = full_df.copy()
        full_df["label"] = labels
        full_df["M"] = k
        full_df["model"] = "full_dipole"
    else:
        Bfull = None
        full_df = None

    return {
        "ds": ds,
        "B": B,
        "labels": labels,
        "info": info,
        "geom": geom,
        "meed": meed,
        "fixed_df": fixed_df,
        "Bfixed": meed["B_meed"],
        "Bfull": Bfull,
        "full_df": full_df,
    }


def dipole_r2_all_map_counts(json_path, all_sets, fit_full=True, fast_full=True, seed=1):
    rows = []
    for entry in all_sets:
        k = len(entry["Maps"])
        result = dipole_models_for_k(json_path, k, fit_full=fit_full, fast_full=fast_full, seed=seed)
        fixed = result["fixed_df"]
        for _, r in fixed.iterrows():
            rows.append({
                "M": k,
                "map_index": int(r["map_index"]),
                "label": r["label"],
                "model": "fixed_MEED_theta_phi",
                "R2": float(r["R2"]),
                "projection_R2": float(r["projection_R2"]),
            })
        if result["full_df"] is not None:
            for _, r in result["full_df"].iterrows():
                rows.append({
                    "M": k,
                    "map_index": int(r["map_index"]),
                    "label": r["label"],
                    "model": "full_dipole",
                    "R2": float(r["R2"]),
                    "projection_R2": np.nan,
                })
    return pd.DataFrame(rows)


def plot_r2_spider_all_map_counts(r2_df, value_col="R2"):
    Ms = sorted(r2_df["M"].unique())
    n = len(Ms)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(4.2 * cols, 4.2 * rows))

    for i, M in enumerate(Ms, start=1):
        ax = fig.add_subplot(rows, cols, i, projection="polar")
        d = r2_df[r2_df["M"] == M].copy()
        labels = [pretty_label(x) for x in d.sort_values("map_index").drop_duplicates("map_index")["label"]]
        k = len(labels)
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        angles_closed = np.r_[angles, angles[0]]

        for model, dm in d.groupby("model"):
            vals = (
                dm.sort_values("map_index")
                .drop_duplicates("map_index")
                [value_col]
                .values
            )
            vals_closed = np.r_[vals, vals[0]]
            ax.plot(angles_closed, vals_closed, marker="o", linewidth=1.8, label=model)
            ax.fill(angles_closed, vals_closed, alpha=0.06)

        ax.set_xticks(angles, labels, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.5, 0.75, 1.0])
        ax.set_title(f"M={M}")
        if i == 1:
            ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.20), fontsize=8)

    fig.suptitle(f"Reconstruction quality across all map counts ({value_col})", fontsize=12)
    fig.tight_layout()
    return fig


def full_dipole_leave_one_electrode_stability(B, labels, S, fast=True, seed=1):
    """Leave-one-electrode-out sensitivity for the full dipole model.

    This is intentionally heavier than fixed MEED stability. It refits the
    nonlinear full dipole after dropping each electrode. We report location
    displacement, moment angular deviation, and R2 change.
    """
    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 1:
        B_arr = B_arr[:, None]
    if B_arr.ndim != 2:
        raise ValueError(f"Expected B to be 1D or 2D, got shape {B_arr.shape}")

    Bfull, full_df = fit_full_dipole(B_arr, S, fast=fast, seed=seed)
    rows = []

    for m, lab in enumerate(labels):
        ref = full_df.iloc[m]
        ref_loc = np.array([ref["loc_x"], ref["loc_y"], ref["loc_z"]], dtype=float)
        ref_mom = np.array([ref["moment_x"], ref["moment_y"], ref["moment_z"]], dtype=float)
        ref_r2 = float(ref["R2"])

        loc_dists = []
        mom_angles = []
        r2_drops = []

        for drop in range(B_arr.shape[0]):
            keep = np.ones(B_arr.shape[0], dtype=bool)
            keep[drop] = False
            try:
                B_drop = B_arr[keep][:, [m]]
                _, df_drop = fit_full_dipole(B_drop, S[keep], fast=fast, seed=seed)
                row = df_drop.iloc[0]
                loc = np.array([row["loc_x"], row["loc_y"], row["loc_z"]], dtype=float)
                mom = np.array([row["moment_x"], row["moment_y"], row["moment_z"]], dtype=float)
                loc_dists.append(float(np.linalg.norm(loc - ref_loc)))
                mom_angles.append(antipodal_angle(ref_mom, mom, degrees=True))
                r2_drops.append(ref_r2 - float(row["R2"]))
            except Exception:
                loc_dists.append(np.nan)
                mom_angles.append(np.nan)
                r2_drops.append(np.nan)

        rows.append({
            "label": lab,
            "full_loc_shift_mean": float(np.nanmean(loc_dists)),
            "full_loc_shift_max": float(np.nanmax(loc_dists)),
            "full_moment_angle_mean_deg": float(np.nanmean(mom_angles)),
            "full_moment_angle_max_deg": float(np.nanmax(mom_angles)),
            "full_R2_drop_mean": float(np.nanmean(r2_drops)),
            "full_R2_drop_max": float(np.nanmax(r2_drops)),
        })
    return pd.DataFrame(rows)


def combined_electrode_sensitivity(B, labels, S_orth, S, fast_full=True, seed=1):
    fixed = fixed_dipole_stability(B, labels, S_orth, S).rename(columns={
        "jackknife_mean_angle_deg": "fixed_angle_mean_deg",
        "jackknife_max_angle_deg": "fixed_angle_max_deg",
        "jackknife_sd_angle_deg": "fixed_angle_sd_deg",
    })
    full = full_dipole_leave_one_electrode_stability(B, labels, S, fast=fast_full, seed=seed)
    return fixed.merge(full, on="label", how="outer")


def fixed_and_full_null_simulations(S_orth, S, n=80, seed=1, fast_full=True):
    """Null simulations for both fixed MEED and full dipole.

    Value: fixed MEED is a strict low-dimensional scaffold. Full dipole is more
    flexible. If full dipole fits noise well, full-dipole R2 is not a strong
    microstate-likeness diagnostic by itself.
    """
    rng = np.random.default_rng(seed)
    Y, degs = real_sph_basis(S, 4)
    rows = []

    for kind in ["iid_random", "low_order_l1_l2", "high_order_l3_l4"]:
        for i in range(n):
            if kind == "iid_random":
                v = rng.normal(size=S.shape[0])
            else:
                coeff = np.zeros(Y.shape[1])
                idx = np.where((degs == 1) | (degs == 2))[0] if kind == "low_order_l1_l2" else np.where((degs == 3) | (degs == 4))[0]
                coeff[idx] = rng.normal(size=len(idx))
                v = Y @ coeff + 0.05 * rng.normal(size=S.shape[0])

            b = center_l2(v)
            q = S_orth.T @ b
            fixed_projection_R2 = float(np.sum(q ** 2))
            rho = float(np.linalg.norm(q))
            if rho > 1e-12:
                b_fixed = S_orth @ (q / rho)
                fixed_theta_phi_R2 = r2_score_polarity_invariant(b, b_fixed)
            else:
                fixed_theta_phi_R2 = np.nan

            try:
                _, full_df = fit_full_dipole(b[:, None], S, fast=fast_full, seed=seed)
                full_R2 = float(full_df.iloc[0]["R2"])
            except Exception:
                full_R2 = np.nan

            rows.append({
                "kind": kind,
                "fixed_projection_R2": fixed_projection_R2,
                "fixed_theta_phi_R2": fixed_theta_phi_R2,
                "fixed_dipolarity": rho,
                "full_dipole_R2": full_R2,
            })
    return pd.DataFrame(rows)


def plot_fixed_full_null_simulations(null_df):
    kinds = ["iid_random", "low_order_l1_l2", "high_order_l3_l4"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True, sharey=True)
    metrics = ["fixed_projection_R2", "fixed_theta_phi_R2", "full_dipole_R2"]
    titles = ["fixed projection R2", "fixed theta-phi R2", "full dipole R2"]

    for ax, metric, title in zip(axes, metrics, titles):
        ax.boxplot([null_df.loc[null_df["kind"] == k, metric].dropna() for k in kinds], labels=kinds, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("R2")
    fig.suptitle("Null simulations: strict fixed MEED versus flexible full dipole", fontsize=12)
    return fig


# ---------------------------------------------------------------------
# v6 corrections
# ---------------------------------------------------------------------
# This block intentionally overrides/extends some earlier functions.
# Changes:
# 1. no PCA comparison;
# 2. no old reconstruction-gallery plot;
# 3. local MEED effects are additive Nch x 2 maps around each microstate;
# 4. reconstruction_r2_table is restored;
# 5. full-dipole electrode sensitivity is made robust and explicit;
# 6. null simulations compare fixed MEED and full dipole.

def safe_unit(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def tangent_basis_at_direction(u):
    """Local angular tangent basis at a unit dipole direction.

    Returns e_theta and e_phi in 3D dipole-coordinate space. Small angular
    changes in radians satisfy:

        u(theta+dtheta, phi+dphi)
        ~= u + e_theta*dtheta + e_phi*dphi

    Therefore scalp maps satisfy the additive approximation:

        V(theta+dtheta, phi+dphi)
        ~= V + (S_orth@e_theta)*dtheta + (S_orth@e_phi)*dphi
    """
    u = safe_unit(u)
    theta0 = anat_azimuth(u[0], u[1])
    phi0 = elevation(u)

    old_az = np.deg2rad(90.0 - theta0)
    phi = np.deg2rad(phi0)

    e_theta = np.array([
        -np.cos(phi) * np.sin(old_az),
        np.cos(phi) * np.cos(old_az),
        0.0,
    ])
    e_phi = np.array([
        -np.sin(phi) * np.cos(old_az),
        -np.sin(phi) * np.sin(old_az),
        np.cos(phi),
    ])

    e_theta = safe_unit(e_theta)
    e_phi = e_phi - e_theta * np.dot(e_theta, e_phi)
    e_phi = safe_unit(e_phi)
    return e_theta, e_phi, theta0, phi0


def local_meed_additive_effects(S_orth, U, radians=True):
    """Per-microstate additive local effects.

    Parameters
    ----------
    S_orth : array, shape Nch x 3
    U : array, shape 3 x M
        Unit MEED directions.
    radians : bool
        If True, effect maps are per 1 radian. If False, per 1 degree.

    Returns
    -------
    effects : list of dict
        Each item contains baseline, dtheta_map, dphi_map, and A_local.
        A_local has shape Nch x 2 and is additive on channels:
            b_new ~= baseline + A_local @ [delta_theta, delta_phi]
    """
    scale = 1.0 if radians else np.pi / 180.0
    effects = []
    for m in range(U.shape[1]):
        u = safe_unit(U[:, m])
        e_theta, e_phi, theta0, phi0 = tangent_basis_at_direction(u)
        baseline = S_orth @ u
        dtheta = (S_orth @ e_theta) * scale
        dphi = (S_orth @ e_phi) * scale
        A = np.column_stack([dtheta, dphi])
        effects.append({
            "map_index": m,
            "theta_deg": theta0,
            "phi_deg": phi0,
            "baseline": baseline,
            "dtheta_map": dtheta,
            "dphi_map": dphi,
            "A_local": A,
            "units": "per_radian" if radians else "per_degree",
        })
    return effects


def apply_local_meed_delta(effect, delta_theta, delta_phi):
    """Apply additive local theta/phi perturbation to one microstate.

    delta_theta and delta_phi must match the units of effect["A_local"].
    If effects were created with radians=True, pass radians.
    """
    delta = np.array([delta_theta, delta_phi], dtype=float)
    return effect["baseline"] + effect["A_local"] @ delta


def plot_local_meed_effects_per_microstate(B_original, B_meed, effects, labels, info, delta_deg=10.0):
    """Visualize additive local MEED effects for each microstate.

    Columns are:
    original, MEED baseline, dV/dtheta, baseline + delta theta,
    dV/dphi, baseline + delta phi.

    The derivative maps are additive Nch vectors. The delta plots show the
    actual first-order local perturbation in sensor space.
    """
    M = len(effects)
    cols = 6
    fig, axes = plt.subplots(M, cols, figsize=(2.55 * cols, 2.35 * M), constrained_layout=True)
    if M == 1:
        axes = axes[None, :]

    vlim_maps = float(max(np.max(np.abs(B_original)), np.max(np.abs(B_meed)), 1e-12))
    delta_rad = np.deg2rad(delta_deg)

    for m, eff in enumerate(effects):
        plus_theta = apply_local_meed_delta(eff, delta_rad, 0.0)
        plus_theta = center_l2(plus_theta)
        plus_phi = apply_local_meed_delta(eff, 0.0, delta_rad)
        plus_phi = center_l2(plus_phi)

        panels = [
            ("original", B_original[:, m], vlim_maps),
            ("MEED baseline", B_meed[:, m], vlim_maps),
            (r"$dV/d\theta$", eff["dtheta_map"], float(max(np.max(np.abs(eff["dtheta_map"])), 1e-12))),
            (rf"$+\Delta\theta={delta_deg:g}^\circ$", plus_theta, vlim_maps),
            (r"$dV/d\phi$", eff["dphi_map"], float(max(np.max(np.abs(eff["dphi_map"])), 1e-12))),
            (rf"$+\Delta\phi={delta_deg:g}^\circ$", plus_phi, vlim_maps),
        ]

        for j, (title, vals, vlim) in enumerate(panels):
            ax = axes[m, j]
            mne.viz.plot_topomap(
                vals, info, axes=ax, show=False, contours=4, sensors=True,
                cmap="RdBu_r", vlim=(-vlim, vlim), sphere="auto",
                extrapolate="head", image_interp="linear",
            )
            if j == 0:
                ax.set_title(f"{pretty_label(labels[m])}\n{title}", fontsize=8)
            else:
                ax.set_title(title, fontsize=8)

    fig.suptitle("Per-microstate additive local MEED effects on the channel vector", fontsize=12)
    return fig


def reconstruction_r2_table(B, reconstructions, labels=None):
    """Return long and summary reconstruction R2 tables.

    This function is intentionally restored because downstream cells rely on it.
    """
    rows = []
    for name, Bhat in reconstructions.items():
        for m in range(B.shape[1]):
            rows.append({
                "representation": name,
                "map_index": m,
                "label": pretty_label(labels[m]) if labels is not None else m,
                "R2": r2_score_polarity_invariant(B[:, m], Bhat[:, m]),
            })
    long = pd.DataFrame(rows)
    summary = long.groupby("representation", as_index=False)["R2"].agg(["mean", "min", "max", "std"]).reset_index()
    return summary, long


def gram_preservation_table(B, reconstructions):
    """Summarize how well each reconstruction preserves map-space Gram structure."""
    B0 = center_l2_cols(B)
    G0 = B0.T @ B0
    iu = np.triu_indices_from(G0, k=1)

    rows = []
    for name, Bhat in reconstructions.items():
        Bh = center_l2_cols(Bhat)
        Gh = Bh.T @ Bh
        diff = Gh - G0

        upper0 = G0[iu]
        upperh = Gh[iu]
        if upper0.size > 1:
            corr = float(np.corrcoef(upper0, upperh)[0, 1])
        else:
            corr = np.nan

        rows.append(
            {
                "representation": name,
                "fro_error": float(np.linalg.norm(diff, ord="fro")),
                "fro_error_rel": float(np.linalg.norm(diff, ord="fro") / np.maximum(np.linalg.norm(G0, ord="fro"), 1e-12)),
                "mean_abs_diff_offdiag": float(np.mean(np.abs(diff[iu]))),
                "max_abs_diff_offdiag": float(np.max(np.abs(diff[iu]))),
                "upper_triangle_corr": corr,
            }
        )
    return pd.DataFrame(rows)


def plot_gram_preservation(B, reconstructions):
    """Plot original and reconstructed Gram matrices side by side."""
    mats = {"original": center_l2_cols(B).T @ center_l2_cols(B)}
    for name, Bhat in reconstructions.items():
        mats[name] = center_l2_cols(Bhat).T @ center_l2_cols(Bhat)

    names = list(mats.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        im = ax.imshow(mats[name], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(name)
        ax.set_xlabel("map")
        ax.set_ylabel("map")
        fig.colorbar(im, ax=ax, shrink=0.78)
    return fig


def plot_reconstruction_spider_for_set(r2_long, title="Reconstruction quality"):
    """Spider plot for one map set.

    Expects columns: representation, label, R2.
    """
    reps = list(r2_long["representation"].unique())
    labels = list(r2_long["label"].unique())
    M = len(labels)

    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    theta_closed = np.r_[theta, theta[0]]

    fig, ax = plt.subplots(figsize=(5.4, 5.4), subplot_kw={"projection": "polar"})
    for rep in reps:
        vals = []
        d = r2_long[r2_long["representation"] == rep]
        for lab in labels:
            vals.append(float(d.loc[d["label"] == lab, "R2"].iloc[0]))
        vals = np.asarray(vals)
        vals_closed = np.r_[vals, vals[0]]
        ax.plot(theta_closed, vals_closed, marker="o", label=rep)
        ax.fill(theta_closed, vals_closed, alpha=0.08)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    return fig


def reconstruction_quality_all_m(all_sets, info_builder=None, fast_full=True, seed=1):
    """Compute fixed MEED and full-dipole R2 for every map count in all_sets.

    Parameters
    ----------
    all_sets : list
        JSON entries with Maps, Labels, and chanlocs.
    info_builder : callable or None
        If None, uses standard_1020 with each entry's channel names.
    """
    rows = []
    for entry in all_sets:
        k = len(entry["Maps"])
        raw = np.asarray(entry["Maps"], dtype=float).T
        B = center_l2_cols(raw)
        labels = list(entry.get("Labels", [f"Map_{i+1}" for i in range(k)]))
        labels = remap_labels(labels, k)

        original_ch_names = [c["labels"] for c in entry["chanlocs"]]
        ch_names = [LEGACY_TO_MODERN.get(ch, ch) for ch in original_ch_names]
        info = mne.create_info(ch_names=ch_names, sfreq=1.0, ch_types="eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="raise")

        geom = sensor_geometry(info)
        Q, Bproj, Bfixed, fixed_df = project_fixed_dipole(B, geom["S_orth"])

        try:
            Bfull, full_df = fit_full_dipole(B, geom["S"], fast=fast_full, seed=seed)
        except Exception:
            Bfull = None
            full_df = None

        for m, lab in enumerate(labels):
            rows.append({
                "M": k,
                "map_index": m,
                "label": pretty_label(lab),
                "representation": "fixed_MEED",
                "R2": r2_score_polarity_invariant(B[:, m], Bfixed[:, m]),
                "projection_R2": float(fixed_df.loc[m, "projection_R2"]),
            })
            if Bfull is not None:
                rows.append({
                    "M": k,
                    "map_index": m,
                    "label": pretty_label(lab),
                    "representation": "full_dipole",
                    "R2": r2_score_polarity_invariant(B[:, m], Bfull[:, m]),
                    "projection_R2": np.nan,
                })
    return pd.DataFrame(rows)


def plot_reconstruction_spiders_all_m(r2_all):
    """Facet spider plots by number of maps M."""
    Ms = sorted(r2_all["M"].unique())
    n = len(Ms)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(5.0 * cols, 4.9 * rows), constrained_layout=True)
    for i, M in enumerate(Ms):
        ax = fig.add_subplot(rows, cols, i + 1, projection="polar")
        dM = r2_all[r2_all["M"] == M]
        labels = list(dM.sort_values("map_index")["label"].drop_duplicates())
        reps = list(dM["representation"].unique())
        theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        theta_closed = np.r_[theta, theta[0]]

        for rep in reps:
            vals = []
            d = dM[dM["representation"] == rep]
            for lab in labels:
                vals.append(float(d.loc[d["label"] == lab, "R2"].iloc[0]))
            vals = np.asarray(vals)
            ax.plot(theta_closed, np.r_[vals, vals[0]], marker="o", label=rep)
            ax.fill(theta_closed, np.r_[vals, vals[0]], alpha=0.08)

        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.5, 0.75, 1.0])
        ax.set_title(f"M={M}")

    handles, labels_legend = ax.get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper right")
    fig.suptitle("Reconstruction quality across all map counts", fontsize=12)
    return fig


def plot_reconstruction_spiders_with_metamaps_all_m(ds, r2_all):
    """Show each solution's R2 spider beside its maps and MEED reconstruction.

    Parameters
    ----------
    ds : dict
        Dataset returned by ``load_microstates``; its ``all_sets`` entries define
        the solutions and their manual map ordering.
    r2_all : pandas.DataFrame
        Output of ``reconstruction_quality_all_m(ds["all_sets"], ...)``.

    Returns
    -------
    matplotlib.figure.Figure
        A ``2 * n_solutions`` by ``(2 * max_k + 2)`` panel grid.  The first two
        columns contain one spider plot spanning the two rows of each solution;
        the remaining columns contain original maps above their fixed-MEED
        reconstructions.
    """
    all_sets = ds["all_sets"]
    n_solutions = len(all_sets)
    if n_solutions == 0:
        raise ValueError("ds['all_sets'] must contain at least one solution")

    max_k = max(len(entry["Maps"]) for entry in all_sets)
    reconstructed = []
    for entry in all_sets:
        B = center_l2_cols(np.asarray(entry["Maps"], dtype=float).T)
        info = make_info_from_entry(entry)
        geom = sensor_geometry(info)
        _, _, B_meed, _ = project_fixed_dipole(B, geom["S_orth"])
        reconstructed.append((B, B_meed, info))

    vmax = max(
        1e-12,
        float(max(np.max(np.abs(mat)) for B, B_meed, _ in reconstructed for mat in (B, B_meed))),
    )
    fig = plt.figure(
        figsize=(2.35 * (max_k + 1), 2.55 * n_solutions),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2 * n_solutions, 2 * max_k + 2)

    for solution_index, entry in enumerate(all_sets):
        k = len(entry["Maps"])
        labels = entry.get("Labels", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:k])
        B, B_meed, info = reconstructed[solution_index]
        grid_row = 2 * solution_index

        # The spider occupies the first 2 x 2 block for this solution.
        ax_spider = fig.add_subplot(grid[grid_row:grid_row + 2, :2], projection="polar")
        dM = r2_all[r2_all["M"] == k]
        spider_labels = list(dM.sort_values("map_index")["label"].drop_duplicates())
        theta = np.linspace(0, 2 * np.pi, len(spider_labels), endpoint=False)
        theta_closed = np.r_[theta, theta[0]]
        for representation in dM["representation"].unique():
            d_representation = dM[dM["representation"] == representation]
            values = np.asarray([
                float(d_representation.loc[d_representation["label"] == label, "R2"].iloc[0])
                for label in spider_labels
            ])
            if representation == "fixed_MEED":
                ax_spider.plot(theta_closed, np.r_[values, values[0]], linestyle='--', marker="d", markerfacecolor="none", label=representation, zorder=3)
            else:
                ax_spider.plot(theta_closed, np.r_[values, values[0]], marker="o", label=representation)


        ax_spider.set_xticks(theta)
        ax_spider.set_xticklabels(spider_labels, fontsize=8)
        ax_spider.set_ylim(0.75, 1.0)
        ax_spider.set_yticks([0.9, 0.95, 1.0])
        if solution_index == 0:
            ax_spider.legend(loc="upper left", bbox_to_anchor=(1.05, 1.15), fontsize=8)

        for map_index, label in enumerate(labels):
            col_start = 2 + 2 * map_index
            panels = [
                (grid_row, "original", B[:, map_index]),
                (grid_row + 1, "fixed MEED", align_polarity(B[:, map_index], B_meed[:, map_index])),
            ]
            for row, row_name, values in panels:
                ax = fig.add_subplot(grid[row, col_start:col_start + 2])
                mne.viz.plot_topomap(
                    values, info, axes=ax, show=False, contours=4, sensors=False,
                    cmap="RdBu_r", vlim=(-vmax, vmax), sphere="auto",
                    extrapolate="head", image_interp="linear",
                )

    fig.suptitle("Reconstruction quality, microstates, and fixed-MEED reconstructions", fontsize=13)
    return fig


def full_dipole_fit_single(B, S, map_index, fast=True, seed=1):
    """Fit full dipole for a single column and return row dict plus prediction."""
    Bsub = B[:, [map_index]]
    Bhat, df = fit_full_dipole(Bsub, S, fast=fast, seed=seed)
    row = df.iloc[0].to_dict()
    row["map_index"] = map_index
    return Bhat[:, 0], row


def full_dipole_fit_single_robust(B, S, map_index, fast=True, seed=1, raise_on_failure=False):
    """Robust single-map full dipole fit with retries.

    Returns
    -------
    pred : ndarray, shape (Nch,)
    row : dict
    ok : bool
        True when a numerically valid full-dipole fit was found.
    """
    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 1:
        B_arr = B_arr[:, None]
    if B_arr.ndim != 2:
        raise ValueError(f"Expected B to be 1D or 2D, got shape {B_arr.shape}")
    if map_index < 0 or map_index >= B_arr.shape[1]:
        raise IndexError(
            f"map_index={map_index} out of bounds for B with {B_arr.shape[1]} column(s)"
        )

    last_exc = None

    # Try several seeds in fast mode first (cheap and usually good enough).
    seeds_fast = [seed, seed + 11, seed + 29, seed + 53]
    for sd in seeds_fast:
        try:
            pred, row = full_dipole_fit_single(B_arr, S, map_index, fast=True, seed=sd)
            if np.all(np.isfinite(pred)) and np.isfinite(row.get("R2", np.nan)):
                return pred, row, True
        except Exception as exc:
            last_exc = exc

    # If requested non-fast, or if fast attempts fail, try one slower fit.
    try:
        pred, row = full_dipole_fit_single(B_arr, S, map_index, fast=False, seed=seed + 1001)
        if np.all(np.isfinite(pred)) and np.isfinite(row.get("R2", np.nan)):
            return pred, row, True
    except Exception as exc:
        last_exc = exc

    if raise_on_failure:
        raise RuntimeError(
            f"full_dipole_fit_single_robust failed for map_index={map_index}, seed={seed}"
        ) from last_exc
    return np.full(B_arr.shape[0], np.nan), {}, False


def fixed_and_full_leave_one_electrode_sensitivity(B, labels, S, fast_full=True, seed=1, verbose=False):
    """Leave-one-electrode sensitivity for fixed MEED and full dipole.

    Per dropped electrode, reports:
    - delta_theta_deg
    - delta_phi_deg
    - stereo_angle_deg (3D antipodal angle)
    - R2_change
    - full dipole only: loc_shift_x/y/z and loc_shift_norm
    """
    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 1:
        B_arr = B_arr[:, None]
    if B_arr.ndim != 2:
        raise ValueError(f"Expected B to be 1D or 2D, got shape {B_arr.shape}")

    S = np.asarray(S, dtype=float)
    if S.ndim != 2:
        raise ValueError(f"Expected S to be 2D, got shape {S.shape}")
    if S.shape[0] != B_arr.shape[0]:
        raise ValueError(
            f"Row mismatch between B and S: B has {B_arr.shape[0]} rows, S has {S.shape[0]} rows"
        )
    if len(labels) != B_arr.shape[1]:
        raise ValueError(
            f"labels length mismatch: got {len(labels)} labels for {B_arr.shape[1]} map column(s)"
        )
    Sc = S - S.mean(axis=0, keepdims=True)
    Minvsqrt, _ = inv_sqrtm(Sc.T @ Sc)
    Sorth = Sc @ Minvsqrt

    Q, _, Bfixed, fixed_df = project_fixed_dipole(B_arr, Sorth)
    U_full = Q / np.maximum(np.linalg.norm(Q, axis=0, keepdims=True), 1e-12)

    # Reference full-dipole fits; fail loudly if any map cannot be fitted.
    full_ref_rows = []
    full_ref_pred = []
    for m in range(B_arr.shape[1]):
        pred, row = full_dipole_fit_single(B_arr, S, m, fast=fast_full, seed=seed + m)
        full_ref_pred.append(pred)
        full_ref_rows.append(row)

    rows = []
    for m, lab in enumerate(labels):
        # Fixed reference orientation in anatomical angles.
        theta_fixed_ref, phi_fixed_ref, u_fixed_ref = canonical_theta_phi_from_vector(U_full[:, m])
        ref_fixed_r2 = float(fixed_df.loc[m, "R2"])

        # Full reference orientation/location.
        ref = full_ref_rows[m]
        ref_mom = np.array(
            [ref.get("moment_x", np.nan), ref.get("moment_y", np.nan), ref.get("moment_z", np.nan)],
            dtype=float,
        )
        theta_full_ref, phi_full_ref, u_full_ref = canonical_theta_phi_from_vector(ref_mom)
        ref_loc = np.array(
            [ref.get("loc_x", np.nan), ref.get("loc_y", np.nan), ref.get("loc_z", np.nan)],
            dtype=float,
        )
        ref_full_r2 = float(ref.get("R2", np.nan))

        for drop in range(B_arr.shape[0]):
            keep = np.ones(B_arr.shape[0], dtype=bool)
            keep[drop] = False

            # Fixed MEED drop-electrode refit.
            Sdrop_raw = S[keep]
            Sc_drop = Sdrop_raw - Sdrop_raw.mean(axis=0, keepdims=True)
            Minv_drop, _ = inv_sqrtm(Sc_drop.T @ Sc_drop)
            Sorth_drop = Sc_drop @ Minv_drop
            b_drop = B_arr[keep, m]
            q_drop = Sorth_drop.T @ b_drop
            u_drop = q_drop / max(np.linalg.norm(q_drop), 1e-12)
            theta_fixed_drop, phi_fixed_drop, u_fixed_drop = canonical_theta_phi_from_vector(u_drop)
            fixed_stereo = float(np.degrees(np.arccos(np.clip(abs(np.dot(u_fixed_ref, u_fixed_drop)), -1, 1))))
            b_fixed_drop = Sorth_drop @ u_drop
            fixed_drop_r2 = r2_score_polarity_invariant(b_drop, b_fixed_drop)
            fixed_r2_change = float(fixed_drop_r2 - ref_fixed_r2)

            rows.append({
                "label": pretty_label(lab),
                "map_index": m,
                "dropped_electrode": drop,
                "model": "fixed_MEED",
                "delta_theta_deg": float(abs(wrap180(theta_fixed_drop - theta_fixed_ref))),
                "delta_phi_deg": float(abs(phi_fixed_drop - phi_fixed_ref)),
                "stereo_angle_deg": fixed_stereo,
                "R2_change": fixed_r2_change,
                "drop_fit_R2": fixed_drop_r2,
                "loc_shift_x": np.nan,
                "loc_shift_y": np.nan,
                "loc_shift_z": np.nan,
                "loc_shift_norm": np.nan,
            })

            # Full dipole drop-electrode refit.
            B_drop = B_arr[keep][:, [m]]
            Bhat_drop, dr = full_dipole_fit_single(B_drop, Sdrop_raw, 0, fast=fast_full, seed=seed + m + drop)
            drop_mom = np.array(
                [dr.get("moment_x", np.nan), dr.get("moment_y", np.nan), dr.get("moment_z", np.nan)],
                dtype=float,
            )
            theta_full_drop, phi_full_drop, u_full_drop = canonical_theta_phi_from_vector(drop_mom)
            full_stereo = float(np.degrees(np.arccos(np.clip(abs(np.dot(u_full_ref, u_full_drop)), -1, 1))))

            drop_loc = np.array(
                [dr.get("loc_x", np.nan), dr.get("loc_y", np.nan), dr.get("loc_z", np.nan)],
                dtype=float,
            )
            loc_delta = drop_loc - ref_loc
            loc_shift_norm = float(np.linalg.norm(loc_delta))

            drop_r2 = float(r2_score_polarity_invariant(B_drop.ravel(), np.asarray(Bhat_drop).ravel()))
            full_r2_change = float(drop_r2 - ref_full_r2)

            rows.append({
                "label": pretty_label(lab),
                "map_index": m,
                "dropped_electrode": drop,
                "model": "full_dipole",
                "delta_theta_deg": float(abs(wrap180(theta_full_drop - theta_full_ref))),
                "delta_phi_deg": float(abs(phi_full_drop - phi_full_ref)),
                "stereo_angle_deg": full_stereo,
                "R2_change": full_r2_change,
                "drop_fit_R2": drop_r2,
                "loc_shift_x": float(loc_delta[0]),
                "loc_shift_y": float(loc_delta[1]),
                "loc_shift_z": float(loc_delta[2]),
                "loc_shift_norm": loc_shift_norm,
            })

    sens_df = pd.DataFrame(rows)

    if verbose:
        print(f"[leave-one-electrode] rows={len(sens_df)}")

    return sens_df


def summarize_electrode_sensitivity(sens_df):
    rows = []
    for (model, label), d in sens_df.groupby(["model", "label"]):
        rows.append({
            "model": model,
            "label": label,
            "delta_theta_mean_deg": d["delta_theta_deg"].mean(),
            "delta_theta_max_deg": d["delta_theta_deg"].max(),
            "delta_phi_mean_deg": d["delta_phi_deg"].mean(),
            "delta_phi_max_deg": d["delta_phi_deg"].max(),
            "stereo_angle_mean_deg": d["stereo_angle_deg"].mean(),
            "stereo_angle_max_deg": d["stereo_angle_deg"].max(),
            "R2_change_mean": d["R2_change"].mean(),
            "abs_R2_change_max": d["R2_change"].abs().max(),
            "loc_shift_x_abs_mean": d["loc_shift_x"].abs().mean(),
            "loc_shift_y_abs_mean": d["loc_shift_y"].abs().mean(),
            "loc_shift_z_abs_mean": d["loc_shift_z"].abs().mean(),
            "loc_shift_norm_mean": d["loc_shift_norm"].mean(),
            "loc_shift_norm_max": d["loc_shift_norm"].max(),
        })
    return pd.DataFrame(rows)


def leave_one_electrode_metrics_table(sens_df):
    """Per-drop table with the requested sensitivity quantities."""
    cols = [
        "model",
        "label",
        "map_index",
        "dropped_electrode",
        "delta_theta_deg",
        "delta_phi_deg",
        "stereo_angle_deg",
        "R2_change",
        "loc_shift_x",
        "loc_shift_y",
        "loc_shift_z",
        "loc_shift_norm",
    ]
    keep = [c for c in cols if c in sens_df.columns]
    return sens_df[keep].copy()


def plot_leave_one_electrode_swarms(sens_df):
    """Swarmplots with x=label, y=metric, hue=model."""
    import seaborn as sns

    metrics = [
        ("delta_theta_deg", "delta theta (deg)"),
        ("delta_phi_deg", "delta phi (deg)"),
        ("stereo_angle_deg", "stereo angle change (deg)"),
        ("R2_change", "delta R2"),
        ("loc_shift_x", "location shift x (m)"),
        ("loc_shift_y", "location shift y (m)"),
        ("loc_shift_z", "location shift z (m)"),
        ("loc_shift_norm", "location shift norm (m)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)
    axes = axes.ravel()

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        sns.swarmplot(
            data=sens_df,
            x="label",
            y=metric,
            hue="model",
            dodge=True,
            size=3,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("label")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, title="model")
        else:
            ax.get_legend().remove()

    fig.suptitle("Leave-one-electrode sensitivity: per dropped channel")
    return fig


def plot_fixed_full_electrode_sensitivity(sens_df):
    """Backward-compatible alias to the swarm-based plot."""
    return plot_leave_one_electrode_swarms(sens_df)


def null_simulations_fixed_and_full(S_orth, S, n=120, seed=1, fast_full=True, return_maps=False):
    """Null simulations for fixed and full dipole.

    Interpretation:
    fixed MEED asks how much of the map is first-order dipolar.
    full dipole asks whether a flexible single dipole can fit the map.
    If full dipole fits random/high-order maps too well, high full-dipole R2
    alone is not a microstate-likeness diagnostic.
    """
    rng = np.random.default_rng(seed)
    Y, degs = real_sph_basis(S, 4)
    rows = []
    map_rows = []

    for kind in ["iid_random", "low_order_l1_l2", "high_order_l3_l4"]:
        for i in range(n):
            if kind == "iid_random":
                v = rng.normal(size=S.shape[0])
            else:
                coeff = np.zeros(Y.shape[1])
                idx = np.where((degs == 1) | (degs == 2))[0] if kind == "low_order_l1_l2" else np.where((degs == 3) | (degs == 4))[0]
                coeff[idx] = rng.normal(size=len(idx))
                v = Y @ coeff + 0.05 * rng.normal(size=S.shape[0])

            b = center_l2(v)
            q = S_orth.T @ b
            fixed_proj_R2 = float(np.sum(q ** 2))
            qnorm = np.linalg.norm(q)
            if qnorm > 1e-12:
                b_fixed = S_orth @ (q / qnorm)
                fixed_theta_phi_R2 = r2_score_polarity_invariant(b, b_fixed)
            else:
                fixed_theta_phi_R2 = np.nan

            rows.append({
                "kind": kind,
                "simulation": i,
                "model": "fixed_MEED",
                "R2": fixed_theta_phi_R2,
                "projection_R2": fixed_proj_R2,
            })

            try:
                Bfull, dfull = fit_full_dipole(b[:, None], S, fast=fast_full, seed=seed + i)
                full_R2 = float(dfull.loc[0, "R2"])
            except Exception:
                full_R2 = np.nan

            rows.append({
                "kind": kind,
                "simulation": i,
                "model": "full_dipole",
                "R2": full_R2,
                "projection_R2": np.nan,
            })

            if return_maps:
                map_rows.append({
                    "kind": kind,
                    "simulation": i,
                    "map": b.copy(),
                    "fixed_R2": fixed_theta_phi_R2,
                    "full_R2": full_R2,
                })

    long_df = pd.DataFrame(rows)
    if return_maps:
        return long_df, pd.DataFrame(map_rows)
    return long_df


def plot_null_fixed_full(null_df):
    kinds = ["iid_random", "low_order_l1_l2", "high_order_l3_l4"]
    models = [m for m in ["fixed_MEED", "full_dipole"] if m in null_df["model"].unique()]

    fig, axes = plt.subplots(1, len(kinds), figsize=(4.3 * len(kinds), 4.2), sharey=True, constrained_layout=True)
    if len(kinds) == 1:
        axes = [axes]

    for ax, kind in zip(axes, kinds):
        data = [
            null_df[(null_df["kind"] == kind) & (null_df["model"] == model)]["R2"].dropna().values
            for model in models
        ]
        ax.boxplot(data, labels=models, showfliers=False)
        ax.set_title(kind)
        nonempty = [vals for vals in data if vals.size > 0]
        if nonempty and any(np.any(vals < 0) for vals in nonempty):
            ymin = min(float(np.min(vals)) for vals in nonempty)
            ax.set_ylim(max(ymin - 0.02, -1.02), 1.02)
        else:
            ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", labelrotation=20)
    axes[0].set_ylabel("R2")
    fig.suptitle("Null simulations: strict fixed MEED vs flexible full dipole")
    return fig


def null_paired_r2_tests(null_df):
    """Paired R2 tests (fixed vs full) within each null kind."""
    from scipy.stats import ttest_rel, wilcoxon

    d = null_df.pivot_table(
        index=["kind", "simulation"],
        columns="model",
        values="R2",
        aggfunc="first",
    ).dropna()

    rows = []
    for kind in d.index.get_level_values("kind").unique():
        dk = d.xs(kind, level="kind")
        x = dk["fixed_MEED"].to_numpy()
        y = dk["full_dipole"].to_numpy()
        diff = y - x
        t_res = ttest_rel(y, x)
        w_res = wilcoxon(diff)
        rows.append({
            "kind": kind,
            "n_pairs": int(len(dk)),
            "mean_fixed_R2": float(np.mean(x)),
            "mean_full_R2": float(np.mean(y)),
            "mean_diff_full_minus_fixed": float(np.mean(diff)),
            "ttest_t": float(t_res.statistic),
            "ttest_p": float(t_res.pvalue),
            "wilcoxon_W": float(w_res.statistic),
            "wilcoxon_p": float(w_res.pvalue),
        })
    return pd.DataFrame(rows)


def plot_null_paired_r2(null_df):
    """Per-kind paired R2 plot: seaborn boxplot + paired black line plot."""
    import seaborn as sns

    kinds = ["iid_random", "low_order_l1_l2", "high_order_l3_l4"]
    model_order = ["fixed_MEED", "full_dipole"]
    d = null_df[null_df["kind"].isin(kinds)].copy()
    d["model"] = pd.Categorical(d["model"], categories=model_order, ordered=True)

    fig, axes = plt.subplots(1, len(kinds), figsize=(4.6 * len(kinds), 4.2), sharey=True, constrained_layout=True)
    if len(kinds) == 1:
        axes = [axes]

    for ax, kind in zip(axes, kinds):
        dk = d[d["kind"] == kind].dropna(subset=["R2"]).copy()
        sns.boxplot(
            data=dk,
            x="model",
            y="R2",
            hue="model",
            order=model_order,
            hue_order=model_order,
            dodge=False,
            showfliers=False,
            ax=ax,
        )
        sns.lineplot(
            data=dk.sort_values(["simulation", "model"]),
            x="model",
            y="R2",
            units="simulation",
            estimator=None,
            color="black",
            lw=0.25,
            alpha=0.35,
            legend=False,
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title(kind)
        ax.set_xlabel("model")
    axes[0].set_ylabel("R2")
    fig.suptitle("Null simulations: paired fixed vs full dipole R2")
    return fig


def plot_null_exemplary_maps_by_fixed_r2(null_maps_df, info, rows=7, cols_per_kind=4):
    """Plot exemplary null maps sampled at evenly spaced fixed-R2 quantiles."""
    import ast

    def _to_float_map(x):
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=float).reshape(-1)
        if isinstance(x, (list, tuple)):
            return np.asarray(x, dtype=float).reshape(-1)
        if isinstance(x, str):
            s = x.strip()
            # Try literal list/array syntax first.
            try:
                parsed = ast.literal_eval(s)
                arr = np.asarray(parsed, dtype=float).reshape(-1)
                if arr.size > 0:
                    return arr
            except Exception:
                pass
            # Fallback: parse numeric tokens from bracketed string.
            arr = np.fromstring(s.replace("[", " ").replace("]", " "), sep=" ", dtype=float)
            if arr.size > 0:
                return arr
        raise TypeError(f"Cannot parse map payload of type {type(x)} into numeric array")

    kinds = ["iid_random", "low_order_l1_l2", "high_order_l3_l4"]
    n_each = rows * cols_per_kind
    total_cols = cols_per_kind * len(kinds)

    fig, axes = plt.subplots(rows, total_cols, figsize=(2.15 * total_cols, 2.0 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]

    parsed_maps = [_to_float_map(v) for v in null_maps_df["map"].values]
    vmax = float(max(np.max(np.abs(v)) for v in parsed_maps if v.size > 0))
    last_im = None

    for k_i, kind in enumerate(kinds):
        d = null_maps_df[null_maps_df["kind"] == kind].copy().sort_values("fixed_R2").reset_index(drop=True)
        if d.empty:
            continue
        idx = np.linspace(0, len(d) - 1, n_each).round().astype(int)
        dsel = d.iloc[idx].reset_index(drop=True)

        for p in range(n_each):
            r = p // cols_per_kind
            c_local = p % cols_per_kind
            c = k_i * cols_per_kind + c_local
            ax = axes[r, c]
            vals = _to_float_map(dsel.loc[p, "map"])
            fixed_r2 = float(dsel.loc[p, "fixed_R2"])
            full_r2 = float(dsel.loc[p, "full_R2"])
            last_im, _ = mne.viz.plot_topomap(
                vals,
                info,
                axes=ax,
                show=False,
                contours=4,
                sensors=True,
                cmap="RdBu_r",
                vlim=(-vmax, vmax),
                sphere="auto",
                extrapolate="head",
                image_interp="linear",
            )
            ax.set_title(
                rf"$R^2_{{\mathrm{{fixed}}}}$={fixed_r2:.2f}, $R^2_{{\mathrm{{full}}}}$={full_r2:.2f}",
                fontsize=7,
            )
            if c_local == 0:
                ax.set_ylabel(f"{kind}\nrow {r+1}", fontsize=8)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.65, label="a.u.")
    fig.suptitle(
        "Exemplary null maps (equal spacing by fixed-MEED R2): iid | low-order | high-order",
        fontsize=12,
    )
    return fig
