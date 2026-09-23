
import json
import math
import re
from xml.etree import ElementTree
from pathlib import Path
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
try:
    from scipy.special import sph_harm_y
    _SPH_HARM_LEGACY_CONVENTION = False
except ImportError:
    from scipy.special import sph_harm as sph_harm_y
    _SPH_HARM_LEGACY_CONVENTION = True

import warnings

warnings.filterwarnings("ignore")

LEGACY_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

CANONICAL_LABELS = ("A", "B", "C", "D", "E")
STATE_COLORS = {
    "A": "#CC79A7", "B": "#0072B2", "C": "#009E73",
    "D": "#E69F00", "E": "#D55E00",
}
CONDITION_HATCH = {"EC": "///", "EO": ""}
PAPER_WIDTH_IN = 6.0


def configure_publication_style(fontsize: float = 10.0) -> None:
    """Apply typography only; scalar voltage maps retain their own colormap."""
    mpl.rcParams.update({
        "font.size": fontsize, "axes.labelsize": fontsize,
        "axes.titlesize": fontsize, "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize, "legend.fontsize": fontsize,
        "svg.fonttype": "none", "savefig.bbox": None, "hatch.linewidth": 0.6,
    })


def set_publication_spines(ax, *, categorical: bool) -> None:
    """Use a left-only categorical axis or left-and-bottom quantitative axis."""
    for name, spine in ax.spines.items():
        spine.set_visible(name == "left" or (name == "bottom" and not categorical))
    ax.tick_params(top=False, right=False)
    if categorical:
        ax.tick_params(axis="x", length=0)


def assert_svg_width(path: str | Path, expected_inches: float = PAPER_WIDTH_IN) -> None:
    """Verify the emitted SVG canvas, including artists inside its margins."""
    width = ElementTree.parse(path).getroot().attrib.get("width", "")
    match = re.fullmatch(r"([0-9.eE+-]+)(pt|in|px|mm|cm)", width)
    if match is None:
        raise ValueError(f"Unrecognized SVG width: {width!r}")
    factors = {"pt": 1 / 72, "in": 1.0, "px": 1 / 96, "mm": 1 / 25.4, "cm": 1 / 2.54}
    actual = float(match[1]) * factors[match[2]]
    if not math.isclose(actual, expected_inches, rel_tol=0, abs_tol=1e-6):
        raise AssertionError(f"SVG width {actual:g} != {expected_inches:g} inches")

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






    
#%% non-orthogonal base characterization


















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




def polarity_sign(y, yhat):
    """Return +1 or -1 to maximize map correlation under polarity equivalence."""
    y0 = center_l2(y)
    h0 = center_l2(yhat)
    return 1.0 if float(np.dot(y0, h0)) >= 0 else -1.0


def align_polarity(y, yhat):
    """Align prediction polarity to target map polarity."""
    return polarity_sign(y, yhat) * np.asarray(yhat, dtype=float)


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
        Contains theta, phi, dipolarity, and the canonical fixed-FORM fit R2.
        For centered unit-L2 maps, R2 is exactly dipolarity squared.
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
                "R2": dipolarity ** 2,
                "theta_deg": theta,
                "phi_deg": phi,
            }
        )

    Bunit = np.column_stack(Bunit_cols)
    return Q, Bproj, Bunit, pd.DataFrame(rows)

















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








def draw_plane(ax):
    t = np.linspace(0, 2 * np.pi, 180)
    ax.plot(np.cos(t), np.sin(t), 0 * t, color="black", linewidth=0.9)
    nose = np.array([[-0.08, 1.0, 0], [0, 1.13, 0], [0.08, 1.0, 0]])
    ax.plot(nose[:, 0], nose[:, 1], nose[:, 2], color="black", linewidth=0.9)


def plot_literature_bidirectional_dipole_view(
    directions, labels, *, elev: float, azim: float, figsize=(4.0, 4.0)
):
    """Render the legacy dense literature-dipole object from one camera view.

    Each literature map contributes its polarity-invariant axis ``[-q, +q]``;
    ``q`` retains its fitted-dipole magnitude. Thus changing a map's polarity
    does not change the rendered segment. Labels are used only for the
    established canonical A--E colour lookup.
    """
    vectors = np.asarray(directions, dtype=float)
    labels = tuple(map(str, labels))
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) != len(labels):
        raise ValueError("directions must be an n-by-3 array with one label per row")
    if len(vectors) == 0 or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("literature dipole directions must be finite non-zero 3D vectors")

    figure = plt.figure(figsize=figsize)
    axis = figure.add_subplot(111, projection="3d")
    # Preserve the legacy 3D panes/grid but remove the separate opaque 2D axes patch.
    axis.patch.set_visible(False)
    for vector, label in zip(vectors, labels):
        axis.plot(
            [-vector[0], vector[0]], [-vector[1], vector[1]], [-vector[2], vector[2]],
            color=STATE_COLORS.get(label, "0.35"), linewidth=.55, alpha=.90,
        )
    draw_plane(axis)
    axis.set_xlim(-1.15, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_zlim(-1.15, 1.15)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=elev, azim=azim)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_zticklabels([])
    axis.tick_params(length=0)
    axis.grid(True)
    return figure













def fixed_meed(B, S_orth):
    """Fixed-location microstate-equivalent electric dipole representation.

    Returns exact angular FORM coordinates D_FORM with shape 2 x M.
    `dipolarity` and the canonical `R2 = dipolarity**2` are diagnostics only,
    not retained coordinates.
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


























#%% controlled spatial-spectrum simulation (F03)

CONTROLLED_SPECTRUM_CONDITIONS = ("Pure FORM", "5:1", "2:1", "White noise")


def _controlled_real_harmonic(degree: int, order: int, directions: np.ndarray) -> np.ndarray:
    """Evaluate one real spherical harmonic on the fitted sensor sphere."""
    directions = np.asarray(directions, dtype=float)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    polar = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    azimuth = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2 * np.pi)
    if _SPH_HARM_LEGACY_CONVENTION:
        value = sph_harm_y(abs(order), degree, azimuth, polar)
    else:
        value = sph_harm_y(degree, abs(order), polar, azimuth)
    if order < 0:
        return np.sqrt(2.0) * (-1) ** order * value.imag
    if order > 0:
        return np.sqrt(2.0) * (-1) ** order * value.real
    return value.real


def _controlled_harmonic_blocks(sensor_directions: np.ndarray, s_orth: np.ndarray, lmax: int) -> dict[int, np.ndarray]:
    """Build a centered orthogonal finite-montage basis with FORM as degree 1."""
    directions = np.asarray(sensor_directions, dtype=float)
    first_order = np.asarray(s_orth, dtype=float)
    if first_order.shape != (directions.shape[0], 3):
        raise ValueError("FORM basis and sensor directions have incompatible shapes")
    blocks = {1: first_order}
    for degree in range(2, lmax + 1):
        candidate = np.column_stack([
            _controlled_real_harmonic(degree, order, directions)
            for order in range(-degree, degree + 1)
        ])
        candidate -= candidate.mean(axis=0, keepdims=True)
        lower = np.column_stack([blocks[value] for value in sorted(blocks)])
        lower, _ = np.linalg.qr(lower, mode="reduced")
        residual = candidate - lower @ (lower.T @ candidate)
        left, singular, _ = np.linalg.svd(residual, full_matrices=False)
        rank = int(np.sum(singular > 1e-8 * max(1.0, singular[0] if singular.size else 1.0)))
        blocks[degree] = left[:, :rank]
    return blocks


def _controlled_mode_variances(blocks: dict[int, np.ndarray], alpha: float) -> dict[int, float]:
    if np.isinf(alpha):
        return {degree: float(degree == 1) for degree in blocks}
    return {degree: degree ** (-alpha) for degree in blocks}


def simulate_controlled_spectra(
    sensor_directions: np.ndarray,
    s_orth: np.ndarray,
    *,
    n_simulations: int = 1000,
    seed: int = 42,
    lmax: int = 8,
) -> dict:
    """Sample the Figure-F03 spectra in the centered finite-montage FORM space."""
    if n_simulations < 3:
        raise ValueError("At least three simulations are required for representative maps")
    blocks = _controlled_harmonic_blocks(sensor_directions, s_orth, lmax)
    ranks = {degree: block.shape[1] for degree, block in blocks.items()}
    if ranks[1] != 3 or ranks.get(2, 0) == 0:
        raise ValueError("Controlled spectra require non-empty degree-1 and degree-2 blocks")
    alphas = (np.inf, np.log2(5.0 * ranks[2] / ranks[1]), np.log2(2.0 * ranks[2] / ranks[1]), 0.0)
    generator = np.random.default_rng(seed)
    rows: list[dict] = []
    examples: dict[tuple[str, int], np.ndarray] = {}
    representatives: list[dict] = []

    for condition, alpha in zip(CONTROLLED_SPECTRUM_CONDITIONS, alphas):
        variances = _controlled_mode_variances(blocks, float(alpha))
        coefficients = {
            degree: generator.normal(scale=np.sqrt(variances[degree]), size=(ranks[degree], n_simulations))
            for degree in blocks
        }
        energies = {degree: np.sum(values ** 2, axis=0) for degree, values in coefficients.items()}
        # The manuscript conditions specify realized degree-1:degree-2 energy,
        # not merely its expectation over finite random draws.
        target_ratio = {"5:1": 5.0, "2:1": 2.0}.get(condition)
        if target_ratio is not None:
            scale = np.sqrt(target_ratio * energies[2] / np.maximum(energies[1], np.finfo(float).tiny))
            coefficients[1] *= scale[None, :]
            energies[1] = np.sum(coefficients[1] ** 2, axis=0)
        total = sum(energies.values())
        rho2 = energies[1] / total
        for simulation_id in range(n_simulations):
            rows.append({
                "condition": condition, "simulation_id": simulation_id, "rho2": float(rho2[simulation_id]),
                "degree1_energy": float(energies[1][simulation_id]),
                "degree2_energy": float(energies[2][simulation_id]), "total_energy": float(total[simulation_id]),
            })
        for target in np.quantile(rho2, (.2, .5, .8)):
            simulation_id = int(np.argmin(np.abs(rho2 - target)))
            values = sum(blocks[degree] @ coefficients[degree][:, simulation_id] for degree in blocks)
            examples[(condition, simulation_id)] = values / np.linalg.norm(values)
            representatives.append({"condition": condition, "simulation_id": simulation_id})

    return {
        "rows": pd.DataFrame(rows), "examples": examples, "metadata": {
            "n_simulations": n_simulations, "seed": seed, "lmax": lmax,
            "conditions": list(CONTROLLED_SPECTRUM_CONDITIONS), "basis": "centered finite-montage real spherical harmonics; degree 1 is FORM",
            "block_ranks": ranks, "finite_montage_rank": int(sum(ranks.values())), "representative_maps": representatives,
            "expected_degree1_to_degree2_ratio": {"Pure FORM": "infinite", "5:1": 5.0, "2:1": 2.0, "White noise": ranks[1] / ranks[2]},
        },
    }


def plot_controlled_spectra(simulation: dict, info: mne.Info):
    """Render the retained F03 layout from finite-montage simulation maps."""
    rows = simulation["rows"]
    grouped = [rows.loc[rows.condition.eq(condition), "rho2"].to_numpy() for condition in CONTROLLED_SPECTRUM_CONDITIONS]
    figure = plt.figure(figsize=(6, 3.5))
    outer = GridSpec(1, 2, width_ratios=[1, 1.75], wspace=.14, figure=figure)
    violin_axis = figure.add_subplot(outer[0, 0])
    colours = plt.cm.viridis(np.linspace(.15, .85, len(CONTROLLED_SPECTRUM_CONDITIONS)))
    display_values = [values.copy() for values in grouped]
    display_values[0] = np.clip(display_values[0] + np.linspace(-.002, .002, len(display_values[0])), 0, 1)
    violin = violin_axis.violinplot(display_values, positions=np.arange(1, 5), widths=.8, showmeans=False, showmedians=False, showextrema=False)
    for colour, body in zip(colours, violin["bodies"]):
        body.set(facecolor=colour, edgecolor=colour, alpha=.35, linewidth=1.4)
    violin_axis.boxplot(grouped, positions=np.arange(1, 5), widths=.18, patch_artist=True, showfliers=False,
                        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.2},
                        whiskerprops={"color": "black", "linewidth": 1.2}, capprops={"color": "black", "linewidth": 1.2},
                        medianprops={"color": "black", "linewidth": 1.3})
    violin_axis.hlines(1, .7, 1.3, color=colours[0], linewidth=2.2, zorder=4)
    violin_axis.hlines(1, .84, 1.16, color="black", linewidth=1.2, zorder=5)
    violin_axis.set(xticks=np.arange(1, 5), xticklabels=CONTROLLED_SPECTRUM_CONDITIONS, ylabel="ρ²", ylim=(0, 1.02), title="FORM reconstruction quality")
    for label in violin_axis.get_xticklabels():
        label.set(rotation=45, ha="right", rotation_mode="anchor")
    violin_axis.spines[["top", "right", "bottom"]].set_visible(False)
    violin_axis.tick_params(axis="x", length=0, pad=2)
    violin_axis.grid(True, axis="y", alpha=.2)

    grid = outer[0, 1].subgridspec(3, 5, width_ratios=[1, 1, 1, 1, .08], wspace=.05, hspace=.08)
    maximum = max(np.max(np.abs(values)) for values in simulation["examples"].values())
    scalar_map = mpl.cm.ScalarMappable(norm=Normalize(-maximum, maximum), cmap="RdBu_r")
    for column, condition in enumerate(CONTROLLED_SPECTRUM_CONDITIONS):
        condition_rows = rows.loc[rows.condition.eq(condition)]
        values = condition_rows.rho2.to_numpy()
        ids = condition_rows.simulation_id.to_numpy()
        selected_ids = [int(ids[np.argmin(np.abs(values - target))]) for target in np.quantile(values, (.2, .5, .8))]
        for row, simulation_id in enumerate(selected_ids):
            axis = figure.add_subplot(grid[row, column])
            topography = simulation["examples"][(condition, simulation_id)].copy()
            if condition == "Pure FORM":
                topography += np.random.default_rng(10_000 + row).uniform(-.01, .01, len(topography))
                topography /= np.linalg.norm(topography)
            mne.viz.plot_topomap(topography, info, axes=axis, show=False, sensors=False, contours=6,
                                 cmap="RdBu_r", vlim=(-maximum, maximum), sphere="auto")
            if row == 2:
                axis.text(.5, -.13, condition, transform=axis.transAxes, rotation=45, ha="right", va="top",
                          rotation_mode="anchor", fontsize=11)
    colorbar = figure.colorbar(scalar_map, cax=figure.add_subplot(grid[:, 4]))
    colorbar.set_label("normalized potential", fontsize=11)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    colorbar.ax.tick_params(labelsize=10)
    figure.subplots_adjust(left=.07, right=.98, top=.95, bottom=.25, wspace=.14)
    return figure
