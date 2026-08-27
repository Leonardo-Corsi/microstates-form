#!/usr/bin/env python3
"""
meed_realdata_spaces.py

Read an MS Template Explorer DistMat .mat file directly and generate BOTH:

1) MEED + 3D MDS figure, with all study-template points colored by the selected
   meta-cluster solution.

2) MEED topomap-layout figure, where every study-template point is replaced by
   a small scalp topography.

Default outputs:
    meed_vs_mds_K4.svg
    meed_topomap_layout.svg
    meed_topomap_angle_grid.svg

An interactive 3D dipole figure is also opened by default.  Each panel shows
one meta-cluster solution, with every study-template dipole drawn as a line
colored by its assigned label.

Example:
    python meed_realdata_spaces.py metamaps_struct_lemon.mat
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne
import yaml
from scipy.io import loadmat
from scipy.optimize import minimize


DEFAULT_AZIM = 170.2980
DEFAULT_ELEV = -1.4568
TOPOMAP_INTERPOLATION = "linear"
TOPOMAP_EXTRAPOLATE = "local"
TOPOMAP_GRID_INTERPOLATION = "cubic"
TOPOMAP_GRID_EXTRAPOLATE = "head"
TOPOMAP_GRID_MONTAGE = "standard_1005"
TOPOMAP_GRID_RESOLUTION = 128
TOPOMAP_LAYOUT_RESOLUTION = 64
TOPOMAP_LAYOUT_FIGSIZE = (11.33, 7.5)
TOPOMAP_GRID_FIGSIZE = (11.33, 11.33)
TOPOMAP_SIZE_INCH = 0.4
TOPOMAP_GRID_ANGLES_DEG = np.arange(-75, 76, 15)

# Match the fixed MATLAB MDS axis order and signs for this MAT export.
MDS_AXIS_ORDER = (1, 0, 2)
MDS_AXIS_SIGN = np.array([1.0, -1.0, -1.0])


def load_distmat(path):
    mat = loadmat(path, simplify_cells=True)
    if "DistMat" not in mat:
        raise KeyError("MAT file does not contain DistMat")
    return mat["DistMat"]


def chanloc_xyz(chanlocs):
    xyz_eeglab = np.asarray([
        [
            float(ch.get("X", np.nan)),
            float(ch.get("Y", np.nan)),
            float(ch.get("Z", np.nan)),
        ]
        for ch in chanlocs
    ], dtype=float)
    # EEGLAB: X anterior, Y left, Z superior.
    # MNE/anatomical: X right, Y anterior, Z superior.
    return np.column_stack([-xyz_eeglab[:, 1], xyz_eeglab[:, 0], xyz_eeglab[:, 2]])


def chanloc_xy(chanlocs):
    xyz = chanloc_xyz(chanlocs)
    xy = xyz[:, :2]

    valid = np.isfinite(xy).all(axis=1)
    if valid.any():
        r = np.sqrt(np.sum(xy[valid] ** 2, axis=1))
        scale = np.nanmax(r)
        if np.isfinite(scale) and scale > 0:
            return xy / scale

    out = []
    for ch in chanlocs:
        theta = float(ch.get("theta", np.nan))
        radius = float(ch.get("radius", np.nan))
        if np.isfinite(theta) and np.isfinite(radius):
            th = np.deg2rad(theta)
            out.append([radius * np.sin(th), radius * np.cos(th)])
        else:
            out.append([np.nan, np.nan])
    out = np.asarray(out, float)
    r = np.sqrt(np.nansum(out ** 2, axis=1))
    scale = np.nanmax(r)
    return out / (scale if np.isfinite(scale) and scale > 0 else 1.0)


def fixed_dipole_meed(values, chanlocs):
    q = fixed_dipole_vector(values, chanlocs)
    if not np.isfinite(q).all():
        return np.nan, np.nan, np.nan

    theta = np.degrees(np.arctan2(q[0], q[1]))
    phi = np.degrees(np.arctan2(q[2], np.hypot(q[0], q[1])))
    rho = np.linalg.norm(q)
    return theta, phi, rho


def fixed_dipole_vector(values, chanlocs):
    """Return the canonical fixed-dipole vector fitted to one topography."""
    x = np.asarray(values, float).reshape(-1)
    xyz = chanloc_xyz(chanlocs)

    good = np.isfinite(x) & np.isfinite(xyz).all(axis=1)
    x = x[good]
    xyz = xyz[good]

    G = xyz - xyz.mean(axis=0, keepdims=True)
    gram = G.T @ G
    ev, evec = np.linalg.eigh(gram)
    ev = np.maximum(ev, np.finfo(float).eps)
    S = G @ evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T

    x = x - x.mean()
    nrm = np.linalg.norm(x)
    if nrm == 0:
        return np.full(3, np.nan)
    x /= nrm

    q = S.T @ x

    # Canonicalize to the anterior half-space.  This makes theta run from
    # -90 deg (left) through 0 deg (midline) to +90 deg (right).
    if q[1] < 0:
        q = -q
    return q


def all_meed(distmat):
    return np.asarray([
        fixed_dipole_meed(tm["Map"], tm["chanlocs"])
        for tm in distmat["TemplateMap"]
    ], float)


def get_solution(distmat, k):
    cp = distmat["ClusterMaps"]["msinfo"]["ClustPar"]
    min_k = int(cp["MinClasses"])
    max_k = int(cp["MaxClasses"])
    if not min_k <= k <= max_k:
        raise ValueError(f"Available solutions: {min_k}..{max_k}")

    assignments = np.asarray(
        distmat["ClusterAssignment"][k - 1], int
    ).reshape(-1)

    ms = distmat["ClusterMaps"]["msinfo"]["MSMaps"][k - 1]
    colors = np.asarray(ms["ColorMap"], float)
    meta_maps = np.asarray(ms["Maps"], float)
    return assignments, colors, meta_maps


def template_map_correlation(template_a, template_b):
    """Correlate two maps over their shared channel labels."""
    values_a = np.asarray(template_a["Map"], float).reshape(-1)
    values_b = np.asarray(template_b["Map"], float).reshape(-1)

    def channel_values(values, chanlocs):
        out = {}
        for value, ch in zip(values, chanlocs):
            label = str(ch.get("labels", "")).strip().strip("'").lower()
            if label and np.isfinite(value):
                out[label] = value
        return out

    a_by_channel = channel_values(values_a, template_a["chanlocs"])
    b_by_channel = channel_values(values_b, template_b["chanlocs"])
    common = sorted(a_by_channel.keys() & b_by_channel.keys())
    if len(common) < 3:
        return 0.0

    a = np.asarray([a_by_channel[ch] for ch in common], float)
    b = np.asarray([b_by_channel[ch] for ch in common], float)
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def label_polarity_signs(distmat, k):
    """Find one consistent topographic polarity for every K-solution label.

    Within a label, maps can occur as two groups: maps are positively correlated
    inside each group and negatively correlated across the groups.  A
    correlation-weighted sign synchronization identifies the two groups, then
    flips one of them.  The largest internally correlated group defines the
    retained polarity for that label.
    """
    assignments, _, _ = get_solution(distmat, k)
    templates = distmat["TemplateMap"]
    signs = np.ones(len(templates), dtype=float)

    for label in np.unique(assignments):
        idx = np.flatnonzero(assignments == label)
        if len(idx) < 2:
            continue

        corr = np.eye(len(idx))
        for row, i in enumerate(idx):
            for col in range(row):
                value = template_map_correlation(templates[i], templates[idx[col]])
                corr[row, col] = value
                corr[col, row] = value
        anchor = int(np.argmax(np.sum(np.abs(corr), axis=1)))
        label_signs = np.sign(corr[:, anchor])
        label_signs[label_signs == 0] = 1.0

        # Refine the initial split using all pairwise correlations in the label.
        for _ in range(20):
            updated = np.sign(corr @ label_signs)
            updated[updated == 0] = 1.0
            updated *= updated[anchor]
            if np.array_equal(updated, label_signs):
                break
            label_signs = updated

        signs[idx] = label_signs

    return signs


def classical_mds(D, n_components=3):
    """Deterministic classical MDS from the stored MS Template Explorer distance matrix."""
    D = np.asarray(D, float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J

    ev, evec = np.linalg.eigh(B)
    order = np.argsort(ev)[::-1]
    ev = np.maximum(ev[order][:n_components], 0)
    evec = evec[:, order][:, :n_components]
    return evec * np.sqrt(ev)


def mds_sstress(D, n_components=3):
    """Reproduce MATLAB mdscale(..., 'Criterion', 'sstress', 'Start', 'cmdscale')."""
    D = np.asarray(D, float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    n = D.shape[0]
    D2 = D ** 2
    X0 = classical_mds(D, n_components)

    iu = np.triu_indices(n, 1)
    denom = np.sum(D2[iu] ** 2)
    if denom <= 0:
        denom = 1.0

    def objective_and_gradient(flat):
        X = flat.reshape(n, n_components)
        squared_norms = np.sum(X * X, axis=1)
        R2 = squared_norms[:, None] + squared_norms[None, :] - 2.0 * X @ X.T
        E = R2 - D2
        np.fill_diagonal(E, 0.0)

        objective = 0.5 * np.sum(E * E) / denom
        gradient = 4.0 * (E.sum(axis=1)[:, None] * X - E @ X) / denom
        return objective, gradient.ravel()

    result = minimize(
        objective_and_gradient,
        X0.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 100000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 50},
    )
    X = result.x.reshape(n, n_components)
    return X[:, MDS_AXIS_ORDER] * MDS_AXIS_SIGN


def zscore_topomap_values(values):
    values = np.asarray(values, float).copy()
    good = np.isfinite(values)
    if good.sum() < 3:
        return None

    values[good] -= values[good].mean()
    std = values[good].std()
    if not np.isfinite(std) or std == 0:
        return None
    values[good] /= std
    return values


def plot_meed_mds(distmat, k, out_file, azim, elev):
    meed = all_meed(distmat)
    assignments, colors, meta_maps = get_solution(distmat, k)
    point_colors = colors[assignments - 1]

    mds = mds_sstress(np.asarray(distmat["Dist"], float), 3)

    fig = plt.figure(figsize=(13.33, 7.5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axhline(0, linewidth=0.7)
    ax1.axvline(0, linewidth=0.7)
    ax1.scatter(
        meed[:, 0], meed[:, 1],
        c=point_colors,
        s=22,
        edgecolors="black",
        linewidths=0.25,
    )
    ax1.set_xlim(-92, 92)
    ax1.set_ylim(-90, 90)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel(r"$\theta$ anatomical azimuth (deg)")
    ax1.set_ylabel(r"$\phi$ elevation (deg)")
    ax1.set_title(f"MEED space - K={k} meta-cluster colors")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        mds[:, 0], mds[:, 1], mds[:, 2],
        c=point_colors,
        s=20,
        depthshade=False,
        edgecolors="0.3",
        linewidths=0.25,
    )
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.grid(False)
    ax2.set_title(f"3D MDS - view [{azim:.4f}, {elev:.4f}]")

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def plot_topomap_layout(distmat, out_file, k, figsize=TOPOMAP_LAYOUT_FIGSIZE, topomap_size=TOPOMAP_SIZE_INCH):
    meed = all_meed(distmat)
    templates = distmat["TemplateMap"]
    polarity_signs = label_polarity_signs(distmat, k)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.07, 0.10, 0.90, 0.84])

    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set_xlim(-92, 92)
    ax.set_ylim(-90, 90)
    ax.set_aspect("auto")
    ax.set_xlabel(r"$\theta$ anatomical azimuth (deg)")
    ax.set_ylabel(r"$\phi$ elevation (deg)")
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig_w, fig_h = figsize
    box_w = topomap_size / fig_w
    box_h = topomap_size / fig_h

    pos = ax.get_position()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    for i, tm in enumerate(templates):
        theta, phi = meed[i, :2]

        cx = pos.x0 + (theta - x0) / (x1 - x0) * pos.width
        cy = pos.y0 + (phi - y0) / (y1 - y0) * pos.height

        inset = fig.add_axes(
            [cx - box_w / 2, cy - box_h / 2, box_w, box_h],
            zorder=5,
        )
        values = zscore_topomap_values(polarity_signs[i] * tm["Map"])
        if values is None:
            inset.axis("off")
            continue
        vmax = np.nanmax(np.abs(values))
        mne.viz.plot_topomap(
            values,
            chanloc_xy(tm["chanlocs"]),
            axes=inset,
            show=False,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            contours=0,
            sensors=False,
            sphere=(0.0, 0.0, 0.0, 1.0),
            extrapolate=TOPOMAP_EXTRAPOLATE,
            image_interp=TOPOMAP_INTERPOLATION,
            res=TOPOMAP_LAYOUT_RESOLUTION,
        )

    fig.savefig(out_file)
    plt.close(fig)


class Linked3DViews:
    """Keep a group of 3D Matplotlib axes at the same camera angle."""

    def __init__(self, fig, axes):
        self.fig = fig
        self.axes = list(axes)
        self.view = self._view(self.axes[0])
        self.updating = False
        self.connection_id = fig.canvas.mpl_connect("draw_event", self.sync)

    @staticmethod
    def _view(ax):
        return ax.elev, ax.azim, getattr(ax, "roll", 0.0)

    def sync(self, _event):
        if self.updating:
            return

        changed_axes = [ax for ax in self.axes if self._view(ax) != self.view]
        if not changed_axes:
            return

        # A 3D rotation updates the active axes first.  Copy that view to every
        # other panel, then request one redraw to refresh their projections.
        elev, azim, roll = self._view(changed_axes[-1])
        self.updating = True
        for ax in self.axes:
            if ax is not changed_axes[-1]:
                try:
                    ax.view_init(elev=elev, azim=azim, roll=roll)
                except TypeError:  # Matplotlib versions without camera roll.
                    ax.view_init(elev=elev, azim=azim)
        self.view = elev, azim, roll
        self.updating = False
        self.fig.canvas.draw_idle()


def draw_head_plane_and_sphere(ax):
    """Draw the scalp outline on z=0 and an edge-only upper hemisphere."""
    t = np.linspace(0, 2 * np.pi, 180)
    ax.plot(np.cos(t), np.sin(t), np.zeros_like(t), color="black", linewidth=1.6)
    nose = np.array([[-0.08, 1.0, 0], [0, 1.13, 0], [0.08, 1.0, 0]])
    ax.plot(nose[:, 0], nose[:, 1], nose[:, 2], color="black", linewidth=1.6)

    longitude = np.linspace(0, 2 * np.pi, 40)
    latitude = np.linspace(0, np.pi / 2, 20)
    longitude, latitude = np.meshgrid(longitude, latitude)
    x = np.sin(latitude) * np.cos(longitude)
    y = np.sin(latitude) * np.sin(longitude)
    z = np.cos(latitude)
    ax.plot_surface(
        x, y, z,
        color=(1.0, 1.0, 1.0, 0.0),
        edgecolor="0.75",
        linewidth=0.25,
        shade=False,
        zorder=0,
    )


def plot_dipole_3d_solutions(distmat, solutions, azim, elev, only_k=None):
    """Interactively compare all study-map dipoles across K-solutions."""
    templates = distmat["TemplateMap"]
    dipoles = np.asarray([
        fixed_dipole_vector(tm["Map"], tm["chanlocs"])
        for tm in templates
    ], float)

    n_solutions = len(solutions)
    n_cols = min(3, n_solutions)
    n_rows = int(np.ceil(n_solutions / n_cols))
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    axes = []

    for panel, k in enumerate(solutions, start=1):
        if only_k is not None and not ((isinstance(k, int) and k == only_k) or (isinstance(only_k, list) and k in only_k)):
            continue
        assignments, colors, _ = get_solution(distmat, k)
        ax = fig.add_subplot(n_rows, n_cols, panel, projection="3d")
        axes.append(ax)

        for dipole, label in zip(dipoles, assignments):
            if not np.isfinite(dipole).all():
                continue
            direction = dipole / np.linalg.norm(dipole)
            color = colors[label - 1]
            ax.plot(
                [-direction[0], -0.05 * direction[0]],
                [-direction[1], -0.05 * direction[1]],
                [-direction[2], -0.05 * direction[2]],
                color=color,
                linewidth=1.2,
                alpha=0.8,
            )
            ax.plot(
                [0.05 * direction[0], direction[0]],
                [0.05 * direction[1], direction[1]],
                [0.05 * direction[2], direction[2]],
                color=color,
                linewidth=1.2,
                alpha=0.8,
            )

        ax.scatter([0], [0], [0], color="black", s=12, depthshade=False)
        draw_head_plane_and_sphere(ax)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("right-left")
        ax.set_ylabel("anterior-posterior")
        ax.set_zlabel("superior-inferior")
        ax.set_title(f"Study-map fixed dipoles — K={k}")
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle("Drag any panel to rotate all K-solutions together", y=0.98)
    fig.tight_layout()
    # Keep a strong reference: Matplotlib stores bound event callbacks weakly.
    fig._linked_3d_views = Linked3DViews(fig, axes)
    plt.show()


def standard_grid_info():
    """Return a dense standard EEG montage and unit sensor directions."""
    montage = mne.channels.make_standard_montage(TOPOMAP_GRID_MONTAGE)
    positions = montage.get_positions()["ch_pos"]
    unique_names = []
    unique_positions = []
    seen_positions = set()
    for name in montage.ch_names:
        position = np.asarray(positions[name], float)
        key = tuple(np.round(position, 10))
        if key in seen_positions:
            continue
        seen_positions.add(key)
        unique_names.append(name)
        unique_positions.append(position)

    info = mne.create_info(unique_names, sfreq=1.0, ch_types="eeg")
    info.set_montage(montage)
    xyz = np.asarray(unique_positions, float)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return info, xyz


def topomap_from_meed_angles(theta_deg, phi_deg, sensor_directions):
    """Forward central-dipole potential on a dense standard electrode grid."""

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    q = np.array([
        np.cos(phi) * np.sin(theta),
        np.cos(phi) * np.cos(theta),
        np.sin(phi),
    ])

    return sensor_directions @ q


def plot_topomap_angle_grid(distmat, out_file):
    """Draw fixed-dipole topomaps at regularly spaced MEED angles."""
    grid_info, sensor_directions = standard_grid_info()
    angles = TOPOMAP_GRID_ANGLES_DEG
    figsize = TOPOMAP_GRID_FIGSIZE

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.09, 0.09, 0.86, 0.86])
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\theta$ anatomical azimuth (deg)")
    ax.set_ylabel(r"$\phi$ elevation (deg)")
    ax.set_xticks(np.arange(-90, 91, 15))
    ax.set_yticks(np.arange(-90, 91, 15))
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig_w, fig_h = figsize
    pos = ax.get_position()
    topomap_size = min(pos.width * fig_w, pos.height * fig_h) / 12
    box_w = topomap_size / fig_w
    box_h = topomap_size / fig_h
    for theta in angles:
        for phi in angles:
            cx = pos.x0 + (theta + 90) / 180 * pos.width
            cy = pos.y0 + (phi + 90) / 180 * pos.height
            inset = fig.add_axes(
                [cx - box_w / 2, cy - box_h / 2, box_w, box_h],
                zorder=5,
            )
            values = topomap_from_meed_angles(theta, phi, sensor_directions)
            values = zscore_topomap_values(values)
            if values is None:
                inset.axis("off")
                continue
            vmax = np.nanmax(np.abs(values))
            mne.viz.plot_topomap(
                values,
                grid_info,
                axes=inset,
                show=False,
                cmap="RdBu_r",
                vlim=(-vmax, vmax),
                contours=0,
                sensors=False,
                sphere="auto",
                extrapolate=TOPOMAP_GRID_EXTRAPOLATE,
                image_interp=TOPOMAP_GRID_INTERPOLATION,
                res=TOPOMAP_GRID_RESOLUTION,
            )

    fig.savefig(out_file)
    plt.close(fig)


def main():
    with open(Path(__file__).with_name("config.yml"), encoding="utf-8") as f:
        project_paths = yaml.safe_load(f)["paths"]
    results_root = Path(project_paths["results_root"])
    p = argparse.ArgumentParser()
    p.add_argument("--mat_file","-m", type=Path, default=Path("metamaps_struct_lemon.mat"))
    p.add_argument("--solution", "-k", type=int, default=4)
    p.add_argument("--meed-mds-output", type=Path, default=None)
    p.add_argument("--topomap-output", type=Path, default=None)
    p.add_argument("--topomap-angle-grid-output", type=Path, default=None)
    p.add_argument(
        "--dipole-3d-solutions",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="K-solutions to show in the linked interactive 3D dipole figure "
             "(default: every available solution).",
    )
    p.add_argument(
        "--no-dipole-3d",
        action="store_true",
        help="Do not open the interactive linked 3D dipole figure.",
    )
    args = p.parse_args()

    if args.meed_mds_output is None:
        args.meed_mds_output = results_root / f"meed_vs_mds_K{args.solution}.svg"
    if args.topomap_output is None:
        args.topomap_output = results_root / "meed_topomap_layout.svg"
    if args.topomap_angle_grid_output is None:
        args.topomap_angle_grid_output = results_root / "meed_topomap_angle_grid.svg"

    for output_path in (
        args.meed_mds_output,
        args.topomap_output,
        args.topomap_angle_grid_output,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    d = load_distmat(args.mat_file)
    cp = d["ClusterMaps"]["msinfo"]["ClustPar"]
    min_k = int(cp["MinClasses"])
    max_k = int(cp["MaxClasses"])
    dipole_solutions = (
        args.dipole_3d_solutions
        if args.dipole_3d_solutions is not None
        else list(range(min_k, max_k + 1))
    )
    invalid_solutions = [k for k in dipole_solutions if not min_k <= k <= max_k]
    if invalid_solutions:
        raise ValueError(
            f"Invalid dipole 3D solution(s) {invalid_solutions}; "
            f"available solutions: {min_k}..{max_k}"
        )

    # ALWAYS generate both outputs.
    plot_meed_mds(
        d,
        args.solution,
        args.meed_mds_output,
        DEFAULT_AZIM,
        DEFAULT_ELEV,
    )

    plot_topomap_layout(
        d,
        args.topomap_output,
        args.solution,
        figsize=TOPOMAP_LAYOUT_FIGSIZE,
        topomap_size=TOPOMAP_SIZE_INCH,
    )
    plot_topomap_angle_grid(d, args.topomap_angle_grid_output)

    if not args.no_dipole_3d:
        plot_dipole_3d_solutions(
            d,
            dipole_solutions,
            DEFAULT_AZIM,
            DEFAULT_ELEV,
            only_k=4
        )

    print(f"Generated: {args.meed_mds_output}")
    print(f"Generated: {args.topomap_output}")
    print(f"Generated: {args.topomap_angle_grid_output}")


if __name__ == "__main__":
    main()
