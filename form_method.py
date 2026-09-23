#!/usr/bin/env python3
"""Run the reproducible fixed-FORM method diagnostics and optional comparisons."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from matplotlib import patches
from mpl_toolkits.mplot3d import proj3d

import form_method_utils as u

SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yml")
    parser.add_argument("--meta-json", type=Path, default=SCRIPT_DIR / "metamaps_export_lemon.json")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--spatial-spectrum", action="store_true", help="Write F03 controlled fixed-FORM spectra.")
    parser.add_argument("--literature-conformity", action="store_true")
    parser.add_argument("--literature-mds", action="store_true", help="Write F01 literature MDS and Custo-ordered reference maps.")
    parser.add_argument("--orientation-assets", action="store_true")
    parser.add_argument("--export-glb", action="store_true",
                        help="Write literature GLB result assets under outdir/literature_glb; opens no viewer.")
    return parser.parse_args()


def _literature_info(chanlocs) -> "u.mne.Info":
    """Build an MNE info object from one MAT map's native EEGLAB coordinates."""
    names, positions = [], []
    for index, channel in enumerate(np.atleast_1d(chanlocs)):
        names.append(str(channel.get("labels", f"EEG{index:03d}")))
        positions.append([-float(channel["Y"]), float(channel["X"]), float(channel["Z"])])
    positions = np.asarray(positions, dtype=float)
    info = u.mne.create_info(names, sfreq=1.0, ch_types="eeg")
    montage = u.mne.channels.make_dig_montage(
        ch_pos=dict(zip(names, positions)), coord_frame="head",
        nasion=[0.0, float(np.max(positions[:, 1])) * 1.08, 0.0],
        lpa=[float(np.min(positions[:, 0])) * 1.08, 0.0, 0.0],
        rpa=[float(np.max(positions[:, 0])) * 1.08, 0.0, 0.0],
    )
    info.set_montage(montage, on_missing="ignore")
    return info


def literature_form_rows(mat_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project all 313 MAT templates in their own documented sensor geometry."""
    distmat = loadmat(mat_path, simplify_cells=True)["DistMat"]
    assignment = np.asarray(distmat["ClusterAssignment"][4], dtype=int).reshape(-1)
    source_to_label = {1: "E", 2: "C", 3: "D", 4: "A", 5: "B"}  # K=5 ECDAB source slots.
    rows = []
    for index, template in enumerate(distmat["TemplateMap"]):
        info = _literature_info(template["chanlocs"])
        values = np.asarray(template["Map"], dtype=float).reshape(-1, 1)
        normalized = u.center_l2_cols(values)
        geometry = u.sensor_geometry(info)
        q, _, _, fit = u.project_fixed_dipole(normalized, geometry["S_orth"])
        row = fit.iloc[0].to_dict()
        row.update({"template_id": f"literature_{index:03d}", "source_index": index,
                    "source_cluster_id": int(assignment[index]),
                    "label": source_to_label[int(assignment[index])],
                    "n_channels": len(info.ch_names),
                    "sphere_radius_m": float(geometry["radius_m"]),
                    "geometry": "native_EEGLAB_coordinates_to_fitted_sphere"})
        rows.append(row)
    literature = pd.DataFrame(rows).sort_values(["label", "template_id"])
    meta = u.load_microstates(SCRIPT_DIR / "metamaps_export_lemon.json", k=5)
    q, _, _, fit = u.project_fixed_dipole(meta["B"], u.sensor_geometry(meta["info"])["S_orth"])
    fit.insert(0, "label", meta["labels"])
    fit.insert(1, "template_id", [f"koenig_{label}" for label in meta["labels"]])
    return literature, fit


MDS_AZIM = -193.4439
MDS_ELEV = 13.7453
MDS_AXIS_ORDER = (1, 0, 2)
MDS_AXIS_SIGN = np.array([1.0, -1.0, -1.0])
MDS_SOURCE_TO_LABEL = {1: "E", 2: "C", 3: "D", 4: "A", 5: "B"}


def _classical_mds_start(distance: np.ndarray) -> np.ndarray:
    """Classical-MDS initialization used by the MS Template Explorer s-stress fit."""
    n_points = len(distance)
    centering = np.eye(n_points) - np.ones((n_points, n_points)) / n_points
    gram = -.5 * centering @ (distance ** 2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:3]
    return vectors[:, order] * np.sqrt(np.maximum(values[order], 0))


def _matlab_like_sstress_mds(distance: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reproduce the recovered MATLAB-like 3D s-stress MDS convention exactly."""
    distance = np.asarray(distance, float)
    distance = .5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.)
    squared_distance = distance ** 2
    upper = np.triu_indices(len(distance), 1)
    denominator = float(np.sum(squared_distance[upper] ** 2)) or 1.

    def objective_and_gradient(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coordinates = flat.reshape(-1, 3)
        norm = np.sum(coordinates * coordinates, axis=1)
        squared_fit = norm[:, None] + norm[None, :] - 2 * coordinates @ coordinates.T
        error = squared_fit - squared_distance
        np.fill_diagonal(error, 0.)
        objective = .5 * np.sum(error * error) / denominator
        gradient = 4 * (error.sum(axis=1)[:, None] * coordinates - error @ coordinates) / denominator
        return float(objective), gradient.ravel()

    result = minimize(
        objective_and_gradient, _classical_mds_start(distance).ravel(), jac=True,
        method="L-BFGS-B", options={"maxiter": 100000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 50},
    )
    coordinates = result.x.reshape(-1, 3)[:, MDS_AXIS_ORDER] * MDS_AXIS_SIGN
    return coordinates, {"success": bool(result.success), "message": str(result.message),
                         "iterations": int(result.nit), "sstress": float(result.fun)}


def _style_matlab_mds_axis(axis) -> None:
    """Match the recovered MS Template Explorer presentation conventions."""
    axis.grid(False)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.pane.fill = False
        coordinate_axis.pane.set_edgecolor("0.65")
    axis.set(xticks=[], yticks=[], zticks=[])


def _thin_topomap_outlines(axis, linewidth: float = .25) -> None:
    """Keep the individual scalp outline subordinate to the colored class ring."""
    for line in axis.lines:
        line.set_linewidth(linewidth)


def save_literature_mds(mat_path: Path, outdir: Path) -> None:
    """F01: recovered MATLAB-like literature MDS and actual Custo-ordered meta maps."""
    distmat = loadmat(mat_path, simplify_cells=True)["DistMat"]
    assignments = np.asarray(distmat["ClusterAssignment"][4], int).reshape(-1)
    coordinates, solver = _matlab_like_sstress_mds(np.asarray(distmat["Dist"], float))
    if len(coordinates) != len(distmat["TemplateMap"]) or len(assignments) != len(coordinates):
        raise RuntimeError("MDS coordinate, assignment, and literature-map counts must agree")
    labels = np.array([MDS_SOURCE_TO_LABEL[int(value)] for value in assignments])
    study_indices = np.asarray(distmat["StudyIdx"], int).reshape(-1)
    study_labels = np.asarray(distmat["StudyLabel"], str).reshape(-1)
    map_labels = np.asarray(distmat["Labels"], str).reshape(-1)
    if not np.all((study_indices >= 1) & (study_indices <= len(study_labels))):
        raise RuntimeError("MAT StudyIdx values are outside the StudyLabel table")
    table = pd.DataFrame({
        "template_id": [f"literature_{index:03d}" for index in range(len(coordinates))],
        "source_index": np.arange(len(coordinates)), "source_map_label": map_labels,
        "study_index": study_indices, "study_label": study_labels[study_indices - 1],
        "source_cluster_id": assignments, "label": labels,
        "x": coordinates[:, 0], "y": coordinates[:, 1], "z": coordinates[:, 2],
    })
    if table["template_id"].duplicated().any() or tuple(sorted(table.label.unique())) != u.CANONICAL_LABELS:
        raise RuntimeError("F01 requires unique map IDs and the complete canonical A-E assignment")
    outdir.mkdir(parents=True, exist_ok=True)

    meta = u.load_microstates(SCRIPT_DIR / "metamaps_export_lemon.json", k=5)
    if tuple(meta["labels"]) != u.CANONICAL_LABELS:
        raise RuntimeError("F01 reference maps must use the manually enforced Custo A-E order")
    figure = plt.figure(figsize=(7.0, 5.25))
    # Matplotlib 3D axes cannot host child axes. Projecting with this same camera
    # gives each 2D topomap the exact displayed location of its 3D MDS point.
    axis = figure.add_axes([.015, .025, .97, .95], projection="3d")
    limits = []
    for dimension in range(3):
        lower, upper = np.min(coordinates[:, dimension]), np.max(coordinates[:, dimension])
        margin = .05 * (upper - lower) or .05
        limits.append((lower - margin, upper + margin))
    axis.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
    axis.set_box_aspect((1, 1, 1), zoom=1.8)
    axis.view_init(elev=MDS_ELEV, azim=MDS_AZIM)
    _style_matlab_mds_axis(axis)
    figure.canvas.draw()
    projected = np.column_stack(proj3d.proj_transform(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], axis.get_proj()
    ))
    display_xy = axis.transData.transform(projected[:, :2])
    figure_xy = figure.transFigure.inverted().transform(display_xy)
    table[["display_x", "display_y", "display_depth"]] = np.column_stack([figure_xy, projected[:, 2]])
    table.to_csv(outdir / "literature_mds_coordinates.csv", index=False)

    # Far thumbnails are drawn first; nearer topomaps preserve the selected 3D occlusion order.
    thumbnail_size = .034
    template_order = np.argsort(projected[:, 2])[::-1]
    for depth_rank, index in enumerate(template_order):
        x, y = figure_xy[index]
        map_axis = figure.add_axes([x - thumbnail_size / 2, y - thumbnail_size / 2,
                                    thumbnail_size, thumbnail_size], zorder=10 + depth_rank)
        template = distmat["TemplateMap"][index]
        mne.viz.plot_topomap(np.asarray(template["Map"], float), _literature_info(template["chanlocs"]),
                             axes=map_axis, show=False, sensors=False, contours=0, cmap="RdBu_r", sphere="auto")
        _thin_topomap_outlines(map_axis)
        map_axis.patch.set_alpha(0)
        map_axis.add_patch(patches.Circle((.5, .5), .48, transform=map_axis.transAxes, fill=False,
                                          edgecolor=u.STATE_COLORS[labels[index]], linewidth=.8,
                                          clip_on=False, zorder=2))

    # Filled diamonds mark the mean 3D position of the five meta-clusters above all thumbnails.
    centers = np.array([coordinates[labels == label].mean(axis=0) for label in u.CANONICAL_LABELS])
    center_projection = np.column_stack(proj3d.proj_transform(
        centers[:, 0], centers[:, 1], centers[:, 2], axis.get_proj()
    ))
    center_xy = figure.transFigure.inverted().transform(axis.transData.transform(center_projection[:, :2]))
    marker_axis = figure.add_axes([0, 0, 1, 1], zorder=400)
    marker_axis.scatter(center_xy[:, 0], center_xy[:, 1], marker="D", s=38,
                        c=[u.STATE_COLORS[label] for label in u.CANONICAL_LABELS],
                        edgecolors="black", linewidths=.55)
    marker_axis.set(xlim=(0, 1), ylim=(0, 1)); marker_axis.axis("off")

    # The actual K=5 meta maps remain an independent right-column reference.
    for row, label in enumerate(u.CANONICAL_LABELS):
        map_axis = figure.add_axes([.815, .795 - row * .187, .17, .17], zorder=500)
        mne.viz.plot_topomap(meta["B"][:, row], meta["info"], axes=map_axis, show=False,
                             sensors=False, contours=0, cmap="RdBu_r", sphere="auto")
        _thin_topomap_outlines(map_axis)
        map_axis.add_patch(patches.Circle((.5, .5), .48, transform=map_axis.transAxes,
                                          fill=False, edgecolor=u.STATE_COLORS[label], linewidth=2.0,
                                          clip_on=False, zorder=2))
        map_axis.set_title(label, color=u.STATE_COLORS[label], fontsize=10, fontweight="bold", pad=1)
    _save_figure(figure, outdir / "figure_literature_mds.svg", width=7.0)
    plt.close(figure)
    provenance = {
        "n_templates": int(len(table)), "n_studies": int(len(study_labels)),
        "k": 5, "assignment": "MAT ClusterAssignment[4]; source slots E,C,D,A,B -> Custo A-E",
        "mds": {"criterion": "sstress", "start": "cmdscale", "axis_order": list(MDS_AXIS_ORDER),
                "axis_sign": MDS_AXIS_SIGN.tolist(), **solver},
        "camera": {"azim": MDS_AZIM, "elev": MDS_ELEV, "projection": "camera-projected 2D topomap axes at retained 3D MDS coordinates"},
        "caption": "Group-wise microstate templates across 70 studies in the data-driven MDS positions from the meta-clustering proposed in the MATLAB EEGTemplates app (Koenig et al., 2024); circle colors correspond to the five-template solution, with the center of each meta-cluster displayed at right as the manually Custo et al. (2017)-ordered A-E template.",
    }
    (outdir / "literature_mds_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def _bootstrap_median_ci(values: np.ndarray, *, seed: int, n_resamples: int = 20000) -> tuple[float, float]:
    """Deterministic percentile bootstrap 90% confidence interval for a median."""
    values = np.asarray(values, float)
    if len(values) == 0:
        raise ValueError("Cannot estimate a median CI without literature templates")
    generator = np.random.default_rng(seed)
    medians = np.median(values[generator.integers(0, len(values), size=(n_resamples, len(values)))], axis=1)
    return tuple(float(value) for value in np.quantile(medians, [.05, .95]))


def save_literature_rho2_table(literature: pd.DataFrame, mat_path: Path, outdir: Path) -> None:
    """Write F04's K=4..8 literature and exact meta-template rho-squared table."""
    distmat = loadmat(mat_path, simplify_cells=True)["DistMat"]
    base = literature.set_index("source_index")["R2"]
    records = []
    for k in range(4, 9):
        assignments = np.asarray(distmat["ClusterAssignment"][k - 1], int).reshape(-1)
        manual_labels = u.labels_from_manual_order(k)
        if manual_labels is None or len(assignments) != len(base):
            raise RuntimeError(f"F04 requires the full K={k} assignment and manual label order")
        meta = u.load_microstates(SCRIPT_DIR / "metamaps_export_lemon.json", k=k)
        _, _, _, meta_fit = u.project_fixed_dipole(meta["B"], u.sensor_geometry(meta["info"])["S_orth"])
        meta_rho2 = dict(zip(meta["labels"], meta_fit["R2"].to_numpy(float)))
        assigned_labels = np.asarray([manual_labels[int(value) - 1] for value in assignments])
        for label in meta["labels"]:
            values = base.loc[np.flatnonzero(assigned_labels == label)].to_numpy(float)
            n = len(values)
            median_ci_low, median_ci_high = _bootstrap_median_ci(values, seed=k * 100 + ord(label))
            records.append({
                "k": k, "microstate": label, "n_literature_templates": n,
                "literature_mean_rho2": float(np.mean(values)), "literature_sd_rho2": float(np.std(values, ddof=1)),
                "literature_median_rho2": float(np.median(values)),
                "literature_median_ci90_low": median_ci_low, "literature_median_ci90_high": median_ci_high,
                "meta_rho2": float(meta_rho2[label]),
            })
    summary = pd.DataFrame(records).sort_values(["k", "microstate"])
    summary.to_csv(outdir / "literature_rho2_summary_k4_k8.csv", index=False)

    presentation = pd.DataFrame({
        "K": summary["k"], "microstate": summary["microstate"],
        "Literature mean rho² ± SD (N)": [
            f"{row.literature_mean_rho2:.3f} ± {row.literature_sd_rho2:.3f} (N={row.n_literature_templates})"
            for row in summary.itertuples(index=False)
        ],
        "Literature median rho² [90% CI] (N)": [
            f"{row.literature_median_rho2:.3f} [{row.literature_median_ci90_low:.3f}, {row.literature_median_ci90_high:.3f}] (N={row.n_literature_templates})"
            for row in summary.itertuples(index=False)
        ],
        "Exact meta-template rho²": [f"{row.meta_rho2:.3f}" for row in summary.itertuples(index=False)],
    })
    presentation.to_csv(outdir / "literature_rho2_table_k4_k8.csv", index=False, sep=";")
    presentation.loc[presentation["K"].eq(5)].to_csv(outdir / "literature_rho2_table_k5.csv", index=False, sep=";")
    (outdir / "literature_rho2_table_provenance.json").write_text(json.dumps({
        "k_solutions": [4, 5, 6, 7, 8], "literature_metric": "R2=rho_squared",
        "summary": "mean ± sample SD; median with deterministic percentile-bootstrap 90% CI",
        "bootstrap_resamples": 20000,
        "labels": "manual K-specific order in form_method_utils.MANUAL_REORDER_STR",
    }, indent=2) + "\n")


def save_literature_conformity(mat_path: Path, outdir: Path) -> None:
    """Write F04's one-piece K=4..8 literature-conformity figure and source tables."""
    literature, meta = literature_form_rows(mat_path)
    if len(literature) != 313 or not np.allclose(literature["R2"], literature["dipolarity"] ** 2):
        raise RuntimeError("Literature FORM rows must cover all maps with R2 equal to rho squared")
    literature.to_csv(outdir / "literature_form_map_rows.csv", index=False)
    meta.to_csv(outdir / "literature_form_meta_rows.csv", index=False)
    save_literature_rho2_table(literature, mat_path, outdir)

    # One compact boxplot axis per stored solution; assignments are read, never re-clustered.
    distmat = loadmat(mat_path, simplify_cells=True)["DistMat"]
    fig, axes = plt.subplots(5, 1, figsize=(u.PAPER_WIDTH_IN, 15.0), sharey=True, layout="constrained")
    base = literature.set_index("source_index")["R2"]
    for panel, (axis, k) in enumerate(zip(axes, range(4, 9))):
        assignments = np.asarray(distmat["ClusterAssignment"][k - 1], int).reshape(-1)
        manual_labels = u.labels_from_manual_order(k)
        if manual_labels is None or len(assignments) != len(base):
            raise RuntimeError(f"F04 requires the full K={k} assignment and manual label order")
        solution = u.load_microstates(SCRIPT_DIR / "metamaps_export_lemon.json", k=k)
        frame = pd.DataFrame({"R2": base.to_numpy(),
                              "label": [manual_labels[int(value) - 1] for value in assignments]})
        order = list(solution["labels"])
        palette = {label: u.STATE_COLORS.get(label, "0.55") for label in order}
        sns.boxplot(data=frame, x="label", y="R2", order=order, hue="label", hue_order=order,
                    palette=palette, dodge=False, showfliers=False, width=.55, linewidth=.7,
                    legend=False, ax=axis)
        sns.stripplot(data=frame, x="label", y="R2", order=order, hue="label", hue_order=order,
                      palette=palette, dodge=False, jitter=.16, size=2.2, alpha=.35, linewidth=0,
                      legend=False, ax=axis)
        _, _, _, meta_fit = u.project_fixed_dipole(solution["B"], u.sensor_geometry(solution["info"])["S_orth"])
        meta_points = pd.DataFrame({"label": order, "R2": meta_fit["R2"].to_numpy(float)})
        sns.scatterplot(data=meta_points, x=np.arange(len(order)), y="R2", hue="label", hue_order=order,
                        palette=palette, marker="D", s=42, edgecolor="black", linewidth=.6,
                        legend=False, ax=axis, zorder=4)
        axis.set(title=f"K = {k}", xlim=(-.6, len(order) - .4),
                 ylabel=r"FORM conformity $\rho^2$" if panel == 2 else "")
        for tick in axis.get_xticklabels():
            tick.set_color(u.STATE_COLORS.get(tick.get_text(), "0.45"))
        u.set_publication_spines(axis, categorical=True)
    _save_figure(fig, outdir / "figure_literature_conformity.svg")
    plt.close(fig)
    (outdir / "literature_form_provenance.json").write_text(json.dumps({
        "n_maps": len(literature), "assignments": "MAT ClusterAssignment K4-K8 with manual A-H order",
        "metric": "R2=rho_squared", "geometry": "per-map native EEGLAB coordinates -> fitted sphere",
        "figure": "single 6-inch-wide, 15-inch-tall K4-K8 stacked boxplot figure; F-H use neutral grey"}, indent=2) + "\n")

def _minimal_angle_reference(axis) -> None:
    """Replace conventional angle axes with zero crosshairs and 15-degree bars."""
    axis.set(xlabel="", ylabel="", xticks=[], yticks=[])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.axhline(0, color="0.25", lw=.8, zorder=20)
    axis.axvline(0, color="0.25", lw=.8, zorder=20)
    axis.text(2, 2, "0", fontsize=10, color="0.2", ha="left", va="bottom", zorder=21)
    x0, y0, length = -78, -78, 15
    axis.plot([x0, x0 + length], [y0, y0], color="0.15", lw=3, solid_capstyle="butt", zorder=21)
    axis.plot([x0, x0], [y0, y0 + length], color="0.15", lw=3, solid_capstyle="butt", zorder=21)
    axis.text(x0 + length / 2, y0 - 5, "15°", fontsize=10, ha="center", va="top", zorder=21)
    axis.text(x0 - 5, y0 + length / 2, "15°", fontsize=10, ha="right", va="center", rotation=90, zorder=21)


def save_orientation_assets(mat_path: Path, outdir: Path) -> None:
    """Write F07's three vector views, dense field family, and angle asset."""
    literature, _ = literature_form_rows(mat_path)
    vectors = literature[["q_x", "q_y", "q_z"]].to_numpy(float)
    views = ((30, -60, "orientation_vectors_isometric.svg"),
             (-1.4568, 170.2980, "orientation_vectors_lateral.svg"),
             (90, -90, "orientation_vectors_top.svg"))
    for elevation, azimuth, name in views:
        figure = u.plot_literature_bidirectional_dipole_view(
            vectors, literature.label, elev=elevation, azim=azimuth, figsize=(4.0, 4.0)
        )
        output = outdir / name
        figure.savefig(output, format="svg", bbox_inches=None)
        plt.close(figure)
        u.assert_svg_width(output, expected_inches=4.0)
    # Dense 11x11 FORM field family. The former implementation lived in a
    # notebook-only helper; this direct construction keeps the approved grid
    # reproducible from the bundled canonical montage.
    dataset = u.load_microstates(SCRIPT_DIR / "metamaps_export_lemon.json", k=5)
    geometry = u.sensor_geometry(dataset["info"])
    field_path = outdir / "orientation_field_family.svg"
    field_figure, field_axes = plt.subplots(11, 11, figsize=(2.5, 2.5))
    angles = np.linspace(-75.0, 75.0, 11)
    for row, phi in enumerate(angles[::-1]):
        for column, theta in enumerate(angles):
            axis = field_axes[row, column]
            field = geometry["S_orth"] @ u.unit_from_angles(theta, phi)
            mne.viz.plot_topomap(field, dataset["info"], axes=axis, show=False, sensors=False,
                                 contours=0, cmap="RdBu_r", vlim=(-1.0, 1.0), sphere="auto")
            axis.set_axis_off()
    field_figure.subplots_adjust(0, 0, 1, 1, 0, 0)
    field_figure.savefig(field_path, format="svg", bbox_inches=None)
    plt.close(field_figure)
    u.assert_svg_width(field_path, expected_inches=2.5)

    # Literature angles with the five canonical K=5 metamaps alongside them.
    figure = plt.figure(figsize=(3.5, 2.5), layout="constrained")
    grid = figure.add_gridspec(5, 2, width_ratios=(4.4, 1.2), wspace=.08)
    axis = figure.add_subplot(grid[:, 0])
    for label in u.CANONICAL_LABELS:
        data = literature.loc[literature.label.eq(label)]
        axis.scatter(data.theta_deg, data.phi_deg, s=7, alpha=.58, color=u.STATE_COLORS[label], label=label)
    axis.set(xlim=(-90,90), ylim=(-90,90))
    _minimal_angle_reference(axis)
    # Grid rows run from top to bottom: E, D, C, B, A, so A is lowest.
    for row, label in enumerate(reversed(u.CANONICAL_LABELS)):
        topomap_axis = figure.add_subplot(grid[row, 1])
        index = list(dataset["labels"]).index(label)
        mne.viz.plot_topomap(dataset["B"][:, index], dataset["info"], axes=topomap_axis,
                             show=False, sensors=False, contours=0, cmap="RdBu_r", sphere="auto")
        topomap_axis.add_patch(patches.Circle((.5, .5), .46, transform=topomap_axis.transAxes,
                                              fill=False, lw=2.0, color=u.STATE_COLORS[label]))
        topomap_axis.text(1.08, .5, label, transform=topomap_axis.transAxes, va="center",
                          fontweight="bold", color=u.STATE_COLORS[label], fontsize=10)
    figure.savefig(outdir / "orientation_literature_angles.svg", format="svg", bbox_inches=None); plt.close(figure)
    widths = {name: width for _, _, name in views for width in [4.0]}
    widths.update({"orientation_field_family.svg": 2.5, "orientation_literature_angles.svg": 3.5})
    (outdir / "orientation_assets_metadata.json").write_text(json.dumps({
        "width_inches": widths, "n_literature_axes": int(len(literature)),
        "vector_representation": "full polarity-invariant literature dipole axes [-q,+q]",
        "coordinates": "theta/phi are polarity-canonical display angles; F07 vector views show every literature q as its antipodal axis [-q,+q]",
    }, indent=2) + "\n")

HEAD_RADIUS = 1.0
TIP_RADIUS = 0.018
TUBE_RADIUS = 0.006
CYLINDER_SIDES = 6
SPHERE_RINGS = 4
SPHERE_SEGMENTS = 8


def _glb_fixed_dipole_vector(values, chanlocs):
    """Match the fixed-dipole direction used by the existing literature plot."""
    x = np.asarray(values, dtype=float).reshape(-1)
    xyz = np.asarray([[c["X"], c["Y"], c["Z"]] for c in chanlocs], dtype=float)
    good = np.isfinite(x) & np.isfinite(xyz).all(axis=1)
    x, xyz = x[good], xyz[good]
    geometry = xyz - xyz.mean(axis=0, keepdims=True)
    eigvals, eigvecs = np.linalg.eigh(geometry.T @ geometry)
    eigvals = np.maximum(eigvals, np.finfo(float).eps)
    whitening = geometry @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    x = x - x.mean()
    norm = np.linalg.norm(x)
    if norm == 0:
        return np.full(3, np.nan)
    q = whitening.T @ (x / norm)
    return -q if q[1] < 0 else q


def safe_geometry(directions, requested_tube_radius, requested_tip_radius):
    """Choose a common inner endpoint so non-collinear tubes cannot touch.

    Two radial cylinders at angular separation theta have centre-line distance
    ``2*r*sin(theta/2)`` at radius r.  The exporter starts at radius 0.72
    and reduces the render-only radii, when necessary, to 80% of the nearest
    half-separation.  This leaves a gap for every distinct source direction.
    """
    dots = np.clip(directions @ directions.T, -1.0, 1.0)
    angles = np.arccos(dots)
    angles[np.diag_indices_from(angles)] = np.inf
    positive = angles[angles > 1e-6]
    minimum = float(np.min(positive)) if positive.size else np.pi
    # Keep cylinders clear: tightly clustered source directions receive thinner render-only geometry.
    start = 0.72
    tube_radius = min(requested_tube_radius, 0.80 * start * np.sin(minimum / 2.0))
    tip_radius = min(requested_tip_radius, 0.80 * (HEAD_RADIUS - requested_tip_radius) * np.sin(minimum / 2.0))
    if tube_radius <= 0 or tip_radius <= 0:
        raise ValueError("Fixed-dipole directions must have a non-zero angular separation.")
    return start, tube_radius, tip_radius, minimum


def cylinder_between(a, b, radius, sides=CYLINDER_SIDES):
    axis = b - a
    length = np.linalg.norm(axis)
    if length == 0:
        return np.empty((0, 3), np.float32), np.empty((0,), np.uint32)
    axis /= length
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.arange(sides) * (2 * np.pi / sides)
    ring = radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    vertices = np.vstack((a + ring, b + ring)).astype(np.float32)
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.extend((i, j, sides + j, i, sides + j, sides + i))
    return vertices, np.asarray(faces, dtype=np.uint32)


def uv_sphere(center, radius, rings=SPHERE_RINGS, segments=SPHERE_SEGMENTS):
    vertices = [center + np.array([0.0, 0.0, radius]), center + np.array([0.0, 0.0, -radius])]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        for segment in range(segments):
            theta = 2 * np.pi * segment / segments
            vertices.append(center + radius * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]))
    faces, first_ring, last_ring = [], 2, 2 + (rings - 2) * segments
    for s in range(segments):
        nxt = (s + 1) % segments
        faces.extend((0, first_ring + nxt, first_ring + s))
        faces.extend((1, last_ring + s, last_ring + nxt))
    for ring in range(rings - 2):
        base, nxt_base = first_ring + ring * segments, first_ring + (ring + 1) * segments
        for s in range(segments):
            nxt = (s + 1) % segments
            faces.extend((base + s, base + nxt, nxt_base + nxt, base + s, nxt_base + nxt, nxt_base + s))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def merge_meshes(meshes):
    vertices, faces, offset = [], [], 0
    for points, indices in meshes:
        vertices.append(points)
        faces.append(indices + offset)
        offset += len(points)
    return np.vstack(vertices), np.concatenate(faces)


def head_ring():
    points = np.array([[np.cos(t), np.sin(t), 0.0] for t in np.linspace(0, 2 * np.pi, 97)[:-1]])
    meshes = [cylinder_between(points[i], points[(i + 1) % len(points)], 0.008, 6) for i in range(len(points))]
    # The small nose makes the circle's anatomical orientation unambiguous.
    nose = np.array([[-0.075, 1.0, 0.0], [0.0, 1.11, 0.0], [0.075, 1.0, 0.0]])
    meshes.extend(cylinder_between(nose[i], nose[i + 1], 0.008, 6) for i in range(2))
    return merge_meshes(meshes)


def gltf_asset(meshes, materials, name, bin_name):
    """Create a compact glTF document and its one binary buffer."""
    blob, buffer_views, accessors, gltf_meshes = bytearray(), [], [], []

    def append(data, target):
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        raw = data.tobytes()
        blob.extend(raw)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        return len(buffer_views) - 1

    for mesh_name, points, faces, material in meshes:
        point_view = append(np.asarray(points, dtype='<f4'), 34962)
        face_view = append(np.asarray(faces, dtype='<u4'), 34963)
        position = len(accessors)
        accessors.append({"bufferView": point_view, "componentType": 5126, "count": len(points), "type": "VEC3", "min": points.min(0).tolist(), "max": points.max(0).tolist()})
        indices = len(accessors)
        accessors.append({"bufferView": face_view, "componentType": 5125, "count": len(faces), "type": "SCALAR"})
        gltf_meshes.append({"name": mesh_name, "primitives": [{"attributes": {"POSITION": position}, "indices": indices, "material": material}]})

    document = {
        "asset": {"version": "2.0", "generator": "form_method"},
        "scene": 0, "scenes": [{"name": name, "nodes": list(range(len(gltf_meshes)))}],
        "nodes": [{"name": mesh["name"], "mesh": i} for i, mesh in enumerate(gltf_meshes)],
        "meshes": gltf_meshes, "materials": materials,
        "buffers": [{"uri": bin_name, "byteLength": len(blob)}],
        "bufferViews": buffer_views, "accessors": accessors,
    }
    return document, bytes(blob)


def write_exports(outdir, k, directions, assignments, colors):
    start, tube_radius, tip_radius, min_angle = safe_geometry(directions, TUBE_RADIUS, TIP_RADIUS)
    materials = [{"name": f"cluster_{i + 1}", "pbrMetallicRoughness": {"baseColorFactor": [*color, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.72}} for i, color in enumerate(colors)]
    materials.append({"name": "head_outline", "pbrMetallicRoughness": {"baseColorFactor": [0.03, 0.03, 0.03, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.8}})
    meshes = []
    for label in range(1, len(colors) + 1):
        cluster = []
        for direction in directions[assignments == label]:
            for sign in (-1.0, 1.0):
                tip = sign * (HEAD_RADIUS - tip_radius) * direction
                cluster.append(cylinder_between(sign * start * direction, tip, tube_radius))
                cluster.append(uv_sphere(sign * HEAD_RADIUS * direction, tip_radius))
        points, faces = merge_meshes(cluster)
        meshes.append((f"K{k}_cluster_{label}", points, faces, label - 1))
    points, faces = head_ring()
    meshes.append(("head_outline", points, faces, len(materials) - 1))
    stem = f"literature_fixed_dipoles_K{k}"
    document, blob = gltf_asset(meshes, materials, stem, f"{stem}.bin")
    (outdir / f"{stem}.gltf").write_text(json.dumps(document, separators=(",", ":")))
    (outdir / f"{stem}.bin").write_bytes(blob)
    embedded = json.loads(json.dumps(document))
    embedded["buffers"][0].pop("uri")
    json_chunk = json.dumps(embedded, separators=(",", ":")).encode()
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    bin_chunk = blob + b"\0" * ((-len(blob)) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk + struct.pack("<I4s", len(bin_chunk), b"BIN\0") + bin_chunk
    (outdir / f"{stem}.glb").write_bytes(glb)
    return {"k": k, "maps": int(len(directions)), "start_radius": start, "tube_radius": tube_radius, "tip_radius": tip_radius, "minimum_angular_separation_degrees": float(np.degrees(min_angle)), "glb": str(outdir / f"{stem}.glb")}





def export_literature_glb(mat_path: Path, outdir: Path) -> list[dict]:
    """Export existing literature fixed-dipole solutions as result-only GLB assets."""
    distmat = loadmat(mat_path, simplify_cells=True)["DistMat"]
    templates = distmat["TemplateMap"]
    directions = np.asarray([_glb_fixed_dipole_vector(item["Map"], item["chanlocs"]) for item in templates], dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    if not np.isfinite(directions).all():
        raise ValueError("At least one literature template has no finite fixed-dipole direction.")
    keep = np.ones(len(directions), dtype=bool)
    for index in range(1, len(directions)):
        keep[index] = not np.any(np.linalg.norm(directions[:index] - directions[index], axis=1) < 1e-6)
    directions = directions[keep]
    parameters = distmat["ClusterMaps"]["msinfo"]["ClustPar"]
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k in range(int(parameters["MinClasses"]), int(parameters["MaxClasses"]) + 1):
        solution = distmat["ClusterMaps"]["msinfo"]["MSMaps"][k - 1]
        assignments = np.asarray(distmat["ClusterAssignment"][k - 1], dtype=int).reshape(-1)[keep]
        solution_labels = u.labels_from_manual_order(k) or [str(index) for index in range(1, k + 1)]
        fallback_colors = np.asarray(solution["ColorMap"], dtype=float)
        colors = np.asarray([u.STATE_COLORS.get(label, fallback_colors[index])
                             for index, label in enumerate(solution_labels)], dtype=object)
        colors = np.asarray([tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
                             if isinstance(color, str) else color for color in colors], dtype=float)
        manifest.append(write_exports(outdir, k, directions, assignments, colors))
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

def save_literature_geometry_correspondence(mat_path: Path, outdir: Path) -> dict:
    """Compare literature topographic distances with polarity-invariant FORM distances."""
    distmat = loadmat(mat_path, simplify_cells=True)["DistMat"]
    literature, _ = literature_form_rows(mat_path)
    literature = literature.sort_values("source_index").reset_index(drop=True)
    n_templates = len(literature)
    if n_templates != 313 or not np.array_equal(literature.source_index.to_numpy(), np.arange(n_templates)):
        raise ValueError("Literature FORM rows must contain source indices 0 through 312 exactly once")
    d_literature = np.asarray(distmat["Dist"], dtype=float)
    if (d_literature.shape != (n_templates, n_templates) or not np.isfinite(d_literature).all()
            or not np.allclose(d_literature, d_literature.T, atol=1e-12, rtol=0)
            or not np.allclose(np.diag(d_literature), 0.0, atol=1e-12, rtol=0)):
        raise ValueError("DistMat.Dist must be finite, symmetric, and have a zero diagonal")
    q = literature.loc[:, ["q_x", "q_y", "q_z"]].to_numpy(float)
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(q).all() or np.any(norms == 0):
        raise ValueError("Literature FORM directions must be finite and nonzero")
    directions = q / norms[:, None]
    d_form = np.arccos(np.clip(np.abs(directions @ directions.T), 0.0, 1.0))
    first, second = np.triu_indices(n_templates, k=1)
    literature_pairs = d_literature[first, second]
    form_pairs = d_form[first, second]
    rho = float(spearmanr(literature_pairs, form_pairs).statistic)
    expected_rho = 0.9869466151207816
    if not np.isclose(rho, expected_rho, atol=1e-10, rtol=0):
        raise RuntimeError(f"Unexpected literature-FORM geometry Spearman rho: {rho:.16f}")
    pairs = pd.DataFrame({
        "source_index_i": first, "source_index_j": second,
        "template_id_i": literature.template_id.to_numpy()[first],
        "template_id_j": literature.template_id.to_numpy()[second],
        "literature_distance": literature_pairs,
        "form_angular_distance_rad": form_pairs,
    })
    pairs.to_csv(outdir / "literature_form_geometry_pairwise.csv", index=False)
    summary = {
        "n_templates": n_templates, "n_unique_pairs": len(pairs), "spearman_rho": rho,
        "literature_distance_source": "DistMat.Dist from metamaps_struct_lemon.mat",
        "form_orientation_source": "q_x, q_y, q_z from literature_form_rows(), normalized to unit length",
        "form_distance": "arccos(abs(u_i dot u_j))", "mds_coordinates_used": False,
        "cluster_information_used": False, "inferential_p_value_reported": False,
        "distmat_max_symmetry_error": float(np.max(np.abs(d_literature - d_literature.T))),
        "distmat_max_abs_diagonal": float(np.max(np.abs(np.diag(d_literature)))),
        "form_unit_norm_max_error": float(np.max(np.abs(np.linalg.norm(directions, axis=1) - 1.0))),
    }
    (outdir / "literature_form_geometry_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary

def _save_figure(fig, path: Path, *, width: float = u.PAPER_WIDTH_IN) -> None:
    fig.set_size_inches(width, fig.get_size_inches()[1], forward=True)
    fig.savefig(path, format="svg", bbox_inches=None)
    u.assert_svg_width(path, expected_inches=width)


def main() -> None:
    args = _parse_cli()
    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    outdir = args.outdir or ((args.config.parent / cfg["output_root"]).resolve() / "m_method")
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    u.configure_publication_style()
    mat_path = SCRIPT_DIR / "metamaps_struct_lemon.mat"
    if args.literature_conformity:
        save_literature_geometry_correspondence(mat_path, outdir)

    # Input: exact K=5 meta maps. Transform: existing radial normalization,
    # average referencing, and symmetric orthonormalization. Output: q/rho/R2.
    dataset = u.load_microstates(args.meta_json, k=5)
    labels, maps, info = dataset["labels"], dataset["B"], dataset["info"]
    if tuple(labels) != u.CANONICAL_LABELS:
        raise RuntimeError(f"K=5 maps must already be A-E ordered, got {labels}")
    geometry = u.sensor_geometry(info)
    q, projected, unit_reconstruction, fit = u.project_fixed_dipole(maps, geometry["S_orth"])
    if not np.allclose(geometry["S_orth"].T @ geometry["S_orth"], np.eye(3), atol=1e-10):
        raise RuntimeError("FORM basis must be orthonormal")
    if not np.allclose(fit["R2"].to_numpy(), np.sum(projected ** 2, axis=0), atol=1e-12):
        raise RuntimeError("FORM projection R2 must equal projected energy")
    if not np.allclose(fit["R2"].to_numpy(), fit["dipolarity"].to_numpy() ** 2, atol=1e-12):
        raise RuntimeError("FORM projection R2 must equal rho squared")
    if not np.allclose(np.linalg.norm(unit_reconstruction, axis=0), 1.0, atol=1e-12):
        raise RuntimeError("FORM reconstructed directions must be normalized")
    fit.insert(0, "label", labels)
    fit.to_csv(outdir / "form_k5_projection.csv", index=False)

    # Static map polarity is antipodally equivalent. Signed q is retained here;
    # it must not be substituted into adjacent-sample signed TD calculations.
    pd.DataFrame([{"design_condition": geometry["M_condition"],
                   "orthonormality_error": geometry["S_orth_residual_fro"]}]).to_csv(
        outdir / "form_k5_geometry_diagnostics.csv", index=False)
    fig = u.plot_fixed_space(maps, unit_reconstruction, labels, fit, info)
    _save_figure(fig, outdir / "figure_form_k5_projection.svg")

    if args.spatial_spectrum:
        spectrum = u.simulate_controlled_spectra(geometry["S"], geometry["S_orth"])
        spectrum_rows = spectrum["rows"]
        if not (spectrum_rows.groupby("condition").size() == 1000).all():
            raise RuntimeError("Controlled spectra require 1,000 maps per condition")
        if not np.allclose(spectrum_rows.loc[spectrum_rows.condition.eq("Pure FORM"), "rho2"], 1.0, atol=1e-12):
            raise RuntimeError("Pure FORM spectra must have rho squared equal to one")
        for condition, ratio in (("5:1", 5.0), ("2:1", 2.0)):
            values = spectrum_rows.loc[spectrum_rows.condition.eq(condition)]
            if not np.allclose(values.degree1_energy / values.degree2_energy, ratio, atol=1e-10):
                raise RuntimeError(f"{condition} spectrum has the wrong degree-1:degree-2 energy ratio")
        blocks = u._controlled_harmonic_blocks(geometry["S"], geometry["S_orth"], lmax=8)
        basis = np.column_stack([blocks[degree] for degree in sorted(blocks)])
        if not np.allclose(basis.mean(axis=0), 0.0, atol=1e-10) or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10):
            raise RuntimeError("Controlled harmonic basis must be centered and orthonormal")
        ranks = spectrum["metadata"]["block_ranks"]
        white = spectrum_rows.loc[spectrum_rows.condition.eq("White noise")]
        if spectrum["metadata"]["finite_montage_rank"] != basis.shape[1] or not np.isclose(white.degree1_energy.mean() / ranks[1], white.degree2_energy.mean() / ranks[2], rtol=0.15):
            raise RuntimeError("White-noise spectrum must have equal expected energy per retained mode")
        spectrum["rows"].to_csv(outdir / "controlled_spectrum_simulations.csv", index=False)
        (outdir / "controlled_spectrum_metadata.json").write_text(
            json.dumps(spectrum["metadata"], indent=2) + "\n", encoding="utf-8"
        )
        figure = u.plot_controlled_spectra(spectrum, info)
        _save_figure(figure, outdir / "figure_spatial_spectrum_selectivity.svg", width=6.0)
        plt.close(figure)

    if args.literature_conformity:
        save_literature_conformity(mat_path, outdir)
    if args.literature_mds:
        save_literature_mds(mat_path, outdir)
    if args.orientation_assets:
        save_orientation_assets(SCRIPT_DIR / "metamaps_struct_lemon.mat", outdir)
    if args.export_glb:
        export_literature_glb(SCRIPT_DIR / "metamaps_struct_lemon.mat", outdir / "literature_glb")

    provenance = {"k": 5, "labels": labels, "r2_definition": "rho_squared", "meta_json": str(args.meta_json),
                  "sphere_origin_m": geometry["origin_m"].tolist(),
                  "sphere_radius_m": float(geometry["radius_m"])}
    (outdir / "form_method_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote fixed-FORM method results to {outdir}")


if __name__ == "__main__":
    main()
