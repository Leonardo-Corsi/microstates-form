"""Fit and cache whole-group microstate solutions from recording-level maps.

Run after form_validation_a_subject_level.py. All recording-selected centroids are
pooled and fitted with native Pycrostates for K=1..15. The primary backfitting solution
is K=5, while the independently computed metacriterion and every K model remain
available for replication checks.
"""

import argparse
import atexit
import json
from pathlib import Path
import shutil
import threading

import numpy as np
import pandas as pd
import psutil
import yaml
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

import form_method_utils as u
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import form_validation_abc_utils as mvud


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yml"

CLUSTERING_IMPLEMENTATION = "Pycrostates ModKMeans"
CRITERIA_COLUMNS = (
    "Gamma", "Silhouette", "Davies_Bouldin", "Point_Biserial", "Dunn",
    "Krzanowski_Lai", "Cross_Validation",
)
CRITERIA_MINIMIZE = {"Davies_Bouldin", "Cross_Validation"}


def _format_ram(n_bytes: int) -> str:
    return f"{n_bytes / 1024 ** 3:.2f} GiB"


def _start_ram_monitor(interval_seconds: float):
    """Periodically report main and recursively spawned-process RSS."""
    stop_event = threading.Event()
    main_process = psutil.Process()

    def report():
        while not stop_event.is_set():
            try:
                worker_rss = 0
                worker_count = 0
                for process in main_process.children(recursive=True):
                    try:
                        worker_rss += process.memory_info().rss
                        worker_count += 1
                    except psutil.Error:
                        pass
                system_memory = psutil.virtual_memory()
                tqdm.write(
                    "RAM | "
                    f"main={_format_ram(main_process.memory_info().rss)} | "
                    f"spawned={_format_ram(worker_rss)} ({worker_count} processes) | "
                    f"system={_format_ram(system_memory.used)}/{_format_ram(system_memory.total)}"
                )
            except psutil.Error:
                pass
            stop_event.wait(max(float(interval_seconds), 0.1))

    thread = threading.Thread(target=report, name="ram-monitor", daemon=True)
    thread.start()
    atexit.register(stop_event.set)
    return stop_event, thread


def _parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def save_group_criteria_summary(criteria: pd.DataFrame, selected_k: int,
                                metacriterion_selected_k: int, output_file: Path) -> None:
    """Write a readable fixed-width group-level seven-criterion K summary."""
    missing = set(CRITERIA_COLUMNS).difference(criteria.columns)
    if missing:
        raise ValueError(f"Group criteria are missing columns: {sorted(missing)}")
    criteria = criteria.sort_values("k")
    names = {
        "gamma": "Gamma ↑", "silhouette": "Silhouette ↑",
        "davies_bouldin": "Davies–Bouldin ↓", "point_biserial": "Point-biserial ↑",
        "dunn": "Dunn ↑", "krzanowski_lai": "Krzanowski–Lai ↑",
        "cross_validation": "Cross-validation ↓",
    }
    u.configure_publication_style()
    figure, axes = plt.subplots(2, 4, figsize=(u.PAPER_WIDTH_IN, 4.35), constrained_layout=True)
    for index, (axis, criterion) in enumerate(zip(axes.ravel(), CRITERIA_COLUMNS)):
        axis.plot(criteria["k"], criteria[criterion], color="#333333", marker="o", markersize=2.5)
        axis.axvline(selected_k, color="#333333", linewidth=0.9, linestyle="--")
        axis.axvline(metacriterion_selected_k, color="#777777", linewidth=0.9, linestyle=":")
        axis.set_title(names.get(str(criterion).lower(), str(criterion).replace("_", "–")), fontsize=10)
        axis.set_xlim(criteria["k"].min() - .35, criteria["k"].max() + .35)
        axis.set_xticks([1, 5, 10, 15])
        axis.set_xlabel("K" if index >= 4 else "")
        axis.set_ylabel("value" if index % 4 == 0 else "")
        axis.tick_params(labelsize=10)
        u.set_publication_spines(axis, categorical=False)
    axes.ravel()[-1].axis("off")
    figure.suptitle("Stage-B criteria (dashed: configured K; dotted: metacriterion K)", fontsize=10)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, format="svg", bbox_inches=None)
    u.assert_svg_width(output_file)
    figure.savefig(output_file.with_suffix(".png"), dpi=300, bbox_inches=None)
    plt.close(figure)


def _mat_info(chanlocs) -> mne.Info:
    names, positions = [], []
    for index, channel in enumerate(np.atleast_1d(chanlocs)):
        names.append(str(channel.get("labels", f"EEG{index:03d}")))
        positions.append([-float(channel["Y"]), float(channel["X"]), float(channel["Z"])])
    positions = np.asarray(positions, dtype=float)
    info = mne.create_info(names, sfreq=1.0, ch_types="eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, positions)), coord_frame="head"), on_missing="ignore")
    return info


def _aligned_abs_correlation(first_maps, first_names, second_maps, second_names) -> tuple[np.ndarray, list[str]]:
    first_lookup = {str(name).lower(): i for i, name in enumerate(first_names)}
    second_lookup = {str(name).lower(): i for i, name in enumerate(second_names)}
    shared = [name for name in first_lookup if name in second_lookup]
    if len(shared) < 3:
        raise ValueError("Need at least three shared channels for map correspondence")
    first = np.asarray(first_maps, float)[[first_lookup[name] for name in shared]]
    second = np.asarray(second_maps, float)[[second_lookup[name] for name in shared]]
    first = u.center_l2_cols(first)
    second = u.center_l2_cols(second)
    return np.abs(first.T @ second), shared


def save_lemon_correspondence(
    recording_index: pd.DataFrame,
    criteria: pd.DataFrame,
    votes: pd.DataFrame,
    group_maps: np.ndarray,
    meta_maps: np.ndarray,
    meta_names: list[str],
    meta_info: mne.Info,
    mat_file: Path,
    output_file: Path,
) -> None:
    """Render F05 from cache-backed selection and exact Zanesco 2020 map inputs."""
    distmat = loadmat(mat_file, simplify_cells=True)["DistMat"]
    source = next(item for item in distmat["TemplateSet"] if item["setname"] == "Zanesco 2020")
    historic = np.asarray(source["msinfo"]["MSMaps"][4]["Maps"], float).T
    historic_info = _mat_info(source["chanlocs"])
    historic_corr, shared_names = _aligned_abs_correlation(
        meta_maps, meta_names, historic, historic_info.ch_names)
    historic_order = linear_sum_assignment(-historic_corr)[1]
    historic = historic[:, historic_order]
    historic_corr = historic_corr[:, historic_order]
    current_corr, _ = _aligned_abs_correlation(group_maps, meta_names, meta_maps, meta_names)
    # Reordering map columns never changes the electrode montage; using the
    # five map indices as channel indices previously reduced this to a spurious
    # five-channel correlation.
    current_historic_corr, _ = _aligned_abs_correlation(group_maps, meta_names, historic, historic_info.ch_names)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"recording_id": recording_index["recording_id"], "condition": recording_index["condition"],
                  "selected_k": recording_index["selected_k"]}).to_csv(output_file.parent / "lemon_recording_selected_k.csv", index=False)
    votes.to_csv(output_file.parent / "lemon_group_criterion_choices.csv", index=False)
    pd.DataFrame(historic_corr, index=list(u.CANONICAL_LABELS), columns=list(u.CANONICAL_LABELS)).to_csv(output_file.parent / "lemon_meta_zanesco_correlation.csv")
    pd.DataFrame(current_corr, index=list(u.CANONICAL_LABELS), columns=list(u.CANONICAL_LABELS)).to_csv(output_file.parent / "lemon_current_meta_correlation.csv")
    pd.DataFrame(current_historic_corr, index=list(u.CANONICAL_LABELS), columns=list(u.CANONICAL_LABELS)).to_csv(output_file.parent / "lemon_current_zanesco_correlation.csv")
    pd.DataFrame({"shared_channel": shared_names}).to_csv(output_file.parent / "lemon_zanesco_channel_correspondence.csv", index=False)

    fig = plt.figure(figsize=(u.PAPER_WIDTH_IN, 8.0), layout="constrained")
    grid = fig.add_gridspec(6, 5, height_ratios=[1.1, 1, 1, 1, 1.1, 1.1])
    selection = fig.add_subplot(grid[0, :3])
    counts = recording_index.groupby(["condition", "selected_k"]).size().rename("n").reset_index()
    for condition, hatch in u.CONDITION_HATCH.items():
        subset = counts.loc[counts.condition.eq(condition)]
        selection.bar(subset.selected_k + (-0.18 if condition == "EC" else 0.18), subset.n, width=.34,
                      color="0.7", edgecolor="0.2", hatch=hatch, label=condition)
    selection.set(xlabel="recording selected K", ylabel="recordings")
    selection.legend(frameon=False, ncol=2)
    u.set_publication_spines(selection, categorical=False)
    selection.text(.98, .95, f"primary K=5; vote K={int(criteria.loc[criteria.metacriterion_selected, 'k'].iloc[0])}",
                   transform=selection.transAxes, ha="right", va="top", fontsize=10)
    criterion_axis = fig.add_subplot(grid[0, 3:])
    criterion_axis.axis("off")
    criterion_axis.table(cellText=votes[["criterion", "optimal_k"]].values, colLabels=["criterion", "K"], loc="center", cellLoc="left")

    rows = [("Meta (Koenig et al., 2024)", meta_maps, meta_info),
            ("Historical (Zanesco, 2020; re-rendered)", historic, historic_info),
            ("Current", group_maps, meta_info)]
    for row_index, (title, maps, info) in enumerate(rows, start=1):
        for column, label in enumerate(u.CANONICAL_LABELS):
            axis = fig.add_subplot(grid[row_index, column])
            mne.viz.plot_topomap(maps[:, column], info, axes=axis, show=False, contours=0,
                                 sensors=False, cmap="RdBu_r", sphere="auto")
            axis.set_title(label if row_index == 1 else "", color=u.STATE_COLORS[label], fontsize=10)
            if column == 0: axis.set_ylabel(title, fontsize=10)
    for row_index, (matrix, title) in enumerate(((current_corr, "Current vs Koenig |r|"),
                                                  (current_historic_corr, "Current vs Zanesco |r|")), start=4):
        axis = fig.add_subplot(grid[row_index, :])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set(xticks=range(5), yticks=range(5), xticklabels=u.CANONICAL_LABELS, yticklabels=u.CANONICAL_LABELS, title=title)
        for i in range(5):
            for j in range(5): axis.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=10)
        fig.colorbar(image, ax=axis, fraction=.03, pad=.02, label="|r|")
    fig.savefig(output_file, format="svg", bbox_inches=None)
    u.assert_svg_width(output_file)
    plt.close(fig)



def save_f05_combined_figure(
    recording_criteria: pd.DataFrame,
    group_maps: np.ndarray,
    meta_maps: np.ndarray,
    meta_names: list[str],
    meta_info: mne.Info,
    mat_file: Path,
    output_file: Path,
) -> None:
    """F05: recording-wise criteria, metacriterion counts, templates, and correlations."""
    missing = set(CRITERIA_COLUMNS).difference(recording_criteria.columns)
    if missing:
        raise ValueError(f"Recording criteria are missing columns: {sorted(missing)}")
    distmat = loadmat(mat_file, simplify_cells=True)["DistMat"]
    source = next(item for item in distmat["TemplateSet"] if item["setname"] == "Zanesco 2020")
    historic = np.asarray(source["msinfo"]["MSMaps"][4]["Maps"], float).T
    historic_info = _mat_info(source["chanlocs"])
    meta_historic_corr, _ = _aligned_abs_correlation(meta_maps, meta_names, historic, historic_info.ch_names)
    historic_order = linear_sum_assignment(-meta_historic_corr)[1]
    historic = historic[:, historic_order]
    current_meta_corr, _ = _aligned_abs_correlation(group_maps, meta_names, meta_maps, meta_names)
    current_historic_corr, _ = _aligned_abs_correlation(group_maps, meta_names, historic, historic_info.ch_names)

    names = {
        "Gamma": "Gamma", "Silhouette": "Silhouette", "Davies_Bouldin": "Davies–Bouldin",
        "Point_Biserial": "Point-biserial", "Dunn": "Dunn",
        "Krzanowski_Lai": "Krzanowski–Lai", "Cross_Validation": "Cross-validation",
    }
    direction = {"Gamma": "↑", "Silhouette": "↑", "Davies_Bouldin": "↓",
                 "Point_Biserial": "↑", "Dunn": "↑", "Krzanowski_Lai": "↑", "Cross_Validation": "↓"}
    criteria = recording_criteria.copy()
    k_values = sorted(int(k) for k in criteria["k"].dropna().unique())
    colors = dict(zip(CRITERIA_COLUMNS, plt.cm.viridis(np.linspace(.1, .9, len(CRITERIA_COLUMNS)))))
    winning_k = group_maps.shape[1]

    u.configure_publication_style()
    fig = plt.figure(figsize=(u.PAPER_WIDTH_IN, 10.5), layout="constrained")
    fig.set_constrained_layout_pads(h_pad=.10, hspace=.12)
    outer = fig.add_gridspec(6, 1, height_ratios=(3.2, 1.0, 1.0, 1.0, 1.08, 1.08), hspace=.12)
    criterion_grid = outer[0].subgridspec(2, 4, wspace=.33, hspace=.42)
    for index, criterion in enumerate(CRITERIA_COLUMNS):
        axis = fig.add_subplot(criterion_grid[index])
        grouped = criteria.groupby("k")[criterion]
        median = grouped.median().reindex(k_values)
        q25 = grouped.quantile(.25).reindex(k_values)
        q75 = grouped.quantile(.75).reindex(k_values)
        color = colors[criterion]
        axis.fill_between(k_values, q25.to_numpy(float), q75.to_numpy(float), color=color, alpha=.25, linewidth=0)
        axis.plot(k_values, median.to_numpy(float), color=color, lw=1.5, marker="o", ms=2.8)
        axis.axvline(winning_k, color="0.2", linestyle="--", linewidth=1.0, zorder=0)
        axis.set(title=names[criterion], xlabel="k")
        axis.text(.02, .96, direction[criterion], transform=axis.transAxes, ha="left", va="top", fontsize=10)
        u.set_publication_spines(axis, categorical=False)
    selection_axis = fig.add_subplot(criterion_grid[-1])
    counts = (criteria.loc[criteria["metacriterion_selected"].fillna(False)].groupby("k").size()
              .reindex(k_values, fill_value=0))
    bar_colors = np.repeat("0.25", len(k_values)).astype(object)
    if winning_k in k_values:
        bar_colors[k_values.index(winning_k)] = plt.cm.viridis(.75)
    selection_axis.bar(k_values, counts.to_numpy(), color=bar_colors, width=.72)
    selection_axis.axvline(winning_k, color="0.2", linestyle="--", linewidth=1.0, zorder=0)
    selection_axis.set(title="Metacriterion", xlabel="", ylabel="")
    u.set_publication_spines(selection_axis, categorical=False)

    rows = [("Meta\n2024", meta_maps, meta_info),
            ("LEMON\n2020", historic, historic_info),
            ("Current", group_maps, meta_info)]
    for outer_index, (row_name, maps, info) in enumerate(rows, start=1):
        map_grid = outer[outer_index].subgridspec(1, 5, wspace=.03)
        for column, label in enumerate(u.CANONICAL_LABELS):
            axis = fig.add_subplot(map_grid[column])
            mne.viz.plot_topomap(maps[:, column], info, axes=axis, show=False, contours=0,
                                 sensors=False, cmap="RdBu_r", sphere="auto")
            if outer_index == 1:
                axis.set_title(label, color=u.STATE_COLORS[label], fontsize=10)
            if column == 0:
                axis.set_ylabel(row_name, fontsize=10)

    def correlation_axis(spec, matrix, title):
        axis = fig.add_subplot(spec)
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set(xticks=range(5), yticks=range(5), xticklabels=u.CANONICAL_LABELS,
                 yticklabels=u.CANONICAL_LABELS, title=title)
        for row in range(5):
            for column in range(5):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=10)
        fig.colorbar(image, ax=axis, fraction=.03, pad=.02, label="|r|")
        return axis

    correlation_axis(outer[4], current_meta_corr, "Current vs Koenig |r|")
    correlation_axis(outer[5], current_historic_corr, "Current vs Zanesco |r|")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="svg", bbox_inches=None)
    u.assert_svg_width(output_file)
    plt.close(fig)


def main():
    cli = _parse_cli()
    with cli.config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    ram_stop, ram_thread = _start_ram_monitor(cfg["ram_monitor_interval_seconds"])

    base = cli.config.resolve().parent
    group = cfg["b_group_level"]
    output_root = base / Path(cfg["output_root"])
    meta_json = base / Path(cfg["meta_json"])
    # Pool recording-level centroids with their source metadata.
    summaries_dir = output_root / "a_subject_level" / "summaries"
    if not summaries_dir.exists():
        raise FileNotFoundError(
            f"Canonical cache summaries not found: {summaries_dir}. Run form_validation_a_subject_level.py first."
        )
    print("[B] Loading atomized subject-level summaries.")
    recording_index = mvud.load_recording_jsons(summaries_dir, "*_summary.json")
    if recording_index.empty:
        raise RuntimeError("The recording-level index contains no successful recordings.")

    stage_dir = output_root / "b_group_level"
    model_dir = stage_dir / "models"
    stage_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    recording_index["selected_model_file"] = recording_index.apply(
        lambda row: str(
            output_root / "a_subject_level" / "models" / str(row["recording_id"])
            / f"k-{int(row['selected_k']):02d}_modkmeans.fif"
        ),
        axis=1,
    )

    print("[B] Pooling selected recording centroids.")
    template = mvud.load_meta_template(meta_json, k=group["selected_k"])
    with threadpool_limits(limits=1):
        pooled_maps, pooled_metadata, pooled_info = mvud.pool_selected_recording_centers(
            recording_index
        )
    pooled_file = stage_dir / "pooled_recording_centroids.npz"
    np.savez_compressed(
        pooled_file,
        maps=pooled_maps.astype(np.float32),
        **pooled_metadata,
    )

    # Fit independent pooled group solutions and evaluate the selection criteria.
    group_data = mvud.make_chdata(pooled_maps, pooled_info)
    models_by_k = {}
    model_paths = {
        k: model_dir / f"k-{k:02d}_modkmeans.fif"
        for k in range(group["k_min"], group["k_max"] + 1)
    }
    print(
        f"[B] Fitting group K={group['k_min']}..{group['k_max']} with "
        "native Pycrostates ModKMeans."
    )
    with tqdm(total=len(model_paths), desc="B group K fits", position=0) as progress:
        with threadpool_limits(limits=1):
            for k, model_file in model_paths.items():
                if model_file.exists() and not cfg["overwrite"]:
                    model = mvud.read_cluster_model(model_file)
                else:
                    seed = mvud.stable_seed(cfg["random_seed"], "group", k)
                    model = mvud.fit_modkmeans(
                        group_data,
                        n_clusters=k,
                        n_init=group["n_init"],
                        max_iter=group["max_iter"],
                        tol=group["tol"],
                        random_state=seed,
                        n_jobs=group["n_jobs_clustering"],
                    )
                    mvud.save_cluster_model(model, model_file, overwrite=True)
                models_by_k[k] = model
                progress.update()

    criteria, votes, metacriterion_selected_k = mvud.compute_zanesco_metacriterion(
        models_by_k,
        pooled_maps,
        corr_threshold=group["criteria_corr_threshold"],
        criteria_min_k=group["criteria_min_k"],
        max_maps=group["criteria_max_maps"],
        max_pairs=group["criteria_max_pairs"],
        random_seed=mvud.stable_seed(cfg["random_seed"], "group", "criteria"),
    )
    criteria_file = stage_dir / "group_criteria.csv"
    votes_file = stage_dir / "group_votes.csv"
    criteria.to_csv(criteria_file, index=False)
    votes.to_csv(votes_file, index=False)
    save_group_criteria_summary(
        criteria,
        selected_k=int(group["selected_k"]),
        metacriterion_selected_k=int(metacriterion_selected_k),
        output_file=stage_dir / "group_criteria_summary.svg",
    )

    # Align the primary pooled solution to canonical A-E labels and polarity.
    primary_k = int(group["selected_k"])
    primary_model = models_by_k[primary_k]
    primary_model_file = model_paths[primary_k]
    primary_unordered_maps = np.asarray(primary_model.cluster_centers_, dtype=float).copy()
    primary_raw_labels = [f"group{index + 1}" for index in range(primary_unordered_maps.shape[0])]
    np.savez_compressed(
        stage_dir / "group_maps_raw.npz",
        raw_maps=primary_unordered_maps.astype(np.float32).T,
        raw_labels=np.asarray(primary_raw_labels, dtype="U"),
        ch_names=np.asarray(template["ch_names"], dtype="U"),
        selected_k=np.asarray(primary_k, dtype=np.int16),
    )
    ordered_maps, primary_matching = mvud.match_group_maps_to_meta(
        primary_unordered_maps,
        template["maps"],
        template["labels"],
        raw_labels=primary_raw_labels,
    )
    # Koenig assigns canonical labels once: every later condition fit targets this A-E array.
    rematched_group, group_identity = mvud.match_group_maps_to_meta(
        ordered_maps.T, template["maps"], template["labels"], raw_labels=list(template["labels"])
    )
    identity = group_identity.sort_values("canonical_output_index")["raw_map_index"].to_numpy(int)
    if not np.array_equal(identity, np.arange(primary_k)) or not np.allclose(rematched_group, ordered_maps, atol=1e-6):
        raise AssertionError("Canonical pooled A-E maps must rematch to Koenig with identity [0,1,2,3,4]")
    primary_model.reorder_clusters(
        order=primary_matching["group_original_index"].to_numpy(dtype=int).tolist()
    )
    primary_model.rename_clusters(new_names=template["labels"])
    mvud.save_cluster_model(primary_model, primary_model_file, overwrite=True)
    selected_copy = stage_dir / "selected_group_cluster_raw.fif"
    shutil.copy2(primary_model_file, selected_copy)
    matching_file = stage_dir / "meta_matching.csv"
    primary_matching.to_csv(matching_file, index=False)
    primary_matching.to_csv(stage_dir / "group_to_koenig_matching.csv", index=False)
    group_maps_file = stage_dir / "group_maps_A-E.npz"
    np.savez_compressed(
        group_maps_file,
        group_maps=ordered_maps.astype(np.float32),
        original_group_maps=primary_unordered_maps.astype(np.float32).T,
        meta_maps=template["maps"].astype(np.float32),
        labels=np.asarray(template["labels"], dtype="U"),
        ch_names=np.asarray(template["ch_names"], dtype="U"),
        selected_k=np.asarray(primary_k, dtype=np.int16),
        metacriterion_selected_k=np.asarray(metacriterion_selected_k, dtype=np.int16),
    )
    figure_file = mvud.save_group_topomap_figure(
        ordered_maps,
        template["maps"],
        template["labels"],
        template["info"],
        stage_dir / "group_maps_A-E.svg",
    )

    save_lemon_correspondence(
        recording_index, criteria, votes, ordered_maps, template["maps"],
        template["ch_names"], template["info"], base / "metamaps_struct_lemon.mat",
        stage_dir / "figure_lemon_correspondence.svg",
    )
    recording_criteria = pd.concat(
        [pd.read_csv(output_root / "a_subject_level" / "criteria" / f"{recording_id}_criteria.csv")
         for recording_id in recording_index["recording_id"]],
        ignore_index=True,
    )
    save_f05_combined_figure(
        recording_criteria, ordered_maps, template["maps"], template["ch_names"], template["info"],
        base / "metamaps_struct_lemon.mat", stage_dir / "figure_F05_combined.svg",
    )


    selected_criterion = criteria.loc[criteria["k"].eq(primary_k)].iloc[0]
    summary = {
        "n_recordings": int(len(recording_index)),
        "n_subjects": int(recording_index["subject"].nunique()),
        "n_pooled_recording_centroids": int(pooled_maps.shape[1]),
        "n_channels": int(pooled_maps.shape[0]),
        "clustering_implementation": CLUSTERING_IMPLEMENTATION,
        "primary_selected_k": primary_k,
        "metacriterion_selected_k": int(metacriterion_selected_k),
        "primary_model_GEV_fraction": float(primary_model.GEV_),
        "primary_model_GEV_percent": 100.0 * float(primary_model.GEV_),
        "primary_criterion_assigned_fraction": float(
            selected_criterion["assigned_fraction"]
        ),
        "minimum_meta_match_absolute_correlation": float(
            primary_matching["absolute_spatial_correlation"].min()
        ),
        "mean_meta_match_absolute_correlation": float(
            primary_matching["absolute_spatial_correlation"].mean()
        ),
        "selection_implementation": "Zanesco seven-criterion metacriterion",
    }
    summary_file = mvud.write_json(summary, stage_dir / "group_summary.json")
    print(
        "Group-level stage complete: "
        f"{pooled_maps.shape[1]} recording centroids, primary K={primary_k}, "
        f"metacriterion K={metacriterion_selected_k}. "
        f"Maps: {group_maps_file}"
    )
    ram_stop.set()
    ram_thread.join(timeout=1)


if __name__ == "__main__":
    main()
