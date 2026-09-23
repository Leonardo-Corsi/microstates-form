#!/usr/bin/env python3
"""
Prepare EC and EO LEMON EEG recordings in a Zanesco-style sensor space.

This script does not compute GFP, DISS, GFP peaks, microstate clusters,
labels, or GEV. It only prepares the EC and EO recordings for later analysis:

1. Load preprocessed EEGLAB EC and EO .set files.
2. Keep scalp EEG channels.
3. Add missing target channels as bad flat channels.
4. Interpolate bad/missing channels onto the fixed target channel list.
5. Apply average reference.
6. Save each recording as FIF.
7. Write recording length summaries.
8. Exclude subjects by a selectable approximation of Zanesco's examples, or
   by a user-supplied ID list.
9. Optionally write metadata-derived exclusion candidate tables.

Important limitation:
Zanesco reports excluding 12 participants with current psychological diagnoses,
with examples including substance abuse and unspecified hallucinations. The exact
12 subject IDs are not published in the paper. The built-in exclusion modes below
are transparent approximations based on the metadata table, not a guaranteed
reconstruction of the unpublished Zanesco list.
"""

import argparse
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

LEMON61 = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
    "AF7", "AF3", "AF4", "AF8",
    "F5", "F1", "F2", "F6",
    "FT7", "FC3", "FC4", "FT8",
    "C5", "C1", "C2", "C6",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "P5", "P1", "P2", "P6",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "Iz",
]

# Fixed interpolation target for this Zanesco (2020)-style LEMON analysis.
# LEMON61 plus Fpz and FCz gives 63 channels. This is our explicit target,
# not a reconstruction of an unpublished 64-channel list.
LEMON_TARGET_CHANNELS = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "PO9", "O1", "Oz", "O2", "PO10", "Iz",
]

BIOSEMI64 = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7",
    "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
    "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
    "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]

TARGETS = {
    "lemon": LEMON_TARGET_CHANNELS,
    "biosemi64": BIOSEMI64,
}

STRICT_CURRENT_SUBSTANCE_OR_HALLUCINATION_IDS = {
    "sub-010199",
    "sub-010240",
    "sub-010088",
    "sub-010241",
    "sub-010142",
    "sub-010036",
    "sub-010141",
    "sub-010288",
}

BROAD_SUBSTANCE_OR_HALLUCINATION_IDS = {
    "sub-010199",
    "sub-010081",
    "sub-010240",
    "sub-010090",
    "sub-010088",
    "sub-010241",
    "sub-010142",
    "sub-010036",
    "sub-010141",
    "sub-010288",
    "sub-010300",
    "sub-010317",
    "sub-010231",
}

EXCLUSION_MODES = {
    "none": set(),
    "strict_current_substance_or_hallucination": STRICT_CURRENT_SUBSTANCE_OR_HALLUCINATION_IDS,
    "broad_substance_or_hallucination": BROAD_SUBSTANCE_OR_HALLUCINATION_IDS,
}


def parse_subject_id(path: Path) -> str:
    match = re.search(r"sub-[0-9]{6}", path.name)
    if not match:
        raise ValueError(f"Cannot parse BIDS subject id from {path}")
    return match.group(0)


def find_set_files(data_root: Path, recursive: bool) -> list[Path]:
    patterns = ["*_EC*.set", "*_EO*.set"]
    direct = sorted(path for pattern in patterns for path in data_root.glob(pattern))
    if direct and not recursive:
        return direct
    return sorted(path for pattern in patterns for path in data_root.glob(f"**/{pattern}"))


def recording_condition(path: Path) -> str:
    match = re.search(r"_(EC|EO)(?:[_\.]|$)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot determine EC/EO condition from {path}")
    return match.group(1).upper()


def resolve_metadata_csv(data_root: Path, metadata_csv: Path | None) -> Path | None:
    if metadata_csv is not None:
        return metadata_csv

    candidates = [
        data_root / "Behavioural_Data_MPILMBB_LEMON" / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv",
        data_root / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv",
    ]
    candidates.extend(
        parent / "Behavioural_Data_MPILMBB_LEMON" / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
        for parent in data_root.parents
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_exclusion_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    ids: set[str] = set()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "ID" in df.columns:
            vals = df["ID"].dropna().astype(str)
        elif "subject" in df.columns:
            vals = df["subject"].dropna().astype(str)
        else:
            vals = df.iloc[:, 0].dropna().astype(str)
        ids.update(v.strip() for v in vals if v.strip())
    else:
        with open(path, "r", encoding="utf-8") as f:
            ids.update(line.strip() for line in f if line.strip())
    return ids


def get_exclusion_ids(mode: str, exclude_ids_path: Path | None) -> tuple[set[str], str]:
    if mode not in EXCLUSION_MODES and mode != "custom":
        raise ValueError(f"Unknown exclusion mode: {mode}")
    if mode == "custom":
        if exclude_ids_path is None:
            raise ValueError("--exclusion_mode custom requires --exclude_ids")
        return read_exclusion_ids(exclude_ids_path), "custom"
    built_in = set(EXCLUSION_MODES[mode])
    manual = read_exclusion_ids(exclude_ids_path)
    if manual:
        return built_in | manual, f"{mode}+manual"
    return built_in, mode


def clean_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"^(Green|Yellow|Red|Blue|White|Black)_[0-9]+_", "", name)
    name = name.replace(" ", "")
    return name


def standardize_channel_names(raw: mne.io.BaseRaw) -> None:
    mapping = {ch: clean_name(ch) for ch in raw.ch_names}
    raw.rename_channels(mapping)


def add_missing_channels(raw: mne.io.BaseRaw, target_chs: list[str]) -> list[str]:
    missing = [ch for ch in target_chs if ch not in raw.ch_names]
    if not missing:
        return []
    info = mne.create_info(missing, sfreq=raw.info["sfreq"], ch_types="eeg")
    data = np.zeros((len(missing), raw.n_times), dtype=float)
    add_raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.add_channels([add_raw], force_update_info=True)
    raw.info["bads"].extend(missing)
    return missing


def export_sxyz(target_chs: list[str], out_path: Path) -> None:
    montage = mne.channels.make_standard_montage("standard_1020")
    pos = montage.get_positions()["ch_pos"]
    missing = [ch for ch in target_chs if ch not in pos]
    if missing:
        raise RuntimeError("Missing standard_1020 coordinates for: " + ", ".join(missing))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(str(len(target_chs)) + "\n")
        for ch in target_chs:
            x, y, z = pos[ch]
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {ch}\n")


def text_from_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index)
    return df[present].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()


def write_exclusion_candidates(metadata_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(metadata_csv)
    needed = [
        "ID",
        "SKID_Diagnoses",
        "SKID_Diagnoses 1",
        "SKID_Diagnoses 2",
        "Comments_SKID_assessment",
        "DRUG",
        "DRUG_0=negative_1=Positive",
        "AUDIT",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    text_cols = [
        "SKID_Diagnoses",
        "SKID_Diagnoses 1",
        "SKID_Diagnoses 2",
        "Comments_SKID_assessment",
        "DRUG",
    ]
    txt = text_from_cols(df, text_cols)

    current = txt.str.contains(r"\bcurrent\b", regex=True)
    hallucination = txt.str.contains(r"halluc", regex=True)
    current_hallucination = txt.str.contains(
        r"(?:current[^|]*halluc|halluc[^|]*current)", regex=True
    )
    strict_current_substance = txt.str.contains(
        r"current[^|]*(?:alcohol|cannabis|substance|drug|dependence|abuse|addiction)",
        regex=True,
    )
    strict_current_substance_or_hallucination = (
        strict_current_substance | current_hallucination
    )
    broad_substance = txt.str.contains(
        r"alcohol|cannabis|substance|drug|dependence|abuse|addiction", regex=True
    )
    broad_substance_or_hallucination = broad_substance | hallucination

    drug_numeric = pd.to_numeric(df["DRUG_0=negative_1=Positive"], errors="coerce").fillna(0)
    drug_positive = drug_numeric.ne(0)
    non_none_skid = (
        df["SKID_Diagnoses"].notna()
        & df["SKID_Diagnoses"].astype(str).str.strip().str.lower().ne("none")
        & df["SKID_Diagnoses"].astype(str).str.strip().ne("")
    )

    out = df[needed].copy()
    out["builtin_strict_id"] = out["ID"].astype(str).isin(STRICT_CURRENT_SUBSTANCE_OR_HALLUCINATION_IDS)
    out["builtin_broad_id"] = out["ID"].astype(str).isin(BROAD_SUBSTANCE_OR_HALLUCINATION_IDS)
    out["candidate_current_anywhere"] = current
    out["candidate_current_hallucination"] = current_hallucination
    out["candidate_strict_current_substance"] = strict_current_substance
    out["candidate_strict_current_substance_or_hallucination"] = strict_current_substance_or_hallucination
    out["candidate_broad_substance_or_hallucination"] = broad_substance_or_hallucination
    out["candidate_drug_screen_positive"] = drug_positive
    out["candidate_non_none_SKID_Diagnoses"] = non_none_skid
    out["candidate_current_or_positive_drug"] = current | drug_positive

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "exclusion_candidates_all_metadata.csv", index=False)
    out.loc[out["builtin_strict_id"]].to_csv(
        out_dir / "builtin_strict_current_substance_or_hallucination.csv", index=False
    )
    out.loc[out["builtin_broad_id"]].to_csv(
        out_dir / "builtin_broad_substance_or_hallucination.csv", index=False
    )
    out.loc[strict_current_substance_or_hallucination].to_csv(
        out_dir / "regex_strict_current_substance_or_hallucination.csv", index=False
    )
    out.loc[broad_substance_or_hallucination].to_csv(
        out_dir / "regex_broad_substance_or_hallucination.csv", index=False
    )
    out.loc[current].to_csv(out_dir / "regex_current_anywhere.csv", index=False)
    out.loc[non_none_skid].to_csv(out_dir / "regex_non_none_SKID.csv", index=False)

    summary = pd.DataFrame(
        [
            {"criterion": "builtin_strict_current_substance_or_hallucination", "n": len(STRICT_CURRENT_SUBSTANCE_OR_HALLUCINATION_IDS)},
            {"criterion": "builtin_broad_substance_or_hallucination", "n": len(BROAD_SUBSTANCE_OR_HALLUCINATION_IDS)},
            {"criterion": "regex_current_anywhere", "n": int(current.sum())},
            {"criterion": "regex_current_hallucination", "n": int(current_hallucination.sum())},
            {"criterion": "regex_strict_current_substance", "n": int(strict_current_substance.sum())},
            {"criterion": "regex_strict_current_substance_or_hallucination", "n": int(strict_current_substance_or_hallucination.sum())},
            {"criterion": "regex_broad_substance_or_hallucination", "n": int(broad_substance_or_hallucination.sum())},
            {"criterion": "drug_screen_positive", "n": int(drug_positive.sum())},
            {"criterion": "non_none_SKID_Diagnoses", "n": int(non_none_skid.sum())},
            {"criterion": "current_or_positive_drug", "n": int((current | drug_positive).sum())},
        ]
    )
    summary.to_csv(out_dir / "exclusion_candidate_counts.csv", index=False)


def process_one(path: Path, out_fif: Path, target_chs: list[str]) -> dict:
    subject = parse_subject_id(path)
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    standardize_channel_names(raw)

    present_target = [ch for ch in target_chs if ch in raw.ch_names]
    extra = [ch for ch in raw.ch_names if ch not in target_chs]
    if not present_target:
        raise RuntimeError(f"No target scalp channels found in {path}")

    raw.pick(present_target)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names}, verbose="ERROR")

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose="ERROR")

    missing = add_missing_channels(raw, target_chs)
    raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")

    raw.reorder_channels(target_chs)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    out_fif.parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_fif, overwrite=True, verbose="ERROR")

    return {
        "subject": subject,
        "input_file": str(path),
        "output_file": str(out_fif),
        "sfreq": float(raw.info["sfreq"]),
        "n_times": int(raw.n_times),
        "duration_sec": float(raw.n_times / raw.info["sfreq"]),
        "duration_min": float(raw.n_times / raw.info["sfreq"] / 60.0),
        "n_channels_output": len(raw.ch_names),
        "missing_interpolated": ";".join(missing),
        "n_missing_interpolated": len(missing),
        "extra_channels_dropped": ";".join(extra),
    }


def write_ids(path: Path, ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(subject + "\n" for subject in sorted(ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yml"))
    parser.add_argument("--data_root", type=Path, default=None,
                        help="Downloaded EEGLAB files; defaults to lemon_root in config or the parent of data_root.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Prepared FIF directory; defaults to data_root in config.")
    parser.add_argument("--metadata_csv", type=Path, default=None)
    parser.add_argument(
        "--exclude_ids",
        type=Path,
        default=None,
        help="Optional CSV or TXT with subject IDs. With built-in modes these IDs are added. With custom mode this is the full exclusion list.",
    )
    parser.add_argument(
        "--exclusion_mode",
        choices=["none", "strict_current_substance_or_hallucination", "broad_substance_or_hallucination", "custom"],
        default="broad_substance_or_hallucination",
        help="Default removes the 13-subject broader substance/hallucination set.",
    )
    parser.add_argument("--target_montage", "--target64", dest="target_montage",
                        choices=[*TARGETS, "lemon_plus64"], default="lemon",
                        help="Fixed target: lemon (63 channels) or biosemi64. "
                             "--target64 and lemon_plus64 are legacy aliases.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Read the flat project config only after argument parsing, so --help
    # does not require a config file or access to the EEG directory.
    try:
        with args.config.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("config must contain a mapping")
        config_dir = args.config.resolve().parent
        if args.out is None:
            args.out = config_dir / Path(config["data_root"])
        if args.data_root is None:
            args.data_root = config_dir / Path(config.get("lemon_root", Path(config["data_root"]).parent))
        if args.metadata_csv is None and config.get("metadata_csv"):
            metadata = config_dir / Path(config["metadata_csv"])
            if metadata.is_file():
                args.metadata_csv = metadata
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(f"Cannot load preparation paths from {args.config}: {error}")
    if args.target_montage == "lemon_plus64":
        args.target_montage = "lemon"

    args.out.mkdir(parents=True, exist_ok=True)
    summary_dir = args.out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    args.metadata_csv = resolve_metadata_csv(args.data_root, args.metadata_csv)
    if args.metadata_csv is not None:
        write_exclusion_candidates(args.metadata_csv, summary_dir)

    exclude_ids, exclusion_label = get_exclusion_ids(args.exclusion_mode, args.exclude_ids)
    write_ids(summary_dir / "excluded_subject_ids_requested.txt", exclude_ids)

    target_chs = TARGETS[args.target_montage]
    with open(summary_dir / "target_channels.txt", "w", encoding="utf-8") as f:
        f.writelines(ch + "\n" for ch in target_chs)
    export_sxyz(target_chs, args.out / "montage" / f"{args.target_montage}.sxyz")

    set_files = find_set_files(args.data_root, args.recursive)

    rows = []
    skipped = []
    for path in tqdm(set_files, desc="Processing EC/EO files"):
        subject = parse_subject_id(path)
        condition = recording_condition(path)
        if subject in exclude_ids:
            skipped.append({
                "subject": subject,
                "condition": condition,
                "input_file": str(path),
                "reason": exclusion_label,
            })
            continue
        out_fif = (
            args.out / f"fif_{condition.lower()}_avgref"
            / f"{subject}_{condition}_avgref_raw.fif"
        )
        # Reuse historical paths when present rather than creating a second
        # FIF for the same subject-condition in an existing analysis tree.
        legacy_fif = (
            args.out / f"fif_{condition.lower()}_64avgref"
            / f"{subject}_{condition}_64avgref_raw.fif"
        )
        if legacy_fif.exists():
            if out_fif.exists():
                raise FileExistsError(f"Duplicate prepared recordings: {out_fif} and {legacy_fif}")
            out_fif = legacy_fif
        if args.dry_run:
            rows.append({
                "subject": subject,
                "condition": condition,
                "input_file": str(path),
                "output_file": str(out_fif),
                "dry_run": True,
            })
        else:
            row = process_one(path, out_fif, target_chs)
            row["condition"] = condition
            rows.append(row)

    pd.DataFrame(rows).to_csv(summary_dir / "recording_summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(summary_dir / "excluded_subjects_applied.csv", index=False)

    if rows and not args.dry_run and "duration_min" in rows[0]:
        durations = pd.DataFrame(rows).groupby("condition")["duration_min"]
        dur_summary = durations.agg(
            n_recordings="count",
            mean_duration_min="mean",
            sd_duration_min_sample="std",
            min_duration_min="min",
            max_duration_min="max",
        ).reset_index()
        dur_summary.to_csv(summary_dir / "duration_summary.csv", index=False)

    print(f"Target montage: {args.target_montage} ({len(target_chs)} channels)")
    print(f"Found EC/EO files: {len(set_files)}")
    print(f"Exclusion mode: {args.exclusion_mode}")
    print(f"Requested exclusions: {len(exclude_ids)}")
    print(f"Excluded matching EC/EO files: {len(skipped)}")
    print(f"Processed or listed: {len(rows)}")
    print(f"Metadata CSV: {args.metadata_csv if args.metadata_csv is not None else 'not found'}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
