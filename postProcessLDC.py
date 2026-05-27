#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REFERENCE_RE = re.compile(r"^(uxRMS|uyRMS|uxuy|ux|uy)_(cx|cy)_(\d+)_1\.csv$")
CASE_RE_RE = re.compile(r"(?:^|[_-])re(\d+)(?:[_-]|$)", re.IGNORECASE)
FIELDS = ("ux", "uz", "ux2", "uz2", "uxuz")

VARIABLE_MAP = {
    "ux": ("ux", "ux", 1.0),
    "uy": ("uz", "uz", 1.0),
    "uxRMS": ("ux_rms", "ux RMS", 10.0),
    "uyRMS": ("uz_rms", "uz RMS", 10.0),
    "uxuy": ("uxuz", "uxuz covariance", 500.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LDC profiles against digitized Prasad/Koseff reference curves.")
    parser.add_argument("case_id", nargs="?", default="default", help="Case id under output/<ID>.")
    parser.add_argument("--output-root", default="output", help="Root output directory.")
    parser.add_argument("--reference-dir", default=None, help="Directory containing reference CSV files.")
    parser.add_argument("--step-min", "--tstep-min", type=int, default=None, help="First timestep to include.")
    parser.add_argument("--step-max", "--tstep-max", type=int, default=None, help="Last timestep to include.")
    parser.add_argument("--chunk-samples", type=int, default=4096, help="Profile samples to process per chunk.")
    return parser.parse_args()


def case_reynolds(case_id: str) -> str:
    match = CASE_RE_RE.search(case_id)
    if not match:
        raise ValueError(f"case id must include Reynolds number like re10000: {case_id}")
    return match.group(1)


def read_named_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, autostrip=True)
    if data.dtype.names is None:
        raise ValueError(f"{path} does not have a header")

    return {
        name.strip(): np.atleast_1d(np.asarray(data[name], dtype=float))
        for name in data.dtype.names
    }


def read_key_value_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["key", "value"]:
            raise ValueError(f"{path} must contain key,value columns")
        return {row["key"]: row["value"] for row in reader}


def read_reference_curve(path: Path, profile: str) -> tuple[np.ndarray, np.ndarray]:
    columns = read_named_csv(path)
    lower_names = {name.strip().lower(): name for name in columns}
    if "x" not in lower_names or "y" not in lower_names:
        raise ValueError(f"{path} must contain x and y columns")

    x = columns[lower_names["x"]]
    y = columns[lower_names["y"]]

    if profile == "cy":
        return y, x
    if profile == "cx":
        return x, y

    raise ValueError(f"Unsupported profile orientation: {profile}")


def sorted_unique_average(s: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(s) & np.isfinite(q)
    s = s[valid]
    q = q[valid]
    order = np.argsort(s)
    s = s[order]
    q = q[order]

    unique_s, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    if unique_s.size != s.size:
        sums = np.zeros_like(unique_s, dtype=float)
        np.add.at(sums, inverse, q)
        q = sums / counts
        s = unique_s

    return s, q


def dtype_from_metadata(metadata: dict[str, str]) -> np.dtype:
    real_t_bytes = int(metadata["real_t_bytes"])
    if real_t_bytes == 4:
        return np.dtype("<f4")
    if real_t_bytes == 8:
        return np.dtype("<f8")
    raise ValueError(f"unsupported real_t_bytes: {real_t_bytes}")


def read_steps(path: Path) -> np.ndarray:
    columns = read_named_csv(path)
    if "step" not in columns:
        raise ValueError(f"{path} must contain a step column")
    return columns["step"].astype(np.int64)


def read_coordinates(path: Path, expected_length: int) -> np.ndarray:
    columns = read_named_csv(path)
    if "s" not in columns:
        raise ValueError(f"{path} must contain an s column")
    s = columns["s"]
    if s.size != expected_length:
        raise ValueError(f"{path} has {s.size} coordinates, expected {expected_length}")
    return s


def check_sample_file(path: Path, sample_count: int, length: int, dtype: np.dtype) -> None:
    expected = sample_count * length * len(FIELDS)
    bytes_expected = expected * dtype.itemsize
    bytes_actual = path.stat().st_size
    if bytes_actual != bytes_expected:
        values_actual = bytes_actual // dtype.itemsize
        raise ValueError(f"{path} has {values_actual} values, expected {expected}")


def select_steps(steps: np.ndarray, step_min: int | None, step_max: int | None) -> tuple[np.ndarray, int]:
    _, reverse_indices = np.unique(steps[::-1], return_index=True)
    keep_last_duplicate = np.zeros(steps.shape, dtype=bool)
    keep_last_duplicate[steps.size - 1 - reverse_indices] = True

    mask = keep_last_duplicate.copy()
    if step_min is not None:
        mask &= steps >= step_min
    if step_max is not None:
        mask &= steps <= step_max
    if not np.any(mask):
        raise ValueError("selected timestep window contains no profile samples")
    dropped_duplicates = int(steps.size - keep_last_duplicate.sum())
    return mask, dropped_duplicates


def compute_window_profile(
    sample_path: Path,
    sample_count: int,
    coordinates: np.ndarray,
    mask: np.ndarray,
    dtype: np.dtype,
    u_char: float,
    sample_velocity_scale: float,
    chunk_samples: int,
) -> dict[str, np.ndarray]:
    if chunk_samples <= 0:
        raise ValueError("--chunk-samples must be positive")

    length = coordinates.size
    check_sample_file(sample_path, sample_count, length, dtype)
    samples = np.memmap(sample_path, dtype=dtype, mode="r", shape=(sample_count, length, len(FIELDS)))

    selected_count = int(mask.sum())
    if selected_count == 0:
        raise ValueError("selected timestep window contains no profile samples")

    sums = np.zeros((length, len(FIELDS)), dtype=np.float64)
    for start in range(0, sample_count, chunk_samples):
        stop = min(start + chunk_samples, sample_count)
        chunk_mask = mask[start:stop]
        if not np.any(chunk_mask):
            continue

        chunk = np.asarray(samples[start:stop][chunk_mask], dtype=np.float64)
        sums += chunk.sum(axis=0)

    mean = sums / float(selected_count)
    if sample_velocity_scale <= 0.0:
        raise ValueError("sample_velocity_scale must be positive")

    inv_sample_velocity_scale = 1.0 / sample_velocity_scale
    inv_sample_velocity_scale2 = inv_sample_velocity_scale * inv_sample_velocity_scale
    mean_ux = mean[:, 0] * inv_sample_velocity_scale
    mean_uz = mean[:, 1] * inv_sample_velocity_scale
    mean_ux2 = mean[:, 2] * inv_sample_velocity_scale2
    mean_uz2 = mean[:, 3] * inv_sample_velocity_scale2
    mean_uxuz = mean[:, 4] * inv_sample_velocity_scale2

    ux_rms = np.sqrt(np.maximum(mean_ux2 - mean_ux * mean_ux, 0.0))
    uz_rms = np.sqrt(np.maximum(mean_uz2 - mean_uz * mean_uz, 0.0))
    uxuz_cov = mean_uxuz - mean_ux * mean_uz

    return {
        "s": coordinates,
        "ux": mean_ux / u_char,
        "uz": mean_uz / u_char,
        "ux_rms": ux_rms / u_char,
        "uz_rms": uz_rms / u_char,
        "uxuz": uxuz_cov / (u_char * u_char),
        "ux_mean_raw": mean_ux,
        "uz_mean_raw": mean_uz,
        "ux_rms_raw": ux_rms,
        "uz_rms_raw": uz_rms,
        "uxuz_cov_raw": uxuz_cov,
    }


def write_profile(path: Path, profile: dict[str, np.ndarray], sample_count: int) -> None:
    columns = [
        "s",
        "ux",
        "uz",
        "ux_rms",
        "uz_rms",
        "uxuz",
        "ux_mean_raw",
        "uz_mean_raw",
        "ux_rms_raw",
        "uz_rms_raw",
        "uxuz_cov_raw",
        "samples",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(profile["s"].size):
            writer.writerow([profile[name][i] if name != "samples" else sample_count for name in columns])


def print_extracted_profile_summary(profiles: dict[str, dict[str, np.ndarray]], sample_count: int) -> None:
    print("extracted profile coordinates:", flush=True)
    for name in ("cx", "cy"):
        s = profiles[name]["s"]
        print(
            f"  centerline_{name}: s_min={float(np.min(s)):.17g}, "
            f"s_max={float(np.max(s)):.17g}, points={s.size}, samples={sample_count}",
            flush=True,
        )

    wall_nodes = (
        ("west wall", "cx", int(np.argmin(profiles["cx"]["s"]))),
        ("east wall", "cx", int(np.argmax(profiles["cx"]["s"]))),
        ("bottom wall", "cy", int(np.argmin(profiles["cy"]["s"]))),
        ("lid wall", "cy", int(np.argmax(profiles["cy"]["s"]))),
    )

    print("wall-node fluctuation statistics (nondimensional):", flush=True)
    for wall_name, profile_name, index in wall_nodes:
        profile = profiles[profile_name]
        print(
            f"  {wall_name} centerline_{profile_name} s={float(profile['s'][index]):.17g}: "
            f"ux_rms={float(profile['ux_rms'][index]):.17g}, "
            f"uz_rms={float(profile['uz_rms'][index]):.17g}, "
            f"uxuz_cov={float(profile['uxuz'][index]):.17g}",
            flush=True,
        )


def compute_metrics(s_ref: np.ndarray, q_ref: np.ndarray, q_sim: np.ndarray) -> dict[str, float]:
    err = q_sim - q_ref
    trapezoid = getattr(np, "trapezoid", np.trapz)
    ref_l2 = float(trapezoid(q_ref * q_ref, s_ref))
    err_l2 = float(trapezoid(err * err, s_ref))
    rel_l2 = math.sqrt(err_l2 / ref_l2) if ref_l2 > 0.0 else math.nan

    ref_peak_idx = int(np.argmax(q_ref))
    sim_peak_idx = int(np.argmax(q_sim))
    ref_min_idx = int(np.argmin(q_ref))
    sim_min_idx = int(np.argmin(q_sim))

    ref_integral = float(trapezoid(q_ref, s_ref))
    sim_integral = float(trapezoid(q_sim, s_ref))

    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "relative_l2": rel_l2,
        "linf": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "peak_value_error": float(np.max(q_sim) - np.max(q_ref)),
        "peak_location_error": float(s_ref[sim_peak_idx] - s_ref[ref_peak_idx]),
        "minimum_value_error": float(np.min(q_sim) - np.min(q_ref)),
        "minimum_location_error": float(s_ref[sim_min_idx] - s_ref[ref_min_idx]),
        "reference_integral": ref_integral,
        "simulation_integral": sim_integral,
        "trapezoidal_integral_mismatch": sim_integral - ref_integral,
    }


def make_plot(
    plot_path: Path,
    profile: str,
    variable: str,
    s_ref: np.ndarray,
    q_ref: np.ndarray,
    q_sim: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)

    if profile == "cy":
        ax.plot(q_ref, s_ref, "o", markersize=4, label="reference")
        ax.plot(q_sim, s_ref, "-", linewidth=1.8, label="simulation")
        ax.set_xlabel(variable)
        ax.set_ylabel("y")
    else:
        ax.plot(s_ref, q_ref, "o", markersize=4, label="reference")
        ax.plot(s_ref, q_sim, "-", linewidth=1.8, label="simulation")
        ax.set_xlabel("x")
        ax.set_ylabel(variable)

    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(frameon=False)
    ax.set_title(plot_path.stem)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    reference_dir = Path(args.reference_dir).expanduser() if args.reference_dir else repo_root / "referenceData"
    case_dir = Path(args.output_root).expanduser() / args.case_id
    profile_dir = case_dir / "profiles"
    post_dir = case_dir / "postProcessing"
    plot_dir = post_dir / "plots"
    processed_profile_dir = post_dir / "profiles"

    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference directory not found: {reference_dir}")
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"profile directory not found: {profile_dir}")

    post_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    processed_profile_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_key_value_csv(profile_dir / "metadata.csv")
    cx_length = int(metadata["cx_length"])
    cy_length = int(metadata["cy_length"])
    u_char = float(metadata["u_char"])
    sample_velocity_scale = float(metadata.get("sample_velocity_scale", "3.0"))
    dtype = dtype_from_metadata(metadata)
    target_reynolds = case_reynolds(args.case_id)

    steps = read_steps(profile_dir / "sample_steps.csv")
    sample_count = int(steps.size)
    if sample_count == 0:
        raise ValueError("sample_steps.csv contains no profile samples")

    mask, dropped_duplicate_samples = select_steps(steps, args.step_min, args.step_max)
    selected_steps = steps[mask]
    selected_sample_count = int(mask.sum())

    profiles = {
        "cx": None,
        "cy": None,
    }
    print(f"selected steps {int(selected_steps.min())}..{int(selected_steps.max())} ({selected_sample_count} samples)", flush=True)
    if dropped_duplicate_samples:
        print(f"ignored {dropped_duplicate_samples} duplicate timestep samples, keeping the last occurrence", flush=True)
    print(f"using reference Reynolds {target_reynolds}", flush=True)
    print(f"sample velocity scale to physical units: {sample_velocity_scale:g}", flush=True)
    print(f"processing centerline_cx_samples.bin in chunks of {args.chunk_samples} samples", flush=True)
    profiles["cx"] = compute_window_profile(
        profile_dir / "centerline_cx_samples.bin",
        sample_count,
        read_coordinates(profile_dir / "centerline_cx_coordinates.csv", cx_length),
        mask,
        dtype,
        u_char,
        sample_velocity_scale,
        args.chunk_samples,
    )
    print(f"processing centerline_cy_samples.bin in chunks of {args.chunk_samples} samples", flush=True)
    profiles["cy"] = compute_window_profile(
        profile_dir / "centerline_cy_samples.bin",
        sample_count,
        read_coordinates(profile_dir / "centerline_cy_coordinates.csv", cy_length),
        mask,
        dtype,
        u_char,
        sample_velocity_scale,
        args.chunk_samples,
    )

    print_extracted_profile_summary(profiles, selected_sample_count)

    write_profile(processed_profile_dir / "centerline_cx.csv", profiles["cx"], selected_sample_count)
    write_profile(processed_profile_dir / "centerline_cy.csv", profiles["cy"], selected_sample_count)

    for stale_plot in plot_dir.glob("*.png"):
        stale_plot.unlink()

    rows: list[dict[str, object]] = []
    for reference_path in sorted(reference_dir.glob("*.csv")):
        match = REFERENCE_RE.match(reference_path.name)
        if not match:
            continue

        reference_variable, profile, reynolds = match.groups()
        if reynolds != target_reynolds:
            continue

        sim_column, label, reference_plot_scale = VARIABLE_MAP[reference_variable]
        profile_data = profiles[profile]

        s_ref, q_ref = read_reference_curve(reference_path, profile)
        q_ref = q_ref / reference_plot_scale
        s_ref, q_ref = sorted_unique_average(s_ref, q_ref)
        s_sim, q_sim_profile = sorted_unique_average(profile_data["s"], profile_data[sim_column])

        if s_ref.size < 2:
            raise ValueError(f"{reference_path} must contain at least two finite points")
        if s_sim.size < 2:
            raise ValueError(f"simulation profile centerline_{profile}.csv must contain at least two finite points")
        if s_ref[0] < s_sim[0] or s_ref[-1] > s_sim[-1]:
            raise ValueError(
                f"{reference_path} reference range [{s_ref[0]}, {s_ref[-1]}] exceeds "
                f"simulation range [{s_sim[0]}, {s_sim[-1]}]"
            )

        q_sim = np.interp(s_ref, s_sim, q_sim_profile)
        metrics = compute_metrics(s_ref, q_ref, q_sim)
        plot_path = plot_dir / f"{reference_path.stem}.png"
        make_plot(plot_path, profile, label, s_ref, q_ref, q_sim)

        rows.append(
            {
                "file": reference_path.name,
                "profile": profile,
                "reference_variable": reference_variable,
                "simulation_column": sim_column,
                "reynolds": reynolds,
                "reference_plot_scale": reference_plot_scale,
                "window_step_min": int(selected_steps.min()),
                "window_step_max": int(selected_steps.max()),
                "window_samples": selected_sample_count,
                "points": s_ref.size,
                **metrics,
                "plot": str(plot_path),
            }
        )

    if not rows:
        raise RuntimeError(f"no reference CSV files matched expected LDC names in {reference_dir}")

    metrics_path = post_dir / "ldc_reference_metrics.csv"
    fieldnames = list(rows[0].keys())
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {metrics_path}")
    print(f"wrote plots to {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
