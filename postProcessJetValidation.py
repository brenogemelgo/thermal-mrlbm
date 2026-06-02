#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELDS = (
    "mean_ux",
    "mean_uy",
    "mean_uz",
    "mean_T",
    "mean_ux2",
    "mean_uy2",
    "mean_uz2",
    "mean_T2",
    "mean_uzT",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process thermal MR-LBM jet validation statistics.")
    parser.add_argument("statistics_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--radial-bins", type=int, default=128)
    parser.add_argument("--center-radius", type=float, default=1.0)
    parser.add_argument("--stations", type=str, default=None, help="Comma-separated z/D stations.")
    parser.add_argument("--overlay-papanicolaou", action="store_true")
    return parser.parse_args()


def read_metadata(statistics_dir):
    metadata_path = statistics_dir / "statistics_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def open_field(statistics_dir, name, shape):
    path = statistics_dir / f"{name}.bin"
    expected_bytes = int(np.prod(shape)) * np.dtype("<f8").itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path} has {actual_bytes} bytes, expected {expected_bytes}")
    return np.memmap(path, dtype="<f8", mode="r", shape=shape)


def safe_ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=np.abs(denominator) > 1.0e-30)
    return result


def bin_mean(values, bins, counts, radial_bins):
    sums = np.bincount(bins, weights=values, minlength=radial_bins).astype(np.float64)
    result = np.full(radial_bins, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def interpolation_width(r, profile, center_value):
    if not np.isfinite(center_value) or center_value <= 0.0:
        return np.nan

    target = center_value / math.e
    previous_r = r[0]
    previous_value = profile[0]

    if np.isfinite(previous_value) and previous_value <= target:
        return previous_r

    for i in range(1, len(r)):
        value = profile[i]
        if not np.isfinite(value) or not np.isfinite(previous_value):
            previous_r = r[i]
            previous_value = value
            continue
        if previous_value > target and value <= target:
            denominator = value - previous_value
            if abs(denominator) < 1.0e-30:
                return r[i]
            fraction = (target - previous_value) / denominator
            return previous_r + fraction * (r[i] - previous_r)
        previous_r = r[i]
        previous_value = value

    return np.nan


def parse_stations(stations_arg, z_over_d):
    max_station = float(np.nanmax(z_over_d))
    if stations_arg:
        stations = [float(value.strip()) for value in stations_arg.split(",") if value.strip()]
    else:
        candidates = [5.0, 10.0, 15.0, 20.0, 25.0]
        stations = [station for station in candidates if station <= max_station]
        if not stations:
            stations = [0.25 * max_station, 0.5 * max_station, 0.75 * max_station]

    return [station for station in stations if 0.0 <= station <= max_station]


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def compute_profiles(fields, metadata, radial_bins, center_radius):
    nx = int(metadata["NX"])
    ny = int(metadata["NY"])
    nz = int(metadata["NZ"])
    diam = float(metadata["DIAM"])
    t_ref = float(metadata["T_REF"])

    y, x = np.indices((ny, nx), dtype=np.float64)
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    flat_radius = radius.ravel()

    r_edges = np.linspace(0.0, float(np.nanmax(flat_radius)), radial_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    bins = np.searchsorted(r_edges, flat_radius, side="right") - 1
    bins = np.clip(bins, 0, radial_bins - 1)
    counts = np.bincount(bins, minlength=radial_bins).astype(np.float64)

    center_mask = (flat_radius <= center_radius)
    if not np.any(center_mask):
        raise ValueError("center-radius selects no cells")

    radial_w = np.empty((nz, radial_bins), dtype=np.float64)
    radial_t = np.empty((nz, radial_bins), dtype=np.float64)
    radial_ux_rms = np.empty((nz, radial_bins), dtype=np.float64)
    radial_uy_rms = np.empty((nz, radial_bins), dtype=np.float64)
    radial_uz_rms = np.empty((nz, radial_bins), dtype=np.float64)
    radial_t_rms = np.empty((nz, radial_bins), dtype=np.float64)

    center_w = np.empty(nz, dtype=np.float64)
    center_t = np.empty(nz, dtype=np.float64)
    center_ux_rms = np.empty(nz, dtype=np.float64)
    center_uy_rms = np.empty(nz, dtype=np.float64)
    center_uz_rms = np.empty(nz, dtype=np.float64)
    center_t_rms = np.empty(nz, dtype=np.float64)

    for z in range(nz):
        ux = np.asarray(fields["mean_ux"][z], dtype=np.float64).ravel()
        uy = np.asarray(fields["mean_uy"][z], dtype=np.float64).ravel()
        uz = np.asarray(fields["mean_uz"][z], dtype=np.float64).ravel()
        temperature = np.asarray(fields["mean_T"][z], dtype=np.float64).ravel()

        ux_rms = np.sqrt(np.maximum(np.asarray(fields["mean_ux2"][z]).ravel() - ux * ux, 0.0))
        uy_rms = np.sqrt(np.maximum(np.asarray(fields["mean_uy2"][z]).ravel() - uy * uy, 0.0))
        uz_rms = np.sqrt(np.maximum(np.asarray(fields["mean_uz2"][z]).ravel() - uz * uz, 0.0))
        t_rms = np.sqrt(np.maximum(np.asarray(fields["mean_T2"][z]).ravel() - temperature * temperature, 0.0))
        t_excess = temperature - t_ref

        radial_w[z] = bin_mean(uz, bins, counts, radial_bins)
        radial_t[z] = bin_mean(t_excess, bins, counts, radial_bins)
        radial_ux_rms[z] = bin_mean(ux_rms, bins, counts, radial_bins)
        radial_uy_rms[z] = bin_mean(uy_rms, bins, counts, radial_bins)
        radial_uz_rms[z] = bin_mean(uz_rms, bins, counts, radial_bins)
        radial_t_rms[z] = bin_mean(t_rms, bins, counts, radial_bins)

        center_w[z] = np.mean(uz[center_mask])
        center_t[z] = np.mean(t_excess[center_mask])
        center_ux_rms[z] = np.mean(ux_rms[center_mask])
        center_uy_rms[z] = np.mean(uy_rms[center_mask])
        center_uz_rms[z] = np.mean(uz_rms[center_mask])
        center_t_rms[z] = np.mean(t_rms[center_mask])

    z = np.arange(nz, dtype=np.float64)
    z_over_d = z / diam
    b_w = np.array([interpolation_width(r_centers, radial_w[i], center_w[i]) for i in range(nz)])
    b_t = np.array([interpolation_width(r_centers, radial_t[i], center_t[i]) for i in range(nz)])

    return {
        "r": r_centers,
        "z": z,
        "z_over_d": z_over_d,
        "radial_w": radial_w,
        "radial_t": radial_t,
        "radial_ux_rms": radial_ux_rms,
        "radial_uy_rms": radial_uy_rms,
        "radial_uz_rms": radial_uz_rms,
        "radial_t_rms": radial_t_rms,
        "center_w": center_w,
        "center_t": center_t,
        "center_ux_rms": center_ux_rms,
        "center_uy_rms": center_uy_rms,
        "center_uz_rms": center_uz_rms,
        "center_t_rms": center_t_rms,
        "b_w": b_w,
        "b_t": b_t,
    }


def write_outputs(data, metadata, output_dir, stations, overlay_papanicolaou):
    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    diam = float(metadata["DIAM"])
    u_char = float(metadata["U_CHAR"])
    delta_t = float(metadata["T_HOT"]) - float(metadata["T_REF"])
    z_over_d = data["z_over_d"]

    write_csv(
        csv_dir / "centerline_validation.csv",
        ["z", "z_over_D", "w_c", "T_excess_c", "U_CHAR_over_w_c", "DeltaT_over_T_c"],
        zip(
            data["z"],
            z_over_d,
            data["center_w"],
            data["center_t"],
            safe_ratio(u_char, data["center_w"]),
            safe_ratio(delta_t, data["center_t"]),
        ),
    )

    write_csv(
        csv_dir / "width_growth.csv",
        ["z", "z_over_D", "b_w", "b_T", "b_w_over_D", "b_T_over_D"],
        zip(data["z"], z_over_d, data["b_w"], data["b_t"], data["b_w"] / diam, data["b_t"] / diam),
    )

    write_csv(
        csv_dir / "rms_centerline.csv",
        ["z", "z_over_D", "ux_rms_c", "uy_rms_c", "uz_rms_c", "T_rms_c", "uz_rms_over_w_c", "T_rms_over_T_c"],
        zip(
            data["z"],
            z_over_d,
            data["center_ux_rms"],
            data["center_uy_rms"],
            data["center_uz_rms"],
            data["center_t_rms"],
            safe_ratio(data["center_uz_rms"], data["center_w"]),
            safe_ratio(data["center_t_rms"], data["center_t"]),
        ),
    )

    radial_rows = []
    for iz, z_value in enumerate(data["z"]):
        for ir, r_value in enumerate(data["r"]):
            radial_rows.append(
                (
                    z_value,
                    z_over_d[iz],
                    r_value,
                    r_value / diam,
                    data["radial_w"][iz, ir],
                    data["radial_t"][iz, ir],
                    data["radial_ux_rms"][iz, ir],
                    data["radial_uy_rms"][iz, ir],
                    data["radial_uz_rms"][iz, ir],
                    data["radial_t_rms"][iz, ir],
                )
            )

    write_csv(
        csv_dir / "radial_profiles.csv",
        ["z", "z_over_D", "r", "r_over_D", "w", "T_excess", "ux_rms", "uy_rms", "uz_rms", "T_rms"],
        radial_rows,
    )

    reference_z = np.linspace(float(np.nanmin(z_over_d)), float(np.nanmax(z_over_d)), 512)

    plt.figure()
    plt.plot(z_over_d, safe_ratio(u_char, data["center_w"]), label="U_CHAR / w_c")
    plt.plot(z_over_d, safe_ratio(delta_t, data["center_t"]), label="(T_HOT - T_REF) / T_c")
    if overlay_papanicolaou:
        plt.plot(reference_z, 0.149 * (reference_z + 2.56), "k--", label="Papanicolaou velocity")
        plt.plot(reference_z, 0.157 * (reference_z + 4.35), "k:", label="Papanicolaou temperature")
    plt.xlabel("z/D")
    plt.ylabel("decay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "centerline_decay.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(z_over_d, data["b_w"] / diam, label="b_w / D")
    plt.plot(z_over_d, data["b_t"] / diam, label="b_T / D")
    if overlay_papanicolaou:
        plt.plot(reference_z, 0.104 * (reference_z + 2.58), "k--", label="Papanicolaou velocity")
        plt.plot(reference_z, 0.139 * (reference_z - 2.51), "k:", label="Papanicolaou temperature")
    plt.xlabel("z/D")
    plt.ylabel("width")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "width_growth.png", dpi=200)
    plt.close()

    station_indices = [int(np.argmin(np.abs(z_over_d - station))) for station in stations]

    plt.figure()
    plotted = False
    for iz in station_indices:
        if np.isfinite(data["b_w"][iz]) and data["b_w"][iz] > 0.0 and abs(data["center_w"][iz]) > 1.0e-30:
            plt.plot(data["r"] / data["b_w"][iz], data["radial_w"][iz] / data["center_w"][iz], label=f"z/D={z_over_d[iz]:.2f}")
            plotted = True
    plt.xlabel("r / b_w")
    plt.ylabel("w / w_c")
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "profile_collapse_velocity.png", dpi=200)
    plt.close()

    plt.figure()
    plotted = False
    for iz in station_indices:
        if np.isfinite(data["b_t"][iz]) and data["b_t"][iz] > 0.0 and abs(data["center_t"][iz]) > 1.0e-30:
            plt.plot(data["r"] / data["b_t"][iz], data["radial_t"][iz] / data["center_t"][iz], label=f"z/D={z_over_d[iz]:.2f}")
            plotted = True
    plt.xlabel("r / b_T")
    plt.ylabel("T_excess / T_c")
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "profile_collapse_temperature.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(z_over_d, safe_ratio(data["center_uz_rms"], data["center_w"]), label="uz_rms,c / w_c")
    plt.plot(z_over_d, safe_ratio(data["center_t_rms"], data["center_t"]), label="T_rms,c / T_c")
    plt.xlabel("z/D")
    plt.ylabel("centerline RMS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "centerline_rms.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    statistics_dir = args.statistics_dir
    output_dir = args.output if args.output is not None else statistics_dir

    metadata = read_metadata(statistics_dir)
    shape = (int(metadata["NZ"]), int(metadata["NY"]), int(metadata["NX"]))
    fields = {name: open_field(statistics_dir, name, shape) for name in FIELDS}

    data = compute_profiles(fields, metadata, args.radial_bins, args.center_radius)
    stations = parse_stations(args.stations, data["z_over_d"])
    write_outputs(data, metadata, output_dir, stations, args.overlay_papanicolaou)

    print(f"statistics: {statistics_dir}")
    print(f"plots: {output_dir / 'plots'}")
    print(f"csv: {output_dir / 'csv'}")


if __name__ == "__main__":
    main()
