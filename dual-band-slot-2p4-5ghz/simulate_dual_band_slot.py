#!/usr/bin/env python3
"""Finite-ground forked quarter-wave dual-band slot for 2.4/5 GHz on GPU openEMS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0


SOLVER = Path("/home/shanda/openEMS-gpu/local/bin/openEMS")
DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "results"

TARGET_FREQUENCIES = np.array([2.4e9, 5.0e9])
FREQUENCY_MIN = 1.5e9
FREQUENCY_MAX = 6.2e9
REFERENCE_IMPEDANCE = 430.0
BOX_SIZE = np.array([220.0, 200.0, 150.0])


def build_model(run_dir: Path, args):
    """Build and serialize the finite-ground forked quarter-wave dual-band slot model."""
    unit = 1e-3
    excitation_f0 = (FREQUENCY_MIN + FREQUENCY_MAX) / 2.0
    excitation_fc = (FREQUENCY_MAX - FREQUENCY_MIN) / 2.0
    maximum_cell = C0 / FREQUENCY_MAX / unit / 30.0

    fdtd = openEMS(NrTS=100000, EndCriteria=1e-5)
    fdtd.SetGaussExcite(excitation_f0, excitation_fc)
    fdtd.SetBoundaryCond(["PML_8"] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)

    half_box = BOX_SIZE / 2.0
    mesh.AddLine("x", [-half_box[0], half_box[0]])
    mesh.AddLine("y", [-half_box[1], half_box[1]])
    mesh.AddLine("z", [-half_box[2], 0.0, half_box[2]])

    half_ground_x = args.ground_x / 2.0
    half_ground_y = args.ground_y / 2.0
    half_separation = args.slot_separation / 2.0
    half_slot_width = args.slot_width / 2.0
    half_connector = args.connector_width / 2.0

    aperture_rectangles = [
        (
            half_connector,
            half_connector + args.low_slot_length,
            half_separation - half_slot_width,
            half_separation + half_slot_width,
        ),
        (
            -half_connector - args.high_slot_length,
            -half_connector,
            -half_separation - half_slot_width,
            -half_separation + half_slot_width,
        ),
        (
            -half_connector,
            half_connector,
            -half_separation - half_slot_width,
            half_separation + half_slot_width,
        ),
    ]

    x_edges = sorted(
        set(
            [-half_ground_x, half_ground_x]
            + [value for rect in aperture_rectangles for value in rect[:2]]
        )
    )
    y_edges = sorted(
        set(
            [-half_ground_y, half_ground_y]
            + [value for rect in aperture_rectangles for value in rect[2:]]
            + [-args.port_length / 2.0, args.port_length / 2.0]
        )
    )
    mesh.AddLine("x", x_edges)
    mesh.AddLine("y", y_edges)

    def inside_aperture(x_value, y_value):
        return any(
            xmin < x_value < xmax and ymin < y_value < ymax
            for xmin, xmax, ymin, ymax in aperture_rectangles
        )

    ground = csx.AddMetal("finite_PEC_ground_with_forked_slot")
    ground_tiles = 0
    for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
        for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
            midpoint_x = (x0 + x1) / 2.0
            midpoint_y = (y0 + y1) / 2.0
            if inside_aperture(midpoint_x, midpoint_y):
                continue
            ground.AddBox([x0, y0, 0.0], [x1, y1, 0.0], priority=10)
            ground_tiles += 1

    port_start = [-half_connector, -args.port_length / 2.0, 0.0]
    port_stop = [half_connector, args.port_length / 2.0, 0.0]
    port = fdtd.AddLumpedPort(
        1,
        REFERENCE_IMPEDANCE,
        port_start,
        port_stop,
        "x",
        excite=1.0,
        priority=20,
        edges2grid="xy",
    )

    mesh.SmoothMeshLines("all", maximum_cell, 1.4)
    nf2ff = None if args.tuning_only else fdtd.CreateNF2FFBox()

    run_dir.mkdir(parents=True, exist_ok=True)
    xml_path = run_dir / "dual_band_slot.xml"
    fdtd.Write2XML(str(xml_path))

    mesh_counts = {axis: int(len(mesh.GetLines(axis))) for axis in "xyz"}
    model_info = {
        "topology": "finite-ground forked quarter-wave dual-band slot",
        "target_frequencies_hz": TARGET_FREQUENCIES.tolist(),
        "frequency_sweep_hz": [FREQUENCY_MIN, FREQUENCY_MAX],
        "reference_impedance_ohm": REFERENCE_IMPEDANCE,
        "geometry_mm": {
            "low_band_slot_length": args.low_slot_length,
            "high_band_slot_length": args.high_slot_length,
            "slot_width": args.slot_width,
            "slot_center_separation": args.slot_separation,
            "connector_width": args.connector_width,
            "port_length": args.port_length,
            "ground_size": [args.ground_x, args.ground_y],
            "simulation_box": BOX_SIZE.tolist(),
        },
        "ground_tile_count": ground_tiles,
        "tuning_only": args.tuning_only,
        "mesh_lines": mesh_counts,
        "mesh_cells": int(np.prod([mesh_counts[a] - 1 for a in "xyz"])),
        "maximum_cell_mm": float(maximum_cell),
        "boundary_conditions": ["PML_8"] * 6,
        "xml_file": str(xml_path),
        "solver": str(SOLVER),
    }
    (run_dir / "model_info.json").write_text(
        json.dumps(model_info, indent=2) + "\n", encoding="utf-8"
    )
    return port, nf2ff, xml_path, model_info


def clean_solver_outputs(run_dir: Path):
    """Remove only known generated outputs inside one intentional run directory."""
    exact_names = {
        "et",
        "ht",
        "port_ut_1",
        "port_it_1",
        "openEMS_run_stats.txt",
        "openEMS_stats.txt",
        "solver.log",
        "nf2ff_2p4ghz.h5",
        "nf2ff_5ghz.h5",
    }
    for name in exact_names:
        path = run_dir / name
        if path.is_file():
            path.unlink()
    for pattern in ("nf2ff_E_*.h5", "nf2ff_H_*.h5"):
        for path in run_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def run_solver(xml_path: Path, run_dir: Path, gpu_device: int):
    if not SOLVER.is_file():
        raise FileNotFoundError(f"Custom openEMS solver not found: {SOLVER}")
    clean_solver_outputs(run_dir)
    command = [
        str(SOLVER),
        str(xml_path),
        "--engine=gpu",
        f"--gpu-device={gpu_device}",
        "--gpu-kernel=auto",
        "--dump-statistics",
        "--verbose=1",
    ]
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=run_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (run_dir / "solver.log").write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f"openEMS exited with status {completed.returncode}; "
            f"see {run_dir / 'solver.log'}"
        )


def interpolated_complex(x: float, xp: np.ndarray, values: np.ndarray) -> complex:
    return complex(np.interp(x, xp, values.real), np.interp(x, xp, values.imag))


def threshold_bands(freq: np.ndarray, values_db: np.ndarray, threshold: float):
    indices = np.flatnonzero(values_db <= threshold)
    if indices.size == 0:
        return []
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return [[float(freq[g[0]]), float(freq[g[-1]])] for g in groups]


def target_metrics(target, window, freq, s11, s11_db, zin, bands):
    mask = (freq >= window[0]) & (freq <= window[1])
    local_indices = np.flatnonzero(mask)
    resonance_index = int(local_indices[np.argmin(s11_db[mask])])
    target_s11 = interpolated_complex(target, freq, s11)
    target_zin = interpolated_complex(target, freq, zin)
    target_band = next(
        (band for band in bands if band[0] <= target <= band[1]), None
    )
    return {
        "target_frequency_hz": float(target),
        "resonance_frequency_hz": float(freq[resonance_index]),
        "minimum_s11_db": float(s11_db[resonance_index]),
        "target_s11_db": float(
            20.0 * np.log10(max(abs(target_s11), 1e-15))
        ),
        "target_zin_ohm": {
            "real": target_zin.real,
            "imag": target_zin.imag,
        },
        "target_vswr": float(
            (1.0 + abs(target_s11)) / (1.0 - abs(target_s11))
        ),
        "minus_10db_band_hz": target_band,
        "minus_10db_bandwidth_hz": (
            float(target_band[1] - target_band[0]) if target_band else None
        ),
        "fractional_bandwidth_percent": (
            float(100.0 * (target_band[1] - target_band[0]) / target)
            if target_band else None
        ),
    }


def postprocess(run_dir: Path, port, nf2ff, args):
    required = [run_dir / "port_ut_1", run_dir / "port_it_1"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing solver result(s): " + ", ".join(missing))

    freq = np.unique(np.r_[
        np.linspace(FREQUENCY_MIN, FREQUENCY_MAX, 2351),
        TARGET_FREQUENCIES,
    ])
    port.CalcPort(str(run_dir), freq)
    s11 = port.uf_ref / port.uf_inc
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-15))
    zin = port.uf_tot / port.if_tot
    bands = threshold_bands(freq, s11_db, -10.0)

    windows = [(1.7e9, 3.4e9), (3.7e9, 6.1e9)]
    keys = ["2p4ghz", "5ghz"]
    metrics = {
        key: target_metrics(target, window, freq, s11, s11_db, zin, bands)
        for key, target, window in zip(keys, TARGET_FREQUENCIES, windows)
    }

    with (run_dir / "frequency_response.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frequency_hz", "s11_db", "zin_real_ohm", "zin_imag_ohm"]
        )
        writer.writerows(zip(freq, s11_db, zin.real, zin.imag))

    fig, axis = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    axis.plot(freq / 1e9, s11_db, color="navy", linewidth=1.8)
    axis.axhline(-10.0, color="0.4", linestyle="--", linewidth=1.0)
    for target, color in zip(TARGET_FREQUENCIES, ("crimson", "darkorange")):
        axis.axvline(target / 1e9, color=color, linestyle=":")
    axis.set(
        xlabel="Frequency (GHz)",
        ylabel="$S_{11}$ (dB)",
        title="Forked quarter-wave dual-band slot: input match",
    )
    axis.set_xlim(FREQUENCY_MIN / 1e9, FREQUENCY_MAX / 1e9)
    axis.grid(True, alpha=0.3)
    fig.savefig(run_dir / "s11.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    axis.plot(freq / 1e9, zin.real, label="Re($Z_{in}$)")
    axis.plot(freq / 1e9, zin.imag, label="Im($Z_{in}$)")
    for target in TARGET_FREQUENCIES:
        axis.axvline(target / 1e9, color="0.4", linestyle=":")
    axis.set(
        xlabel="Frequency (GHz)",
        ylabel="Impedance (ohm)",
        title="Dual-band bowtie feed-point impedance",
    )
    axis.set_xlim(FREQUENCY_MIN / 1e9, FREQUENCY_MAX / 1e9)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(run_dir / "impedance.png", dpi=180)
    plt.close(fig)

    summary = {
        "all_minus_10db_bands_hz": bands,
        "targets": metrics,
    }

    include_farfield = nf2ff is not None and not args.skip_farfield
    if include_farfield:
        theta = np.arange(0.0, 181.0, 2.0)
        phi = np.arange(0.0, 360.0, 5.0)
        branch_phis = [0.0, 0.0]
        for key, target, branch_phi in zip(
            keys, TARGET_FREQUENCIES, branch_phis
        ):
            print(f"Calculating 3D far field at {target / 1e9:g} GHz...", flush=True)
            farfield = nf2ff.CalcNF2FF(
                str(run_dir),
                target,
                theta,
                phi,
                outfile=f"nf2ff_{key}.h5",
                read_cached=False,
                verbose=True,
            )
            directivity_db = float(10.0 * np.log10(farfield.Dmax[0]))
            accepted_power = float(np.interp(target, freq, port.P_acc))
            radiated_power = float(farfield.Prad[0])
            efficiency_raw = radiated_power / accepted_power
            efficiency = min(max(efficiency_raw, 0.0), 1.0)
            target_s11 = interpolated_complex(target, freq, s11)
            mismatch_efficiency = 1.0 - abs(target_s11) ** 2
            realized_gain_db = directivity_db + 10.0 * np.log10(
                max(efficiency * mismatch_efficiency, 1e-15)
            )

            e_norm = farfield.E_norm[0]
            peak_index = np.unravel_index(np.argmax(e_norm), e_norm.shape)
            metrics[key]["farfield"] = {
                "directivity_dbi": directivity_db,
                "realized_gain_dbi": float(realized_gain_db),
                "peak_theta_deg": float(theta[peak_index[0]]),
                "peak_phi_deg": float(phi[peak_index[1]]),
                "radiated_power_w": radiated_power,
                "accepted_power_w": accepted_power,
                "radiation_efficiency": float(efficiency),
                "raw_prad_over_paccepted": float(efficiency_raw),
                "power_closure_error_percent": float(
                    100.0 * (efficiency_raw - 1.0)
                ),
            }

            e_db = (
                20.0
                * np.log10(np.maximum(e_norm, 1e-30) / np.max(e_norm))
                + directivity_db
            )
            fig = plt.figure(figsize=(7.0, 5.2), constrained_layout=True)
            axis = fig.add_subplot(111, projection="polar")
            wanted_planes = [
                (branch_phi, "slot-axis plane"),
                ((branch_phi + 90.0) % 360.0, "transverse plane"),
            ]
            for wanted_phi, label in wanted_planes:
                phi_index = int(np.argmin(np.abs(phi - wanted_phi)))
                closed_theta = np.deg2rad(
                    np.r_[theta, 360.0 - theta[-2:0:-1]]
                )
                closed_gain = np.r_[
                    e_db[:, phi_index], e_db[-2:0:-1, phi_index]
                ]
                axis.plot(closed_theta, closed_gain, label=label)
            axis.set_rlim(directivity_db - 35.0, directivity_db)
            axis.set_title(f"{target / 1e9:g} GHz directivity cuts (dBi)")
            axis.legend(
                loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2
            )
            fig.savefig(run_dir / f"radiation_pattern_{key}.png", dpi=180)
            plt.close(fig)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--low-slot-length", type=float, default=33.48)
    parser.add_argument("--high-slot-length", type=float, default=14.71)
    parser.add_argument("--slot-width", type=float, default=1.8)
    parser.add_argument("--slot-separation", type=float, default=12.0)
    parser.add_argument("--connector-width", type=float, default=1.2)
    parser.add_argument("--port-length", type=float, default=1.0)
    parser.add_argument("--ground-x", type=float, default=120.0)
    parser.add_argument("--ground-y", type=float, default=100.0)
    parser.add_argument(
        "--tuning-only",
        action="store_true",
        help="omit NF2FF surfaces for fast port-only tuning runs",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="write XML/model metadata but do not execute openEMS",
    )
    parser.add_argument(
        "--post-only",
        action="store_true",
        help="reuse existing solver outputs and only post-process",
    )
    parser.add_argument("--skip-farfield", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.generate_only and args.post_only:
        raise SystemExit("--generate-only and --post-only are mutually exclusive")
    run_dir = args.run_dir.resolve()
    port, nf2ff, xml_path, model_info = build_model(run_dir, args)
    print(json.dumps(model_info, indent=2))
    if args.generate_only:
        print(f"Generated model only: {xml_path}")
        return
    if not args.post_only:
        run_solver(xml_path, run_dir, args.gpu_device)
    postprocess(run_dir, port, nf2ff, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
