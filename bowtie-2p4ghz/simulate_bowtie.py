#!/usr/bin/env python3
"""Free-space, center-fed bowtie antenna model for the custom GPU openEMS build."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "results_tuned"

TARGET_FREQUENCY = 2.4e9
FREQUENCY_MIN = 1.2e9
FREQUENCY_MAX = 3.6e9
REFERENCE_IMPEDANCE = 50.0

# Geometry dimensions in millimetres.  Each arm is a triangular PEC sheet.
ARM_LENGTH = 17.05
ARM_WIDTH = 25.575
FEED_GAP = 0.5683333333
FEED_WIDTH = 1.1366666667
BOX_SIZE = np.array([200.0, 200.0, 150.0])


def build_model(run_dir: Path):
    """Build and serialize the bowtie model; return post-processing handles."""
    unit = 1e-3
    excitation_fc = 1.2e9
    maximum_cell = C0 / FREQUENCY_MAX / unit / 30.0

    fdtd = openEMS(NrTS=60000, EndCriteria=1e-5)
    fdtd.SetGaussExcite(TARGET_FREQUENCY, excitation_fc)
    fdtd.SetBoundaryCond(["PML_8"] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)

    half_box = BOX_SIZE / 2.0
    mesh.AddLine("x", [-half_box[0], half_box[0]])
    mesh.AddLine("y", [-half_box[1], half_box[1]])
    mesh.AddLine("z", [-half_box[2], 0.0, half_box[2]])

    inner_x = FEED_GAP / 2.0
    outer_x = inner_x + ARM_LENGTH
    half_width = ARM_WIDTH / 2.0

    bowtie = csx.AddMetal("bowtie_PEC")
    left = bowtie.AddPolygon(
        [[-inner_x, -outer_x, -outer_x], [0.0, -half_width, half_width]],
        norm_dir=2,
        elevation=0.0,
        priority=10,
    )
    right = bowtie.AddPolygon(
        [[inner_x, outer_x, outer_x], [0.0, half_width, -half_width]],
        norm_dir=2,
        elevation=0.0,
        priority=10,
    )

    # Resolve triangle extrema and the small feed region explicitly.
    mesh.AddLine("x", [-outer_x, -inner_x, inner_x, outer_x])
    mesh.AddLine("y", [-half_width, -FEED_WIDTH / 2.0, 0.0,
                       FEED_WIDTH / 2.0, half_width])
    fdtd.AddEdges2Grid(
        dirs="xy", primitives=[left, right], metal_edge_res=maximum_cell / 2.0
    )

    port_start = [-inner_x, -FEED_WIDTH / 2.0, 0.0]
    port_stop = [inner_x, FEED_WIDTH / 2.0, 0.0]
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
    nf2ff = fdtd.CreateNF2FFBox()

    run_dir.mkdir(parents=True, exist_ok=True)
    xml_path = run_dir / "bowtie_2p4ghz.xml"
    fdtd.Write2XML(str(xml_path))

    mesh_counts = {axis: int(len(mesh.GetLines(axis))) for axis in "xyz"}
    model_info = {
        "target_frequency_hz": TARGET_FREQUENCY,
        "frequency_sweep_hz": [FREQUENCY_MIN, FREQUENCY_MAX],
        "reference_impedance_ohm": REFERENCE_IMPEDANCE,
        "geometry_mm": {
            "arm_length": ARM_LENGTH,
            "arm_width": ARM_WIDTH,
            "feed_gap": FEED_GAP,
            "feed_width": FEED_WIDTH,
            "simulation_box": BOX_SIZE.tolist(),
        },
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
    return fdtd, port, nf2ff, xml_path, model_info


def clean_solver_outputs(run_dir: Path):
    """Remove only known generated solver outputs before an intentional rerun."""
    exact_names = {
        "et",
        "ht",
        "port_ut_1",
        "port_it_1",
        "openEMS_run_stats.txt",
        "openEMS_stats.txt",
        "solver.log",
        "nf2ff.h5",
        "nf2ff_2p4ghz.h5",
    }
    for name in exact_names:
        path = run_dir / name
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
    """Return sampled contiguous frequency bands at or below a threshold."""
    indices = np.flatnonzero(values_db <= threshold)
    if indices.size == 0:
        return []
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return [[float(freq[g[0]]), float(freq[g[-1]])] for g in groups]


def postprocess(run_dir: Path, port, nf2ff, include_farfield: bool):
    required = [run_dir / "port_ut_1", run_dir / "port_it_1"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing solver result(s): " + ", ".join(missing))

    freq = np.linspace(FREQUENCY_MIN, FREQUENCY_MAX, 1201)
    port.CalcPort(str(run_dir), freq)
    s11 = port.uf_ref / port.uf_inc
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-15))
    zin = port.uf_tot / port.if_tot

    resonance_index = int(np.argmin(s11_db))
    resonance_frequency = float(freq[resonance_index])
    target_s11 = interpolated_complex(TARGET_FREQUENCY, freq, s11)
    target_zin = interpolated_complex(TARGET_FREQUENCY, freq, zin)
    bands = threshold_bands(freq, s11_db, -10.0)
    target_band = next(
        (band for band in bands if band[0] <= TARGET_FREQUENCY <= band[1]), None
    )

    with (run_dir / "frequency_response.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frequency_hz", "s11_db", "zin_real_ohm", "zin_imag_ohm"]
        )
        writer.writerows(zip(freq, s11_db, zin.real, zin.imag))

    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.plot(freq / 1e9, s11_db, color="navy", linewidth=1.8)
    axis.axhline(-10.0, color="0.4", linestyle="--", linewidth=1.0)
    axis.axvline(TARGET_FREQUENCY / 1e9, color="crimson", linestyle=":")
    axis.set(xlabel="Frequency (GHz)", ylabel="$S_{11}$ (dB)",
             title="2.4 GHz free-space bowtie: input match")
    axis.grid(True, alpha=0.3)
    fig.savefig(run_dir / "s11.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.plot(freq / 1e9, zin.real, label="Re($Z_{in}$)")
    axis.plot(freq / 1e9, zin.imag, label="Im($Z_{in}$)")
    axis.axvline(TARGET_FREQUENCY / 1e9, color="crimson", linestyle=":")
    axis.set(xlabel="Frequency (GHz)", ylabel="Impedance (ohm)",
             title="Bowtie feed-point impedance")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(run_dir / "impedance.png", dpi=180)
    plt.close(fig)

    summary = {
        "resonance_frequency_hz": resonance_frequency,
        "minimum_s11_db": float(s11_db[resonance_index]),
        "target_frequency_hz": TARGET_FREQUENCY,
        "target_s11_db": float(20.0 * np.log10(max(abs(target_s11), 1e-15))),
        "target_zin_ohm": {"real": target_zin.real, "imag": target_zin.imag},
        "target_vswr": float((1.0 + abs(target_s11)) / (1.0 - abs(target_s11))),
        "minus_10db_bands_hz": bands,
        "target_minus_10db_band_hz": target_band,
        "target_minus_10db_bandwidth_hz": (
            float(target_band[1] - target_band[0]) if target_band else None
        ),
    }

    if include_farfield:
        theta = np.arange(0.0, 181.0, 2.0)
        phi = np.arange(0.0, 360.0, 5.0)
        print("Calculating 3D far field at 2.4 GHz...", flush=True)
        farfield = nf2ff.CalcNF2FF(
            str(run_dir),
            TARGET_FREQUENCY,
            theta,
            phi,
            outfile="nf2ff_2p4ghz.h5",
            read_cached=False,
            verbose=True,
        )
        directivity_db = float(10.0 * np.log10(farfield.Dmax[0]))
        accepted_power = float(np.interp(TARGET_FREQUENCY, freq, port.P_acc))
        radiated_power = float(farfield.Prad[0])
        radiation_efficiency_raw = radiated_power / accepted_power
        # A small NF2FF/port power mismatch is numerical integration error.
        # Bound the physical efficiency used for realized gain, while retaining
        # the raw ratio as a power-closure diagnostic.
        radiation_efficiency = min(max(radiation_efficiency_raw, 0.0), 1.0)
        mismatch_efficiency = 1.0 - abs(target_s11) ** 2
        realized_gain_db = directivity_db + 10.0 * np.log10(
            max(radiation_efficiency * mismatch_efficiency, 1e-15)
        )
        summary["farfield_2p4ghz"] = {
            "directivity_dbi": directivity_db,
            "realized_gain_dbi": float(realized_gain_db),
            "radiated_power_w": radiated_power,
            "accepted_power_w": accepted_power,
            "radiation_efficiency": float(radiation_efficiency),
            "raw_prad_over_paccepted": float(radiation_efficiency_raw),
            "power_closure_error_percent": float(
                100.0 * (radiation_efficiency_raw - 1.0)
            ),
        }

        e_db = (
            20.0
            * np.log10(
                np.maximum(farfield.E_norm[0], 1e-30)
                / np.max(farfield.E_norm[0])
            )
            + directivity_db
        )
        fig = plt.figure(figsize=(7.0, 5.2), constrained_layout=True)
        axis = fig.add_subplot(111, projection="polar")
        for wanted_phi, label in [(0.0, "x-z plane"), (90.0, "y-z plane")]:
            index = int(np.argmin(np.abs(phi - wanted_phi)))
            closed_theta = np.deg2rad(np.r_[theta, 360.0 - theta[-2:0:-1]])
            closed_gain = np.r_[e_db[:, index], e_db[-2:0:-1, index]]
            axis.plot(closed_theta, closed_gain, label=label)
        axis.set_rlim(directivity_db - 35.0, directivity_db)
        axis.set_title("2.4 GHz directivity cuts (dBi)")
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        fig.savefig(run_dir / "radiation_pattern.png", dpi=180)
        plt.close(fig)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--gpu-device", type=int, default=0)
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
    parser.add_argument("--no-farfield", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.generate_only and args.post_only:
        raise SystemExit("--generate-only and --post-only are mutually exclusive")
    run_dir = args.run_dir.resolve()
    _, port, nf2ff, xml_path, model_info = build_model(run_dir)
    print(json.dumps(model_info, indent=2))
    if args.generate_only:
        print(f"Generated model only: {xml_path}")
        return
    if not args.post_only:
        run_solver(xml_path, run_dir, args.gpu_device)
    postprocess(run_dir, port, nf2ff, include_farfield=not args.no_farfield)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
