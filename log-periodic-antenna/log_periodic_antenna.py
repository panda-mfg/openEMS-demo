#!/usr/bin/env python3
"""GPU openEMS model of a balanced log-periodic dipole array (LPDA)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
OPENEMS_GPU_ROOT = SCRIPT_DIR.parent
CUSTOM_PREFIX = OPENEMS_GPU_ROOT / "local"
CUSTOM_PYTHON = CUSTOM_PREFIX / "python-packages"

# Always prefer the bindings installed from openEMS-Project over system copies.
sys.path.insert(0, str(CUSTOM_PYTHON))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import CSXCAD
import openEMS as openems_package
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0


UNIT = 1e-3  # Geometry coordinates are millimetres.
F_LOW = 0.8e9
F_HIGH = 3.0e9
F_CENTER = 0.5 * (F_LOW + F_HIGH)
F_CUTOFF = 0.5 * (F_HIGH - F_LOW)
FEED_OHMS = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "simulation",
        help="simulation/result directory",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="write the XML model without running FDTD",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="reuse an existing FDTD result",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the generated XML in AppCSXCAD",
    )
    parser.add_argument(
        "--skip-far-field",
        action="store_true",
        help="calculate only the port response",
    )
    parser.add_argument("--engine", default="gpu", choices=("gpu", "multithreaded", "basic"))
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--nr-ts", type=int, default=80_000)
    parser.add_argument("--end-criteria", type=float, default=1e-4)
    return parser.parse_args()


def require_custom_bindings() -> None:
    expected = CUSTOM_PREFIX.resolve()
    modules = {
        "openEMS": Path(openems_package.__file__).resolve(),
        "CSXCAD": Path(CSXCAD.__file__).resolve(),
    }
    outside = {name: path for name, path in modules.items() if expected not in path.parents}
    if outside:
        details = ", ".join(f"{name}={path}" for name, path in outside.items())
        raise RuntimeError(f"Refusing to run with non-custom bindings: {details}")
    print(f"Custom openEMS bindings: {modules['openEMS']}")
    print(f"Custom CSXCAD bindings: {modules['CSXCAD']}")


def lpda_dimensions() -> tuple[np.ndarray, np.ndarray]:
    """Return full element lengths and x positions in millimetres."""
    element_count = 9
    tau = 0.86
    sigma = 0.16
    longest = 0.48 * C0 / F_LOW / UNIT
    lengths = longest * tau ** np.arange(element_count)
    positions = np.zeros(element_count)
    positions[1:] = np.cumsum(sigma * lengths[:-1])
    return lengths, positions


def build_model(nr_ts: int, end_criteria: float):
    lengths, positions = lpda_dimensions()

    strip_width = 4.0
    boom_width = 4.0
    boom_separation = 4.0
    feed_x = positions[-1] + 12.0
    boom_start = -8.0

    # About lambda/4 at the low edge gives the antenna room before the PML.
    padding = 90.0
    domain = {
        "x": np.array([boom_start - padding, feed_x + padding]),
        "y": np.array([-lengths[0] / 2 - padding, lengths[0] / 2 + padding]),
        "z": np.array([-padding, padding]),
    }

    fdtd = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
    fdtd.SetGaussExcite(F_CENTER, F_CUTOFF)
    fdtd.SetBoundaryCond(["PML_8"] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    for axis in "xyz":
        mesh.AddLine(axis, domain[axis])

    # Resolve the upper/lower booms, the differential feed, and every strip edge.
    mesh.AddLine("z", [-boom_separation / 2, 0.0, boom_separation / 2])
    mesh.AddLine("x", [boom_start, feed_x - strip_width, feed_x])
    mesh.AddLine("x", np.ravel([positions - strip_width / 2, positions + strip_width / 2]))
    mesh.AddLine("y", [-boom_width / 2, 0.0, boom_width / 2])
    mesh.AddLine("y", np.ravel([-lengths / 2, lengths / 2]))

    metal = csx.AddMetal("lpda_metal")
    z_lower = -boom_separation / 2
    z_upper = boom_separation / 2

    # The two conductors form the balanced boom transmission line.
    metal.AddBox(
        priority=10,
        start=[boom_start, -boom_width / 2, z_lower],
        stop=[feed_x, boom_width / 2, z_lower],
    )
    metal.AddBox(
        priority=10,
        start=[boom_start, -boom_width / 2, z_upper],
        stop=[feed_x, boom_width / 2, z_upper],
    )

    # Adjacent split dipoles are transposed between the two boom conductors.
    # This supplies the 180-degree phase reversal used by a physical LPDA.
    for index, (length, x_pos) in enumerate(zip(lengths, positions)):
        if index % 2 == 0:
            z_negative, z_positive = z_lower, z_upper
        else:
            z_negative, z_positive = z_upper, z_lower
        metal.AddBox(
            priority=10,
            start=[x_pos - strip_width / 2, -length / 2, z_negative],
            stop=[x_pos + strip_width / 2, -boom_width / 2, z_negative],
        )
        metal.AddBox(
            priority=10,
            start=[x_pos - strip_width / 2, boom_width / 2, z_positive],
            stop=[x_pos + strip_width / 2, length / 2, z_positive],
        )

    # Add metal-edge mesh lines before smoothing the surrounding air mesh.
    max_cell = C0 / F_HIGH / UNIT / 20.0
    fdtd.AddEdges2Grid(dirs="xy", properties=metal, metal_edge_res=1.5)
    mesh.SmoothMeshLines("all", max_cell, ratio=1.4)

    port = fdtd.AddLumpedPort(
        port_nr=1,
        R=FEED_OHMS,
        start=[feed_x - strip_width, -boom_width / 2, z_lower],
        stop=[feed_x, boom_width / 2, z_upper],
        p_dir="z",
        excite=1.0,
        priority=20,
        edges2grid="xy",
    )

    nf2ff = fdtd.CreateNF2FFBox(opt_resolution=[max_cell] * 3)
    design = {
        "frequency_band_ghz": [F_LOW / 1e9, F_HIGH / 1e9],
        "feed_ohms": FEED_OHMS,
        "tau": 0.86,
        "sigma": 0.16,
        "element_lengths_mm": lengths.tolist(),
        "element_positions_mm": positions.tolist(),
        "boom_separation_mm": boom_separation,
        "mesh_max_cell_mm": max_cell,
        "mesh_cells": {axis: len(mesh.GetLines(axis)) - 1 for axis in "xyz"},
        "domain_mm": {axis: values.tolist() for axis, values in domain.items()},
    }
    return fdtd, port, nf2ff, design


def appcsxcad_path() -> Path:
    candidates = (
        CUSTOM_PREFIX / "bin" / "AppCSXCAD",
        OPENEMS_GPU_ROOT / "openEMS-Project" / "AppCSXCAD" / "build-codex" / "AppCSXCAD",
        Path("/home/shanda/opt/openEMS/bin/AppCSXCAD"),
        Path("/usr/local/bin/AppCSXCAD"),
        Path("/usr/bin/AppCSXCAD"),
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("AppCSXCAD was not found in the local openEMS installations")


def show_geometry(xml_path: Path) -> None:
    app = appcsxcad_path()
    print(f"Opening geometry with {app}")
    subprocess.Popen([str(app), str(xml_path)], start_new_session=True)


def save_port_results(output_dir: Path, port) -> tuple[np.ndarray, np.ndarray, float]:
    frequency = np.linspace(F_LOW, F_HIGH, 601)
    port.CalcPort(str(output_dir), frequency)
    s11 = np.asarray(port.uf_ref / port.uf_inc)
    impedance = np.asarray(port.uf_tot / port.if_tot)
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-12))
    best_index = int(np.nanargmin(s11_db))
    best_frequency = float(frequency[best_index])

    np.savez(
        output_dir / "port_results.npz",
        frequency_hz=frequency,
        s11=s11,
        s11_db=s11_db,
        impedance_ohm=impedance,
        accepted_power_w=np.asarray(port.P_acc),
    )

    figure, (s_axis, z_axis) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    s_axis.plot(frequency / 1e9, s11_db, color="navy", linewidth=1.8)
    s_axis.axhline(-10, color="0.4", linestyle="--", linewidth=1)
    s_axis.set_ylabel("S11 (dB)")
    s_axis.set_ylim(min(-35, float(np.nanmin(s11_db)) - 2), 1)
    s_axis.grid(True, alpha=0.3)
    z_axis.plot(frequency / 1e9, impedance.real, label="Real", linewidth=1.6)
    z_axis.plot(frequency / 1e9, impedance.imag, label="Imaginary", linewidth=1.6)
    z_axis.axhline(FEED_OHMS, color="0.4", linestyle="--", linewidth=1)
    z_axis.set_xlabel("Frequency (GHz)")
    z_axis.set_ylabel("Input impedance (ohm)")
    z_axis.grid(True, alpha=0.3)
    z_axis.legend()
    figure.suptitle("9-element log-periodic dipole array")
    figure.tight_layout()
    figure.savefig(output_dir / "s11_impedance.png", dpi=160)
    plt.close(figure)
    return frequency, s11_db, best_frequency


def save_far_field(output_dir: Path, nf2ff, port, frequency: np.ndarray, best_frequency: float):
    phi = np.arange(-180.0, 180.1, 2.0)
    result = nf2ff.CalcNF2FF(
        str(output_dir),
        best_frequency,
        90.0,
        phi,
        center=[0.0, 0.0, 0.0],
        outfile="nf2ff_azimuth.h5",
        verbose=1,
    )
    field = np.asarray(result.E_norm[0]).squeeze()
    dmax_db = float(10.0 * np.log10(result.Dmax[0]))
    normalized = np.maximum(np.abs(field) / np.nanmax(np.abs(field)), 1e-6)
    directivity_db = 20.0 * np.log10(normalized) + dmax_db
    accepted_power = float(np.interp(best_frequency, frequency, np.asarray(port.P_acc)))
    efficiency = float(result.Prad[0] / accepted_power) if accepted_power > 0 else float("nan")

    np.savez(
        output_dir / "far_field_results.npz",
        frequency_hz=best_frequency,
        phi_deg=phi,
        directivity_db=directivity_db,
        dmax_db=dmax_db,
        radiated_power_w=float(result.Prad[0]),
        radiation_efficiency=efficiency,
    )

    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(111, projection="polar")
    axis.plot(np.deg2rad(phi), directivity_db, linewidth=1.8)
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_rlim(max(-30.0, dmax_db - 30.0), max(1.0, dmax_db + 1.0))
    axis.set_title(f"Azimuth directivity at {best_frequency / 1e9:.3f} GHz")
    axis.grid(True, alpha=0.4)
    figure.tight_layout()
    figure.savefig(output_dir / "far_field_azimuth.png", dpi=160)
    plt.close(figure)
    return dmax_db, efficiency


def main() -> int:
    args = parse_args()
    require_custom_bindings()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fdtd, port, nf2ff, design = build_model(args.nr_ts, args.end_criteria)
    xml_path = output_dir / "log_periodic_antenna.xml"
    fdtd.Write2XML(str(xml_path))
    (output_dir / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    print(f"Wrote model: {xml_path}")
    cells = design["mesh_cells"]
    print(f"Mesh cells: {cells['x']} x {cells['y']} x {cells['z']}")

    if args.show:
        show_geometry(xml_path)
    if args.generate_only:
        return 0

    if not args.postprocess_only:
        run_options = {"cleanup": True, "verbose": 1, "engine": args.engine, "dump_statistics": True}
        if args.engine == "gpu":
            run_options["gpu_device"] = args.gpu_device
        print(f"Running custom openEMS engine={args.engine}, GPU device={args.gpu_device}")
        error_code = fdtd.Run(str(output_dir), **run_options)
        if error_code not in (None, 0):
            raise RuntimeError(f"openEMS setup failed with error code {error_code}")

    frequency, s11_db, best_frequency = save_port_results(output_dir, port)
    summary = {
        "engine": args.engine,
        "gpu_device": args.gpu_device if args.engine == "gpu" else None,
        "best_frequency_ghz": best_frequency / 1e9,
        "minimum_s11_db": float(np.nanmin(s11_db)),
    }
    below_10 = frequency[s11_db <= -10.0]
    if below_10.size:
        mask = s11_db <= -10.0
        starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
        stops = np.flatnonzero(mask & np.r_[~mask[1:], True])
        summary["s11_below_minus_10_db_bands_ghz"] = [
            [float(frequency[start] / 1e9), float(frequency[stop] / 1e9)]
            for start, stop in zip(starts, stops)
        ]

    if not args.skip_far_field:
        dmax_db, efficiency = save_far_field(output_dir, nf2ff, port, frequency, best_frequency)
        summary["peak_directivity_dbi"] = dmax_db
        summary["radiation_efficiency"] = efficiency

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
