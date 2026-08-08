#!/usr/bin/env python3
"""GPU openEMS model of a 5 GHz planar inverted-F antenna (PIFA) on PTFE."""

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
sys.path.insert(0, str(CUSTOM_PREFIX / "python-packages"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import CSXCAD
import openEMS as openems_package
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0


UNIT = 1e-3  # Geometry coordinates are millimetres.
TARGET_FREQUENCY = 5.0e9
SWEEP_LOW = 4.5e9
SWEEP_HIGH = 5.5e9
EXCITATION_CUTOFF = 0.8e9
FEED_OHMS = 50.0
PTFE_EPSILON = 2.1
PTFE_LOSS_TANGENT = 2.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "simulation",
        help="simulation/result directory",
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--skip-far-field", action="store_true")
    parser.add_argument("--engine", default="gpu", choices=("gpu", "multithreaded", "basic"))
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--nr-ts", type=int, default=80_000)
    parser.add_argument("--end-criteria", type=float, default=1e-4)
    parser.add_argument("--plate-length", type=float, default=8.0, help="top plate length in mm")
    parser.add_argument("--plate-width", type=float, default=4.0, help="top plate width in mm")
    parser.add_argument("--feed-offset", type=float, default=2.0, help="feed distance from the short in mm")
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


def build_model(nr_ts: int, end_criteria: float, requested_length: float, requested_width: float, requested_offset: float):
    # Half-millimetre dimensional increments keep tuning meshes comparable.
    plate_length = round(requested_length * 2.0) / 2.0
    plate_width = round(requested_width * 2.0) / 2.0
    feed_offset = round(requested_offset * 2.0) / 2.0
    if not 1.0 <= feed_offset <= plate_length - 1.0:
        raise ValueError("feed offset must remain between the short and open end")
    if plate_width < 2.0:
        raise ValueError("plate width must be at least 2 mm")

    substrate_length = 30.0
    substrate_width = 20.0
    substrate_height = 1.6
    short_plate_width = 2.0
    short_width = 1.0
    feed_pad = 1.0

    plate_start = -plate_length / 2.0
    plate_stop = plate_length / 2.0
    feed_x = plate_start + feed_offset

    padding_xy = 20.0
    domain = {
        "x": np.array([-substrate_length / 2 - padding_xy, substrate_length / 2 + padding_xy]),
        "y": np.array([-substrate_width / 2 - padding_xy, substrate_width / 2 + padding_xy]),
        "z": np.array([-20.0, substrate_height + 30.0]),
    }

    fdtd = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
    fdtd.SetGaussExcite(TARGET_FREQUENCY, EXCITATION_CUTOFF)
    fdtd.SetBoundaryCond(["PML_8"] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)
    for axis in "xyz":
        mesh.AddLine(axis, domain[axis])

    # Regular PCB-region lines plus exact material and feed interfaces.
    mesh.AddLine("x", np.arange(-substrate_length / 2, substrate_length / 2 + 0.1, 1.0))
    mesh.AddLine("x", [plate_start, plate_start + short_width, plate_stop])
    mesh.AddLine("x", [feed_x - feed_pad / 2, feed_x + feed_pad / 2])
    mesh.AddLine("y", np.arange(-substrate_width / 2, substrate_width / 2 + 0.1, 1.0))
    mesh.AddLine("y", [-plate_width / 2, -feed_pad / 2, 0.0, feed_pad / 2, plate_width / 2])
    mesh.AddLine("z", np.linspace(0.0, substrate_height, 5))

    ptfe_kappa = 2.0 * np.pi * TARGET_FREQUENCY * EPS0 * PTFE_EPSILON * PTFE_LOSS_TANGENT
    substrate = csx.AddMaterial("PTFE", epsilon=PTFE_EPSILON, kappa=ptfe_kappa)
    substrate.AddBox(
        priority=1,
        start=[-substrate_length / 2, -substrate_width / 2, 0.0],
        stop=[substrate_length / 2, substrate_width / 2, substrate_height],
    )

    ground = csx.AddMetal("ground_plane")
    ground.AddBox(
        priority=10,
        start=[-substrate_length / 2, -substrate_width / 2, 0.0],
        stop=[substrate_length / 2, substrate_width / 2, 0.0],
    )

    radiator = csx.AddMetal("inverted_f_radiator")
    radiator.AddBox(
        priority=10,
        start=[plate_start, -plate_width / 2, substrate_height],
        stop=[plate_stop, plate_width / 2, substrate_height],
    )
    radiator.AddBox(
        priority=10,
        start=[plate_start, -short_plate_width / 2, 0.0],
        stop=[plate_start + short_width, short_plate_width / 2, substrate_height],
    )

    max_cell = C0 / (TARGET_FREQUENCY + EXCITATION_CUTOFF) / UNIT / 20.0
    mesh.SmoothMeshLines("all", max_cell, ratio=1.4)

    port = fdtd.AddLumpedPort(
        port_nr=1,
        R=FEED_OHMS,
        start=[feed_x - feed_pad / 2, -feed_pad / 2, 0.0],
        stop=[feed_x + feed_pad / 2, feed_pad / 2, substrate_height],
        p_dir="z",
        excite=1.0,
        priority=20,
    )
    nf2ff = fdtd.CreateNF2FFBox(opt_resolution=[max_cell] * 3)

    design = {
        "antenna_type": "planar inverted-F antenna (PIFA)",
        "target_frequency_ghz": TARGET_FREQUENCY / 1e9,
        "sweep_ghz": [SWEEP_LOW / 1e9, SWEEP_HIGH / 1e9],
        "feed_ohms": FEED_OHMS,
        "top_plate_length_mm": plate_length,
        "top_plate_width_mm": plate_width,
        "feed_offset_mm": feed_offset,
        "short_width_mm": short_width,
        "short_plate_width_mm": short_plate_width,
        "substrate": {
            "material": "PTFE",
            "relative_permittivity": PTFE_EPSILON,
            "loss_tangent": PTFE_LOSS_TANGENT,
            "conductivity_s_per_m": ptfe_kappa,
            "size_mm": [substrate_length, substrate_width, substrate_height],
        },
        "mesh_max_cell_mm": max_cell,
        "mesh_cells": {axis: len(mesh.GetLines(axis)) - 1 for axis in "xyz"},
        "domain_mm": {axis: values.tolist() for axis, values in domain.items()},
    }
    return fdtd, port, nf2ff, design


def appcsxcad_path() -> Path:
    candidates = (
        OPENEMS_GPU_ROOT / "openEMS-Project" / "AppCSXCAD" / "build-codex" / "AppCSXCAD",
        CUSTOM_PREFIX / "bin" / "AppCSXCAD",
        Path("/home/shanda/opt/openEMS/bin/AppCSXCAD"),
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("AppCSXCAD was not found")


def show_geometry(xml_path: Path) -> None:
    app = appcsxcad_path()
    environment = os.environ.copy()
    local_lib = str(CUSTOM_PREFIX / "lib")
    environment["LD_LIBRARY_PATH"] = local_lib + ":" + environment.get("LD_LIBRARY_PATH", "")
    environment["QT_X11_NO_MITSHM"] = "1"
    log = (xml_path.parent / "appcsxcad.log").open("ab")
    print(f"Opening geometry with {app}")
    subprocess.Popen(
        [str(app), str(xml_path)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )


def matching_bands(frequency: np.ndarray, s11_db: np.ndarray) -> list[list[float]]:
    mask = s11_db <= -10.0
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return [
        [float(frequency[start] / 1e9), float(frequency[stop] / 1e9)]
        for start, stop in zip(starts, stops)
    ]


def save_port_results(output_dir: Path, port):
    frequency = np.linspace(SWEEP_LOW, SWEEP_HIGH, 601)
    port.CalcPort(str(output_dir), frequency)
    s11 = np.asarray(port.uf_ref / port.uf_inc)
    impedance = np.asarray(port.uf_tot / port.if_tot)
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-12))
    target_s11 = float(np.interp(TARGET_FREQUENCY, frequency, s11_db))
    target_impedance = np.interp(TARGET_FREQUENCY, frequency, impedance.real) + 1j * np.interp(
        TARGET_FREQUENCY, frequency, impedance.imag
    )

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
    s_axis.axvline(5.0, color="darkred", linestyle=":", linewidth=1)
    s_axis.set_ylabel("S11 (dB)")
    s_axis.set_ylim(min(-35, float(np.nanmin(s11_db)) - 2), 1)
    s_axis.grid(True, alpha=0.3)
    z_axis.plot(frequency / 1e9, impedance.real, label="Real", linewidth=1.6)
    z_axis.plot(frequency / 1e9, impedance.imag, label="Imaginary", linewidth=1.6)
    z_axis.axhline(FEED_OHMS, color="0.4", linestyle="--", linewidth=1)
    z_axis.axvline(5.0, color="darkred", linestyle=":", linewidth=1)
    z_axis.set_xlabel("Frequency (GHz)")
    z_axis.set_ylabel("Input impedance (ohm)")
    z_axis.grid(True, alpha=0.3)
    z_axis.legend()
    figure.suptitle("5 GHz PTFE inverted-F antenna")
    figure.tight_layout()
    figure.savefig(output_dir / "s11_impedance.png", dpi=160)
    plt.close(figure)
    return frequency, s11_db, target_s11, target_impedance


def save_far_field(output_dir: Path, nf2ff, port, frequency: np.ndarray):
    theta = np.arange(0.0, 180.1, 2.0)
    phi = np.arange(0.0, 360.0, 4.0)
    result = nf2ff.CalcNF2FF(
        str(output_dir),
        TARGET_FREQUENCY,
        theta,
        phi,
        center=[0.0, 0.0, 0.0],
        outfile="nf2ff_3d.h5",
        verbose=1,
    )
    field = np.asarray(result.E_norm[0]).squeeze()
    dmax_db = float(10.0 * np.log10(result.Dmax[0]))
    directivity_db = 20.0 * np.log10(
        np.maximum(np.abs(field) / np.nanmax(np.abs(field)), 1e-6)
    ) + dmax_db
    peak_theta_index, peak_phi_index = np.unravel_index(np.nanargmax(directivity_db), directivity_db.shape)
    accepted_power = float(np.interp(TARGET_FREQUENCY, frequency, np.asarray(port.P_acc)))
    efficiency = float(result.Prad[0] / accepted_power) if accepted_power > 0 else float("nan")

    np.savez(
        output_dir / "far_field_results.npz",
        frequency_hz=TARGET_FREQUENCY,
        theta_deg=theta,
        phi_deg=phi,
        directivity_db=directivity_db,
        dmax_db=dmax_db,
        peak_theta_deg=float(theta[peak_theta_index]),
        peak_phi_deg=float(phi[peak_phi_index]),
        radiated_power_w=float(result.Prad[0]),
        radiation_efficiency=efficiency,
    )

    phi_0 = int(np.argmin(np.abs(phi - 0.0)))
    phi_90 = int(np.argmin(np.abs(phi - 90.0)))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(theta, directivity_db[:, phi_0], label="phi = 0°", linewidth=1.8)
    axis.plot(theta, directivity_db[:, phi_90], label="phi = 90°", linewidth=1.8)
    axis.set_xlabel("Theta (degrees; 0° is +z)")
    axis.set_ylabel("Directivity (dBi)")
    axis.set_title("5.000 GHz elevation cuts")
    axis.grid(True, alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "far_field_elevation.png", dpi=160)
    plt.close(figure)
    return dmax_db, float(theta[peak_theta_index]), float(phi[peak_phi_index]), efficiency


def main() -> int:
    args = parse_args()
    require_custom_bindings()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fdtd, port, nf2ff, design = build_model(
        args.nr_ts, args.end_criteria, args.plate_length, args.plate_width, args.feed_offset
    )
    xml_path = output_dir / "inverted_f_5ghz_ptfe.xml"
    fdtd.Write2XML(str(xml_path))
    (output_dir / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    cells = design["mesh_cells"]
    print(f"Wrote model: {xml_path}")
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

    frequency, s11_db, target_s11, target_impedance = save_port_results(output_dir, port)
    best_index = int(np.nanargmin(s11_db))
    summary = {
        "engine": args.engine,
        "gpu_device": args.gpu_device if args.engine == "gpu" else None,
        "target_frequency_ghz": TARGET_FREQUENCY / 1e9,
        "s11_at_target_db": target_s11,
        "impedance_at_target_ohm": {
            "real": float(target_impedance.real),
            "imaginary": float(target_impedance.imag),
        },
        "best_frequency_ghz": float(frequency[best_index] / 1e9),
        "minimum_s11_db": float(s11_db[best_index]),
        "s11_below_minus_10_db_bands_ghz": matching_bands(frequency, s11_db),
    }
    if not args.skip_far_field:
        dmax, peak_theta, peak_phi, efficiency = save_far_field(output_dir, nf2ff, port, frequency)
        summary.update(
            {
                "peak_directivity_dbi": dmax,
                "peak_theta_deg": peak_theta,
                "peak_phi_deg": peak_phi,
                "radiation_efficiency": efficiency,
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
