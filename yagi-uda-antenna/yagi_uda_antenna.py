#!/usr/bin/env python3
"""GPU openEMS model of a five-element Yagi-Uda antenna for 2.4 GHz."""

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
from openEMS.physical_constants import C0


UNIT = 1e-3  # millimetres
TARGET_FREQUENCY = 2.4e9
SWEEP_LOW = 2.2e9
SWEEP_HIGH = 2.6e9
EXCITATION_CUTOFF = 0.35e9
FEED_OHMS = 50.0


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
    parser.add_argument("--nr-ts", type=int, default=60_000)
    parser.add_argument("--end-criteria", type=float, default=1e-4)
    parser.add_argument(
        "--driven-scale",
        type=float,
        default=0.408,
        help="driven-element full length divided by the 2.4 GHz wavelength",
    )
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


def build_model(nr_ts: int, end_criteria: float, driven_scale: float):
    wavelength = C0 / TARGET_FREQUENCY / UNIT
    element_width = 3.0
    feed_gap = 2.0
    folded_spacing = 6.0

    names = ("reflector", "driven", "director_1", "director_2", "director_3")
    positions = wavelength * np.array([-0.20, 0.0, 0.15, 0.32, 0.50])
    length_scales = np.array([0.482, driven_scale, 0.424, 0.414, 0.405])
    lengths = np.round(wavelength * length_scales)

    padding = 60.0
    domain = {
        "x": np.array([positions.min() - padding, positions.max() + padding]),
        "y": np.array([-lengths.max() / 2 - padding, lengths.max() / 2 + padding]),
        "z": np.array([-padding, padding]),
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
    mesh.AddLine("z", 0.0)
    mesh.AddLine("x", np.ravel([positions - element_width / 2, positions + element_width / 2]))
    mesh.AddLine(
        "x",
        [positions[1] - folded_spacing - element_width / 2,
         positions[1] - folded_spacing + element_width / 2],
    )
    mesh.AddLine("y", np.ravel([-lengths / 2, lengths / 2]))
    mesh.AddLine("y", [-feed_gap / 2, 0.0, feed_gap / 2])

    metal = csx.AddMetal("yagi_elements")
    for index, (x_pos, length) in enumerate(zip(positions, lengths)):
        x_start = x_pos - element_width / 2
        x_stop = x_pos + element_width / 2
        if index == 1:
            # A folded driven element raises the Yagi feed resistance.
            return_x_start = x_pos - folded_spacing - element_width / 2
            return_x_stop = x_pos - folded_spacing + element_width / 2
            metal.AddBox(
                priority=10,
                start=[x_start, -length / 2, 0.0],
                stop=[x_stop, -feed_gap / 2, 0.0],
            )
            metal.AddBox(
                priority=10,
                start=[x_start, feed_gap / 2, 0.0],
                stop=[x_stop, length / 2, 0.0],
            )
            metal.AddBox(
                priority=10,
                start=[return_x_start, -length / 2, 0.0],
                stop=[return_x_stop, length / 2, 0.0],
            )
            metal.AddBox(
                priority=10,
                start=[return_x_start, -length / 2, 0.0],
                stop=[x_stop, -length / 2 + element_width, 0.0],
            )
            metal.AddBox(
                priority=10,
                start=[return_x_start, length / 2 - element_width, 0.0],
                stop=[x_stop, length / 2, 0.0],
            )
        else:
            metal.AddBox(
                priority=10,
                start=[x_start, -length / 2, 0.0],
                stop=[x_stop, length / 2, 0.0],
            )

    max_cell = C0 / (TARGET_FREQUENCY + EXCITATION_CUTOFF) / UNIT / 20.0
    mesh.SmoothMeshLines("all", max_cell, ratio=1.4)

    driven_x = positions[1]
    port = fdtd.AddLumpedPort(
        port_nr=1,
        R=FEED_OHMS,
        start=[driven_x - element_width / 2, -feed_gap / 2, 0.0],
        stop=[driven_x + element_width / 2, feed_gap / 2, 0.0],
        p_dir="y",
        excite=1.0,
        priority=20,
    )
    nf2ff = fdtd.CreateNF2FFBox(opt_resolution=[max_cell] * 3)

    design = {
        "target_frequency_ghz": TARGET_FREQUENCY / 1e9,
        "sweep_ghz": [SWEEP_LOW / 1e9, SWEEP_HIGH / 1e9],
        "feed_ohms": FEED_OHMS,
        "element_names": names,
        "element_length_scales": length_scales.tolist(),
        "element_lengths_mm": lengths.tolist(),
        "element_positions_mm": positions.tolist(),
        "element_width_mm": element_width,
        "feed_gap_mm": feed_gap,
        "driven_element": "folded dipole",
        "folded_spacing_mm": folded_spacing,
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
    frequency = np.linspace(SWEEP_LOW, SWEEP_HIGH, 501)
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
    s_axis.axvline(TARGET_FREQUENCY / 1e9, color="darkred", linestyle=":", linewidth=1)
    s_axis.set_ylabel("S11 (dB)")
    s_axis.set_ylim(min(-35, float(np.nanmin(s11_db)) - 2), 1)
    s_axis.grid(True, alpha=0.3)
    z_axis.plot(frequency / 1e9, impedance.real, label="Real", linewidth=1.6)
    z_axis.plot(frequency / 1e9, impedance.imag, label="Imaginary", linewidth=1.6)
    z_axis.axhline(FEED_OHMS, color="0.4", linestyle="--", linewidth=1)
    z_axis.axvline(TARGET_FREQUENCY / 1e9, color="darkred", linestyle=":", linewidth=1)
    z_axis.set_xlabel("Frequency (GHz)")
    z_axis.set_ylabel("Input impedance (ohm)")
    z_axis.grid(True, alpha=0.3)
    z_axis.legend()
    figure.suptitle("Five-element 2.4 GHz Yagi-Uda")
    figure.tight_layout()
    figure.savefig(output_dir / "s11_impedance.png", dpi=160)
    plt.close(figure)
    return frequency, s11_db, target_s11, target_impedance


def save_far_field(output_dir: Path, nf2ff, port, frequency: np.ndarray):
    phi = np.arange(-180.0, 180.1, 2.0)
    result = nf2ff.CalcNF2FF(
        str(output_dir),
        TARGET_FREQUENCY,
        90.0,
        phi,
        center=[0.0, 0.0, 0.0],
        outfile="nf2ff_azimuth.h5",
        verbose=1,
    )
    field = np.asarray(result.E_norm[0]).squeeze()
    dmax_db = float(10.0 * np.log10(result.Dmax[0]))
    directivity_db = 20.0 * np.log10(
        np.maximum(np.abs(field) / np.nanmax(np.abs(field)), 1e-6)
    ) + dmax_db
    peak_index = int(np.nanargmax(directivity_db))
    front = float(directivity_db[np.argmin(np.abs(phi))])
    back = float(directivity_db[np.argmin(np.abs(np.abs(phi) - 180.0))])
    accepted_power = float(np.interp(TARGET_FREQUENCY, frequency, np.asarray(port.P_acc)))
    efficiency = float(result.Prad[0] / accepted_power) if accepted_power > 0 else float("nan")

    np.savez(
        output_dir / "far_field_results.npz",
        frequency_hz=TARGET_FREQUENCY,
        phi_deg=phi,
        directivity_db=directivity_db,
        dmax_db=dmax_db,
        peak_phi_deg=float(phi[peak_index]),
        front_to_back_db=front - back,
        radiated_power_w=float(result.Prad[0]),
        radiation_efficiency=efficiency,
    )

    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(111, projection="polar")
    axis.plot(np.deg2rad(phi), directivity_db, linewidth=1.8)
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_rlim(max(-30.0, dmax_db - 30.0), max(1.0, dmax_db + 1.0))
    axis.set_title("Azimuth directivity at 2.400 GHz")
    axis.grid(True, alpha=0.4)
    figure.tight_layout()
    figure.savefig(output_dir / "far_field_azimuth.png", dpi=160)
    plt.close(figure)
    return dmax_db, float(phi[peak_index]), front - back, efficiency


def main() -> int:
    args = parse_args()
    require_custom_bindings()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fdtd, port, nf2ff, design = build_model(args.nr_ts, args.end_criteria, args.driven_scale)
    xml_path = output_dir / "yagi_uda_2p4ghz.xml"
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
        dmax, peak_phi, front_to_back, efficiency = save_far_field(output_dir, nf2ff, port, frequency)
        summary.update(
            {
                "peak_directivity_dbi": dmax,
                "peak_azimuth_deg": peak_phi,
                "front_to_back_db": front_to_back,
                "radiation_efficiency": efficiency,
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
