#!/usr/bin/env python3
"""1.447-1.617 GHz conductor-only, dual-port quadrature QHA."""

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


UNIT = 1e-3
TARGET_FREQUENCY = 1.532e9
TARGET_BAND_LOW = 1.447e9
TARGET_BAND_HIGH = 1.617e9
SWEEP_LOW = 1.2e9
SWEEP_HIGH = 1.85e9
EXCITATION_CUTOFF = 0.45e9
REFERENCE_OHMS = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "simulation_quadrature_1447_1617mhz",
        help="simulation and result directory",
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--skip-far-field", action="store_true")
    parser.add_argument("--engine", default="gpu", choices=("gpu", "multithreaded", "basic"))
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--nr-ts", type=int, default=120_000)
    parser.add_argument("--end-criteria", type=float, default=1e-4)
    parser.add_argument("--long-diameter", type=float, default=67.1, help="port-1 loop diameter in mm")
    parser.add_argument("--short-diameter", type=float, default=67.1, help="port-2 loop diameter in mm")
    parser.add_argument("--long-height", type=float, default=33.6, help="port-1 loop axial height in mm")
    parser.add_argument("--short-height", type=float, default=32.5, help="port-2 loop axial height in mm")
    parser.add_argument("--turns", type=float, default=0.20, help="fractional helix turns")
    parser.add_argument("--wire-radius", type=float, default=0.50, help="PEC wire radius in mm")
    parser.add_argument("--feed-gap", type=float, default=2.0, help="balanced feed gap in mm")
    parser.add_argument("--port-2-amplitude", type=float, default=0.90, help="port-2 source amplitude for equal incident waves")
    parser.add_argument("--winding", type=int, default=1, choices=(-1, 1))
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


def snap_hundredth_mm(value: float) -> float:
    return round(value * 100.0) / 100.0


def add_wire(metal, points: np.ndarray, radius: float, priority: int = 10):
    return metal.AddWire(np.asarray(points, dtype=float), radius=radius, priority=priority)


def helix_points(radius: float, height: float, turns: float, phase: float, winding: int) -> np.ndarray:
    samples = max(41, int(np.ceil(abs(turns) * 240.0)) + 1)
    u = np.linspace(0.0, 1.0, samples)
    angle = phase + winding * 2.0 * np.pi * turns * u
    return np.vstack((radius * np.cos(angle), radius * np.sin(angle), height * u))


def build_model(
    nr_ts: int,
    end_criteria: float,
    requested_long_diameter: float,
    requested_short_diameter: float,
    requested_long_height: float,
    requested_short_height: float,
    requested_turns: float,
    requested_wire_radius: float,
    requested_feed_gap: float,
    requested_port_2_amplitude: float,
    winding: int,
):
    long_diameter = snap_hundredth_mm(requested_long_diameter)
    short_diameter = snap_hundredth_mm(requested_short_diameter)
    long_height = snap_hundredth_mm(requested_long_height)
    short_height = snap_hundredth_mm(requested_short_height)
    turns = round(requested_turns, 3)
    wire_radius = snap_hundredth_mm(requested_wire_radius)
    feed_gap = snap_hundredth_mm(requested_feed_gap)
    port_2_amplitude = float(requested_port_2_amplitude)

    if min(long_diameter, short_diameter, long_height, short_height) <= 0:
        raise ValueError("QHA diameters and heights must be positive")
    if not 0.1 <= turns <= 1.5:
        raise ValueError("turn count must be between 0.1 and 1.5")
    if not 0.0 < port_2_amplitude <= 2.0:
        raise ValueError("port-2 amplitude must be in (0, 2]")
    if wire_radius <= 0 or feed_gap <= 2.0 * wire_radius:
        raise ValueError("feed gap must exceed the PEC wire diameter")
    if abs(long_height - short_height) <= 2.0 * wire_radius:
        raise ValueError("top crossovers require more than one wire diameter of vertical separation")

    long_radius = long_diameter / 2.0
    short_radius = short_diameter / 2.0
    outer_radius = max(long_radius, short_radius) + wire_radius
    maximum_height = max(long_height, short_height) + wire_radius
    port_1_feed_z = -feed_gap / 2.0
    port_2_feed_z = -3.0 * feed_gap / 2.0
    quadrature_delay = 3.0 / (4.0 * TARGET_FREQUENCY)

    padding_xy = 60.0
    padding_z = 60.0
    domain = {
        "x": np.array([-outer_radius - padding_xy, outer_radius + padding_xy]),
        "y": np.array([-outer_radius - padding_xy, outer_radius + padding_xy]),
        "z": np.array([port_2_feed_z - padding_z, maximum_height + padding_z]),
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

    mesh.AddLine(
        "x",
        [-long_radius, -short_radius, -feed_gap / 2.0, -wire_radius,
         0.0, wire_radius, feed_gap / 2.0, short_radius, long_radius],
    )
    mesh.AddLine(
        "y",
        [-long_radius, -short_radius, -feed_gap / 2.0, -wire_radius,
         0.0, wire_radius, feed_gap / 2.0, short_radius, long_radius],
    )
    mesh.AddLine(
        "z",
        [port_2_feed_z, port_1_feed_z, 0.0, short_height, long_height],
    )

    metal = csx.AddMetal("QHA_PEC")
    phases = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
    radii = [long_radius, short_radius, long_radius, short_radius]
    heights = [long_height, short_height, long_height, short_height]
    arms = [helix_points(radii[k], heights[k], turns, phases[k], winding) for k in range(4)]
    for points in arms:
        add_wire(metal, points, wire_radius)

    # Each pair of opposite helices is one independently driven bifilar loop.
    # The 1 mm top-height offset is a physical air crossover, not self-phasing.
    add_wire(metal, np.column_stack((arms[0][:, -1], arms[2][:, -1])), wire_radius)
    add_wire(metal, np.column_stack((arms[1][:, -1], arms[3][:, -1])), wire_radius)

    # Compact balanced feeds occupy two z planes so the orthogonal feed lines
    # remain isolated in air. Port 1 drives the x pair; port 2 drives the y pair.
    port_1_nodes = (
        np.array([-feed_gap / 2.0, 0.0, port_1_feed_z]),
        np.array([feed_gap / 2.0, 0.0, port_1_feed_z]),
    )
    port_2_nodes = (
        np.array([0.0, -feed_gap / 2.0, port_2_feed_z]),
        np.array([0.0, feed_gap / 2.0, port_2_feed_z]),
    )
    for arm_index, node in ((2, port_1_nodes[0]), (0, port_1_nodes[1])):
        start = arms[arm_index][:, 0]
        route = np.column_stack((start, [start[0], start[1], port_1_feed_z], node))
        add_wire(metal, route, wire_radius)
    for arm_index, node in ((3, port_2_nodes[0]), (1, port_2_nodes[1])):
        start = arms[arm_index][:, 0]
        route = np.column_stack((start, [start[0], start[1], port_2_feed_z], node))
        add_wire(metal, route, wire_radius)

    max_cell = C0 / (TARGET_FREQUENCY + EXCITATION_CUTOFF) / UNIT / 22.0
    mesh.SmoothMeshLines("all", max_cell, ratio=1.4)

    port_1 = fdtd.AddLumpedPort(
        port_nr=1,
        R=REFERENCE_OHMS,
        start=port_1_nodes[0],
        stop=port_1_nodes[1],
        p_dir="x",
        excite=1.0,
        priority=20,
        delay=0.0,
    )
    port_2 = fdtd.AddLumpedPort(
        port_nr=2,
        R=REFERENCE_OHMS,
        start=port_2_nodes[0],
        stop=port_2_nodes[1],
        p_dir="y",
        excite=port_2_amplitude,
        priority=20,
        delay=quadrature_delay,
    )
    ports = [port_1, port_2]
    nf2ff = fdtd.CreateNF2FFBox(opt_resolution=[max_cell] * 3)

    wavelength_mm = C0 / TARGET_FREQUENCY / UNIT
    design = {
        "antenna_type": "conductor-only dual-port quadrature fractional-turn QHA",
        "target_frequency_ghz": TARGET_FREQUENCY / 1e9,
        "target_band_ghz": [TARGET_BAND_LOW / 1e9, TARGET_BAND_HIGH / 1e9],
        "sweep_ghz": [SWEEP_LOW / 1e9, SWEEP_HIGH / 1e9],
        "reference_impedance_ohm_per_port": REFERENCE_OHMS,
        "material": "PEC only; no substrate, support, radome, or ground plane",
        "excitation": {
            "port_1_phase_deg_at_center": 0.0,
            "port_2_phase_deg_at_center": 90.0,
            "port_1_source_amplitude": 1.0,
            "port_2_source_amplitude": port_2_amplitude,
            "amplitude_note": "port 2 calibrated for equal incident waves at center",
            "port_2_time_delay_s": quadrature_delay,
            "combination": "simultaneous linear FDTD superposition",
        },
        "turns": turns,
        "winding_sign": winding,
        "wire_radius_mm": wire_radius,
        "balanced_feed_gap_mm": feed_gap,
        "port_1_x_bifilar_loop": {
            "diameter_mm": long_diameter,
            "height_mm": long_height,
            "diameter_over_lambda0": long_diameter / wavelength_mm,
            "height_over_lambda0": long_height / wavelength_mm,
        },
        "port_2_y_bifilar_loop": {
            "diameter_mm": short_diameter,
            "height_mm": short_height,
            "diameter_over_lambda0": short_diameter / wavelength_mm,
            "height_over_lambda0": short_height / wavelength_mm,
        },
        "mesh_max_cell_mm": max_cell,
        "mesh_cells": {axis: len(mesh.GetLines(axis)) - 1 for axis in "xyz"},
        "domain_mm": {axis: values.tolist() for axis, values in domain.items()},
    }
    return fdtd, ports, nf2ff, design


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
    environment["LD_LIBRARY_PATH"] = str(CUSTOM_PREFIX / "lib") + ":" + environment.get(
        "LD_LIBRARY_PATH", ""
    )
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


def wrapped_phase_deg(values: np.ndarray) -> np.ndarray:
    return (np.rad2deg(np.angle(values)) + 180.0) % 360.0 - 180.0


def interp_complex(x: float, xp: np.ndarray, values: np.ndarray) -> complex:
    return complex(np.interp(x, xp, values.real), np.interp(x, xp, values.imag))


def save_port_results(output_dir: Path, ports):
    frequency = np.linspace(SWEEP_LOW, SWEEP_HIGH, 801)
    for port in ports:
        port.CalcPort(str(output_dir), frequency, ref_impedance=REFERENCE_OHMS)

    incident = np.vstack([np.asarray(port.uf_inc) for port in ports])
    reflected = np.vstack([np.asarray(port.uf_ref) for port in ports])
    impedance = np.vstack([np.asarray(port.uf_tot / port.if_tot) for port in ports])
    active_gamma = reflected / np.where(np.abs(incident) > 1e-15, incident, 1e-15)
    active_gamma_db = 20.0 * np.log10(np.maximum(np.abs(active_gamma), 1e-12))
    incident_power = np.vstack([np.asarray(port.P_inc) for port in ports])
    reflected_power = np.vstack([np.asarray(port.P_ref) for port in ports])
    accepted_power = np.vstack([np.asarray(port.P_acc) for port in ports])
    total_incident_power = np.sum(incident_power, axis=0)
    total_reflected_power = np.sum(reflected_power, axis=0)
    total_accepted_power = np.sum(accepted_power, axis=0)
    combined_gamma = np.sqrt(
        np.maximum(total_reflected_power, 0.0)
        / np.maximum(total_incident_power, 1e-30)
    )
    combined_reflection_db = 20.0 * np.log10(np.maximum(combined_gamma, 1e-12))
    incident_ratio = incident[1] / np.where(np.abs(incident[0]) > 1e-15, incident[0], 1e-15)
    relative_phase_deg = wrapped_phase_deg(incident_ratio)
    incident_voltage_ratio = np.abs(incident_ratio)

    target_active = [interp_complex(TARGET_FREQUENCY, frequency, row) for row in active_gamma]
    target_impedance = [interp_complex(TARGET_FREQUENCY, frequency, row) for row in impedance]
    target_combined_db = float(np.interp(TARGET_FREQUENCY, frequency, combined_reflection_db))
    unwrapped_phase = np.unwrap(np.angle(incident_ratio))
    target_phase_deg = float(np.rad2deg(np.interp(TARGET_FREQUENCY, frequency, unwrapped_phase)))
    target_phase_deg = float((target_phase_deg + 180.0) % 360.0 - 180.0)
    target_voltage_ratio = float(np.interp(TARGET_FREQUENCY, frequency, incident_voltage_ratio))
    target_accepted_power = float(np.interp(TARGET_FREQUENCY, frequency, total_accepted_power))
    band_mask = (frequency >= TARGET_BAND_LOW) & (frequency <= TARGET_BAND_HIGH)

    np.savez(
        output_dir / "port_results.npz",
        frequency_hz=frequency,
        port_incident_voltage=incident,
        port_reflected_voltage=reflected,
        port_active_reflection=active_gamma,
        port_active_reflection_db=active_gamma_db,
        port_impedance_ohm=impedance,
        port_incident_power_w=incident_power,
        port_reflected_power_w=reflected_power,
        port_accepted_power_w=accepted_power,
        total_accepted_power_w=total_accepted_power,
        combined_active_reflection=combined_gamma,
        combined_active_reflection_db=combined_reflection_db,
        incident_port_2_over_port_1=incident_ratio,
        incident_relative_phase_deg=relative_phase_deg,
        incident_voltage_ratio=incident_voltage_ratio,
    )

    figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(frequency / 1e9, combined_reflection_db, color="black", linewidth=2.0,
                 label="combined reflected/incident power")
    axes[0].plot(frequency / 1e9, active_gamma_db[0], linewidth=1.2, label="port 1 active")
    axes[0].plot(frequency / 1e9, active_gamma_db[1], linewidth=1.2, label="port 2 active")
    axes[0].axhline(-10.0, color="0.4", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Active reflection (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(frequency / 1e9, impedance[0].real, label="Re Z1", linewidth=1.3)
    axes[1].plot(frequency / 1e9, impedance[0].imag, label="Im Z1", linewidth=1.3)
    axes[1].plot(frequency / 1e9, impedance[1].real, label="Re Z2", linewidth=1.3)
    axes[1].plot(frequency / 1e9, impedance[1].imag, label="Im Z2", linewidth=1.3)
    axes[1].axhline(REFERENCE_OHMS, color="0.4", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Active impedance (ohm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2)
    axes[2].plot(frequency / 1e9, relative_phase_deg, color="darkgreen", label="port 2 / port 1 phase")
    axes[2].axhline(90.0, color="0.4", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("Frequency (GHz)")
    axes[2].set_ylabel("Incident phase (deg)")
    axes[2].grid(True, alpha=0.3)
    for axis in axes:
        axis.axvspan(TARGET_BAND_LOW / 1e9, TARGET_BAND_HIGH / 1e9, color="gold", alpha=0.12)
        axis.axvline(TARGET_FREQUENCY / 1e9, color="darkred", linestyle=":", linewidth=1.0)
    figure.suptitle("Dual-port 0/90 degree quadrature QHA")
    figure.tight_layout()
    figure.savefig(output_dir / "combined_port_response.png", dpi=160)
    plt.close(figure)

    summary = {
        "port_1_phase_command_deg": 0.0,
        "port_2_phase_command_deg": 90.0,
        "incident_relative_phase_at_center_deg": target_phase_deg,
        "incident_voltage_ratio_port_2_over_port_1_at_center": target_voltage_ratio,
        "combined_active_reflection_at_center_db": target_combined_db,
        "combined_below_minus_10_db_bands_ghz": matching_bands(frequency, combined_reflection_db),
        "combined_active_reflection_worst_in_target_band_db": float(np.max(combined_reflection_db[band_mask])),
        "combined_active_reflection_best_in_target_band_db": float(np.min(combined_reflection_db[band_mask])),
        "total_accepted_power_at_center_w": target_accepted_power,
        "port_1_active_reflection_at_center_db": float(20.0 * np.log10(max(abs(target_active[0]), 1e-12))),
        "port_2_active_reflection_at_center_db": float(20.0 * np.log10(max(abs(target_active[1]), 1e-12))),
        "port_1_active_impedance_at_center_ohm": {"real": target_impedance[0].real, "imaginary": target_impedance[0].imag},
        "port_2_active_impedance_at_center_ohm": {"real": target_impedance[1].real, "imaginary": target_impedance[1].imag},
    }
    return frequency, total_accepted_power, summary


def axial_ratio_db(cprh: np.ndarray, cplh: np.ndarray) -> np.ndarray:
    right = np.abs(cprh)
    left = np.abs(cplh)
    ratio = (right + left) / np.maximum(np.abs(right - left), 1e-15)
    return 20.0 * np.log10(np.maximum(ratio, 1.0))


def save_far_field(output_dir: Path, nf2ff, total_accepted_power: np.ndarray, frequency: np.ndarray):
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
    cprh = np.asarray(result.E_cprh[0]).squeeze()
    cplh = np.asarray(result.E_cplh[0]).squeeze()
    dmax_db = float(10.0 * np.log10(result.Dmax[0]))
    field_max = np.nanmax(np.abs(field))
    directivity_db = 20.0 * np.log10(np.maximum(np.abs(field) / field_max, 1e-6)) + dmax_db
    cprh_db = 20.0 * np.log10(np.maximum(np.abs(cprh) / field_max, 1e-6)) + dmax_db
    cplh_db = 20.0 * np.log10(np.maximum(np.abs(cplh) / field_max, 1e-6)) + dmax_db
    ar_db = axial_ratio_db(cprh, cplh)
    peak_theta_index, peak_phi_index = np.unravel_index(np.nanargmax(directivity_db), directivity_db.shape)
    accepted_power = float(np.interp(TARGET_FREQUENCY, frequency, total_accepted_power))
    efficiency = float(result.Prad[0] / accepted_power) if accepted_power > 0 else float("nan")
    phi_0 = int(np.argmin(np.abs(phi)))
    plus_z_rhcp = float(cprh_db[0, phi_0])
    plus_z_lhcp = float(cplh_db[0, phi_0])
    minus_z_rhcp = float(cprh_db[-1, phi_0])
    minus_z_lhcp = float(cplh_db[-1, phi_0])
    plus_z_hand = "CPRH" if plus_z_rhcp >= plus_z_lhcp else "CPLH"
    minus_z_hand = "CPRH" if minus_z_rhcp >= minus_z_lhcp else "CPLH"

    np.savez(
        output_dir / "far_field_results.npz",
        frequency_hz=TARGET_FREQUENCY,
        theta_deg=theta,
        phi_deg=phi,
        directivity_db=directivity_db,
        cprh_directivity_db=cprh_db,
        cplh_directivity_db=cplh_db,
        axial_ratio_db=ar_db,
        dmax_db=dmax_db,
        peak_theta_deg=float(theta[peak_theta_index]),
        peak_phi_deg=float(phi[peak_phi_index]),
        radiated_power_w=float(result.Prad[0]),
        radiation_efficiency=efficiency,
    )

    figure, (pattern_axis, ar_axis) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    pattern_axis.plot(theta, directivity_db[:, phi_0], label="Total", linewidth=1.8)
    pattern_axis.plot(theta, cprh_db[:, phi_0], label="CPRH", linewidth=1.5)
    pattern_axis.plot(theta, cplh_db[:, phi_0], label="CPLH", linewidth=1.5)
    pattern_axis.set_ylabel("Directivity (dBi)")
    pattern_axis.set_title(f"{TARGET_FREQUENCY / 1e9:.3f} GHz combined-quadrature QHA, phi = 0 deg")
    pattern_axis.grid(True, alpha=0.35)
    pattern_axis.legend()
    ar_axis.plot(theta, np.minimum(ar_db[:, phi_0], 30.0), color="darkgreen", linewidth=1.8)
    ar_axis.axhline(3.0, color="0.4", linestyle="--", linewidth=1.0)
    ar_axis.set_xlabel("Theta (degrees; 0 is +z)")
    ar_axis.set_ylabel("Axial ratio (dB)")
    ar_axis.set_ylim(0.0, 30.0)
    ar_axis.grid(True, alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_dir / "far_field_circular_polarization.png", dpi=160)
    plt.close(figure)

    return {
        "peak_directivity_dbi": dmax_db,
        "peak_theta_deg": float(theta[peak_theta_index]),
        "peak_phi_deg": float(phi[peak_phi_index]),
        "combined_accepted_power_w": accepted_power,
        "radiated_power_w": float(result.Prad[0]),
        "radiation_efficiency": efficiency,
        "plus_z_axial_ratio_db": float(ar_db[0, phi_0]),
        "plus_z_dominant_circular_component": plus_z_hand,
        "plus_z_cprh_directivity_dbi": plus_z_rhcp,
        "plus_z_cplh_directivity_dbi": plus_z_lhcp,
        "minus_z_axial_ratio_db": float(ar_db[-1, phi_0]),
        "minus_z_dominant_circular_component": minus_z_hand,
        "minus_z_cprh_directivity_dbi": minus_z_rhcp,
        "minus_z_cplh_directivity_dbi": minus_z_lhcp,
    }


def main() -> int:
    args = parse_args()
    require_custom_bindings()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fdtd, ports, nf2ff, design = build_model(
        args.nr_ts,
        args.end_criteria,
        args.long_diameter,
        args.short_diameter,
        args.long_height,
        args.short_height,
        args.turns,
        args.wire_radius,
        args.feed_gap,
        args.port_2_amplitude,
        args.winding,
    )
    xml_path = output_dir / "dual_port_quadrature_qha_1447_1617mhz_pec.xml"
    fdtd.Write2XML(str(xml_path))
    (output_dir / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    cells = design["mesh_cells"]
    print(f"Wrote model: {xml_path}")
    print("Mesh cells: {} x {} x {}".format(cells["x"], cells["y"], cells["z"]))

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

    frequency, total_accepted_power, port_summary = save_port_results(output_dir, ports)
    summary = {
        "engine": args.engine,
        "gpu_device": args.gpu_device if args.engine == "gpu" else None,
        "target_frequency_ghz": TARGET_FREQUENCY / 1e9,
        "target_band_ghz": [TARGET_BAND_LOW / 1e9, TARGET_BAND_HIGH / 1e9],
        "reference_impedance_ohm_per_port": REFERENCE_OHMS,
        "excitation_combination": "port 1 at 0 deg plus port 2 at +90 deg",
    }
    summary.update(port_summary)
    if not args.skip_far_field:
        summary.update(save_far_field(output_dir, nf2ff, total_accepted_power, frequency))

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
