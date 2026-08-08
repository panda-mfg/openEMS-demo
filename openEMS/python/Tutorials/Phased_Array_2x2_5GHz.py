#!/usr/bin/env python3
"""Simulate a square planar patch phased array at a configurable frequency.

The probe-fed patches are driven simultaneously.  A requested steering
direction is converted to an excitation delay for each 50 ohm lumped port, so
mutual coupling, active impedance, substrate loss, and the steered far field
are all included in one FDTD run.

Default 2 x 2 channel numbering, viewed from above (+z):

    CH3 (-x,+y)    CH4 (+x,+y)
    CH1 (-x,-y)    CH2 (+x,-y)

The script deliberately invokes the standalone solver given by --binary.  It
does not use the in-process FDTD.Run() method from the Python extension.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0


DEFAULT_BINARY = Path("/home/shanda/openEMS-gpu/local/bin/openEMS")
DEFAULT_RESULTS_ROOT = Path("/home/shanda/openEMS-gpu/results")
MM = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Square planar patch phased array")
    parser.add_argument(
        "--sim-path",
        type=Path,
        default=None,
        help="result directory (default: /home/shanda/openEMS-gpu/results/...) ",
    )
    parser.add_argument(
        "--binary", type=Path, default=DEFAULT_BINARY,
        help="standalone local openEMS executable",
    )
    parser.add_argument(
        "--array-size", type=int, default=2,
        help="number of patch elements along x and y (2 gives 2x2)",
    )
    parser.add_argument("--frequency-ghz", type=float, default=5.0)
    parser.add_argument(
        "--corner-ghz", type=float, default=0.65,
        help="Gaussian-pulse 20 dB corner offset from the center frequency",
    )
    parser.add_argument(
        "--theta-deg", type=float, default=20.0,
        help="beam elevation from broadside (+z), 0 to 60 degrees",
    )
    parser.add_argument(
        "--phi-deg", type=float, default=0.0,
        help="beam azimuth from +x toward +y",
    )
    parser.add_argument(
        "--spacing-lambda", type=float, default=0.5,
        help="equal x/y center spacing in free-space wavelengths",
    )
    parser.add_argument("--substrate-epsilon", type=float, default=3.48)
    parser.add_argument("--substrate-loss-tangent", type=float, default=0.0037)
    parser.add_argument("--substrate-thickness-mm", type=float, default=1.524)
    parser.add_argument(
        "--patch-length-mm", type=float, default=None,
        help="override the calibrated transmission-line-model resonant length",
    )
    parser.add_argument(
        "--patch-length-scale", type=float, default=0.975,
        help="calibration factor applied to the closed-form patch length",
    )
    parser.add_argument(
        "--patch-width-mm", type=float, default=None,
        help="override the transmission-line-model patch width",
    )
    parser.add_argument(
        "--feed-offset-mm", type=float, default=None,
        help="feed offset from patch center toward -y (default: 0.232*length)",
    )
    parser.add_argument(
        "--feed-size-mm", type=float, default=0.8,
        help="square footprint of each vertical lumped feed",
    )
    parser.add_argument(
        "--mesh-cells-per-wavelength", type=float, default=20.0,
        help="maximum air-cell resolution at the upper excitation corner",
    )
    parser.add_argument("--timesteps", type=int, default=60000)
    parser.add_argument("--end-criteria", type=float, default=1e-4)
    parser.add_argument(
        "--engine",
        choices=["fastest", "basic", "sse", "sse-compressed",
                 "multithreaded", "gpu"],
        default="gpu",
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--farfield-step-deg", type=float, default=5.0,
        help="angular sampling used by the NF2FF calculation",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="write XML and model summary without launching the solver",
    )
    parser.add_argument(
        "--setup-only", action="store_true",
        help="run solver preprocessing (--no-simulation) only",
    )
    parser.add_argument(
        "--post-only", action="store_true",
        help="reuse solver output already present in --sim-path",
    )
    parser.add_argument(
        "--s-parameters", action="store_true",
        help="run one excitation per port and export the complete S-matrix",
    )
    parser.add_argument(
        "--excite-port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-nf2ff", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--force", action="store_true",
        help="remove only known openEMS outputs in --sim-path before a run",
    )
    return parser.parse_args()


def patch_dimensions(frequency: float, epsilon_r: float, height_mm: float) -> tuple[float, float, float]:
    """Return (width_mm, length_mm, effective_epsilon) from the TL model."""
    height = height_mm * MM
    width = C0 / (2.0 * frequency) * math.sqrt(2.0 / (epsilon_r + 1.0))
    epsilon_eff = (
        (epsilon_r + 1.0) / 2.0
        + (epsilon_r - 1.0) / 2.0
        / math.sqrt(1.0 + 12.0 * height / width)
    )
    delta_length = (
        0.412
        * height
        * ((epsilon_eff + 0.3) * (width / height + 0.264))
        / ((epsilon_eff - 0.258) * (width / height + 0.8))
    )
    effective_length = C0 / (2.0 * frequency * math.sqrt(epsilon_eff))
    length = effective_length - 2.0 * delta_length
    return width / MM, length / MM, epsilon_eff


def channel_positions(spacing_mm: float, array_size: int) -> np.ndarray:
    """Return row-major element centers, starting at the -x/-y corner."""
    coordinates = (
        np.arange(array_size, dtype=float) - (array_size - 1.0) / 2.0
    ) * spacing_mm
    return np.array([
        [x_position, y_position]
        for y_position in coordinates
        for x_position in coordinates
    ])


def steering_delays(
    positions_mm: np.ndarray,
    theta_deg: float,
    phi_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nonnegative feed delays and desired relative phases.

    For transmission, w_n = exp(-j*k*r_n.u).  A delayed openEMS excitation
    has exp(-j*omega*delay), hence delay_n = r_n.u/c.  A common delay is then
    added to make every delay nonnegative; it has no effect on beam direction.
    """
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    projection_m = (
        positions_mm[:, 0] * math.sin(theta) * math.cos(phi)
        + positions_mm[:, 1] * math.sin(theta) * math.sin(phi)
    ) * MM
    raw_delays = projection_m / C0
    delays = raw_delays - np.min(raw_delays)
    return delays, projection_m


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.array_size <= 8:
        raise ValueError("array-size must be between 1 and 8")
    if not 0 <= args.excite_port <= args.array_size ** 2:
        raise ValueError("excite-port is outside the array's port range")
    if args.frequency_ghz <= 0 or args.corner_ghz <= 0:
        raise ValueError("frequency and corner frequency must be positive")
    if not 0 <= args.theta_deg <= 60:
        raise ValueError("theta must be between 0 and 60 degrees")
    if not 0.35 <= args.spacing_lambda <= 0.8:
        raise ValueError("spacing-lambda must be between 0.35 and 0.8")
    if args.substrate_epsilon <= 1 or args.substrate_thickness_mm <= 0:
        raise ValueError("invalid substrate parameters")
    if args.feed_size_mm <= 0:
        raise ValueError("feed-size-mm must be positive")
    if not 10 <= args.mesh_cells_per_wavelength <= 60:
        raise ValueError("mesh-cells-per-wavelength must be between 10 and 60")
    if not 0.8 <= args.patch_length_scale <= 1.2:
        raise ValueError("patch-length-scale must be between 0.8 and 1.2")
    if args.timesteps <= 0 or not 0 < args.end_criteria < 1:
        raise ValueError("invalid FDTD stopping parameters")
    if args.farfield_step_deg <= 0 or args.farfield_step_deg > 15:
        raise ValueError("farfield-step-deg must be in (0, 15]")
    exclusive = sum((args.generate_only, args.setup_only, args.post_only))
    if exclusive > 1:
        raise ValueError("generate-only, setup-only, and post-only are exclusive")
    if args.s_parameters and (args.generate_only or args.setup_only):
        raise ValueError("s-parameters cannot be combined with generate-only or setup-only")


def result_path(args: argparse.Namespace) -> Path:
    if args.sim_path is not None:
        return args.sim_path.resolve()
    theta = f"{args.theta_deg:g}".replace("-", "m").replace(".", "p")
    phi = f"{args.phi_deg % 360:g}".replace(".", "p")
    size_label = f"{args.array_size}x{args.array_size}"
    frequency_label = f"{args.frequency_ghz:g}".replace(".", "p")
    return (DEFAULT_RESULTS_ROOT /
            f"patch_array_{size_label}_{frequency_label}GHz_theta{theta}_phi{phi}").resolve()


def clean_known_outputs(sim_path: Path) -> None:
    """Delete solver products only; never delete the simulation directory."""
    patterns = (
        "et", "ht", "port_ut*", "port_it*", "nf2ff*.h5",
        "*.vtk", "*.vtr", "*.vtp", "*.pvd",
        "openEMS_run_stats.txt", "openEMS_stats.txt", "debugCSX.xml",
    )
    for pattern in patterns:
        for candidate in sim_path.glob(pattern):
            if candidate.is_file():
                candidate.unlink()


def existing_solver_output(sim_path: Path) -> bool:
    return any(sim_path.glob("port_ut*")) or any(sim_path.glob("nf2ff_E_*.h5"))


def build_model(args: argparse.Namespace) -> dict:
    frequency = args.frequency_ghz * 1e9
    corner = args.corner_ghz * 1e9
    wavelength_mm = C0 / frequency / MM
    spacing_mm = args.spacing_lambda * wavelength_mm
    width_model, length_model, epsilon_eff = patch_dimensions(
        frequency, args.substrate_epsilon, args.substrate_thickness_mm)
    patch_width = args.patch_width_mm or width_model
    patch_length = (
        args.patch_length_mm
        if args.patch_length_mm is not None
        else args.patch_length_scale * length_model
    )
    feed_offset = args.feed_offset_mm
    if feed_offset is None:
        feed_offset = 0.232 * patch_length
    if not 0 <= feed_offset < patch_length / 2.0:
        raise ValueError("feed-offset-mm must be in [0, patch_length/2)")
    if spacing_mm <= max(patch_width, patch_length):
        raise ValueError("element spacing is too small for the patch dimensions")

    positions = channel_positions(spacing_mm, args.array_size)
    delays, projections_m = steering_delays(
        positions, args.theta_deg, args.phi_deg)
    desired_phases_deg = -360.0 * frequency * projections_m / C0
    if args.excite_port:
        excitation_amplitudes = np.zeros(args.array_size ** 2)
        excitation_amplitudes[args.excite_port - 1] = 1.0
        excitation_delays = np.zeros(args.array_size ** 2)
        excitation_phases_deg = np.zeros(args.array_size ** 2)
    else:
        excitation_amplitudes = np.ones(args.array_size ** 2)
        excitation_delays = delays
        excitation_phases_deg = desired_phases_deg

    # A finite, continuous substrate and ground retain inter-element surface
    # waves and coupling.  The margin is 0.10 lambda on every board edge.
    board_margin = 0.10 * wavelength_mm
    array_span = (args.array_size - 1) * spacing_mm
    substrate_x = array_span + patch_width + 2.0 * board_margin
    substrate_y = array_span + patch_length + 2.0 * board_margin

    # Air regions leave room for the Huygens surface before the eight-cell PML.
    air_xy = 0.75 * wavelength_mm
    air_below = 0.40 * wavelength_mm
    air_above = 0.75 * wavelength_mm
    x_bounds = np.array([-substrate_x / 2.0 - air_xy,
                          substrate_x / 2.0 + air_xy])
    y_bounds = np.array([-substrate_y / 2.0 - air_xy,
                          substrate_y / 2.0 + air_xy])
    z_bounds = np.array([-air_below,
                          args.substrate_thickness_mm + air_above])

    fdtd = openEMS(NrTS=args.timesteps, EndCriteria=args.end_criteria)
    fdtd.SetGaussExcite(frequency, corner)
    fdtd.SetBoundaryCond(["PML_8"] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(MM)
    mesh.AddLine("x", x_bounds)
    mesh.AddLine("y", y_bounds)
    mesh.AddLine("z", z_bounds)

    patch = csx.AddMetal("patches")
    for x_pos, y_pos in positions:
        patch.AddBox(
            priority=10,
            start=[x_pos - patch_width / 2.0,
                   y_pos - patch_length / 2.0,
                   args.substrate_thickness_mm],
            stop=[x_pos + patch_width / 2.0,
                  y_pos + patch_length / 2.0,
                  args.substrate_thickness_mm],
        )

    max_mesh_mm = (
        C0 / (frequency + corner) / MM / args.mesh_cells_per_wavelength)
    fdtd.AddEdges2Grid(
        dirs="xy", properties=patch, metal_edge_res=max_mesh_mm / 2.0)

    substrate_kappa = (
        args.substrate_loss_tangent
        * 2.0 * math.pi * frequency * EPS0 * args.substrate_epsilon
    )
    substrate = csx.AddMaterial(
        "RO4350B", epsilon=args.substrate_epsilon, kappa=substrate_kappa)
    substrate_start = [-substrate_x / 2.0, -substrate_y / 2.0, 0.0]
    substrate_stop = [substrate_x / 2.0, substrate_y / 2.0,
                      args.substrate_thickness_mm]
    substrate.AddBox(priority=0, start=substrate_start, stop=substrate_stop)

    # Five z cells resolve the thin dielectric and the vertical probe feeds.
    mesh.AddLine(
        "z", np.linspace(0.0, args.substrate_thickness_mm, 6))

    ground = csx.AddMetal("ground")
    ground.AddBox(
        priority=10,
        start=[substrate_start[0], substrate_start[1], 0.0],
        stop=[substrate_stop[0], substrate_stop[1], 0.0],
    )
    fdtd.AddEdges2Grid(dirs="xy", properties=ground)

    ports = []
    feed_half_size = args.feed_size_mm / 2.0
    for channel, ((x_pos, y_pos), delay, amplitude) in enumerate(
            zip(positions, excitation_delays, excitation_amplitudes), start=1):
        feed_y = y_pos - feed_offset
        port = fdtd.AddLumpedPort(
            port_nr=channel,
            R=50.0,
            # A finite footprint is intentional.  It represents a small
            # probe/pad and remains a real cell volume after discretization;
            # zero-area vertical boxes are ignored by some solver engines.
            start=[x_pos - feed_half_size, feed_y - feed_half_size, 0.0],
            stop=[x_pos + feed_half_size, feed_y + feed_half_size,
                  args.substrate_thickness_mm],
            p_dir="z",
            excite=float(amplitude),
            priority=50,
            edges2grid="xy",
            delay=float(delay),
        )
        ports.append(port)

    mesh.SmoothMeshLines("all", max_mesh_mm, 1.35)
    nf2ff = None
    if not args.no_nf2ff:
        nf2ff = fdtd.CreateNF2FFBox(name="nf2ff", frequency=frequency)

    grid_lines = {
        axis: np.asarray(mesh.GetLines(axis)) for axis in ("x", "y", "z")
    }
    cell_counts = {axis: int(lines.size - 1) for axis, lines in grid_lines.items()}
    total_cells = int(np.prod(list(cell_counts.values())))

    return {
        "fdtd": fdtd,
        "csx": csx,
        "ports": ports,
        "nf2ff": nf2ff,
        "frequency": frequency,
        "corner": corner,
        "wavelength_mm": wavelength_mm,
        "spacing_mm": spacing_mm,
        "positions_mm": positions,
        "delays_s": excitation_delays,
        "desired_phases_deg": excitation_phases_deg,
        "excitation_amplitudes": excitation_amplitudes,
        "patch_width_mm": patch_width,
        "patch_length_mm": patch_length,
        "patch_length_scale": args.patch_length_scale,
        "feed_offset_mm": feed_offset,
        "feed_size_mm": args.feed_size_mm,
        "effective_epsilon": epsilon_eff,
        "substrate_size_mm": [substrate_x, substrate_y,
                               args.substrate_thickness_mm],
        "simulation_bounds_mm": [x_bounds.tolist(), y_bounds.tolist(),
                                  z_bounds.tolist()],
        "max_mesh_mm": max_mesh_mm,
        "cell_counts": cell_counts,
        "total_cells": total_cells,
    }


def summary_dict(args: argparse.Namespace, model: dict, binary: Path) -> dict:
    channels = []
    for number, (position, delay, phase, amplitude) in enumerate(zip(
            model["positions_mm"], model["delays_s"],
            model["desired_phases_deg"], model["excitation_amplitudes"]), start=1):
        channels.append({
            "channel": number,
            "position_mm": [float(position[0]), float(position[1]),
                            args.substrate_thickness_mm],
            "excitation_amplitude": float(amplitude),
            "excitation_delay_ps": float(delay * 1e12),
            "relative_phase_deg_at_target": float(phase),
            "termination_ohm": 50.0,
        })
    return {
        "model": (f"{args.array_size ** 2}-channel {args.array_size}x"
                  f"{args.array_size} probe-fed planar patch phased array"),
        "array_size": [args.array_size, args.array_size],
        "number_of_channels": args.array_size ** 2,
        "solver_binary": str(binary),
        "target_frequency_ghz": model["frequency"] / 1e9,
        "gaussian_corner_offset_ghz": model["corner"] / 1e9,
        "steering_theta_deg": args.theta_deg,
        "steering_phi_deg": args.phi_deg % 360.0,
        "free_space_wavelength_mm": model["wavelength_mm"],
        "element_spacing_mm": model["spacing_mm"],
        "element_spacing_lambda": args.spacing_lambda,
        "patch_width_mm": model["patch_width_mm"],
        "patch_length_mm": model["patch_length_mm"],
        "closed_form_patch_length_scale": model["patch_length_scale"],
        "feed_offset_toward_minus_y_mm": model["feed_offset_mm"],
        "feed_footprint_mm": [model["feed_size_mm"], model["feed_size_mm"]],
        "substrate": {
            "material": "RO4350B",
            "epsilon_r": args.substrate_epsilon,
            "loss_tangent_at_target": args.substrate_loss_tangent,
            "size_mm": model["substrate_size_mm"],
        },
        "mesh": {
            "maximum_cell_mm": model["max_mesh_mm"],
            "cells_per_wavelength_at_upper_corner": (
                args.mesh_cells_per_wavelength),
            "cell_counts_xyz": model["cell_counts"],
            "total_cells": model["total_cells"],
        },
        "channels": channels,
    }


def run_solver(args: argparse.Namespace, sim_path: Path, xml_file: Path) -> None:
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"openEMS binary is not executable: {binary}")

    command = [
        str(binary), str(xml_file), f"--engine={args.engine}",
        "--verbose=1", "--dump-statistics",
    ]
    if args.engine == "gpu":
        command.append(f"--gpu-device={args.gpu_device}")
    if args.engine == "multithreaded" and args.threads > 0:
        command.append(f"--numThreads={args.threads}")
    if args.setup_only:
        command.append("--no-simulation")

    environment = os.environ.copy()
    local_lib = str(binary.parent.parent / "lib")
    old_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        local_lib if not old_library_path else local_lib + os.pathsep + old_library_path)

    print("Solver command:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=sim_path, env=environment, check=False)
    # This build returns 1 after a successful --no-simulation preprocessing
    # pass; a normal time-domain run must still return zero.
    if completed.returncode != 0 and not (
            args.setup_only and completed.returncode == 1):
        raise subprocess.CalledProcessError(completed.returncode, command)


def nearest_index(values: np.ndarray, target: float) -> int:
    circular_distance = np.abs((values - target + 180.0) % 360.0 - 180.0)
    return int(np.argmin(circular_distance))


def signed_upper_cut(
    directivity_dbi: np.ndarray,
    theta_values: np.ndarray,
    phi_values: np.ndarray,
    positive_phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    positive_phi %= 360.0
    negative_phi = (positive_phi + 180.0) % 360.0
    upper = np.where(theta_values <= 90.0 + 1e-9)[0]
    theta_upper = theta_values[upper]
    positive = directivity_dbi[upper, nearest_index(phi_values, positive_phi)]
    negative = directivity_dbi[upper, nearest_index(phi_values, negative_phi)]
    signed_theta = np.r_[-theta_upper[:0:-1], theta_upper]
    signed_pattern = np.r_[negative[:0:-1], positive]
    return signed_theta, signed_pattern


def postprocess(
    args: argparse.Namespace,
    model: dict,
    sim_path: Path,
    summary: dict,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = model["frequency"]
    corner = model["corner"]
    frequencies = np.linspace(frequency - corner, frequency + corner, 401)
    target_index = int(np.argmin(np.abs(frequencies - frequency)))

    port_rows = []
    active_reflections = []
    accepted_power = []
    for channel, port in enumerate(model["ports"], start=1):
        port.CalcPort(str(sim_path), frequencies, ref_impedance=50.0)
        gamma = port.uf_ref / port.uf_inc
        active_reflections.append(gamma)
        accepted_power.append(float(np.real(port.P_acc[target_index])))
        impedance = port.uf_tot / port.if_tot
        for index, freq in enumerate(frequencies):
            port_rows.append([
                freq / 1e9,
                channel,
                20.0 * math.log10(max(abs(gamma[index]), 1e-15)),
                math.degrees(np.angle(gamma[index])),
                float(np.real(impedance[index])),
                float(np.imag(impedance[index])),
            ])

    with (sim_path / "active_port_results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frequency_GHz", "channel", "active_reflection_dB",
            "active_reflection_phase_deg", "active_impedance_real_ohm",
            "active_impedance_imag_ohm",
        ])
        writer.writerows(port_rows)

    fig, axis = plt.subplots(figsize=(8.0, 5.0), tight_layout=True)
    for channel, gamma in enumerate(active_reflections, start=1):
        gamma_db = 20.0 * np.log10(np.maximum(np.abs(gamma), 1e-15))
        axis.plot(frequencies / 1e9, gamma_db, linewidth=1.8,
                  label=f"CH{channel}")
    axis.axvline(frequency / 1e9, color="0.4", linestyle=":", linewidth=1)
    axis.axhline(-10, color="0.4", linestyle="--", linewidth=1)
    axis.grid(True, alpha=0.35)
    axis.set(xlabel="Frequency (GHz)", ylabel="Active reflection (dB)",
             title=(f"{args.array_size}x{args.array_size} active port match, steering "
                    f"theta={args.theta_deg:g} deg, phi={args.phi_deg % 360:g} deg"))
    axis.legend(ncol=2)
    fig.savefig(sim_path / "active_reflection.png", dpi=180)
    plt.close(fig)

    step = args.farfield_step_deg
    theta = np.arange(0.0, 180.0 + 0.5 * step, step)
    phi = np.unique(np.mod(np.r_[
        np.arange(0.0, 360.0, step),
        0.0, 90.0, 180.0, 270.0,
        args.phi_deg, args.phi_deg + 180.0,
    ], 360.0))
    nf_result = model["nf2ff"].CalcNF2FF(
        str(sim_path), frequency, theta, phi,
        center=[0.0, 0.0, args.substrate_thickness_mm * MM],
        outfile="nf2ff_pattern.h5", read_cached=False, verbose=1)

    e_norm = np.asarray(nf_result.E_norm[0])
    dmax = float(nf_result.Dmax[0])
    directivity_dbi = (
        20.0 * np.log10(np.maximum(e_norm / np.max(e_norm), 1e-15))
        + 10.0 * math.log10(dmax)
    )
    peak_index = np.unravel_index(np.argmax(directivity_dbi), directivity_dbi.shape)
    peak_theta = float(theta[peak_index[0]])
    peak_phi = float(phi[peak_index[1]])

    x_angles, x_cut = signed_upper_cut(directivity_dbi, theta, phi, 0.0)
    y_angles, y_cut = signed_upper_cut(directivity_dbi, theta, phi, 90.0)
    steer_angles, steer_cut = signed_upper_cut(
        directivity_dbi, theta, phi, args.phi_deg)

    with (sim_path / "farfield_cuts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "signed_theta_deg", "xz_directivity_dBi",
            "yz_directivity_dBi", "steering_plane_directivity_dBi",
        ])
        writer.writerows(zip(x_angles, x_cut, y_cut, steer_cut))

    fig, axis = plt.subplots(figsize=(8.0, 5.0), tight_layout=True)
    axis.plot(x_angles, x_cut, label="xz plane", linewidth=1.8)
    axis.plot(y_angles, y_cut, label="yz plane", linewidth=1.8)
    if not math.isclose(args.phi_deg % 90.0, 0.0, abs_tol=1e-9):
        axis.plot(steer_angles, steer_cut,
                  label=f"steering plane phi={args.phi_deg % 360:g} deg",
                  linewidth=2.0)
    axis.axvline(args.theta_deg, color="0.4", linestyle=":", linewidth=1)
    axis.grid(True, alpha=0.35)
    axis.set_xlim(-90, 90)
    axis.set_ylim(max(-30.0, float(np.max(directivity_dbi)) - 35.0),
                  float(np.max(directivity_dbi)) + 1.0)
    axis.set(xlabel="Signed elevation from broadside (deg)",
             ylabel="Directivity (dBi)",
             title=(f"{args.array_size}x{args.array_size} patch-array far field "
                    f"at {frequency / 1e9:g} GHz"))
    axis.legend()
    fig.savefig(sim_path / "farfield_cuts.png", dpi=180)
    plt.close(fig)

    total_accepted = float(np.sum(accepted_power))
    radiated_power = float(nf_result.Prad[0])
    efficiency = radiated_power / total_accepted if total_accepted > 0 else float("nan")
    power_balance_error_percent = (
        100.0 * (radiated_power - total_accepted) / total_accepted
        if total_accepted > 0 else float("nan"))
    efficiency_quality = (
        "valid" if 0.0 <= efficiency <= 1.0
        else "invalid_nonphysical_power_imbalance")
    active_at_target = []
    for channel, gamma in enumerate(active_reflections, start=1):
        value = gamma[target_index]
        active_at_target.append({
            "channel": channel,
            "magnitude_dB": float(20.0 * np.log10(max(abs(value), 1e-15))),
            "phase_deg": float(np.angle(value, deg=True)),
            "accepted_power_W": accepted_power[channel - 1],
        })

    summary["results"] = {
        "active_reflection_at_target": active_at_target,
        "total_accepted_power_W": total_accepted,
        "radiated_power_W": radiated_power,
        "radiation_efficiency": efficiency,
        "radiation_efficiency_quality": efficiency_quality,
        "power_balance_error_percent": power_balance_error_percent,
        "maximum_directivity_dBi": 10.0 * math.log10(dmax),
        "sampled_peak_theta_deg": peak_theta,
        "sampled_peak_phi_deg": peak_phi,
        "farfield_angular_step_deg": step,
    }
    with (sim_path / "model_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)

    print("\nSimulation result")
    print(f"  Dmax: {10.0 * math.log10(dmax):.2f} dBi")
    print(f"  sampled beam peak: theta={peak_theta:.1f} deg, phi={peak_phi:.1f} deg")
    print(f"  radiated/accepted power: {radiated_power:.4g}/{total_accepted:.4g} W")
    print(f"  radiation efficiency: {100.0 * efficiency:.1f} %")
    if efficiency_quality != "valid":
        print(
            "  warning: nonphysical radiated/accepted power balance "
            f"({power_balance_error_percent:+.1f} %); do not interpret as efficiency")
    for item in active_at_target:
        print(f"  CH{item['channel']} active reflection: {item['magnitude_dB']:.2f} dB")


def matrix_run_complete(sim_path: Path, number_of_ports: int) -> bool:
    """Return true when every voltage/current probe file is present."""
    expected = [
        sim_path / f"port_{kind}_{port}"
        for port in range(1, number_of_ports + 1)
        for kind in ("ut", "it")
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in expected)


def write_touchstone(
    path: Path,
    frequencies: np.ndarray,
    s_matrix: np.ndarray,
    reference_impedance: float = 50.0,
) -> None:
    """Write a Touchstone 1.1 full N-port matrix in row-major order."""
    number_of_ports = s_matrix.shape[1]
    with path.open("w") as stream:
        stream.write(f"# Hz S RI R {reference_impedance:g}\n")
        stream.write("! Full matrix; row is output port, column is input port.\n")
        stream.write("! Four complex values per continuation line.\n")
        for frequency_index, frequency in enumerate(frequencies):
            first_line = True
            for output_port in range(number_of_ports):
                for first_input in range(0, number_of_ports, 4):
                    if first_line:
                        stream.write(f"{frequency:.12e} ")
                        first_line = False
                    else:
                        stream.write("\n              ")
                    for input_port in range(
                            first_input, min(first_input + 4, number_of_ports)):
                        value = s_matrix[frequency_index, output_port, input_port]
                        stream.write(f"{value.real:+.12e} {value.imag:+.12e}  ")
            stream.write("\n")


def export_sparameter_results(
    args: argparse.Namespace,
    sim_path: Path,
    frequencies: np.ndarray,
    s_matrix: np.ndarray,
    model_summary: dict,
) -> None:
    """Export complex data, Touchstone, CSV summaries, plots, and checks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    number_of_ports = s_matrix.shape[1]
    target_frequency = args.frequency_ghz * 1e9
    target_index = int(np.argmin(np.abs(frequencies - target_frequency)))
    target_matrix = s_matrix[target_index]
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(target_matrix), 1e-15))
    frequency_label = f"{args.frequency_ghz:g}".replace(".", "p")

    np.savez_compressed(
        sim_path / f"patch_array_{number_of_ports}port_smatrix.npz",
        frequency_hz=frequencies,
        s=s_matrix,
        reference_impedance_ohm=50.0,
        convention="s[frequency, output_port, input_port]",
    )
    write_touchstone(
        sim_path / f"patch_array_{number_of_ports}port.s{number_of_ports}p",
        frequencies,
        s_matrix,
    )
    write_touchstone(
        sim_path / (
            f"patch_array_{number_of_ports}port_{frequency_label}GHz."
            f"s{number_of_ports}p"),
        frequencies[target_index:target_index + 1],
        s_matrix[target_index:target_index + 1],
    )

    with (sim_path / f"sparameter_matrix_{frequency_label}GHz.csv").open(
            "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "output_port", "input_port", "real", "imag",
            "magnitude_dB", "phase_deg",
        ])
        for output_port in range(number_of_ports):
            for input_port in range(number_of_ports):
                value = target_matrix[output_port, input_port]
                writer.writerow([
                    output_port + 1,
                    input_port + 1,
                    float(value.real),
                    float(value.imag),
                    float(magnitude_db[output_port, input_port]),
                    float(np.angle(value, deg=True)),
                ])

    return_loss_db = np.array([
        20.0 * np.log10(np.maximum(np.abs(s_matrix[:, port, port]), 1e-15))
        for port in range(number_of_ports)
    ])
    fig, axis = plt.subplots(figsize=(9.0, 5.5), tight_layout=True)
    for port in range(number_of_ports):
        axis.plot(frequencies / 1e9, return_loss_db[port], linewidth=1.1,
                  label=f"S{port + 1},{port + 1}")
    axis.axhline(-10.0, color="0.25", linestyle="--", linewidth=1)
    axis.axvline(target_frequency / 1e9, color="0.25", linestyle=":", linewidth=1)
    axis.grid(True, alpha=0.3)
    axis.set(xlabel="Frequency (GHz)", ylabel="Return loss (dB)",
             title=f"{number_of_ports}-port patch-array return loss")
    axis.legend(ncol=4, fontsize=7)
    fig.savefig(sim_path / "sparameter_return_loss.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 8.0), tight_layout=True)
    image = axis.imshow(magnitude_db, vmin=-50.0, vmax=0.0, cmap="viridis")
    for output_port in range(number_of_ports):
        for input_port in range(number_of_ports):
            axis.text(input_port, output_port,
                      f"{magnitude_db[output_port, input_port]:.0f}",
                      ha="center", va="center", fontsize=4,
                      color="white" if magnitude_db[output_port, input_port] < -22 else "black")
    axis.set_xticks(np.arange(number_of_ports), np.arange(1, number_of_ports + 1))
    axis.set_yticks(np.arange(number_of_ports), np.arange(1, number_of_ports + 1))
    axis.set(xlabel="Input port i", ylabel="Output port j",
             title=f"|Sji| at {frequencies[target_index] / 1e9:.4f} GHz (dB)")
    fig.colorbar(image, ax=axis, label="Magnitude (dB)")
    fig.savefig(
        sim_path / f"sparameter_matrix_{frequency_label}GHz.png", dpi=200)
    plt.close(fig)

    transpose = np.swapaxes(s_matrix, 1, 2)
    reciprocity_difference = s_matrix - transpose
    off_diagonal = ~np.eye(number_of_ports, dtype=bool)
    coupling_location = None
    if number_of_ports > 1:
        # Recover matrix indices from the flattened off-diagonal mask.
        coupling_locations = np.argwhere(off_diagonal)
        coupling_location = coupling_locations[
            int(np.argmax(magnitude_db[off_diagonal]))]
    singular_values = np.linalg.svd(target_matrix, compute_uv=False)
    maximum_singular_value_by_frequency = np.linalg.svd(
        s_matrix, compute_uv=False)[:, 0]
    full_band_singular_index = int(
        np.argmax(maximum_singular_value_by_frequency))
    passivity_tolerance = 1e-6
    passive_samples = maximum_singular_value_by_frequency <= (
        1.0 + passivity_tolerance)
    passive_band_start = target_index
    passive_band_stop = target_index
    if passive_samples[target_index]:
        while passive_band_start > 0 and passive_samples[passive_band_start - 1]:
            passive_band_start -= 1
        while (passive_band_stop + 1 < frequencies.size
               and passive_samples[passive_band_stop + 1]):
            passive_band_stop += 1
        contiguous_passive_band = [
            float(frequencies[passive_band_start] / 1e9),
            float(frequencies[passive_band_stop] / 1e9),
        ]
    else:
        contiguous_passive_band = None
    diagonal_db = np.diag(magnitude_db)
    summary = dict(model_summary)
    summary["sparameter_measurement"] = {
        "frequency_points": int(frequencies.size),
        "frequency_start_ghz": float(frequencies[0] / 1e9),
        "frequency_stop_ghz": float(frequencies[-1] / 1e9),
        "target_frequency_ghz": float(frequencies[target_index] / 1e9),
        "matrix_convention": "S[frequency, output_port_j, input_port_i] = b_j/a_i",
        "touchstone_order": "row-major full matrix",
        "return_loss_at_target_dB": diagonal_db.tolist(),
        "worst_return_loss_at_target_dB": float(np.max(diagonal_db)),
        "best_return_loss_at_target_dB": float(np.min(diagonal_db)),
        "strongest_mutual_coupling_at_target": (
            {
                "output_port": int(coupling_location[0] + 1),
                "input_port": int(coupling_location[1] + 1),
                "magnitude_dB": float(magnitude_db[tuple(coupling_location)]),
            }
            if coupling_location is not None else None
        ),
        "reciprocity_max_abs_difference_at_target": float(
            np.max(np.abs(reciprocity_difference[target_index]))),
        "reciprocity_rms_abs_difference_at_target": float(
            np.sqrt(np.mean(np.abs(reciprocity_difference[target_index]) ** 2))),
        "reciprocity_max_abs_difference_full_band": float(
            np.max(np.abs(reciprocity_difference))),
        "maximum_singular_value_at_target": float(np.max(singular_values)),
        "maximum_singular_value_full_band": float(
            maximum_singular_value_by_frequency[full_band_singular_index]),
        "maximum_singular_value_full_band_frequency_ghz": float(
            frequencies[full_band_singular_index] / 1e9),
        "passivity_check_tolerance": passivity_tolerance,
        "passive_frequency_samples": int(np.count_nonzero(passive_samples)),
        "frequency_samples_total": int(frequencies.size),
        "contiguous_passive_band_about_target_ghz": contiguous_passive_band,
    }
    with (sim_path / "sparameter_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)

    print("\nComplete S-parameter matrix")
    print(f"  output: {sim_path}")
    print(f"  target-frequency return loss: {np.min(diagonal_db):.2f} to {np.max(diagonal_db):.2f} dB")
    if coupling_location is not None:
        print(
            "  strongest coupling: S{},{} = {:.2f} dB".format(
                coupling_location[0] + 1,
                coupling_location[1] + 1,
                magnitude_db[tuple(coupling_location)],
            ))
    print(
        "  reciprocity max |Sji-Sij|: {:.3g}".format(
            np.max(np.abs(reciprocity_difference[target_index]))))
    print(f"  maximum singular value at target: {np.max(singular_values):.5f}")
    print(
        "  contiguous passive band about target: "
        f"{contiguous_passive_band[0]:.5g} to "
        f"{contiguous_passive_band[1]:.5g} GHz"
        if contiguous_passive_band is not None
        else "  target matrix did not pass the numerical passivity check"
    )


def run_sparameter_matrix(args: argparse.Namespace) -> int:
    """Run/resume one simulation per input port and assemble the S-matrix."""
    sim_path = result_path(args)
    sim_path.mkdir(parents=True, exist_ok=True)
    number_of_ports = args.array_size ** 2
    frequencies = np.linspace(
        (args.frequency_ghz - args.corner_ghz) * 1e9,
        (args.frequency_ghz + args.corner_ghz) * 1e9,
        401,
    )
    s_matrix = np.full(
        (frequencies.size, number_of_ports, number_of_ports),
        np.nan + 1j * np.nan,
        dtype=complex,
    )
    completed_ports = []
    model_summary = None

    args.no_nf2ff = True
    args.setup_only = False
    args.generate_only = False
    for input_port in range(1, number_of_ports + 1):
        port_path = sim_path / f"port_{input_port:02d}"
        port_path.mkdir(parents=True, exist_ok=True)
        args.excite_port = input_port
        model = build_model(args)
        xml_file = port_path / (
            f"patch_array_{args.array_size}x{args.array_size}_port{input_port:02d}.xml")
        model["fdtd"].Write2XML(str(xml_file))
        if model_summary is None:
            model_summary = summary_dict(args, model, args.binary.resolve())

        complete = matrix_run_complete(port_path, number_of_ports)
        if args.force and complete:
            clean_known_outputs(port_path)
            complete = False
        if complete:
            print(f"[{input_port:02d}/{number_of_ports:02d}] reuse completed port run")
        else:
            if args.post_only:
                raise FileNotFoundError(
                    f"incomplete S-parameter run for input port {input_port}: {port_path}")
            print(f"[{input_port:02d}/{number_of_ports:02d}] simulate input port {input_port}")
            run_solver(args, port_path, xml_file)

        source_port = model["ports"][input_port - 1]
        for port in model["ports"]:
            port.CalcPort(str(port_path), frequencies, ref_impedance=50.0)
        denominator = source_port.uf_inc
        for output_port, port in enumerate(model["ports"]):
            s_matrix[:, output_port, input_port - 1] = port.uf_ref / denominator
        completed_ports.append(input_port)
        np.savez_compressed(
            sim_path / "sparameter_matrix_partial.npz",
            frequency_hz=frequencies,
            s=s_matrix,
            completed_input_ports=np.asarray(completed_ports),
            convention="s[frequency, output_port, input_port]",
        )

    export_sparameter_results(
        args, sim_path, frequencies, s_matrix, model_summary)
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.phi_deg %= 360.0
    if args.s_parameters:
        return run_sparameter_matrix(args)
    sim_path = result_path(args)
    sim_path.mkdir(parents=True, exist_ok=True)

    if not args.post_only and existing_solver_output(sim_path):
        if not args.force:
            raise FileExistsError(
                f"solver output already exists in {sim_path}; use --post-only "
                "to reuse it or --force to replace known output files")
        clean_known_outputs(sim_path)

    model = build_model(args)
    binary = args.binary.resolve()
    size_label = f"{args.array_size}x{args.array_size}"
    frequency_label = f"{args.frequency_ghz:g}".replace(".", "p")
    xml_file = sim_path / (
        f"patch_array_{size_label}_{frequency_label}GHz.xml")
    model["fdtd"].Write2XML(str(xml_file))
    summary = summary_dict(args, model, binary)
    with (sim_path / "model_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)

    print(f"{args.array_size} x {args.array_size} planar patch phased-array model")
    print(f"  target: {model['frequency'] / 1e9:g} GHz")
    print(f"  patch W x L: {model['patch_width_mm']:.3f} x {model['patch_length_mm']:.3f} mm")
    print(f"  x/y spacing: {model['spacing_mm']:.3f} mm ({args.spacing_lambda:g} lambda0)")
    print(f"  mesh cells: {model['cell_counts']} = {model['total_cells']:,}")
    for channel in summary["channels"]:
        print(
            f"  CH{channel['channel']}: xy={channel['position_mm'][:2]} mm, "
            f"phase={channel['relative_phase_deg_at_target']:+.2f} deg, "
            f"delay={channel['excitation_delay_ps']:.2f} ps")
    print(f"  XML: {xml_file}")

    if args.generate_only:
        return 0
    if not args.post_only:
        run_solver(args, sim_path, xml_file)
    if args.setup_only:
        return 0

    postprocess(args, model, sim_path, summary)
    print(f"  results: {sim_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError,
            subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
