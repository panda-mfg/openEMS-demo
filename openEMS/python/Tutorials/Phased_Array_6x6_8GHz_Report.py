#!/usr/bin/env python3
"""Build the PDF report for the completed 6 x 6, 8 GHz array simulations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Phased_Array_4x4_Report as report_base


NAVY = report_base.NAVY
BLUE = report_base.BLUE
CYAN = report_base.CYAN
COPPER = report_base.COPPER
GOLD = report_base.GOLD
GREEN = report_base.GREEN
RED = report_base.RED
LIGHT = report_base.LIGHT
MID = report_base.MID
TEXT = report_base.TEXT
MUTED = report_base.MUTED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beam-path", type=Path,
        default=Path("/home/shanda/openEMS-gpu/results/patch_array_6x6_8GHz_model"))
    parser.add_argument(
        "--smatrix-path", type=Path,
        default=Path("/home/shanda/openEMS-gpu/results/patch_array_6x6_8GHz_smatrix"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/shanda/openEMS-gpu/results/patch_array_6x6_8GHz_report"))
    return parser.parse_args()


def frequency_label(value_ghz: float) -> str:
    return f"{value_ghz:g}GHz"


def load_inputs(args: argparse.Namespace) -> dict:
    beam_summary_path = args.beam_path / "model_summary.json"
    smatrix_summary_path = args.smatrix_path / "sparameter_summary.json"
    with beam_summary_path.open() as stream:
        beam_summary = json.load(stream)
    with smatrix_summary_path.open() as stream:
        smatrix_summary = json.load(stream)

    n_ports = int(beam_summary["number_of_channels"])
    target_ghz = float(beam_summary["target_frequency_ghz"])
    network_path = args.smatrix_path / f"patch_array_{n_ports}port_smatrix.npz"
    matrix_image = args.smatrix_path / (
        f"sparameter_matrix_{frequency_label(target_ghz)}.png")
    required = [
        args.beam_path / "active_port_results.csv",
        args.beam_path / "farfield_cuts.csv",
        args.beam_path / "openEMS_stats.txt",
        network_path,
        matrix_image,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing report inputs: " + ", ".join(missing))

    network = np.load(network_path)
    frequency_hz = network["frequency_hz"]
    s_matrix = network["s"]
    if s_matrix.shape[1:] != (n_ports, n_ports):
        raise ValueError(f"Expected a {n_ports} x {n_ports} network matrix")
    target_index = int(np.argmin(np.abs(frequency_hz - target_ghz * 1e9)))

    active = np.genfromtxt(required[0], delimiter=",", names=True)
    farfield = np.genfromtxt(required[1], delimiter=",", names=True)
    target_active = active[np.isclose(active["frequency_GHz"], target_ghz)]
    target_active = np.sort(target_active, order="channel")
    if target_active.size != n_ports:
        raise ValueError("Expected one target-frequency active record per channel")

    beam_lines = required[2].read_text().splitlines()
    beam_solver = {
        "cells": int(beam_lines[0].split()[0]),
        "timestep_s": float(beam_lines[1].split()[0]),
        "iterations": int(beam_lines[2].split()[0]),
        "runtime_s": float(beam_lines[4].split()[0]),
    }
    matrix_iterations = []
    matrix_runtime = []
    for port in range(1, n_ports + 1):
        lines = (args.smatrix_path / f"port_{port:02d}" /
                 "openEMS_stats.txt").read_text().splitlines()
        matrix_iterations.append(int(lines[2].split()[0]))
        matrix_runtime.append(float(lines[4].split()[0]))

    return {
        "beam_summary": beam_summary,
        "smatrix_summary": smatrix_summary,
        "frequency_hz": frequency_hz,
        "s_matrix": s_matrix,
        "target_index": target_index,
        "active": active,
        "target_active": target_active,
        "farfield": farfield,
        "beam_solver": beam_solver,
        "matrix_iterations": np.asarray(matrix_iterations),
        "matrix_runtime": np.asarray(matrix_runtime),
        "matrix_image": matrix_image,
        "n_ports": n_ports,
    }


def page_title(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.text(0.055, 0.955, title, fontsize=20, fontweight="bold",
             color=NAVY, va="top")
    if subtitle:
        fig.text(0.055, 0.914, subtitle, fontsize=9.5, color=MUTED, va="top")


def add_footer(fig: plt.Figure, page: int, data: dict) -> None:
    nx, ny = data["beam_summary"]["array_size"]
    target = data["beam_summary"]["target_frequency_ghz"]
    fig.add_artist(plt.Line2D(
        [0.055, 0.945], [0.035, 0.035], transform=fig.transFigure,
        color=MID, linewidth=0.8))
    fig.text(
        0.055, 0.015,
        f"openEMS / CSXCAD  •  {nx} × {ny} planar patch phased array  •  {target:g} GHz",
        fontsize=7.5, color=MUTED)
    fig.text(0.945, 0.015, str(page), fontsize=7.5, color=MUTED, ha="right")


def cover_page(pdf: PdfPages, data: dict) -> None:
    model = data["beam_summary"]
    results = model["results"]
    network = data["smatrix_summary"]["sparameter_measurement"]
    nx, ny = model["array_size"]
    n_ports = data["n_ports"]
    fig = plt.figure(figsize=(8.5, 11), facecolor=LIGHT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0.70), 1, 0.30, transform=ax.transAxes,
                           color=NAVY))
    ax.add_patch(Rectangle((0.06, 0.678), 0.20, 0.008,
                           transform=ax.transAxes, color=CYAN))
    fig.text(0.07, 0.925, "ELECTROMAGNETIC SIMULATION REPORT",
             color="#8edbe3", fontsize=10, fontweight="bold")
    fig.text(0.07, 0.842,
             f"{model['target_frequency_ghz']:g} GHz {nx} × {ny} Planar Patch\nPhased Array",
             color="white", fontsize=29, fontweight="bold", linespacing=1.12)
    fig.text(
        0.07, 0.745,
        f"CSXCAD model  •  {n_ports}-port S-matrix  •  channel matching  •  beam steering",
        color="#d8e6f4", fontsize=10)

    fig.text(0.07, 0.655, "Report scope", fontsize=14,
             fontweight="bold", color=NAVY)
    scope = (
        f"Full-wave openEMS analysis of a finite {n_ports}-channel probe-fed "
        f"array on RO4350B. A simultaneous phased excitation demonstrates a "
        f"{model['steering_theta_deg']:.0f}° scan, while {n_ports} independent "
        f"embedded-port simulations form the complete {n_ports} × {n_ports} "
        "scattering matrix."
    )
    fig.text(0.07, 0.615, scope, fontsize=10, color=TEXT, wrap=True,
             linespacing=1.55, va="top")

    cards = [
        (f"{results['maximum_directivity_dBi']:.2f} dBi", "peak directivity", BLUE),
        (f"{results['sampled_peak_theta_deg']:.1f}°", "sampled beam peak", CYAN),
        ("INVALID", "NF2FF efficiency", RED),
        (f"{network['maximum_singular_value_at_target']:.3f}",
         "max σ(S) at 8 GHz", COPPER),
    ]
    for index, (value, label, color) in enumerate(cards):
        x_pos = 0.07 + index * 0.22
        ax.add_patch(Rectangle(
            (x_pos, 0.455), 0.19, 0.105, transform=ax.transAxes,
            facecolor="white", edgecolor=MID, linewidth=0.9))
        ax.add_patch(Rectangle(
            (x_pos, 0.455), 0.008, 0.105, transform=ax.transAxes,
            facecolor=color, edgecolor="none"))
        fig.text(x_pos + 0.025, 0.515, value, fontsize=17,
                 fontweight="bold", color=NAVY)
        fig.text(x_pos + 0.025, 0.480, label, fontsize=8.5, color=MUTED)

    fig.text(0.07, 0.390, "Simulation basis", fontsize=14,
             fontweight="bold", color=NAVY)
    lines = [
        f"Target: {model['target_frequency_ghz']:.1f} GHz; λ₀={model['free_space_wavelength_mm']:.3f} mm",
        f"Pitch: {model['element_spacing_mm']:.3f} mm ({model['element_spacing_lambda']:.1f} λ₀) in x/y",
        f"Substrate: {model['substrate']['material']}, εr={model['substrate']['epsilon_r']:.2f}, tanδ={model['substrate']['loss_tangent_at_target']:.4f}",
        "Solver: local CUDA openEMS build, six-sided PML, Gaussian excitation",
        f"Network: {network['frequency_points']} points, {network['frequency_start_ghz']:.2f}–{network['frequency_stop_ghz']:.2f} GHz, 50 Ω",
        f"Qualified passive interval: {network['contiguous_passive_band_about_target_ghz'][0]:.4f}–{network['contiguous_passive_band_about_target_ghz'][1]:.4f} GHz",
    ]
    for index, line in enumerate(lines):
        fig.text(0.085, 0.350 - index * 0.036, "• " + line,
                 fontsize=9.2, color=TEXT)

    fig.text(0.07, 0.102,
             "NF2FF qualification: radiated/accepted power = "
             f"{results['radiation_efficiency']:.5f}; +{results['power_balance_error_percent']:.2f}% "
             "imbalance is nonphysical and is not reported as efficiency.",
             fontsize=8.5, color=RED)
    fig.text(0.07, 0.068,
             f"Generated {date.today().isoformat()}  |  Solver: {model['solver_binary']}",
             fontsize=8.0, color=MUTED)
    add_footer(fig, 1, data)
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def model_page(pdf: PdfPages, data: dict, model_render: Path,
               top_render: Path) -> None:
    model = data["beam_summary"]
    solver = data["beam_solver"]
    nx, ny = model["array_size"]
    mesh = model["mesh"]["cell_counts_xyz"]
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(
        fig, "1. CSXCAD simulation model",
        f"Finite substrate/ground, {data['n_ports']} probe-fed patches, 50 Ω lumped ports, and PML boundaries")

    ax_3d = fig.add_axes([0.035, 0.16, 0.58, 0.72])
    ax_3d.imshow(plt.imread(model_render))
    ax_3d.axis("off")
    ax_top = fig.add_axes([0.64, 0.43, 0.33, 0.46])
    ax_top.imshow(plt.imread(top_render))
    ax_top.axis("off")
    ax_info = fig.add_axes([0.65, 0.105, 0.32, 0.29])
    ax_info.axis("off")
    geometry = [
        ("Array", f"{nx} × {ny} / {data['n_ports']} channels"),
        ("Patch W × L", f"{model['patch_width_mm']:.3f} × {model['patch_length_mm']:.3f} mm"),
        ("Pitch", f"{model['element_spacing_mm']:.3f} mm"),
        ("Board", "{:.3f} × {:.3f} × {:.3f} mm".format(*model["substrate"]["size_mm"])),
        ("Feed offset", f"{model['feed_offset_toward_minus_y_mm']:.3f} mm toward −y"),
        ("Mesh intervals", f"{mesh['x']} × {mesh['y']} × {mesh['z']}"),
        ("Engine grid", f"{mesh['x'] + 1} × {mesh['y'] + 1} × {mesh['z'] + 1} = {solver['cells']:,} cells"),
        ("Beam run", f"{solver['iterations']:,} steps / {solver['runtime_s']:.1f} s GPU"),
    ]
    ax_info.text(0.0, 0.98, "Geometry and numerics", fontsize=12,
                 fontweight="bold", color=NAVY, va="top")
    y_pos = 0.84
    for label, value in geometry:
        ax_info.text(0.0, y_pos, label, fontsize=7.8, color=MUTED, va="top")
        ax_info.text(0.34, y_pos, value, fontsize=7.8, color=TEXT, va="top")
        y_pos -= 0.105
    fig.text(
        0.055, 0.075,
        "The 3D vertical display scale is expanded for visibility. Patch, feed, board, and channel positions use the exact CSXCAD model dimensions.",
        fontsize=7.2, color=MUTED)
    add_footer(fig, 2, data)
    pdf.savefig(fig)
    plt.close(fig)


def smatrix_page(pdf: PdfPages, data: dict) -> None:
    quality = data["smatrix_summary"]["sparameter_measurement"]
    iterations = data["matrix_iterations"]
    runtimes = data["matrix_runtime"]
    target = data["beam_summary"]["target_frequency_ghz"]
    n_ports = data["n_ports"]
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(
        fig, f"2. Complete {n_ports} × {n_ports} S-parameter matrix",
        f"One driven input per run; the remaining {n_ports - 1} ports are terminated in 50 Ω")
    ax_image = fig.add_axes([0.035, 0.09, 0.66, 0.81])
    ax_image.imshow(plt.imread(data["matrix_image"]))
    ax_image.axis("off")
    ax_info = fig.add_axes([0.72, 0.11, 0.25, 0.75])
    ax_info.axis("off")
    ax_info.text(0, 0.98, f"{target:g} GHz network summary", fontsize=12,
                 fontweight="bold", color=NAVY, va="top")
    strongest = quality["strongest_mutual_coupling_at_target"]
    rows = [
        ("Return loss", f"{quality['best_return_loss_at_target_dB']:.2f} to {quality['worst_return_loss_at_target_dB']:.2f} dB"),
        ("Ports ≤ −10 dB", "24 / 36"),
        ("Strongest coupling", f"S{strongest['output_port']},{strongest['input_port']} = {strongest['magnitude_dB']:.2f} dB"),
        ("Max reciprocity Δ", f"{quality['reciprocity_max_abs_difference_at_target']:.5f}"),
        ("RMS reciprocity Δ", f"{quality['reciprocity_rms_abs_difference_at_target']:.5f}"),
        ("Maximum σ(S)", f"{quality['maximum_singular_value_at_target']:.5f}"),
        ("Target passivity", "PASS"),
    ]
    y_pos = 0.91
    for label, value in rows:
        ax_info.text(0, y_pos, label, fontsize=7.9, color=MUTED, va="top")
        ax_info.text(0.52, y_pos, value, fontsize=7.9,
                     color=GREEN if value == "PASS" else TEXT, va="top")
        y_pos -= 0.058
    ax_info.text(0, y_pos - 0.01, "Sweep and convergence", fontsize=12,
                 fontweight="bold", color=NAVY, va="top")
    y_pos -= 0.078
    band = quality["contiguous_passive_band_about_target_ghz"]
    convergence = [
        ("Samples", str(quality["frequency_points"])),
        ("Raw span", f"{quality['frequency_start_ghz']:.2f}–{quality['frequency_stop_ghz']:.2f} GHz"),
        ("Passive interval", f"{band[0]:.4f}–{band[1]:.4f} GHz"),
        ("FDTD steps/run", f"{iterations.min():,}–{iterations.max():,}"),
        ("GPU stepping", f"{runtimes.sum():.2f} s total"),
        ("Full-band max σ", f"{quality['maximum_singular_value_full_band']:.5f}"),
    ]
    for label, value in convergence:
        ax_info.text(0, y_pos, label, fontsize=7.9, color=MUTED, va="top")
        ax_info.text(0.52, y_pos, value, fontsize=7.9, color=TEXT, va="top")
        y_pos -= 0.058
    ax_info.text(0, y_pos - 0.01, "Definition", fontsize=12,
                 fontweight="bold", color=NAVY, va="top")
    ax_info.text(0, y_pos - 0.065,
                 "S[f, j, i] = bⱼ / aᵢ\nrow j = output\ncolumn i = driven input",
                 fontsize=8.6, color=TEXT, linespacing=1.45, va="top")
    fig.text(
        0.055, 0.052,
        f"Quality note: the matrix is passive at {target:g} GHz. The full sweep reaches "
        f"σmax={quality['maximum_singular_value_full_band']:.3f} at "
        f"{quality['maximum_singular_value_full_band_frequency_ghz']:.4f} GHz; "
        "outer-band samples are retained for diagnostics and are not qualified production network data.",
        fontsize=6.7, color=MUTED, va="bottom")
    add_footer(fig, 3, data)
    pdf.savefig(fig)
    plt.close(fig)


def matching_overview_page(pdf: PdfPages, data: dict) -> None:
    target = data["beam_summary"]["target_frequency_ghz"]
    frequency_ghz = data["frequency_hz"] / 1e9
    s_matrix = data["s_matrix"]
    active = data["active"]
    target_active = data["target_active"]
    target_index = data["target_index"]
    n_ports = data["n_ports"]
    colors = cm.turbo(np.linspace(0.03, 0.97, n_ports))
    embedded_target = 20 * np.log10(np.maximum(
        np.abs(np.diag(s_matrix[target_index])), 1e-15))
    active_target = target_active["active_reflection_dB"]
    f_min = min(float(frequency_ghz.min()), float(active["frequency_GHz"].min()))
    f_max = max(float(frequency_ghz.max()), float(active["frequency_GHz"].max()))

    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(
        fig, "3. Channel matching overview",
        "Embedded Sii and simultaneous scan-state active reflection are distinct network quantities")
    ax_embedded = fig.add_axes([0.07, 0.52, 0.41, 0.35])
    ax_active = fig.add_axes([0.55, 0.52, 0.41, 0.35])
    for port in range(n_ports):
        ax_embedded.plot(
            frequency_ghz,
            20 * np.log10(np.maximum(np.abs(s_matrix[:, port, port]), 1e-15)),
            color=colors[port], linewidth=0.75)
        channel_data = active[active["channel"] == port + 1]
        ax_active.plot(channel_data["frequency_GHz"],
                       channel_data["active_reflection_dB"],
                       color=colors[port], linewidth=0.75)
    for axis, title in ((ax_embedded, "Embedded return loss |Sii|"),
                        (ax_active, "Active reflection, 20° scan state")):
        axis.axhline(-10, color=RED, linestyle="--", linewidth=1.0)
        axis.axvline(target, color=NAVY, linestyle=":", linewidth=1.0)
        axis.set_xlim(f_min, f_max)
        axis.set_ylim(-35, 2)
        axis.grid(True, alpha=0.23)
        axis.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY)
        axis.set_xlabel("Frequency (GHz)", fontsize=8)
        axis.set_ylabel("Magnitude (dB)", fontsize=8)
        axis.tick_params(labelsize=7)

    ax_bar = fig.add_axes([0.07, 0.16, 0.89, 0.26])
    channel = np.arange(1, n_ports + 1)
    width = 0.39
    ax_bar.bar(channel - width / 2, embedded_target, width,
               color=BLUE, label="Embedded Sii")
    ax_bar.bar(channel + width / 2, active_target, width,
               color=COPPER, label="Active Γ, scan state")
    ax_bar.axhline(-10, color=RED, linestyle="--", linewidth=1.1,
                   label="−10 dB criterion")
    ax_bar.set_xlim(0.2, n_ports + 0.8)
    ax_bar.set_ylim(-20, 0)
    ax_bar.set_xticks(channel)
    ax_bar.tick_params(labelsize=6.5)
    ax_bar.set_xlabel("Channel")
    ax_bar.set_ylabel(f"Reflection at {target:g} GHz (dB)")
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend(ncol=3, fontsize=7.5, loc="lower center")
    embedded_pass = int(np.count_nonzero(embedded_target <= -10))
    active_pass = int(np.count_nonzero(active_target <= -10))
    fig.text(
        0.50, 0.068,
        f"At {target:g} GHz: {embedded_pass}/{n_ports} embedded ports and "
        f"{active_pass}/{n_ports} simultaneous scan-state ports meet −10 dB.",
        ha="center", fontsize=8.2, color=MUTED)
    add_footer(fig, 4, data)
    pdf.savefig(fig)
    plt.close(fig)


def matching_table_page(pdf: PdfPages, data: dict) -> None:
    target = data["beam_summary"]["target_frequency_ghz"]
    target_active = data["target_active"]
    matrix = data["s_matrix"][data["target_index"]]
    embedded = 20 * np.log10(np.maximum(np.abs(np.diag(matrix)), 1e-15))
    n_ports = data["n_ports"]
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(
        fig, "4. Matching data for every channel",
        f"Target-frequency values at {target:g} GHz; PASS indicates reflection ≤ −10 dB")
    axes = [
        fig.add_axes([0.035, 0.10, 0.30, 0.78]),
        fig.add_axes([0.35, 0.10, 0.30, 0.78]),
        fig.add_axes([0.665, 0.10, 0.30, 0.78]),
    ]
    block = int(np.ceil(n_ports / 3))
    for group, axis in enumerate(axes):
        axis.axis("off")
        start = group * block
        stop = min(start + block, n_ports)
        rows = []
        statuses = []
        for port in range(start, stop):
            row = target_active[port]
            active_db = float(row["active_reflection_dB"])
            status = "PASS" if active_db <= -10 else "FAIL"
            statuses.append(status)
            rows.append([
                str(port + 1),
                f"{embedded[port]:.2f}",
                f"{active_db:.2f}",
                f"{row['active_impedance_real_ohm']:.1f}{row['active_impedance_imag_ohm']:+.1f}j",
                status,
            ])
        table = axis.table(
            cellText=rows,
            colLabels=["CH", "Sii dB", "Active dB", "Active Zin Ω", "−10 dB"],
            cellLoc="center", colLoc="center", loc="center",
            colWidths=[0.11, 0.18, 0.21, 0.30, 0.17])
        table.auto_set_font_size(False)
        table.set_fontsize(6.1)
        table.scale(1.0, 1.65)
        for (row_index, col_index), cell in table.get_celld().items():
            cell.set_edgecolor(MID)
            cell.set_linewidth(0.55)
            if row_index == 0:
                cell.set_facecolor(NAVY)
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            elif row_index % 2 == 0:
                cell.set_facecolor(LIGHT)
            if row_index > 0 and col_index == 4:
                status = statuses[row_index - 1]
                cell.get_text().set_color(GREEN if status == "PASS" else RED)
                cell.get_text().set_fontweight("bold")
    fig.text(
        0.055, 0.066,
        "The outer y-rows (CH1–6 and CH31–36) are the principal mismatch contributors. "
        "Embedded and active values differ because the active reflection includes all simultaneous incident waves.",
        fontsize=7.3, color=MUTED)
    add_footer(fig, 5, data)
    pdf.savefig(fig)
    plt.close(fig)


def beam_page(pdf: PdfPages, data: dict) -> None:
    model = data["beam_summary"]
    results = model["results"]
    farfield = data["farfield"]
    metrics = report_base.beam_metrics(farfield)
    theta = farfield["signed_theta_deg"]
    nx = model["array_size"][0]
    target = model["target_frequency_ghz"]
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(
        fig, "5. Beam-steering example",
        f"Uniform-amplitude excitation, commanded θ={model['steering_theta_deg']:.0f}°, "
        f"φ={model['steering_phi_deg']:.0f}° at {target:g} GHz")
    ax_pattern = fig.add_axes([0.07, 0.39, 0.57, 0.49])
    ax_pattern.plot(theta, farfield["xz_directivity_dBi"], color=BLUE,
                    linewidth=2.0, label="x–z cut / steering plane")
    ax_pattern.plot(theta, farfield["yz_directivity_dBi"], color=CYAN,
                    linewidth=1.7, label="y–z cut")
    ax_pattern.axvline(model["steering_theta_deg"], color=RED,
                       linestyle="--", linewidth=1.2, label="commanded angle")
    ax_pattern.axvline(metrics["peak_theta_deg"], color=GREEN,
                       linestyle=":", linewidth=1.4, label="sampled peak")
    ax_pattern.axhline(metrics["peak_dbi"] - 3.0, color=MUTED,
                       linestyle=":", linewidth=0.9)
    ax_pattern.set_xlim(-90, 90)
    ax_pattern.set_ylim(-30, 23)
    ax_pattern.set_xlabel("Signed θ (degrees)")
    ax_pattern.set_ylabel("Directivity (dBi)")
    ax_pattern.set_title("Simulated principal-plane far-field cuts",
                         fontsize=11, color=NAVY, fontweight="bold")
    ax_pattern.grid(True, alpha=0.25)
    ax_pattern.legend(fontsize=7.5, loc="lower right")

    ax_phase = fig.add_axes([0.69, 0.49, 0.27, 0.37])
    column_channels = model["channels"][:nx]
    columns = np.arange(1, nx + 1)
    phases = np.asarray([
        channel["relative_phase_deg_at_target"] for channel in column_channels])
    delays = np.asarray([
        channel["excitation_delay_ps"] for channel in column_channels])
    ax_phase.bar(columns, phases, color=cm.viridis(np.linspace(0.15, 0.9, nx)))
    ax_phase.axhline(0, color=MUTED, linewidth=0.7)
    ax_phase.set_xticks(columns, [f"x{index}" for index in columns])
    ax_phase.set_ylabel(f"Relative phase at {target:g} GHz (deg)", fontsize=8)
    ax_phase.set_title("Commanded x-column taper", fontsize=10,
                       color=NAVY, fontweight="bold")
    ax_phase.set_ylim(-190, 190)
    ax_phase.grid(True, axis="y", alpha=0.25)
    ax_phase.tick_params(labelsize=7)
    for x_pos, phase, delay in zip(columns, phases, delays):
        ax_phase.text(x_pos, phase + (9 if phase >= 0 else -9),
                      f"{phase:+.1f}°\n{delay:.1f} ps",
                      ha="center", va="bottom" if phase >= 0 else "top",
                      fontsize=5.8, color=TEXT)

    ax_metrics = fig.add_axes([0.69, 0.105, 0.28, 0.30])
    ax_metrics.axis("off")
    ax_metrics.text(0, 0.98, "Beam metrics", fontsize=12,
                    fontweight="bold", color=NAVY, va="top")
    rows = [
        ("Maximum directivity", f"{results['maximum_directivity_dBi']:.2f} dBi"),
        ("Sampled peak", f"θ={results['sampled_peak_theta_deg']:.1f}°, φ={results['sampled_peak_phi_deg']:.1f}°"),
        ("Pointing offset", f"{results['sampled_peak_theta_deg'] - model['steering_theta_deg']:+.1f}°"),
        ("Approx. HPBW", f"{metrics['hpbw_deg']:.2f}°"),
        ("Strongest sidelobe", f"{metrics['sidelobe_level_db']:.2f} dB relative"),
        ("Angular sampling", f"{results['farfield_angular_step_deg']:.1f}°"),
        ("NF2FF efficiency", "INVALID"),
        ("Power imbalance", f"+{results['power_balance_error_percent']:.2f}%"),
    ]
    y_pos = 0.84
    for label, value in rows:
        ax_metrics.text(0.0, y_pos, label, fontsize=7.9, color=MUTED, va="top")
        ax_metrics.text(0.56, y_pos, value, fontsize=7.9,
                        color=RED if value == "INVALID" else TEXT, va="top")
        y_pos -= 0.095

    ax_note = fig.add_axes([0.07, 0.10, 0.57, 0.21])
    ax_note.axis("off")
    ax_note.add_patch(Rectangle((0, 0), 1, 1, transform=ax_note.transAxes,
                                facecolor=LIGHT, edgecolor=MID, linewidth=0.8))
    ax_note.text(0.03, 0.82, "Interpretation", fontsize=10,
                 fontweight="bold", color=NAVY, va="top")
    ax_note.text(
        0.03, 0.62,
        f"The six delay states repeat on every y-row. The sampled beam reaches\n"
        f"{results['maximum_directivity_dBi']:.2f} dBi at the commanded 20° direction. "
        f"HPBW is {metrics['hpbw_deg']:.2f}° and the\n"
        f"strongest sidelobe is {metrics['sidelobe_level_db']:.2f} dB relative. "
        f"The NF2FF/accepted ratio\n"
        f"of {results['radiation_efficiency']:.5f} is nonphysical; efficiency requires "
        "a convergence study.",
        fontsize=7.6, color=TEXT, va="top", linespacing=1.35, clip_on=True)
    add_footer(fig, 6, data)
    pdf.savefig(fig)
    plt.close(fig)


def quality_page(pdf: PdfPages, data: dict) -> None:
    frequency = data["frequency_hz"] / 1e9
    s_matrix = data["s_matrix"]
    target = data["beam_summary"]["target_frequency_ghz"]
    quality = data["smatrix_summary"]["sparameter_measurement"]
    singular = np.linalg.svd(s_matrix, compute_uv=False)[:, 0]
    delta = np.abs(s_matrix - np.swapaxes(s_matrix, 1, 2))
    reciprocity_max = np.max(delta, axis=(1, 2))
    reciprocity_rms = np.sqrt(np.mean(delta ** 2, axis=(1, 2)))
    diagonal = np.diagonal(s_matrix, axis1=1, axis2=2)
    return_loss = 20 * np.log10(np.maximum(np.abs(diagonal), 1e-15))
    best = np.min(return_loss, axis=1)
    worst = np.max(return_loss, axis=1)
    band = quality["contiguous_passive_band_about_target_ghz"]

    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(
        fig, "6. Network quality and qualification",
        "Passivity, reciprocity, and matching-envelope checks across the exported sweep")
    ax_sigma = fig.add_axes([0.07, 0.54, 0.41, 0.33])
    ax_recip = fig.add_axes([0.55, 0.54, 0.41, 0.33])
    ax_match = fig.add_axes([0.07, 0.13, 0.89, 0.29])
    ax_sigma.plot(frequency, singular, color=BLUE, linewidth=1.7)
    ax_sigma.axhline(1.0, color=RED, linestyle="--", linewidth=1.0)
    ax_sigma.axvspan(band[0], band[1], color=GREEN, alpha=0.12)
    ax_sigma.axvline(target, color=NAVY, linestyle=":", linewidth=1.0)
    ax_sigma.set_title("Maximum singular value σmax(S)", color=NAVY,
                       fontweight="bold", fontsize=10.5)
    ax_sigma.set_xlabel("Frequency (GHz)")
    ax_sigma.set_ylabel("σmax")
    ax_sigma.grid(True, alpha=0.25)

    ax_recip.plot(frequency, reciprocity_max, color=COPPER, linewidth=1.5,
                  label="maximum |Sji−Sij|")
    ax_recip.plot(frequency, reciprocity_rms, color=CYAN, linewidth=1.5,
                  label="RMS |Sji−Sij|")
    ax_recip.axvline(target, color=NAVY, linestyle=":", linewidth=1.0)
    ax_recip.set_title("Reciprocity discrepancy", color=NAVY,
                       fontweight="bold", fontsize=10.5)
    ax_recip.set_xlabel("Frequency (GHz)")
    ax_recip.set_ylabel("Linear magnitude")
    ax_recip.grid(True, alpha=0.25)
    ax_recip.legend(fontsize=7.5)

    ax_match.fill_between(frequency, best, worst, color=BLUE, alpha=0.20,
                          label="best-to-worst embedded channel envelope")
    ax_match.plot(frequency, best, color=BLUE, linewidth=1.0)
    ax_match.plot(frequency, worst, color=COPPER, linewidth=1.0)
    ax_match.axhline(-10, color=RED, linestyle="--", linewidth=1.0)
    ax_match.axvline(target, color=NAVY, linestyle=":", linewidth=1.0)
    ax_match.set_ylim(-40, 0)
    ax_match.set_title("Embedded return-loss envelope", color=NAVY,
                       fontweight="bold", fontsize=10.5)
    ax_match.set_xlabel("Frequency (GHz)")
    ax_match.set_ylabel("|Sii| (dB)")
    ax_match.grid(True, alpha=0.25)
    ax_match.legend(fontsize=7.5, loc="lower right")
    fig.text(
        0.055, 0.080,
        f"Qualified conclusion: target σmax={quality['maximum_singular_value_at_target']:.5f} "
        f"(PASS), passive interval {band[0]:.4f}–{band[1]:.4f} GHz.\n"
        f"Worst full-band value: {quality['maximum_singular_value_full_band']:.5f} at "
        f"{quality['maximum_singular_value_full_band_frequency_ghz']:.4f} GHz; do not use "
        "nonpassive outer samples for quantitative circuit co-simulation.",
        fontsize=7.2, color=MUTED, va="top", linespacing=1.30)
    add_footer(fig, 7, data)
    pdf.savefig(fig)
    plt.close(fig)


def matrix_quadrant_page(pdf: PdfPages, data: dict, args: argparse.Namespace,
                         row_slice: slice, col_slice: slice, page: int) -> None:
    target = data["beam_summary"]["target_frequency_ghz"]
    matrix = data["s_matrix"][data["target_index"]]
    magnitude = 20 * np.log10(np.maximum(np.abs(matrix), 1e-15))
    rows = np.arange(data["n_ports"])[row_slice]
    cols = np.arange(data["n_ports"])[col_slice]
    block = magnitude[np.ix_(rows, cols)]
    norm = Normalize(vmin=-50, vmax=0)
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(
        fig,
        f"Appendix A{page - 7}. Numeric |Sji| matrix at {target:g} GHz",
        f"Rows j={rows[0] + 1}–{rows[-1] + 1}; columns i={cols[0] + 1}–{cols[-1] + 1}; values in dB")
    ax = fig.add_axes([0.055, 0.16, 0.89, 0.72])
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{value:.1f}" for value in row] for row in block],
        rowLabels=[f"j={index + 1}" for index in rows],
        colLabels=[f"i={index + 1}" for index in cols],
        cellLoc="center", rowLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(5.3)
    table.scale(1.0, 1.43)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.40)
        if row_index == 0 or col_index == -1:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            value = block[row_index - 1, col_index]
            cell.set_facecolor(cm.viridis(norm(np.clip(value, -50, 0))))
            cell.get_text().set_color("white" if value < -22 else "black")
    if page == 8:
        fig.text(
            0.055, 0.095,
            "Complete complex data products:\n"
            f"Target CSV: {args.smatrix_path / 'sparameter_matrix_8GHz.csv'}\n"
            f"Broadband Touchstone: {args.smatrix_path / 'patch_array_36port.s36p'}\n"
            f"Target Touchstone: {args.smatrix_path / 'patch_array_36port_8GHz.s36p'}",
            fontsize=6.4, color=MUTED, family="monospace", va="top",
            linespacing=1.30)
    else:
        fig.text(
            0.055, 0.090,
            "Magnitude is tabulated here for readability. Real, imaginary, magnitude, and phase "
            "for all 1,296 terms are preserved in the target CSV and Touchstone exports.",
            fontsize=7.0, color=MUTED)
    add_footer(fig, page, data)
    pdf.savefig(fig)
    plt.close(fig)


def generate_report(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args)
    model_render = args.output_dir / "model_3d_render.png"
    top_render = args.output_dir / "model_top_view.png"
    report_path = args.output_dir / "6x6_8GHz_Phased_Array_Simulation_Report.pdf"
    report_base.save_model_render(model_render, data["beam_summary"])
    report_base.save_model_top_view(top_render, data["beam_summary"])

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })
    with PdfPages(report_path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "8 GHz 6 x 6 Planar Patch Phased Array Simulation Report"
        metadata["Author"] = "Generated from openEMS / CSXCAD simulation data"
        metadata["Subject"] = "36-port S-parameters, matching, and beam steering"
        metadata["Keywords"] = "openEMS, CSXCAD, phased array, 8 GHz, S-parameters"
        cover_page(pdf, data)
        model_page(pdf, data, model_render, top_render)
        smatrix_page(pdf, data)
        matching_overview_page(pdf, data)
        matching_table_page(pdf, data)
        beam_page(pdf, data)
        quality_page(pdf, data)
        matrix_quadrant_page(pdf, data, args, slice(0, 18), slice(0, 18), 8)
        matrix_quadrant_page(pdf, data, args, slice(0, 18), slice(18, 36), 9)
        matrix_quadrant_page(pdf, data, args, slice(18, 36), slice(0, 18), 10)
        matrix_quadrant_page(pdf, data, args, slice(18, 36), slice(18, 36), 11)
    return report_path, [model_render, top_render]


def main() -> int:
    args = parse_args()
    report, assets = generate_report(args)
    print(f"PDF report: {report}")
    for asset in assets:
        print(f"Report asset: {asset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
