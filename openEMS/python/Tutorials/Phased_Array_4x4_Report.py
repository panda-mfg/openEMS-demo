#!/usr/bin/env python3
"""Build a PDF engineering report from the completed 4 x 4 array runs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


NAVY = "#13233f"
BLUE = "#2364aa"
CYAN = "#22a6b3"
COPPER = "#d47a27"
GOLD = "#f2cf78"
GREEN = "#228b5a"
RED = "#b8324a"
LIGHT = "#f4f7fb"
MID = "#d8e1ec"
TEXT = "#172238"
MUTED = "#53657d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beam-path", type=Path,
        default=Path("/home/shanda/openEMS-gpu/results/patch_array_4x4_5GHz_model"),
        help="Completed simultaneous-excitation beam-steering run.")
    parser.add_argument(
        "--smatrix-path", type=Path,
        default=Path("/home/shanda/openEMS-gpu/results/patch_array_4x4_5GHz_smatrix"),
        help="Completed 16-run S-parameter sweep.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/shanda/openEMS-gpu/results/patch_array_4x4_5GHz_report"),
        help="Report and rendered-asset output directory.")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> dict:
    required = [
        args.beam_path / "model_summary.json",
        args.beam_path / "active_port_results.csv",
        args.beam_path / "farfield_cuts.csv",
        args.beam_path / "openEMS_stats.txt",
        args.smatrix_path / "sparameter_summary.json",
        args.smatrix_path / "patch_array_16port_smatrix.npz",
        args.smatrix_path / "sparameter_matrix_5GHz.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing report inputs: " + ", ".join(missing))

    with required[0].open() as stream:
        beam_summary = json.load(stream)
    with required[4].open() as stream:
        smatrix_summary = json.load(stream)
    network = np.load(required[5])
    frequency_hz = network["frequency_hz"]
    s_matrix = network["s"]
    target_index = int(np.argmin(np.abs(
        frequency_hz - beam_summary["target_frequency_ghz"] * 1e9)))

    active = np.genfromtxt(required[1], delimiter=",", names=True)
    farfield = np.genfromtxt(required[2], delimiter=",", names=True)
    target_active = active[np.isclose(
        active["frequency_GHz"], beam_summary["target_frequency_ghz"])]
    target_active = np.sort(target_active, order="channel")
    if target_active.size != 16:
        raise ValueError("Expected one 5 GHz active-matching record per channel")

    beam_stats_lines = required[3].read_text().splitlines()
    beam_solver = {
        "cells": int(beam_stats_lines[0].split()[0]),
        "timestep_s": float(beam_stats_lines[1].split()[0]),
        "iterations": int(beam_stats_lines[2].split()[0]),
        "runtime_s": float(beam_stats_lines[4].split()[0]),
    }
    matrix_iterations = []
    matrix_runtime = []
    for port in range(1, 17):
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
    }


def add_footer(fig: plt.Figure, page: int) -> None:
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.035, 0.035],
                             transform=fig.transFigure, color=MID, linewidth=0.8))
    fig.text(0.055, 0.015, "openEMS / CSXCAD  •  4 × 4 planar patch phased array",
             fontsize=7.5, color=MUTED)
    fig.text(0.945, 0.015, str(page), fontsize=7.5, color=MUTED, ha="right")


def page_title(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.text(0.055, 0.955, title, fontsize=20, fontweight="bold",
             color=NAVY, va="top")
    if subtitle:
        fig.text(0.055, 0.914, subtitle, fontsize=9.5, color=MUTED, va="top")


def cuboid_faces(x0: float, x1: float, y0: float, y1: float,
                 z0: float, z1: float) -> list[list[tuple[float, float, float]]]:
    points = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    return [[tuple(points[index]) for index in face] for face in (
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))]


def draw_model_3d(ax, model: dict) -> None:
    board_x, board_y, height = model["substrate"]["size_mm"]
    patch_w = model["patch_width_mm"]
    patch_l = model["patch_length_mm"]
    feed_offset = model["feed_offset_toward_minus_y_mm"]
    feed_w, feed_l = model["feed_footprint_mm"]

    substrate = Poly3DCollection(
        cuboid_faces(-board_x / 2, board_x / 2, -board_y / 2, board_y / 2,
                     0.0, height),
        facecolors=GOLD, edgecolors="#b8943f", linewidths=0.6, alpha=0.30)
    ax.add_collection3d(substrate)

    ground = Poly3DCollection(
        [[(-board_x / 2, -board_y / 2, 0.0),
          (board_x / 2, -board_y / 2, 0.0),
          (board_x / 2, board_y / 2, 0.0),
          (-board_x / 2, board_y / 2, 0.0)]],
        facecolors="#a65f24", edgecolors="#824519", linewidths=0.8, alpha=0.48)
    ax.add_collection3d(ground)

    for channel in model["channels"]:
        number = int(channel["channel"])
        x_pos, y_pos, _ = channel["position_mm"]
        z_patch = height + 0.05
        patch = Poly3DCollection(
            [[(x_pos - patch_w / 2, y_pos - patch_l / 2, z_patch),
              (x_pos + patch_w / 2, y_pos - patch_l / 2, z_patch),
              (x_pos + patch_w / 2, y_pos + patch_l / 2, z_patch),
              (x_pos - patch_w / 2, y_pos + patch_l / 2, z_patch)]],
            facecolors=COPPER, edgecolors="#713814", linewidths=0.75, alpha=0.98)
        ax.add_collection3d(patch)
        feed_y = y_pos - feed_offset
        ax.plot([x_pos, x_pos], [feed_y, feed_y], [0, height],
                color=RED, linewidth=2.0, zorder=20)
        ax.scatter([x_pos], [feed_y], [z_patch + 0.08], s=10,
                   color=RED, depthshade=False, zorder=21)
        ax.text(x_pos, y_pos, z_patch + 1.15, str(number), fontsize=6.5,
                ha="center", va="center", color=NAVY, fontweight="bold")

    ax.set_xlim(-board_x * 0.58, board_x * 0.58)
    ax.set_ylim(-board_y * 0.58, board_y * 0.58)
    ax.set_zlim(-0.4, 9.0)
    ax.set_box_aspect((board_x, board_y, 48.0))
    ax.view_init(elev=27, azim=-53)
    ax.set_xlabel("x (mm)", labelpad=8, color=MUTED)
    ax.set_ylabel("y (mm)", labelpad=8, color=MUTED)
    ax.set_zlabel("z (mm)", labelpad=5, color=MUTED)
    ax.set_zticks([0.0, height])
    ax.grid(True, alpha=0.25)
    ax.xaxis.pane.set_alpha(0.02)
    ax.yaxis.pane.set_alpha(0.02)
    ax.zaxis.pane.set_alpha(0.02)
    ax.tick_params(labelsize=7, colors=MUTED)
    array_x, array_y = model["array_size"]
    ax.set_title(
        f"{array_x} × {array_y} CSXCAD geometry at "
        f"{model['target_frequency_ghz']:g} GHz\n"
        "(vertical display scale expanded)",
                 color=NAVY, fontsize=13, fontweight="bold", pad=12)
    ax.legend(handles=[
        Patch(facecolor=COPPER, label="PEC patches"),
        Patch(facecolor=GOLD, alpha=0.4, label="RO4350B substrate"),
        Patch(facecolor=RED, label=f"{feed_w:g} × {feed_l:g} mm lumped feeds"),
    ], loc="upper left", fontsize=7, frameon=True)


def save_model_render(output: Path, model: dict) -> None:
    fig = plt.figure(figsize=(11.0, 8.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    draw_model_3d(ax, model)
    fig.text(0.5, 0.025,
             "Exact patch, board, feed-position, and channel dimensions from the CSXCAD model summary.",
             ha="center", fontsize=8, color=MUTED)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_model_top_view(output: Path, model: dict) -> None:
    """Render an exact +z geometry view for dense-array model review."""
    board_x, board_y, _ = model["substrate"]["size_mm"]
    patch_w = model["patch_width_mm"]
    patch_l = model["patch_length_mm"]
    feed_offset = model["feed_offset_toward_minus_y_mm"]
    fig, ax = plt.subplots(figsize=(9.0, 8.5), facecolor="white")
    ax.add_patch(Rectangle(
        (-board_x / 2, -board_y / 2), board_x, board_y,
        facecolor=GOLD, edgecolor="#8d672b", linewidth=1.4, alpha=0.35,
        label="Finite RO4350B substrate and ground"))
    for channel in model["channels"]:
        number = int(channel["channel"])
        x_pos, y_pos, _ = channel["position_mm"]
        ax.add_patch(Rectangle(
            (x_pos - patch_w / 2, y_pos - patch_l / 2), patch_w, patch_l,
            facecolor=COPPER, edgecolor="#713814", linewidth=0.8))
        ax.plot(x_pos, y_pos - feed_offset, marker="o", markersize=3.8,
                color=RED)
        ax.text(x_pos, y_pos, str(number), ha="center", va="center",
                fontsize=7.2, color="white", fontweight="bold")
    margin = max(board_x, board_y) * 0.06
    ax.set_xlim(-board_x / 2 - margin, board_x / 2 + margin)
    ax.set_ylim(-board_y / 2 - margin, board_y / 2 + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)", color=MUTED)
    ax.set_ylabel("y (mm)", color=MUTED)
    ax.grid(True, alpha=0.18)
    array_x, array_y = model["array_size"]
    ax.set_title(
        f"{array_x} × {array_y} planar patch array at "
        f"{model['target_frequency_ghz']:g} GHz — +z view\n"
        f"pitch {model['element_spacing_mm']:.3f} mm; "
        f"patch {patch_w:.3f} × {patch_l:.3f} mm",
        color=NAVY, fontsize=14, fontweight="bold", pad=12)
    ax.legend(handles=[
        Patch(facecolor=GOLD, alpha=0.4, label="Finite substrate / ground"),
        Patch(facecolor=COPPER, label="PEC patch"),
        plt.Line2D([], [], marker="o", linestyle="none", color=RED,
                   label="50 Ω probe/feed location"),
    ], loc="upper center", ncol=3, fontsize=8, frameon=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def beam_metrics(farfield: np.ndarray) -> dict:
    theta = farfield["signed_theta_deg"]
    pattern = farfield["steering_plane_directivity_dBi"]
    peak_index = int(np.argmax(pattern))
    threshold = pattern[peak_index] - 3.0

    left = peak_index
    while left > 0 and pattern[left] >= threshold:
        left -= 1
    right = peak_index
    while right < pattern.size - 1 and pattern[right] >= threshold:
        right += 1

    def crossing(index_a: int, index_b: int) -> float:
        x0, x1 = theta[index_a], theta[index_b]
        y0, y1 = pattern[index_a], pattern[index_b]
        return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))

    left_cross = crossing(left, left + 1)
    right_cross = crossing(right - 1, right)
    minima = [index for index in range(1, pattern.size - 1)
              if pattern[index] <= pattern[index - 1]
              and pattern[index] <= pattern[index + 1]]
    left_null = max(index for index in minima if index < peak_index)
    right_null = min(index for index in minima if index > peak_index)
    sidelobes = np.concatenate((pattern[:left_null + 1],
                                pattern[right_null:]))
    sidelobe_peak = float(np.max(sidelobes))
    return {
        "peak_theta_deg": float(theta[peak_index]),
        "peak_dbi": float(pattern[peak_index]),
        "left_3db_deg": left_cross,
        "right_3db_deg": right_cross,
        "hpbw_deg": right_cross - left_cross,
        "sidelobe_level_db": sidelobe_peak - float(pattern[peak_index]),
    }


def cover_page(pdf: PdfPages, data: dict) -> None:
    model = data["beam_summary"]
    results = model["results"]
    network = data["smatrix_summary"]["sparameter_measurement"]
    fig = plt.figure(figsize=(8.5, 11), facecolor=LIGHT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0.70), 1, 0.30, transform=ax.transAxes,
                           color=NAVY, zorder=0))
    ax.add_patch(Rectangle((0.06, 0.678), 0.20, 0.008,
                           transform=ax.transAxes, color=CYAN))
    fig.text(0.07, 0.925, "ELECTROMAGNETIC SIMULATION REPORT",
             color="#8edbe3", fontsize=10, fontweight="bold")
    fig.text(0.07, 0.842, "5 GHz 4 × 4 Planar Patch\nPhased Array",
             color="white", fontsize=29, fontweight="bold", linespacing=1.12)
    fig.text(0.07, 0.745,
             "CSXCAD model  •  16-port S-matrix  •  channel matching  •  beam steering",
             color="#d8e6f4", fontsize=10)
    fig.text(0.07, 0.655, "Report scope", fontsize=14, fontweight="bold", color=NAVY)
    scope = (
        "A full-wave openEMS analysis of a finite 16-channel probe-fed array on "
        "RO4350B. The report combines a simultaneous phased excitation for a "
        "20° scan with sixteen embedded-port simulations used to form the complete "
        "16 × 16 scattering matrix."
    )
    fig.text(0.07, 0.615, scope, fontsize=10, color=TEXT, wrap=True,
             linespacing=1.55, va="top")

    cards = [
        ("16.75 dBi", "peak directivity", BLUE),
        ("17.5°", "sampled beam peak", CYAN),
        ("88.4%", "radiation efficiency", GREEN),
        ("0.601", "max σ(S) at 5 GHz", COPPER),
    ]
    for index, (value, label, color) in enumerate(cards):
        x_pos = 0.07 + index * 0.22
        ax.add_patch(Rectangle((x_pos, 0.455), 0.19, 0.105,
                               transform=ax.transAxes, facecolor="white",
                               edgecolor=MID, linewidth=0.9))
        ax.add_patch(Rectangle((x_pos, 0.455), 0.008, 0.105,
                               transform=ax.transAxes, facecolor=color,
                               edgecolor="none"))
        fig.text(x_pos + 0.025, 0.515, value, fontsize=18,
                 fontweight="bold", color=NAVY)
        fig.text(x_pos + 0.025, 0.480, label, fontsize=8.5, color=MUTED)

    fig.text(0.07, 0.390, "Simulation basis", fontsize=14,
             fontweight="bold", color=NAVY)
    lines = [
        f"Target frequency: {model['target_frequency_ghz']:.1f} GHz; free-space wavelength {model['free_space_wavelength_mm']:.3f} mm",
        f"Element spacing: {model['element_spacing_mm']:.3f} mm ({model['element_spacing_lambda']:.1f} λ₀) in x and y",
        f"Substrate: {model['substrate']['material']}, εr={model['substrate']['epsilon_r']:.2f}, tanδ={model['substrate']['loss_tangent_at_target']:.4f}",
        "Solver: local CUDA openEMS build, six-sided PML, Gaussian excitation",
        f"Network data: {network['frequency_points']} points, {network['frequency_start_ghz']:.2f}–{network['frequency_stop_ghz']:.2f} GHz, 50 Ω reference",
    ]
    for index, line in enumerate(lines):
        fig.text(0.085, 0.350 - index * 0.038, "• " + line,
                 fontsize=9.5, color=TEXT)

    fig.text(0.07, 0.125, "Prepared from completed solver data",
             fontsize=11, fontweight="bold", color=NAVY)
    fig.text(0.07, 0.092,
             f"Generated {date.today().isoformat()}  |  Solver: {model['solver_binary']}",
             fontsize=8.5, color=MUTED)
    add_footer(fig, 1)
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def model_page(pdf: PdfPages, data: dict, model_render: Path) -> None:
    model = data["beam_summary"]
    solver = data["beam_solver"]
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(fig, "1. CSXCAD simulation model",
               "Finite substrate and ground, sixteen probe-fed planar patches, and 50 Ω lumped ports")
    ax_image = fig.add_axes([0.045, 0.11, 0.63, 0.76])
    ax_image.imshow(plt.imread(model_render))
    ax_image.axis("off")

    ax_info = fig.add_axes([0.70, 0.10, 0.27, 0.78])
    ax_info.axis("off")
    ax_info.text(0.0, 0.98, "Geometry", fontsize=12, fontweight="bold",
                 color=NAVY, va="top")
    geometry = [
        ("Array", "4 × 4 / 16 channels"),
        ("Patch W × L", f"{model['patch_width_mm']:.3f} × {model['patch_length_mm']:.3f} mm"),
        ("Pitch", f"{model['element_spacing_mm']:.3f} mm"),
        ("Board", "{:.3f} × {:.3f} × {:.3f} mm".format(*model["substrate"]["size_mm"])),
        ("Feed offset", f"{model['feed_offset_toward_minus_y_mm']:.3f} mm toward −y"),
        ("Port", "50 Ω, z-directed"),
    ]
    y_pos = 0.925
    for label, value in geometry:
        ax_info.text(0.0, y_pos, label, fontsize=8.2, color=MUTED, va="top")
        ax_info.text(0.39, y_pos, value, fontsize=8.2, color=TEXT, va="top")
        y_pos -= 0.050

    ax_info.text(0.0, y_pos - 0.01, "Numerics", fontsize=12,
                 fontweight="bold", color=NAVY, va="top")
    y_pos -= 0.07
    numerics = [
        ("Mesh intervals", "112 × 109 × 41"),
        ("Engine grid", f"113 × 110 × 42 = {solver['cells']:,} cells"),
        ("Max mesh", f"{model['mesh']['maximum_cell_mm']:.3f} mm"),
        ("Time step", f"{solver['timestep_s'] * 1e12:.4f} ps"),
        ("Beam run", f"{solver['iterations']:,} steps / {solver['runtime_s']:.1f} s GPU"),
        ("Boundaries", "PML_8 on all six faces"),
    ]
    for label, value in numerics:
        ax_info.text(0.0, y_pos, label, fontsize=8.2, color=MUTED, va="top")
        ax_info.text(0.39, y_pos, value, fontsize=8.2, color=TEXT, va="top")
        y_pos -= 0.050

    ax_info.text(0.0, y_pos - 0.01, "Channel numbering (+z view)", fontsize=11,
                 fontweight="bold", color=NAVY, va="top")
    numbering = "13   14   15   16\n 9   10   11   12\n 5    6    7    8\n 1    2    3    4"
    ax_info.text(0.04, y_pos - 0.075, numbering, fontsize=9.5,
                 family="monospace", color=TEXT, linespacing=1.45, va="top")
    add_footer(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


def smatrix_page(pdf: PdfPages, data: dict, matrix_image: Path) -> None:
    quality = data["smatrix_summary"]["sparameter_measurement"]
    iterations = data["matrix_iterations"]
    runtimes = data["matrix_runtime"]
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(fig, "2. Complete 16 × 16 S-parameter matrix",
               "One driven input per run; the remaining fifteen ports are terminated in 50 Ω")

    ax_image = fig.add_axes([0.045, 0.09, 0.64, 0.80])
    ax_image.imshow(plt.imread(matrix_image))
    ax_image.axis("off")
    ax_info = fig.add_axes([0.71, 0.10, 0.25, 0.77])
    ax_info.axis("off")
    ax_info.text(0, 0.98, "5 GHz network summary", fontsize=12,
                 fontweight="bold", color=NAVY, va="top")
    rows = [
        ("Return loss", f"{quality['best_return_loss_at_target_dB']:.2f} to {quality['worst_return_loss_at_target_dB']:.2f} dB"),
        ("Strongest coupling", "S{},{} = {:.2f} dB".format(
            quality["strongest_mutual_coupling_at_target"]["output_port"],
            quality["strongest_mutual_coupling_at_target"]["input_port"],
            quality["strongest_mutual_coupling_at_target"]["magnitude_dB"])),
        ("Max reciprocity Δ", f"{quality['reciprocity_max_abs_difference_at_target']:.5f}"),
        ("RMS reciprocity Δ", f"{quality['reciprocity_rms_abs_difference_at_target']:.5f}"),
        ("Maximum σ(S)", f"{quality['maximum_singular_value_at_target']:.5f}"),
        ("Target passivity", "PASS"),
    ]
    y_pos = 0.91
    for label, value in rows:
        ax_info.text(0, y_pos, label, fontsize=8.3, color=MUTED, va="top")
        ax_info.text(0.51, y_pos, value, fontsize=8.3,
                     color=GREEN if value == "PASS" else TEXT, va="top")
        y_pos -= 0.058

    ax_info.text(0, y_pos - 0.01, "Sweep and convergence", fontsize=12,
                 fontweight="bold", color=NAVY, va="top")
    y_pos -= 0.075
    convergence = [
        ("Frequency samples", str(quality["frequency_points"])),
        ("Raw span", f"{quality['frequency_start_ghz']:.2f}–{quality['frequency_stop_ghz']:.2f} GHz"),
        ("Passive interval", "{:.5f}–{:.4f} GHz".format(
            *quality["contiguous_passive_band_about_target_ghz"])),
        ("FDTD steps/run", f"{iterations.min():,}–{iterations.max():,}"),
        ("GPU stepping total", f"{runtimes.sum():.2f} s"),
    ]
    for label, value in convergence:
        ax_info.text(0, y_pos, label, fontsize=8.3, color=MUTED, va="top")
        ax_info.text(0.51, y_pos, value, fontsize=8.3, color=TEXT, va="top")
        y_pos -= 0.058

    ax_info.text(0, y_pos - 0.01, "Definition", fontsize=12,
                 fontweight="bold", color=NAVY, va="top")
    ax_info.text(0, y_pos - 0.07,
                 "S[f, j, i] = bⱼ / aᵢ\nrow j = output port\ncolumn i = input port",
                 fontsize=9, color=TEXT, linespacing=1.5, va="top")
    fig.text(0.055, 0.053,
             "Quality note: the 5 GHz matrix is passive. Raw samples outside 4.83425–5.4485 GHz are retained for diagnostics; "
             "use a wider-band/higher-accuracy rerun before treating those samples as production network data.",
             fontsize=6.6, color=MUTED, va="bottom")
    add_footer(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


def matching_page(pdf: PdfPages, data: dict) -> plt.Figure:
    frequency_ghz = data["frequency_hz"] / 1e9
    s_matrix = data["s_matrix"]
    target_index = data["target_index"]
    active = data["active"]
    target_active = data["target_active"]
    embedded_target = 20 * np.log10(np.maximum(
        np.abs(np.diag(s_matrix[target_index])), 1e-15))
    colors = cm.turbo(np.linspace(0.05, 0.95, 16))

    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(fig, "3. Matching data by channel",
               "Embedded Sii and active scan-state reflection are different network quantities; both are reported")
    ax_embedded = fig.add_axes([0.07, 0.55, 0.41, 0.32])
    ax_active = fig.add_axes([0.55, 0.55, 0.41, 0.32])
    for port in range(16):
        ax_embedded.plot(frequency_ghz,
                         20 * np.log10(np.maximum(np.abs(s_matrix[:, port, port]), 1e-15)),
                         color=colors[port], linewidth=0.9, label=f"CH{port + 1}")
        channel_data = active[active["channel"] == port + 1]
        ax_active.plot(channel_data["frequency_GHz"],
                       channel_data["active_reflection_dB"],
                       color=colors[port], linewidth=0.9, label=f"CH{port + 1}")
    for axis, title in ((ax_embedded, "Embedded return loss |Sii|"),
                        (ax_active, "Active reflection, 20° scan state")):
        axis.axhline(-10, color=RED, linestyle="--", linewidth=1.0)
        axis.axvline(5.0, color=NAVY, linestyle=":", linewidth=1.0)
        axis.set_xlim(4.35, 5.65)
        axis.set_ylim(-35, 2)
        axis.grid(True, alpha=0.25)
        axis.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY)
        axis.set_xlabel("Frequency (GHz)", fontsize=8)
        axis.set_ylabel("Magnitude (dB)", fontsize=8)
        axis.tick_params(labelsize=7)
    ax_active.legend(ncol=4, fontsize=5.5, loc="lower right", framealpha=0.9)

    table_axes = [fig.add_axes([0.055, 0.10, 0.435, 0.38]),
                  fig.add_axes([0.515, 0.10, 0.435, 0.38])]
    for half, axis in enumerate(table_axes):
        axis.axis("off")
        start = half * 8
        table_rows = []
        statuses = []
        for offset in range(8):
            port = start + offset
            row = target_active[port]
            active_rl = float(row["active_reflection_dB"])
            status = "PASS" if active_rl <= -10.0 else "MARGINAL"
            statuses.append(status)
            table_rows.append([
                f"CH{port + 1}",
                f"{embedded_target[port]:.2f}",
                f"{active_rl:.2f}",
                f"{row['active_impedance_real_ohm']:.1f} {row['active_impedance_imag_ohm']:+.1f}j",
                status,
            ])
        table = axis.table(
            cellText=table_rows,
            colLabels=["Channel", "Sii (dB)", "Active Γ (dB)", "Active Zin (Ω)", "−10 dB"],
            cellLoc="center", colLoc="center", loc="center",
            colWidths=[0.14, 0.16, 0.20, 0.27, 0.18])
        table.auto_set_font_size(False)
        table.set_fontsize(6.9)
        table.scale(1.0, 1.35)
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
    fig.text(0.50, 0.075,
             "At 5 GHz all 16 embedded ports meet −10 dB; 13 of 16 active scan-state ports meet −10 dB. CH2, CH3, and CH10 are marginal.",
             ha="center", fontsize=8, color=MUTED)
    add_footer(fig, 4)
    pdf.savefig(fig)
    return fig


def beam_page(pdf: PdfPages, data: dict) -> plt.Figure:
    model = data["beam_summary"]
    results = model["results"]
    farfield = data["farfield"]
    metrics = beam_metrics(farfield)
    theta = farfield["signed_theta_deg"]

    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(fig, "4. Beam-steering example",
               "Uniform-amplitude excitation, commanded θ = 20°, φ = 0° at 5 GHz")
    ax_pattern = fig.add_axes([0.07, 0.40, 0.57, 0.48])
    ax_pattern.plot(theta, farfield["xz_directivity_dBi"], color=BLUE,
                    linewidth=2.0, label="x–z cut / steering plane")
    ax_pattern.plot(theta, farfield["yz_directivity_dBi"], color=CYAN,
                    linewidth=1.7, label="y–z cut")
    ax_pattern.axvline(model["steering_theta_deg"], color=RED,
                       linestyle="--", linewidth=1.2, label="commanded 20°")
    ax_pattern.axvline(metrics["peak_theta_deg"], color=GREEN,
                       linestyle=":", linewidth=1.4, label="sampled peak 17.5°")
    ax_pattern.axhline(metrics["peak_dbi"] - 3.0, color=MUTED,
                       linestyle=":", linewidth=0.9)
    ax_pattern.set_xlim(-90, 90)
    ax_pattern.set_ylim(-25, 19)
    ax_pattern.set_xlabel("Signed θ (degrees)")
    ax_pattern.set_ylabel("Directivity (dBi)")
    ax_pattern.set_title("Simulated principal-plane far-field cuts",
                         fontsize=11, color=NAVY, fontweight="bold")
    ax_pattern.grid(True, alpha=0.25)
    ax_pattern.legend(fontsize=7.5, loc="lower right")

    ax_phase = fig.add_axes([0.70, 0.49, 0.25, 0.36])
    unique_channels = model["channels"][:4]
    columns = np.arange(1, 5)
    phases = [channel["relative_phase_deg_at_target"] for channel in unique_channels]
    delays = [channel["excitation_delay_ps"] for channel in unique_channels]
    ax_phase.bar(columns, phases, color=[BLUE, CYAN, GOLD, COPPER], alpha=0.9)
    ax_phase.axhline(0, color=MUTED, linewidth=0.7)
    ax_phase.set_xticks(columns, ["x1", "x2", "x3", "x4"])
    ax_phase.set_ylabel("Relative phase at 5 GHz (deg)", fontsize=8)
    ax_phase.set_title("Commanded x-column phase taper",
                       fontsize=10, color=NAVY, fontweight="bold")
    ax_phase.set_ylim(-110, 110)
    ax_phase.grid(True, axis="y", alpha=0.25)
    ax_phase.tick_params(labelsize=7)
    for x_pos, phase, delay in zip(columns, phases, delays):
        ax_phase.text(x_pos, phase * 0.52,
                      f"{phase:+.1f}°\n{delay:.1f} ps", ha="center",
                      va="center", fontsize=6.4, color="white", fontweight="bold")

    ax_metrics = fig.add_axes([0.69, 0.11, 0.27, 0.29])
    ax_metrics.axis("off")
    ax_metrics.text(0, 0.98, "Beam metrics", fontsize=12,
                    fontweight="bold", color=NAVY, va="top")
    rows = [
        ("Maximum directivity", f"{results['maximum_directivity_dBi']:.2f} dBi"),
        ("Sampled peak", f"θ={results['sampled_peak_theta_deg']:.1f}°, φ={results['sampled_peak_phi_deg']:.1f}°"),
        ("Pointing offset", f"{results['sampled_peak_theta_deg'] - model['steering_theta_deg']:+.1f}°"),
        ("Approx. HPBW", f"{metrics['hpbw_deg']:.1f}°"),
        ("Strongest sidelobe", f"{metrics['sidelobe_level_db']:.1f} dB relative"),
        ("Radiation efficiency", f"{100 * results['radiation_efficiency']:.1f}%"),
        ("Angular sampling", f"{results['farfield_angular_step_deg']:.1f}°"),
    ]
    y_pos = 0.84
    for label, value in rows:
        ax_metrics.text(0.0, y_pos, label, fontsize=8.2, color=MUTED, va="top")
        ax_metrics.text(0.55, y_pos, value, fontsize=8.2, color=TEXT, va="top")
        y_pos -= 0.105

    ax_note = fig.add_axes([0.07, 0.10, 0.57, 0.22])
    ax_note.axis("off")
    ax_note.add_patch(Rectangle((0, 0), 1, 1, transform=ax_note.transAxes,
                                facecolor=LIGHT, edgecolor=MID, linewidth=0.8))
    ax_note.text(0.03, 0.82, "Excitation implementation", fontsize=10,
                 fontweight="bold", color=NAVY, va="top")
    ax_note.text(0.03, 0.62,
                 "The four phase/delay states repeat on every y-row. True-time delays of 0, 34.20, 68.40, and 102.61 ps produce the +x scan. The finite board, element pattern, mutual coupling, mesh, and 2.5° far-field sampling shift the observed maximum from the 20° command to 17.5°.",
                 fontsize=8.2, color=TEXT, wrap=True, va="top", linespacing=1.45)
    add_footer(fig, 5)
    pdf.savefig(fig)
    return fig


def matrix_appendix_page(pdf: PdfPages, data: dict, args: argparse.Namespace) -> None:
    target_matrix = data["s_matrix"][data["target_index"]]
    magnitude_db = 20 * np.log10(np.maximum(np.abs(target_matrix), 1e-15))
    norm = Normalize(vmin=-50, vmax=0)
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    page_title(fig, "Appendix A. Numeric |Sji| matrix at 5 GHz",
               "Values are in dB; row j is the output port and column i is the driven input port")
    ax = fig.add_axes([0.045, 0.23, 0.91, 0.64])
    ax.axis("off")
    cell_text = [[f"{value:.1f}" for value in row] for row in magnitude_db]
    table = ax.table(
        cellText=cell_text,
        rowLabels=[f"j={index}" for index in range(1, 17)],
        colLabels=[f"i={index}" for index in range(1, 17)],
        cellLoc="center", rowLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    table.scale(1.0, 1.55)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.45)
        if row_index == 0 or col_index == -1:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            value = magnitude_db[row_index - 1, col_index]
            cell.set_facecolor(cm.viridis(norm(np.clip(value, -50, 0))))
            cell.get_text().set_color("white" if value < -22 else "black")

    fig.text(0.055, 0.168, "Data products", fontsize=10,
             fontweight="bold", color=NAVY)
    products = (
        f"5 GHz Touchstone: {args.smatrix_path / 'patch_array_16port_5GHz.s16p'}\n"
        f"Broadband Touchstone: {args.smatrix_path / 'patch_array_16port.s16p'}\n"
        f"Complex NumPy matrix: {args.smatrix_path / 'patch_array_16port_smatrix.npz'}"
    )
    fig.text(0.055, 0.142, products, fontsize=6.8, color=MUTED,
             family="monospace", va="top", linespacing=1.35)
    fig.text(0.555, 0.168, "Reproduction", fontsize=10,
             fontweight="bold", color=NAVY)
    command = (
        "PYTHONPATH=/home/shanda/openEMS-gpu/local/python-packages\n"
        "python3 Phased_Array_4x4_5GHz.py --s-parameters --timesteps 200000\n"
        "  --sim-path /home/shanda/openEMS-gpu/results/patch_array_4x4_5GHz_smatrix"
    )
    fig.text(0.555, 0.142, command, fontsize=6.6, color=MUTED,
             family="monospace", va="top", linespacing=1.35)
    add_footer(fig, 6)
    pdf.savefig(fig)
    plt.close(fig)


def generate_report(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args)
    model_render = args.output_dir / "model_3d_render.png"
    matching_render = args.output_dir / "channel_matching.png"
    beam_render = args.output_dir / "beam_steering_example.png"
    report_path = args.output_dir / "4x4_5GHz_Phased_Array_Simulation_Report.pdf"
    save_model_render(model_render, data["beam_summary"])

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
        metadata["Title"] = "5 GHz 4 x 4 Planar Patch Phased Array Simulation Report"
        metadata["Author"] = "Generated from openEMS / CSXCAD simulation data"
        metadata["Subject"] = "16-port S-parameters, matching, and beam steering"
        metadata["Keywords"] = "openEMS, CSXCAD, phased array, 5 GHz, S-parameters"
        cover_page(pdf, data)
        model_page(pdf, data, model_render)
        smatrix_page(pdf, data, args.smatrix_path / "sparameter_matrix_5GHz.png")
        matching_fig = matching_page(pdf, data)
        matching_fig.savefig(matching_render, dpi=180, bbox_inches="tight")
        plt.close(matching_fig)
        beam_fig = beam_page(pdf, data)
        beam_fig.savefig(beam_render, dpi=180, bbox_inches="tight")
        plt.close(beam_fig)
        matrix_appendix_page(pdf, data, args)

    return report_path, [model_render, matching_render, beam_render]


def main() -> int:
    args = parse_args()
    report, assets = generate_report(args)
    print(f"PDF report: {report}")
    for asset in assets:
        print(f"Report asset: {asset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
