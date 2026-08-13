#!/usr/bin/env python3
"""Full-wave openEMS model of the Si-Zhu-Sun CPW-fed dual-band antenna."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0


PAPER_URL = (
    "https://www.researchgate.net/publication/258788884_"
    "A_Compact_Planar_and_CPW-Fed_Metamaterial-Inspired_Dual-Band_Antenna"
)
DOI = "10.1109/LAWP.2013.2249037"

UNIT = 1e-3  # Drawing units are millimetres.
SWEEP_LOW = 1.0e9
SWEEP_HIGH = 5.0e9
PAPER_BANDS = ((2.595e9, 2.654e9), (3.185e9, 4.245e9))
FARFIELD_FREQUENCIES = (2.6e9, 3.6e9)

# Published dimensions (Fig. 1 and Section III), millimetres.
BOARD_WIDTH = 31.7
BOARD_LENGTH = 27.0
SUBSTRATE_THICKNESS = 1.6
COPPER_THICKNESS_M = 35e-6
FR4_EPSILON = 4.4
FR4_LOSS_TANGENT = 0.088 / 4.4

R1 = 3.1
R2 = 4.5
R3 = 7.5
TRACE_WIDTH = 0.6
SPLIT_WIDTH = 0.5
FEED_WIDTH = 1.3
CPW_GAP = 0.2
GROUND_GAP = 0.15
GROUND_TOP_BASE = 2.0
GROUND_BOTTOM_BASE = 15.0
GROUND_HEIGHT = 10.0
TAPER_ALPHA = 0.8
TAPER_WIDTH_1 = TAPER_ALPHA * FEED_WIDTH
TAPER_WIDTH_2 = TAPER_ALPHA**2 * FEED_WIDTH

# In Fig. 1, h is the edge-to-edge clearance between the top of the CPW
# grounds and the outside edge of the r3 trace.  The extra half trace width
# is therefore required when locating the ring by its centreline radius.
RING_CENTER_X = 0.0
RING_CENTER_Y = GROUND_HEIGHT + GROUND_GAP + R3 + TRACE_WIDTH / 2.0
COPPER_Z = SUBSTRATE_THICKNESS

# The paper specifies an edge-mounted 50-ohm SMA but not its internal geometry.
# This compact PTFE coax is a documented simulation launch, not a paper dimension.
COAX_START_Y = -12.0
COAX_STOP_Y = 0.25
COAX_CENTER_Z = COPPER_Z
COAX_INNER_RADIUS = 0.30
COAX_OUTER_INNER_RADIUS = 1.00
COAX_OUTER_RADIUS = 1.20
COAX_EPSILON = 2.1
COAX_REFERENCE_PLANE_Y = 0.0

BOX_X = (-62.0, 62.0)
BOX_Y = (-45.0, 72.0)
BOX_Z = (-50.0, 54.0)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--engine",
        choices=("gpu", "multithreaded", "basic"),
        default=os.environ.get("OPENEMS_ENGINE", "gpu"),
    )
    parser.add_argument(
        "--gpu-device", type=int, default=int(os.environ.get("OPENEMS_GPU_DEVICE", "0"))
    )
    parser.add_argument(
        "--gpu-kernel", default=os.environ.get("OPENEMS_GPU_KERNEL", "auto")
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--nr-ts", type=int, default=120_000)
    parser.add_argument("--end-criteria", type=float, default=1e-5)
    parser.add_argument(
        "--mesh-resolution-mm",
        type=float,
        default=0.20,
        help="fine x/y cell size over the antenna and connector",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="write CSXCAD XML and previews without running FDTD",
    )
    parser.add_argument(
        "--post-only",
        action="store_true",
        help="post-process an existing solver run",
    )
    parser.add_argument(
        "--skip-farfield",
        action="store_true",
        help="calculate only the coaxial-port response",
    )
    parser.add_argument(
        "--pec-copper",
        action="store_true",
        help="use a PEC sheet instead of the published 35 um finite-conductivity copper",
    )
    return parser.parse_args()


def annular_arc_polygon(
    radius: float, width: float, start_angle: float, stop_angle: float
) -> np.ndarray:
    """Create a constant-width annular arc as a closed x/y polygon."""
    angular_span = abs(stop_angle - start_angle)
    count = max(96, int(math.ceil(angular_span / (2.0 * math.pi) * 720)))
    theta = np.linspace(start_angle, stop_angle, count)
    outer_radius = radius + width / 2.0
    inner_radius = radius - width / 2.0
    outer = np.column_stack(
        (
            RING_CENTER_X + outer_radius * np.cos(theta),
            RING_CENTER_Y + outer_radius * np.sin(theta),
        )
    )
    inner = np.column_stack(
        (
            RING_CENTER_X + inner_radius * np.cos(theta[::-1]),
            RING_CENTER_Y + inner_radius * np.sin(theta[::-1]),
        )
    )
    return np.vstack((outer, inner))


def radiator_polygons() -> list[tuple[str, np.ndarray]]:
    """Return the paper's connected CRR/SRR and three-section taper."""
    right_half_gap = SPLIT_WIDTH / (2.0 * R1)
    left_half_gap = SPLIT_WIDTH / (2.0 * R2)
    polygons: list[tuple[str, np.ndarray]] = [
        ("outer_closed_ring", annular_arc_polygon(R3, TRACE_WIDTH, 0.0, 2.0 * math.pi)),
        (
            "middle_split_ring",
            annular_arc_polygon(
                R2,
                TRACE_WIDTH,
                math.pi + left_half_gap,
                3.0 * math.pi - left_half_gap,
            ),
        ),
        (
            "inner_split_ring",
            annular_arc_polygon(
                R1,
                TRACE_WIDTH,
                right_half_gap,
                2.0 * math.pi - right_half_gap,
            ),
        ),
    ]

    # The line is narrowed at each ring exactly as w_n = alpha^n * w_f.
    sections = (
        ("cpw_signal_wf", FEED_WIDTH, 0.0, RING_CENTER_Y - R3),
        ("taper_w1", TAPER_WIDTH_1, RING_CENTER_Y - R3, RING_CENTER_Y - R2),
        ("taper_w2", TAPER_WIDTH_2, RING_CENTER_Y - R2, RING_CENTER_Y - R1),
    )
    for name, width, y0, y1 in sections:
        polygons.append(
            (
                name,
                np.array(
                    [
                        [-width / 2.0, y0],
                        [width / 2.0, y0],
                        [width / 2.0, y1],
                        [-width / 2.0, y1],
                    ]
                ),
            )
        )
    return polygons


def ground_polygons() -> list[tuple[str, np.ndarray]]:
    inner = FEED_WIDTH / 2.0 + CPW_GAP
    right = np.array(
        [
            [inner, 0.0],
            [inner + GROUND_BOTTOM_BASE, 0.0],
            [inner + GROUND_TOP_BASE, GROUND_HEIGHT],
            [inner, GROUND_HEIGHT],
        ]
    )
    left = right.copy()
    left[:, 0] *= -1.0
    left = left[::-1]
    return [("left_ground", left), ("right_ground", right)]


def all_planar_polygons() -> list[tuple[str, np.ndarray]]:
    return ground_polygons() + radiator_polygons()


def add_mesh(mesh, fine_resolution: float) -> None:
    mesh.AddLine("x", BOX_X)
    mesh.AddLine("y", BOX_Y)
    mesh.AddLine("z", BOX_Z)

    fine_x = np.arange(-BOARD_WIDTH / 2.0 - 0.4, BOARD_WIDTH / 2.0 + 0.401, fine_resolution)
    fine_y = np.arange(COAX_START_Y - 0.4, BOARD_LENGTH + 0.401, fine_resolution)
    mesh.AddLine("x", fine_x)
    mesh.AddLine("y", fine_y)

    geometry_x = [
        -BOARD_WIDTH / 2.0,
        BOARD_WIDTH / 2.0,
        -FEED_WIDTH / 2.0 - CPW_GAP,
        -FEED_WIDTH / 2.0,
        FEED_WIDTH / 2.0,
        FEED_WIDTH / 2.0 + CPW_GAP,
        -COAX_OUTER_RADIUS,
        -COAX_OUTER_INNER_RADIUS,
        -COAX_INNER_RADIUS,
        COAX_INNER_RADIUS,
        COAX_OUTER_INNER_RADIUS,
        COAX_OUTER_RADIUS,
    ]
    for radius in (R1, R2, R3):
        geometry_x.extend(
            [
                RING_CENTER_X - radius - TRACE_WIDTH / 2.0,
                RING_CENTER_X - radius + TRACE_WIDTH / 2.0,
                RING_CENTER_X + radius - TRACE_WIDTH / 2.0,
                RING_CENTER_X + radius + TRACE_WIDTH / 2.0,
            ]
        )
    mesh.AddLine("x", geometry_x)
    mesh.AddLine(
        "y",
        [
            COAX_START_Y,
            COAX_STOP_Y,
            0.0,
            GROUND_HEIGHT,
            GROUND_HEIGHT + GROUND_GAP,
            RING_CENTER_Y,
            BOARD_LENGTH,
        ],
    )

    coax_z = np.arange(
        COAX_CENTER_Z - COAX_OUTER_RADIUS - 0.15,
        COAX_CENTER_Z + COAX_OUTER_RADIUS + 0.151,
        0.15,
    )
    mesh.AddLine("z", coax_z)
    mesh.AddLine(
        "z",
        [
            0.0,
            COPPER_Z,
            COAX_CENTER_Z - COAX_OUTER_RADIUS,
            COAX_CENTER_Z - COAX_OUTER_INNER_RADIUS,
            COAX_CENTER_Z - COAX_INNER_RADIUS,
            COAX_CENTER_Z + COAX_INNER_RADIUS,
            COAX_CENTER_Z + COAX_OUTER_INNER_RADIUS,
            COAX_CENTER_Z + COAX_OUTER_RADIUS,
        ],
    )

    maximum_cell = C0 / SWEEP_HIGH / UNIT / math.sqrt(FR4_EPSILON) / 18.0
    mesh.SmoothMeshLines("all", maximum_cell, ratio=1.35)


def add_sheet_polygon(sheet, polygon: np.ndarray, priority: int = 10) -> None:
    sheet.AddPolygon(
        [polygon[:, 0], polygon[:, 1]],
        norm_dir=2,
        elevation=COPPER_Z,
        priority=priority,
    )


def geometry_metadata(args: argparse.Namespace, mesh) -> dict:
    mesh_lines = {axis: int(len(mesh.GetLines(axis))) for axis in "xyz"}
    return {
        "source": {
            "title": "A Compact, Planar, and CPW-Fed Metamaterial-Inspired Dual-Band Antenna",
            "authors": ["Li-Ming Si", "Weiren Zhu", "Hou-Jun Sun"],
            "journal": "IEEE Antennas and Wireless Propagation Letters 12 (2013), 305-308",
            "doi": DOI,
            "url": PAPER_URL,
        },
        "paper_geometry_mm": {
            "board_D_by_L": [BOARD_WIDTH, BOARD_LENGTH],
            "substrate_thickness": SUBSTRATE_THICKNESS,
            "copper_thickness": COPPER_THICKNESS_M / UNIT,
            "ring_radii_r1_r2_r3": [R1, R2, R3],
            "ring_width_w": TRACE_WIDTH,
            "split_width_s": SPLIT_WIDTH,
            "cpw_signal_width_wf": FEED_WIDTH,
            "cpw_gap_g": CPW_GAP,
            "ground_to_feed_gap_h": GROUND_GAP,
            "ground_to_resonator_edge_clearance_h": GROUND_GAP,
            "ground_bases_d1_d2": [GROUND_TOP_BASE, GROUND_BOTTOM_BASE],
            "ground_height_l": GROUND_HEIGHT,
            "taper_alpha": TAPER_ALPHA,
            "taper_widths_w1_w2": [TAPER_WIDTH_1, TAPER_WIDTH_2],
        },
        "materials": {
            "fr4_complex_relative_permittivity_as_printed": "4.4 + 0.088i",
            "fr4_epsilon_real": FR4_EPSILON,
            "fr4_loss_tangent": FR4_LOSS_TANGENT,
            "copper_conductivity_s_per_m": None if args.pec_copper else 56e6,
            "copper_model": "PEC sheet" if args.pec_copper else "35 um conducting sheet",
        },
        "paper_targets": {
            "minus_10db_bands_hz": [list(band) for band in PAPER_BANDS],
            "mode_assignment": {"2.6 GHz": "inner SRR", "3.6 GHz": "outer CRR"},
            "reported_efficiency": {"2.6 GHz": 0.793, "3.6 GHz": 0.956},
        },
        "simulation": {
            "sweep_hz": [SWEEP_LOW, SWEEP_HIGH],
            "boundary_conditions": ["PML_8"] * 6,
            "nr_ts": args.nr_ts,
            "end_criteria": args.end_criteria,
            "fine_mesh_resolution_mm": args.mesh_resolution_mm,
            "mesh_lines": mesh_lines,
            "mesh_cells": int(np.prod([mesh_lines[axis] - 1 for axis in "xyz"])),
            "ring_center_mm": [RING_CENTER_X, RING_CENTER_Y, COPPER_Z],
            "actual_ground_to_resonator_edge_clearance_mm": (
                RING_CENTER_Y - R3 - TRACE_WIDTH / 2.0 - GROUND_HEIGHT
            ),
            "port": "edge-launch coaxial TEM port referenced to 50 ohm at board edge",
            "connector_assumption": {
                "reason": "the paper specifies SMA termination but not connector dimensions",
                "ptfe_epsilon": COAX_EPSILON,
                "inner_radius_mm": COAX_INNER_RADIUS,
                "outer_inner_radius_mm": COAX_OUTER_INNER_RADIUS,
                "outer_radius_mm": COAX_OUTER_RADIUS,
                "length_mm": COAX_STOP_Y - COAX_START_Y,
            },
        },
    }


def build_model(args: argparse.Namespace):
    excitation_center = 0.5 * (SWEEP_LOW + SWEEP_HIGH)
    excitation_cutoff = 0.5 * (SWEEP_HIGH - SWEEP_LOW)
    fdtd = openEMS(NrTS=args.nr_ts, EndCriteria=args.end_criteria)
    fdtd.SetGaussExcite(excitation_center, excitation_cutoff)
    fdtd.SetBoundaryCond(["PML_8"] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(UNIT)
    add_mesh(mesh, args.mesh_resolution_mm)

    fr4_kappa = (
        2.0
        * math.pi
        * excitation_center
        * EPS0
        * FR4_EPSILON
        * FR4_LOSS_TANGENT
    )
    fr4 = csx.AddMaterial("FR4", epsilon=FR4_EPSILON, kappa=fr4_kappa)
    fr4.AddBox(
        [-BOARD_WIDTH / 2.0, 0.0, 0.0],
        [BOARD_WIDTH / 2.0, BOARD_LENGTH, SUBSTRATE_THICKNESS],
        priority=1,
    )

    if args.pec_copper:
        copper = csx.AddMetal("top_copper_PEC")
    else:
        copper = csx.AddConductingSheet(
            "top_copper_35um", conductivity=56e6, thickness=COPPER_THICKNESS_M
        )
    polygons = all_planar_polygons()
    for _, polygon in polygons:
        add_sheet_polygon(copper, polygon)

    connector_metal = csx.AddMetal("SMA_launch_PEC")
    coax_dielectric = csx.AddMaterial("SMA_PTFE", epsilon=COAX_EPSILON)
    coax_start = [0.0, COAX_START_Y, COAX_CENTER_Z]
    coax_stop = [0.0, COAX_STOP_Y, COAX_CENTER_Z]
    port = fdtd.AddCoaxialPort(
        port_nr=1,
        pec_prop=connector_metal,
        mat_prop=coax_dielectric,
        start=coax_start,
        stop=coax_stop,
        prop_dir="y",
        r_i=COAX_INNER_RADIUS,
        r_o=COAX_OUTER_INNER_RADIUS,
        r_os=COAX_OUTER_RADIUS,
        excite_amp=1.0,
        FeedShift=1.0,
        MeasPlaneShift=6.0,
        priority=30,
    )

    surface_current = csx.AddDump(
        "surface_current", dump_type=3, file_type=1, frequency=list(FARFIELD_FREQUENCIES)
    )
    surface_current.AddBox(
        [-BOARD_WIDTH / 2.0, 0.0, COPPER_Z],
        [BOARD_WIDTH / 2.0, BOARD_LENGTH, COPPER_Z],
    )
    nf2ff = fdtd.CreateNF2FFBox(
        opt_resolution=[C0 / max(FARFIELD_FREQUENCIES) / UNIT / 15.0] * 3
    )

    metadata = geometry_metadata(args, mesh)
    metadata["derived"] = {
        "fr4_conductivity_s_per_m_at_3ghz": fr4_kappa,
        "ideal_ptfe_coax_impedance_ohm": (
            60.0
            / math.sqrt(COAX_EPSILON)
            * math.log(COAX_OUTER_INNER_RADIUS / COAX_INNER_RADIUS)
        ),
    }
    return fdtd, port, nf2ff, polygons, metadata


def save_geometry_previews(output_dir: Path, polygons: list[tuple[str, np.ndarray]]) -> None:
    substrate_color = "#5c7652"
    copper_color = "#d89920"

    figure, axis = plt.subplots(figsize=(7.2, 8.2), constrained_layout=True)
    axis.add_patch(
        plt.Rectangle(
            (-BOARD_WIDTH / 2.0, 0.0),
            BOARD_WIDTH,
            BOARD_LENGTH,
            facecolor=substrate_color,
            alpha=0.45,
            edgecolor="black",
            linewidth=1.0,
        )
    )
    for _, polygon in polygons:
        axis.add_patch(PlotPolygon(polygon, closed=True, facecolor=copper_color, edgecolor="none"))
    axis.plot([0.0, 0.0], [COAX_START_Y, 0.0], color="#b9b9b9", linewidth=8.0)
    axis.plot([0.0, 0.0], [COAX_START_Y, 0.0], color="#d89920", linewidth=2.0)
    axis.set_aspect("equal")
    axis.set_xlim(-18.5, 18.5)
    axis.set_ylim(-13.0, 28.0)
    axis.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="Si-Zhu-Sun CPW-fed SRR/CRR antenna — top view",
    )
    axis.grid(True, alpha=0.18)
    figure.savefig(output_dir / "geometry_top.png", dpi=200)
    plt.close(figure)

    figure = plt.figure(figsize=(8.4, 7.0), constrained_layout=True)
    axis3d = figure.add_subplot(111, projection="3d")
    x0, x1 = -BOARD_WIDTH / 2.0, BOARD_WIDTH / 2.0
    y0, y1 = 0.0, BOARD_LENGTH
    z0, z1 = 0.0, SUBSTRATE_THICKNESS
    board_faces = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
    ]
    axis3d.add_collection3d(
        Poly3DCollection(board_faces, facecolor=substrate_color, alpha=0.32, edgecolor="0.35")
    )
    metal_faces = [[(x, y, COPPER_Z + 0.02) for x, y in polygon] for _, polygon in polygons]
    axis3d.add_collection3d(
        Poly3DCollection(metal_faces, facecolor=copper_color, edgecolor="none", alpha=0.98)
    )
    axis3d.plot(
        [0.0, 0.0],
        [COAX_START_Y, 0.0],
        [COAX_CENTER_Z, COAX_CENTER_Z],
        color="#b9b9b9",
        linewidth=8.0,
    )
    axis3d.plot(
        [0.0, 0.0],
        [COAX_START_Y, 0.0],
        [COAX_CENTER_Z, COAX_CENTER_Z],
        color=copper_color,
        linewidth=2.0,
    )
    axis3d.set_box_aspect((BOARD_WIDTH, BOARD_LENGTH - COAX_START_Y, 10.0))
    axis3d.view_init(elev=28.0, azim=-52.0)
    axis3d.set(
        xlim=(-18.0, 18.0),
        ylim=(COAX_START_Y, BOARD_LENGTH),
        zlim=(-3.0, 7.0),
        xlabel="x (mm)",
        ylabel="y (mm)",
        zlabel="z (mm)",
        title="CSXCAD model preview",
    )
    figure.savefig(output_dir / "geometry_3d.png", dpi=200)
    plt.close(figure)


def matching_bands(frequency: np.ndarray, s11_db: np.ndarray) -> list[list[float]]:
    indices = np.flatnonzero(s11_db <= -10.0)
    if not indices.size:
        return []
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return [[float(frequency[group[0]]), float(frequency[group[-1]])] for group in groups]


def resonance_metrics(
    frequency: np.ndarray, s11_db: np.ndarray, impedance: np.ndarray, window: tuple[float, float]
) -> dict:
    mask = (frequency >= window[0]) & (frequency <= window[1])
    candidates = np.flatnonzero(mask)
    index = int(candidates[np.argmin(s11_db[mask])])
    return {
        "frequency_hz": float(frequency[index]),
        "s11_db": float(s11_db[index]),
        "input_impedance_ohm": {
            "real": float(impedance[index].real),
            "imag": float(impedance[index].imag),
        },
    }


def postprocess_port(output_dir: Path, port) -> tuple[np.ndarray, np.ndarray, dict]:
    frequency = np.unique(
        np.r_[np.linspace(SWEEP_LOW, SWEEP_HIGH, 1601), FARFIELD_FREQUENCIES]
    )
    reference_shift = COAX_REFERENCE_PLANE_Y - COAX_START_Y
    port.CalcPort(
        str(output_dir),
        frequency,
        ref_impedance=50.0,
        ref_plane_shift=reference_shift,
    )
    s11 = port.uf_ref / port.uf_inc
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-15))
    impedance = port.uf_tot / port.if_tot

    with (output_dir / "frequency_response.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frequency_hz", "s11_db", "zin_real_ohm", "zin_imag_ohm"])
        writer.writerows(zip(frequency, s11_db, impedance.real, impedance.imag))
    np.savez(
        output_dir / "port_results.npz",
        frequency_hz=frequency,
        s11=s11,
        s11_db=s11_db,
        impedance_ohm=impedance,
        accepted_power_w=port.P_acc,
        incident_power_w=port.P_inc,
    )

    figure, (match_axis, impedance_axis) = plt.subplots(
        2, 1, figsize=(8.0, 7.6), sharex=True, constrained_layout=True
    )
    match_axis.plot(frequency / 1e9, s11_db, color="navy", linewidth=1.6)
    match_axis.axhline(-10.0, color="0.35", linestyle="--", linewidth=1.0)
    for low, high in PAPER_BANDS:
        match_axis.axvspan(low / 1e9, high / 1e9, color="seagreen", alpha=0.12)
    match_axis.set(ylabel="S11 (dB)", title="50-ohm response at the board-edge reference plane")
    match_axis.grid(True, alpha=0.25)
    impedance_axis.plot(frequency / 1e9, impedance.real, label="Re(Zin)")
    impedance_axis.plot(frequency / 1e9, impedance.imag, label="Im(Zin)")
    impedance_axis.axhline(50.0, color="0.35", linestyle="--", linewidth=1.0)
    impedance_axis.set(xlabel="Frequency (GHz)", ylabel="Impedance (ohm)")
    impedance_axis.grid(True, alpha=0.25)
    impedance_axis.legend()
    figure.savefig(output_dir / "port_response.png", dpi=200)
    plt.close(figure)

    bands = matching_bands(frequency, s11_db)
    metrics = {
        "reference_impedance_ohm": 50.0,
        "reference_plane_y_mm": COAX_REFERENCE_PLANE_Y,
        "minus_10db_bands_hz": bands,
        "lower_resonance": resonance_metrics(
            frequency, s11_db, impedance, (2.25e9, 2.95e9)
        ),
        "upper_resonance": resonance_metrics(
            frequency, s11_db, impedance, (3.0e9, 4.6e9)
        ),
    }
    return frequency, np.asarray(port.P_acc), metrics


def farfield_metrics(
    output_dir: Path, nf2ff, frequency: np.ndarray, accepted_power: np.ndarray
) -> dict:
    theta = np.arange(0.0, 180.1, 3.0)
    phi = np.arange(0.0, 360.0, 5.0)
    summaries = []
    cut_data = []
    for value in FARFIELD_FREQUENCIES:
        result = nf2ff.CalcNF2FF(
            str(output_dir),
            value,
            theta,
            phi,
            outfile=f"nf2ff_{value / 1e9:g}ghz.h5",
            read_cached=False,
            verbose=1,
        )
        field = np.asarray(result.E_norm[0]).squeeze()
        peak_field = float(np.nanmax(np.abs(field)))
        maximum_directivity_db = float(10.0 * np.log10(result.Dmax[0]))
        directivity_db = (
            20.0 * np.log10(np.maximum(np.abs(field) / peak_field, 1e-7))
            + maximum_directivity_db
        )
        peak = np.unravel_index(np.nanargmax(directivity_db), directivity_db.shape)
        accepted = float(np.interp(value, frequency, accepted_power))
        radiated = float(result.Prad[0])
        efficiency = radiated / accepted if accepted > 0.0 else None
        gain_db = (
            maximum_directivity_db + 10.0 * math.log10(efficiency)
            if efficiency is not None and efficiency > 0.0
            else None
        )
        summaries.append(
            {
                "frequency_hz": value,
                "maximum_directivity_dbi": maximum_directivity_db,
                "maximum_gain_dbi": gain_db,
                "radiation_efficiency": efficiency,
                "peak_theta_deg": float(theta[peak[0]]),
                "peak_phi_deg": float(phi[peak[1]]),
                "radiated_power_w": radiated,
                "accepted_power_w": accepted,
            }
        )
        cut_data.append((value, directivity_db[:, 0], directivity_db[:, 18]))
        np.savez(
            output_dir / f"farfield_{value / 1e9:g}ghz.npz",
            frequency_hz=value,
            theta_deg=theta,
            phi_deg=phi,
            directivity_db=directivity_db,
        )

    figure, axes = plt.subplots(
        1, 2, figsize=(10.0, 4.6), subplot_kw={"projection": "polar"}, constrained_layout=True
    )
    polar_theta = np.deg2rad(theta)
    for axis, (value, phi0, phi90) in zip(axes, cut_data):
        axis.plot(polar_theta, phi0, label="phi=0 deg")
        axis.plot(polar_theta, phi90, label="phi=90 deg")
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_title(f"{value / 1e9:g} GHz directivity")
        axis.set_rlim(bottom=-25.0)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    figure.savefig(output_dir / "farfield_cuts.png", dpi=200)
    plt.close(figure)
    return {"patterns": summaries}


def validate_post_only(output_dir: Path) -> None:
    required = [
        output_dir / "model_info.json",
        output_dir / "port_ut_1A",
        output_dir / "port_ut_1B",
        output_dir / "port_ut_1C",
        output_dir / "port_it_1A",
        output_dir / "port_it_1B",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("post-only is missing solver output(s): " + ", ".join(missing))


def main() -> int:
    args = parse_args()
    if args.generate_only and args.post_only:
        raise SystemExit("--generate-only and --post-only are mutually exclusive")
    if args.mesh_resolution_mm <= 0.0:
        raise SystemExit("--mesh-resolution-mm must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fdtd, port, nf2ff, polygons, metadata = build_model(args)
    xml_path = output_dir / "cpw_metamaterial_dual_band.xml"

    if args.post_only:
        validate_post_only(output_dir)
    else:
        fdtd.Write2XML(str(xml_path))
        metadata["simulation"]["xml_file"] = xml_path.name
        (output_dir / "model_info.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        save_geometry_previews(output_dir, polygons)

    print(json.dumps(metadata, indent=2), flush=True)
    if args.generate_only:
        print(f"Generated CSXCAD model: {xml_path}", flush=True)
        return 0

    if not args.post_only:
        run_options = {"verbose": 1, "dump_statistics": True}
        if args.engine == "gpu":
            run_options.update(
                engine="gpu", gpu_device=args.gpu_device, gpu_kernel=args.gpu_kernel
            )
        elif args.engine == "multithreaded":
            run_options.update(engine="multithreaded", numThreads=args.threads)
        else:
            run_options.update(engine="basic")
        original_directory = Path.cwd()
        try:
            status = fdtd.Run(str(output_dir), cleanup=True, **run_options)
        finally:
            os.chdir(original_directory)
        if status not in (None, 0):
            raise RuntimeError(f"openEMS setup failed with status {status}")

    frequency, accepted_power, port_summary = postprocess_port(output_dir, port)
    summary = {"paper_bands_hz": [list(band) for band in PAPER_BANDS], "port": port_summary}
    if not args.skip_farfield:
        summary["farfield"] = farfield_metrics(output_dir, nf2ff, frequency, accepted_power)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
