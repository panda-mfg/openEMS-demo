#!/usr/bin/env python3
"""Export full 3-D openEMS NF2FF patterns to CSV, legacy VTK, and PNG."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

from openEMS.nf2ff import nf2ff_results


PATTERNS = (
    ("2p5ghz", 2.5e9, "nf2ff_2p5ghz.h5"),
    ("5ghz", 5.0e9, "nf2ff_5ghz.h5"),
)


def pattern_arrays(result, realized_offset_db, floor_db):
    e_norm = np.asarray(result.E_norm[0])
    e_max = float(np.max(e_norm))
    normalized_db = 20.0 * np.log10(np.maximum(e_norm / e_max, 1e-30))
    normalized_db = np.maximum(normalized_db, floor_db)
    directivity_max_db = float(10.0 * np.log10(result.Dmax[0]))
    directivity_db = normalized_db + directivity_max_db
    realized_gain_db = directivity_db + realized_offset_db
    return normalized_db, directivity_db, realized_gain_db


def write_csv(
    path,
    frequency_hz,
    result,
    normalized_db,
    directivity_db,
    realized_gain_db,
):
    theta_deg = np.rad2deg(result.theta)
    phi_deg = np.rad2deg(result.phi)
    e_theta = np.asarray(result.E_theta[0])
    e_phi = np.asarray(result.E_phi[0])
    e_cprh = np.asarray(result.E_cprh[0])
    e_cplh = np.asarray(result.E_cplh[0])
    p_rad = np.asarray(result.P_rad[0])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "theta_deg",
                "phi_deg",
                "normalized_field_db",
                "directivity_dbi",
                "realized_gain_dbi",
                "e_theta_magnitude",
                "e_theta_phase_deg",
                "e_phi_magnitude",
                "e_phi_phase_deg",
                "rhcp_magnitude",
                "lhcp_magnitude",
                "p_rad_native",
            ]
        )
        for theta_index, theta in enumerate(theta_deg):
            for phi_index, phi in enumerate(phi_deg):
                writer.writerow(
                    [
                        frequency_hz,
                        float(theta),
                        float(phi),
                        float(normalized_db[theta_index, phi_index]),
                        float(directivity_db[theta_index, phi_index]),
                        float(realized_gain_db[theta_index, phi_index]),
                        float(abs(e_theta[theta_index, phi_index])),
                        float(np.rad2deg(np.angle(e_theta[theta_index, phi_index]))),
                        float(abs(e_phi[theta_index, phi_index])),
                        float(np.rad2deg(np.angle(e_phi[theta_index, phi_index]))),
                        float(abs(e_cprh[theta_index, phi_index])),
                        float(abs(e_cplh[theta_index, phi_index])),
                        float(p_rad[theta_index, phi_index]),
                    ]
                )


def write_vtk(
    path,
    frequency_hz,
    result,
    normalized_db,
    directivity_db,
    realized_gain_db,
):
    """Write a normalized 3-D pattern surface as legacy ASCII VTK PolyData."""
    theta, phi = np.meshgrid(result.theta, result.phi, indexing="ij")
    radius = 10.0 ** (normalized_db / 20.0)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    ntheta, nphi = radius.shape
    polygons = []
    for ti in range(ntheta - 1):
        for pi in range(nphi):
            next_pi = (pi + 1) % nphi
            polygons.append(
                (
                    ti * nphi + pi,
                    (ti + 1) * nphi + pi,
                    (ti + 1) * nphi + next_pi,
                    ti * nphi + next_pi,
                )
            )

    scalar_fields = (
        ("normalized_field_dB", normalized_db),
        ("directivity_dBi", directivity_db),
        ("realized_gain_dBi", realized_gain_db),
        ("E_theta_magnitude", np.abs(result.E_theta[0])),
        ("E_phi_magnitude", np.abs(result.E_phi[0])),
    )

    with path.open("w", encoding="ascii") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(
            f"openEMS radiation pattern at {frequency_hz:.9g} Hz; "
            "radius is normalized field magnitude\n"
        )
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {len(points)} float\n")
        for point in points:
            handle.write(f"{point[0]:.9e} {point[1]:.9e} {point[2]:.9e}\n")
        handle.write(f"POLYGONS {len(polygons)} {5 * len(polygons)}\n")
        for polygon in polygons:
            handle.write("4 {} {} {} {}\n".format(*polygon))
        handle.write(f"POINT_DATA {len(points)}\n")
        for name, values in scalar_fields:
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            for value in np.asarray(values).ravel():
                handle.write(f"{float(value):.9e}\n")


def write_3d_png(
    path,
    frequency_hz,
    result,
    normalized_db,
    directivity_db,
):
    theta, phi = np.meshgrid(result.theta, result.phi, indexing="ij")
    radius = 10.0 ** (normalized_db / 20.0)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    norm = colors.Normalize(
        vmin=float(np.min(directivity_db)),
        vmax=float(np.max(directivity_db)),
    )
    color_map = plt.get_cmap("viridis")
    facecolors = color_map(norm(directivity_db))

    figure = plt.figure(figsize=(8.0, 7.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        x,
        y,
        z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    axis.set_box_aspect((1, 1, 1))
    axis.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title=f"{frequency_hz / 1e9:g} GHz normalized 3-D radiation pattern",
    )
    axis.view_init(elev=24, azim=35)
    mapper = cm.ScalarMappable(norm=norm, cmap=color_map)
    mapper.set_array([])
    figure.colorbar(
        mapper, ax=axis, shrink=0.68, pad=0.08, label="Directivity (dBi)"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def export_all(results_dir: Path, floor_db: float):
    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = {
        "description": (
            "Full-sphere radiation-pattern exports. VTK surface radius is "
            "normalized E-field magnitude; scalar arrays carry gain values."
        ),
        "angular_sampling": {
            "theta_deg": "0 to 180 in 2 degree steps",
            "phi_deg": "0 to 355 in 5 degree steps",
        },
        "floor_db": floor_db,
        "patterns": {},
    }

    for key, frequency_hz, h5_name in PATTERNS:
        h5_path = results_dir / h5_name
        if not h5_path.is_file():
            raise FileNotFoundError(f"Missing NF2FF result: {h5_path}")
        result = nf2ff_results(str(h5_path))
        farfield_summary = summary["targets"][key]["farfield"]
        realized_offset_db = (
            farfield_summary["realized_gain_dbi"]
            - farfield_summary["directivity_dbi"]
        )
        normalized_db, directivity_db, realized_gain_db = pattern_arrays(
            result, realized_offset_db, floor_db
        )

        csv_path = results_dir / f"radiation_pattern_3d_{key}.csv"
        vtk_path = results_dir / f"radiation_pattern_3d_{key}.vtk"
        png_path = results_dir / f"radiation_pattern_3d_{key}.png"
        write_csv(
            csv_path,
            frequency_hz,
            result,
            normalized_db,
            directivity_db,
            realized_gain_db,
        )
        write_vtk(
            vtk_path,
            frequency_hz,
            result,
            normalized_db,
            directivity_db,
            realized_gain_db,
        )
        write_3d_png(
            png_path,
            frequency_hz,
            result,
            normalized_db,
            directivity_db,
        )
        manifest["patterns"][key] = {
            "frequency_hz": frequency_hz,
            "samples": int(normalized_db.size),
            "theta_samples": int(normalized_db.shape[0]),
            "phi_samples": int(normalized_db.shape[1]),
            "maximum_directivity_dbi": float(np.max(directivity_db)),
            "maximum_realized_gain_dbi": float(np.max(realized_gain_db)),
            "source_hdf5": h5_name,
            "csv": csv_path.name,
            "vtk": vtk_path.name,
            "preview_png": png_path.name,
        }

    manifest_path = results_dir / "radiation_pattern_exports.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--floor-db",
        type=float,
        default=-35.0,
        help="floor used for normalized surface radius and exported dB values",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export_all(arguments.results_dir.resolve(), arguments.floor_db)
