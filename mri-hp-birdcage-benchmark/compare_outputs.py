#!/usr/bin/env python3
"""Compare CPU/GPU MRI Birdcage text traces and HDF5 field dumps."""

import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent


def numeric_metrics(left, right):
    difference = np.abs(left - right)
    reference = np.abs(left)
    return (
        float(np.nanmax(difference)) if difference.size else 0.0,
        float(np.nansum(difference.astype(np.float64) ** 2)),
        float(np.nansum(reference.astype(np.float64) ** 2)),
    )


def compare_dataset(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return None
    if not np.issubdtype(left.dtype, np.number):
        return {"exact": bool(np.array_equal(left[...], right[...]))}
    max_abs = sum_diff_sq = sum_ref_sq = 0.0
    exact = True
    if left.shape and left.chunks:
        selections = left.iter_chunks()
    elif left.shape and left.shape[0] > 1:
        selections = (
            (slice(index, index + 1),) + (slice(None),) * (left.ndim - 1)
            for index in range(left.shape[0]))
    else:
        selections = (Ellipsis,)
    for selection in selections:
        left_values = left[selection]
        right_values = right[selection]
        exact &= np.array_equal(left_values, right_values, equal_nan=True)
        local_max, local_diff_sq, local_ref_sq = numeric_metrics(
            left_values, right_values)
        max_abs = max(max_abs, local_max)
        sum_diff_sq += local_diff_sq
        sum_ref_sq += local_ref_sq
    return {
        "exact": bool(exact),
        "max_absolute_difference": max_abs,
        "sum_difference_squared": sum_diff_sq,
        "sum_reference_squared": sum_ref_sq,
    }


def main():
    cpu = ROOT / "cpu"
    gpu = ROOT / "gpu"
    report = {
        "text_files": {},
        "hdf5_files": {},
        "arrays": 0,
        "exact_arrays": 0,
        "max_absolute_difference": 0.0,
        "sum_difference_squared": 0.0,
        "sum_reference_squared": 0.0,
        "structurally_equal": True,
    }

    text_names = sorted(
        path.name for path in cpu.iterdir()
        if path.is_file()
        and (path.name in ("et", "ht")
             or path.name.startswith(("port_ut_", "port_it_"))))
    for name in text_names:
        left = np.loadtxt(cpu / name, comments="%")
        right = np.loadtxt(gpu / name, comments="%")
        if left.shape != right.shape:
            report["structurally_equal"] = False
            continue
        maximum, diff_sq, ref_sq = numeric_metrics(left, right)
        exact = np.array_equal(left, right, equal_nan=True)
        report["text_files"][name] = {
            "exact": bool(exact), "max_absolute_difference": maximum}
        report["arrays"] += 1
        report["exact_arrays"] += int(exact)
        report["max_absolute_difference"] = max(
            report["max_absolute_difference"], maximum)
        report["sum_difference_squared"] += diff_sq
        report["sum_reference_squared"] += ref_sq

    cpu_hdf5 = sorted(path.name for path in cpu.glob("*.h5"))
    gpu_hdf5 = sorted(path.name for path in gpu.glob("*.h5"))
    if cpu_hdf5 != gpu_hdf5:
        report["structurally_equal"] = False
    for name in sorted(set(cpu_hdf5) & set(gpu_hdf5)):
        file_report = {}
        with h5py.File(cpu / name, "r") as left_file, \
                h5py.File(gpu / name, "r") as right_file:
            left_names = []
            right_names = []
            left_file.visititems(
                lambda item_name, item: left_names.append(item_name)
                if isinstance(item, h5py.Dataset) else None)
            right_file.visititems(
                lambda item_name, item: right_names.append(item_name)
                if isinstance(item, h5py.Dataset) else None)
            if left_names != right_names:
                report["structurally_equal"] = False
            for dataset_name in sorted(set(left_names) & set(right_names)):
                metrics = compare_dataset(
                    left_file[dataset_name], right_file[dataset_name])
                if metrics is None:
                    report["structurally_equal"] = False
                    continue
                file_report[dataset_name] = metrics
                report["arrays"] += 1
                report["exact_arrays"] += int(metrics["exact"])
                if "max_absolute_difference" in metrics:
                    report["max_absolute_difference"] = max(
                        report["max_absolute_difference"],
                        metrics["max_absolute_difference"])
                    report["sum_difference_squared"] += metrics[
                        "sum_difference_squared"]
                    report["sum_reference_squared"] += metrics[
                        "sum_reference_squared"]
        report["hdf5_files"][name] = file_report

    report["relative_l2_difference"] = float(np.sqrt(
        report["sum_difference_squared"]
        / max(report["sum_reference_squared"], 1e-300)))
    report["passed"] = bool(
        report["structurally_equal"]
        and report["relative_l2_difference"] <= 1e-4)
    output = ROOT / "comparison_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "{}: arrays={} exact={} max_abs={:.9g} relative_l2={:.9g}".format(
            "PASS" if report["passed"] else "FAIL",
            report["arrays"], report["exact_arrays"],
            report["max_absolute_difference"],
            report["relative_l2_difference"]))
    print(f"Detailed report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
