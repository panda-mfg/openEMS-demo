#!/usr/bin/env python3
"""Print concise aggregate norms for fixed-step Horn CPU/GPU outputs."""

import pathlib

import h5py
import numpy as np


def hdf5_datasets(handle):
    names = []
    handle.visititems(
        lambda name, item: names.append(name)
        if isinstance(item, h5py.Dataset) else None)
    return names


def main():
    root = pathlib.Path(__file__).resolve().parent
    cpu = root / "cpu" / "Horn_Antenna_PinFeed"
    gpu = root / "gpu" / "Horn_Antenna_PinFeed"
    for name in ("et", "ht", "port_ut_1", "port_it_1"):
        left = np.loadtxt(cpu / name, comments="%")
        right = np.loadtxt(gpu / name, comments="%")
        difference = np.abs(left - right)
        scale = max(float(np.max(np.abs(left))), 1e-300)
        print(f"{name}: max_abs={np.max(difference):.9g} "
              f"relative_to_peak={np.max(difference)/scale:.9g}")

    arrays = exact = 0
    sum_difference_squared = 0.0
    sum_reference_squared = 0.0
    max_abs = 0.0
    max_peak_relative = 0.0
    worst = ""
    for cpu_path in sorted(cpu.glob("nf2ff_*.h5")):
        gpu_path = gpu / cpu_path.name
        with h5py.File(cpu_path, "r") as left_file, \
                h5py.File(gpu_path, "r") as right_file:
            names = hdf5_datasets(left_file)
            if names != hdf5_datasets(right_file):
                raise RuntimeError(f"dataset list differs for {cpu_path.name}")
            for name in names:
                left = left_file[name][...]
                right = right_file[name][...]
                arrays += 1
                if np.array_equal(left, right, equal_nan=True):
                    exact += 1
                    continue
                difference = left.astype(np.float64) - right.astype(np.float64)
                local_max = float(np.max(np.abs(difference)))
                peak = max(float(np.max(np.abs(left))),
                           float(np.max(np.abs(right))), 1e-300)
                peak_relative = local_max / peak
                if local_max > max_abs:
                    max_abs = local_max
                    worst = f"{cpu_path.name}:{name}"
                max_peak_relative = max(max_peak_relative, peak_relative)
                sum_difference_squared += float(np.sum(difference*difference))
                left64 = left.astype(np.float64)
                sum_reference_squared += float(np.sum(left64*left64))
    relative_l2 = np.sqrt(sum_difference_squared /
                          max(sum_reference_squared, 1e-300))
    print(f"NF2FF: arrays={arrays} exact={exact} close={arrays-exact} "
          f"max_abs={max_abs:.9g} max_peak_relative={max_peak_relative:.9g} "
          f"relative_l2={relative_l2:.9g}")
    print(f"NF2FF worst absolute dataset: {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
