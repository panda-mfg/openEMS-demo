#!/usr/bin/env python3
"""Compare fixed-step openEMS time-series and HDF5 datasets numerically."""
import argparse
import pathlib

import h5py
import numpy as np


def compare_array(name, left, right, stats):
    stats['arrays'] += 1
    if left.shape != right.shape or left.dtype != right.dtype:
        print(f"DIFFERENT {name}: {left.shape}/{left.dtype} vs "
              f"{right.shape}/{right.dtype}")
        return False
    exact = np.array_equal(left, right, equal_nan=True)
    if exact:
        stats['exact'] += 1
        return True
    if not np.issubdtype(left.dtype, np.number):
        print(f"DIFFERENT {name}: non-numeric data")
        return False
    difference = np.abs(left - right)
    max_abs = float(np.nanmax(difference)) if difference.size else 0.0
    scale = np.maximum(np.abs(left), np.abs(right))
    relative = np.divide(difference, scale, out=np.zeros_like(difference,
                         dtype=np.float64), where=scale != 0)
    max_rel = float(np.nanmax(relative)) if relative.size else 0.0
    print(f"CLOSE {name}: max_abs={max_abs:.9g}, max_rel={max_rel:.9g}")
    close = max_abs <= 5e-7 or max_rel <= 5e-6
    if close:
        stats['close'] += 1
    return close


def hdf5_datasets(handle):
    result = {}
    handle.visititems(lambda name, item: result.__setitem__(name, item[...])
                      if isinstance(item, h5py.Dataset) else None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=pathlib.Path)
    parser.add_argument("right", type=pathlib.Path)
    args = parser.parse_args()
    passed = True
    stats = {'arrays': 0, 'exact': 0, 'close': 0}
    for name in ("et", "ht", "port_ut_1", "port_it_1"):
        left = np.loadtxt(args.left / name, comments="%")
        right = np.loadtxt(args.right / name, comments="%")
        passed &= compare_array(name, left, right, stats)
    left_files = sorted(path.name for path in args.left.glob("nf2ff_*.h5"))
    right_files = sorted(path.name for path in args.right.glob("nf2ff_*.h5"))
    if left_files != right_files:
        print("DIFFERENT HDF5 file lists")
        return 1
    for filename in left_files:
        with h5py.File(args.left / filename, "r") as left_file, \
                h5py.File(args.right / filename, "r") as right_file:
            left_data = hdf5_datasets(left_file)
            right_data = hdf5_datasets(right_file)
        if left_data.keys() != right_data.keys():
            print(f"DIFFERENT {filename}: dataset lists")
            passed = False
            continue
        for dataset in left_data:
            passed &= compare_array(f"{filename}:{dataset}",
                                    left_data[dataset], right_data[dataset], stats)
    print(("PASS" if passed else "FAIL") +
          f": arrays={stats['arrays']}, exact={stats['exact']}, "
          f"within_tolerance={stats['close']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
