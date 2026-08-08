#!/usr/bin/env python3
"""Collect repeated Horn tutorial CPU/GPU solver and wall timings."""

import argparse
import os
import pathlib
import re
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=("multithreaded", "gpu"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=2000)
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    label = "cpu" if args.engine == "multithreaded" else "gpu"
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    result_root = root / label
    env = os.environ.copy()
    env.update({
        "HORN_ENGINE": args.engine,
        "HORN_TIMESTEPS": str(args.timesteps),
        "HORN_RESULT_ROOT": str(result_root),
        "HORN_SKIP_POSTPROCESS": "1",
        "MPLBACKEND": "Agg",
    })
    command = [sys.executable, str(root / "launch_horn.py")]
    solver_times = []
    speeds = []
    wall_times = []
    steps = []
    cells = []
    for run in range(1, args.repeats + 1):
        result = subprocess.run(command, env=env, text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        (log_dir / f"{label}_{run}.log").write_text(result.stdout)
        if result.returncode:
            print(result.stdout)
            return result.returncode
        time_match = re.search(
            r"Time for (\d+) iterations with ([0-9.eE+-]+) cells : "
            r"([0-9.eE+-]+) sec", result.stdout)
        speed_match = re.search(r"Speed: ([0-9.]+) MCells/s", result.stdout)
        wall_match = re.search(
            r"HORN_BENCHMARK_WALL_SECONDS=([0-9.]+)", result.stdout)
        if not time_match or not speed_match or not wall_match:
            print(result.stdout)
            raise RuntimeError("Horn timing output was not found")
        steps.append(int(time_match.group(1)))
        cells.append(float(time_match.group(2)))
        solver_times.append(float(time_match.group(3)))
        speeds.append(float(speed_match.group(1)))
        wall_times.append(float(wall_match.group(1)))
        print(f"run={run} solver={solver_times[-1]:.6f}s "
              f"speed={speeds[-1]:.3f} MCells/s "
              f"wall={wall_times[-1]:.6f}s")

    if len(set(steps)) != 1 or len(set(cells)) != 1:
        raise RuntimeError("benchmark runs did not execute identical work")
    print(f"model cells={cells[0]:.0f} timesteps={steps[0]}")
    print(f"median solver={statistics.median(solver_times):.6f}s "
          f"speed={statistics.median(speeds):.3f} MCells/s "
          f"wall={statistics.median(wall_times):.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
