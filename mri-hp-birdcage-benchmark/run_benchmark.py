#!/usr/bin/env python3
"""Benchmark the MRI HP Birdcage tutorial on CPU and GPU."""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
TUTORIAL = ROOT.parent / "openEMS/python/Tutorials/MRI_HP_Birdcage.py"


def run_engine(engine, repeats, timesteps, threads):
    label = "cpu" if engine == "multithreaded" else "gpu"
    result_dir = ROOT / label
    log_dir = ROOT / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update({
        "MPLBACKEND": "Agg",
        "OPENEMS_SIM_PATH": str(result_dir),
        "OPENEMS_ENGINE": engine,
        "OPENEMS_NR_TS": str(timesteps),
        "OPENEMS_FIXED_TIMESTEPS": "1",
        "OPENEMS_SKIP_POSTPROCESS": "1",
        "OPENEMS_SKIP_PLOTS": "1",
        "OPENEMS_DUMP_STATISTICS": "1",
    })
    if engine == "multithreaded":
        environment["OPENEMS_NUM_THREADS"] = str(threads)
        environment.pop("OPENEMS_GPU_DEVICE", None)
    else:
        environment["OPENEMS_GPU_DEVICE"] = "0"
        environment.pop("OPENEMS_NUM_THREADS", None)

    records = []
    for repeat in range(1, repeats + 1):
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(TUTORIAL)],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        wall_seconds = time.perf_counter() - started
        log_path = log_dir / f"{label}_{repeat}.log"
        log_path.write_text(result.stdout)
        if result.returncode:
            print(result.stdout)
            raise RuntimeError(
                f"{label} repeat {repeat} failed; see {log_path}")

        time_match = re.search(
            r"Time for (\d+) iterations with ([0-9.eE+-]+) cells : "
            r"([0-9.eE+-]+) sec", result.stdout)
        speed_match = re.search(r"Speed: ([0-9.eE+-]+) MCells/s", result.stdout)
        if not time_match or not speed_match:
            raise RuntimeError(f"timing output missing from {log_path}")
        solver_seconds = float(time_match.group(3))
        throughput = float(speed_match.group(1))
        internal_timing_valid = solver_seconds > 0 and throughput > 0
        record = {
            "repeat": repeat,
            "iterations": int(time_match.group(1)),
            "cells": int(float(time_match.group(2))),
            "solver_seconds": solver_seconds,
            "throughput_mcells_per_second": throughput,
            "internal_timing_valid": internal_timing_valid,
            "wall_seconds": wall_seconds,
            "log": str(log_path),
        }
        records.append(record)
        internal_text = (
            f"solver={solver_seconds:.3f}s speed={throughput:.3f} MCells/s"
            if internal_timing_valid else "solver timing invalid (clock step)")
        print(f"{label} repeat={repeat} {internal_text} "
              f"wall={wall_seconds:.3f}s", flush=True)

    valid_records = [
        record for record in records if record["internal_timing_valid"]]
    if not valid_records:
        raise RuntimeError(f"all {label} internal solver timings were invalid")
    summary = {
        "engine": engine,
        "label": label,
        "repeats": repeats,
        "timesteps": records[0]["iterations"],
        "cells": records[0]["cells"],
        "threads": threads if engine == "multithreaded" else None,
        "runs": records,
        "median_solver_seconds": statistics.median(
            record["solver_seconds"] for record in valid_records),
        "median_throughput_mcells_per_second": statistics.median(
            record["throughput_mcells_per_second"]
            for record in valid_records),
        "median_wall_seconds": statistics.median(
            record["wall_seconds"] for record in records),
    }
    (ROOT / f"{label}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "engine", choices=("all", "multithreaded", "gpu"), nargs="?", default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1 or args.timesteps < 1 or args.threads < 1:
        parser.error("repeats, timesteps, and threads must be positive")

    summaries = {}
    if args.engine in ("all", "multithreaded"):
        summaries["cpu"] = run_engine(
            "multithreaded", args.repeats, args.timesteps, args.threads)
    if args.engine in ("all", "gpu"):
        summaries["gpu"] = run_engine(
            "gpu", args.repeats, args.timesteps, args.threads)
    if "cpu" in summaries and "gpu" in summaries:
        cpu = summaries["cpu"]
        gpu = summaries["gpu"]
        comparison = {
            "solver_speedup_gpu_over_cpu": (
                cpu["median_solver_seconds"] / gpu["median_solver_seconds"]),
            "throughput_ratio_gpu_over_cpu": (
                gpu["median_throughput_mcells_per_second"]
                / cpu["median_throughput_mcells_per_second"]),
            "wall_speedup_gpu_over_cpu": (
                cpu["median_wall_seconds"] / gpu["median_wall_seconds"]),
            "cpu": cpu,
            "gpu": gpu,
        }
        (ROOT / "benchmark_summary.json").write_text(
            json.dumps(comparison, indent=2) + "\n")
        print(
            "GPU/CPU: solver speedup={:.3f}x, throughput ratio={:.3f}x, "
            "wall speedup={:.3f}x".format(
                comparison["solver_speedup_gpu_over_cpu"],
                comparison["throughput_ratio_gpu_over_cpu"],
                comparison["wall_speedup_gpu_over_cpu"]),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
