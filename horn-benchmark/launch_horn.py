#!/usr/bin/env python3
"""Run the unmodified Horn_Antenna tutorial under benchmark controls."""

import importlib
import os
import pathlib
import runpy
import tempfile
import time

import numpy as np


TUTORIAL = pathlib.Path(
    "/home/shanda/openEMS-gpu/openEMS-Project/openEMS/python/Tutorials/"
    "Horn_Antenna.py")


class SimulationComplete(Exception):
    """Stop the tutorial after its simulation section."""


def main():
    engine = os.environ.get("HORN_ENGINE", "gpu")
    timesteps = int(os.environ.get("HORN_TIMESTEPS", "2000"))
    result_root = pathlib.Path(os.environ["HORN_RESULT_ROOT"]).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    post_only = os.environ.get("HORN_POST_ONLY") == "1"
    skip_post = os.environ.get("HORN_SKIP_POSTPROCESS") == "1"

    openems_module = importlib.import_module("openEMS")
    real_openems = openems_module.openEMS

    class ConfiguredOpenEMS:
        def __init__(self, *args, **kwargs):
            kwargs["NrTS"] = timesteps
            kwargs["EndCriteria"] = 0
            self._fdtd = real_openems(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._fdtd, name)

        def Run(self, sim_path, cleanup=False, setup_only=False, **kwargs):
            if not post_only:
                kwargs["engine"] = engine
                if engine == "gpu":
                    kwargs["gpu_device"] = 0
                    kwargs["gpu_kernel"] = "reference"
                else:
                    kwargs["numThreads"] = 8
                self._fdtd.Run(sim_path, cleanup=cleanup,
                               setup_only=setup_only, **kwargs)
            if skip_post:
                raise SimulationComplete()

    openems_module.openEMS = ConfiguredOpenEMS
    original_gettempdir = tempfile.gettempdir
    tempfile.gettempdir = lambda: str(result_root)

    original_system = os.system

    def filtered_system(command):
        if "AppCSXCAD" in command:
            print("Horn benchmark: suppressed AppCSXCAD launch")
            return 0
        return original_system(command)

    os.system = filtered_system
    started = time.perf_counter()
    namespace = None
    try:
        try:
            namespace = runpy.run_path(str(TUTORIAL), run_name="__main__")
        except SimulationComplete:
            pass
    finally:
        elapsed = time.perf_counter() - started
        os.system = original_system
        tempfile.gettempdir = original_gettempdir
        openems_module.openEMS = real_openems
    if namespace is not None and not skip_post:
        freq = namespace["freq"]
        s11 = namespace["s11"]
        s11_db = namespace["s11_dB"]
        minimum = int(np.argmin(s11_db))
        nf2ff_result = namespace["nf2ff_res"]
        metrics_path = result_root / "Horn_Antenna_PinFeed" / \
            "benchmark_metrics.npz"
        np.savez(metrics_path, freq=freq, s11=s11, Zin=namespace["Zin"],
                 Dmax_dBi=namespace["Dmax_dBi"],
                 Prad=nf2ff_result.Prad[0],
                 aperture_efficiency=namespace["e_a"],
                 E_norm=namespace["E_norm"])
        print(f"HORN_METRIC_S11_MIN_DB={s11_db[minimum]:.12g}")
        print(f"HORN_METRIC_S11_MIN_HZ={freq[minimum]:.12g}")
        print(f"HORN_METRIC_DMAX_DBI={namespace['Dmax_dBi']:.12g}")
        print(f"HORN_METRIC_PRAD_W={nf2ff_result.Prad[0]:.12g}")
        print(f"HORN_METRIC_APERTURE_EFFICIENCY="
              f"{namespace['e_a']:.12g}")
        print(f"HORN_BENCHMARK_METRICS={metrics_path}")
    print(f"HORN_BENCHMARK_WALL_SECONDS={elapsed:.9f}")
    print(f"HORN_BENCHMARK_RESULT_PATH={result_root / 'Horn_Antenna_PinFeed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
