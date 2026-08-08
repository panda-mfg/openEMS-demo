#!/usr/bin/env python3
"""Convenience entry point for the 16-channel 4 x 4 patch-array model."""

from pathlib import Path
import runpy
import sys


sys.argv[1:1] = ["--array-size", "4"]
runpy.run_path(
    str(Path(__file__).with_name("Phased_Array_2x2_5GHz.py")),
    run_name="__main__",
)
