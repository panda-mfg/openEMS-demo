#!/usr/bin/env python3
"""Convenience entry point for the 36-channel 6 x 6 array at 8 GHz."""

from pathlib import Path
import runpy
import sys


sys.argv[1:1] = [
    "--array-size", "6",
    "--frequency-ghz", "8.0",
    "--corner-ghz", "1.04",
]
runpy.run_path(
    str(Path(__file__).with_name("Phased_Array_2x2_5GHz.py")),
    run_name="__main__",
)
