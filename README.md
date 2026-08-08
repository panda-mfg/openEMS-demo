# openEMS Antenna Simulation Demos

This private repository is a source-only collection of antenna simulations from the local `openEMS-gpu` workspace. It includes Python and MATLAB model builders, openEMS/CSXCAD geometry setup, solver launch code, and post-processing utilities.

## Included source

- Standalone Python designs: bowtie, dual-band bowtie, dual-band slot, folded dipole, inverted-F/PIFA, log-periodic, quadrifilar helix, and Yagi-Uda antennas.
- Python openEMS tutorials: bent patch, simple patch, dipole SAR, helical, horn, MRI coils, and 2x2/4x4/6x6 phased arrays.
- MATLAB openEMS tutorials and antenna examples, including patch arrays, horn, helix, dipole, inverted-F, CRLH leaky-wave, radar UWB, and MRI coil models.
- Post-processing and reporting scripts for S-parameters, matching, radiation patterns, phased-array reports, and CPU/GPU comparison.
- Source-only horn, helical, and MATLAB tutorial benchmark harnesses.

## Intentionally excluded

Simulation results and generated artifacts are not committed. This includes field dumps, port time-series data, HDF5/NPZ files, XML models, Touchstone and CSV outputs, plots, PDFs, logs, build trees, installed binaries, caches, and benchmark run directories.

The scripts expect a working openEMS/CSXCAD installation plus their normal Python or MATLAB dependencies. Several Python examples can use the local GPU-enabled openEMS build when the relevant environment variables or executable paths are configured.

## Key phased-array demos

- `openEMS/python/Tutorials/Phased_Array_2x2_5GHz.py`
- `openEMS/python/Tutorials/Phased_Array_4x4_5GHz.py`
- `openEMS/python/Tutorials/Phased_Array_6x6_8GHz.py`
- `openEMS/python/Tutorials/Phased_Array_4x4_Report.py`
- `openEMS/python/Tutorials/Phased_Array_6x6_8GHz_Report.py`

Run outputs should be written to ignored result directories or to a separate workspace.
