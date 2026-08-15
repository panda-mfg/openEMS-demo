# MRI HP Birdcage CPU/GPU benchmark

This harness runs the repository's `MRI_HP_Birdcage.py` tutorial with an
identical fixed-timestep workload on the 8-thread CPU engine and CUDA GPU
engine. It records per-run solver throughput, wall time, logs, and a streamed
numerical comparison of port traces and HDF5 field outputs.

```bash
source ~/.local/openEMS-gpu/venv/bin/activate
cd ~/openEMS-demo/mri-hp-birdcage-benchmark
python run_benchmark.py all --repeats 3 --timesteps 1000 --threads 8
python compare_outputs.py
```

Use `gpu` or `multithreaded` instead of `all` to run one engine only.

The harness uses monotonic external wall timing and excludes any negative
openEMS internal timing sample caused by a WSL host-clock correction.
Generated fields, logs, and JSON summaries are intentionally ignored by Git.
