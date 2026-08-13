# CPW-fed metamaterial-inspired dual-band antenna

This directory implements the antenna from Li-Ming Si, Weiren Zhu, and Hou-Jun Sun, “A Compact, Planar, and CPW-Fed Metamaterial-Inspired Dual-Band Antenna,” *IEEE Antennas and Wireless Propagation Letters*, vol. 12, pp. 305–308, 2013. DOI: [10.1109/LAWP.2013.2249037](https://doi.org/10.1109/LAWP.2013.2249037).

The CSXCAD geometry reproduces the published 31.7 mm × 27 mm × 1.6 mm FR-4 board, 35 µm copper, connected outer CRR and inner SRR, stepped taper, 0.2 mm CPW gaps, and trapezoidal grounds. The paper specifies an edge-mounted 50 Ω SMA but does not publish its internal connector dimensions, so the model uses a documented compact 49.8 Ω PTFE coaxial launch connected directly to the board edge.

Generate the CSXCAD XML and preview images without solving:

```bash
PYTHONPATH=/home/shanda/openEMS-gpu/local/python-packages \
python3 simulate_cpw_metamaterial.py --generate-only
```

Run the complete GPU simulation, including S11, impedance, two near-field-to-far-field transformations, efficiency, and surface-current dumps:

```bash
PYTHONPATH=/home/shanda/openEMS-gpu/local/python-packages \
python3 simulate_cpw_metamaterial.py --engine gpu
```

Important generated files in `results/` are:

- `cpw_metamaterial_dual_band.xml`: openEMS/CSXCAD simulation model
- `geometry_top.png` and `geometry_3d.png`: model previews
- `model_info.json`: paper parameters, mesh, materials, and connector assumption
- `frequency_response.csv`, `port_response.png`, and `port_results.npz`: 50 Ω board-edge response
- `farfield_cuts.png`, `farfield_2.6ghz.npz`, and `farfield_3.6ghz.npz`: radiation results
- `summary.json`: extracted resonances, matched bands, gain, and efficiency

The paper reports measured −10 dB bands of 2.595–2.654 GHz and 3.185–4.245 GHz. Differences should be expected because its HFSS mesh, exact SMA geometry, solder, and fabrication tolerances were not published.
