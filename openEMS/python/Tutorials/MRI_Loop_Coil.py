# -*- coding: utf-8 -*-
"""
Tutorials / 7 T MRI loop coil with a spherical saline phantom.

This is the Python counterpart of ``matlab/Tutorials/MRI_Loop_Coil.m``.
The surface loop is tuned with three lumped capacitors and loaded by a
self-contained, homogeneous saline sphere.  Frequency-domain H-field and
local-SAR slices are recorded in the axial and sagittal planes.

Tested with
 - Python 3.12
 - openEMS v0.37

Environment variables useful for batch runs:

``OPENEMS_SIM_PATH``
    Output directory (default: the system temporary directory).
``OPENEMS_ENGINE``
    Solver engine, for example ``multithreaded`` or ``gpu``.
``OPENEMS_NUM_THREADS`` / ``OPENEMS_GPU_DEVICE`` / ``OPENEMS_GPU_KERNEL``
    Optional engine settings.
``OPENEMS_NR_TS`` / ``OPENEMS_FIXED_TIMESTEPS``
    Limit the timestep count and, optionally, force exactly that many steps.
``OPENEMS_GENERATE_ONLY`` / ``OPENEMS_POST_ONLY``
    Only write the XML model, or only post-process an existing result.
``OPENEMS_SKIP_POSTPROCESS`` / ``OPENEMS_SKIP_PLOTS``
    Disable result processing or interactive figures for automated tests.
``OPENEMS_SHOW_GEOMETRY``
    Open AppCSXCAD before the simulation.

(C) 2013-2026 Thorsten Liebig <thorsten.liebig@gmx.de>
Python translation (C) 2026 openEMS contributors
"""

import os
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

from CSXCAD import AppCSXCAD_BIN, ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, MUE0


def env_flag(name, default=False):
    """Read a conventional boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def add_boundary_padding(mesh, number_of_cells=10):
    """Extend every mesh face by uniform cells, as Matlab ``AddPML`` does."""
    for direction in 'xyz':
        lines = np.unique(mesh.GetLines(direction))
        lower_delta = lines[1] - lines[0]
        upper_delta = lines[-1] - lines[-2]
        lower = lines[0] - lower_delta * np.arange(number_of_cells, 0, -1)
        upper = lines[-1] + upper_delta * np.arange(1, number_of_cells + 1)
        mesh.SetLines(direction, np.r_[lower, lines, upper])


def read_frequency_dump(filename, vector=False):
    """Read native or legacy one-frequency dumps into x/y/z axis order."""
    with h5py.File(filename, 'r') as h5_file:
        fd_group = h5_file['/FieldData/FD']
        axes = tuple(np.asarray(h5_file['/Mesh/' + d]) for d in 'xyz')
        axis_shape = tuple(len(axis) for axis in axes)
        if vector:
            if 'f0' in fd_group:
                values = np.asarray(fd_group['f0'])
            else:
                values = np.asarray(fd_group['f0_real'])
                values = values + 1j * np.asarray(fd_group['f0_imag'])
            if values.shape[0] == 3 and values.shape[1:] == axis_shape:
                # Native layout: component/x/y/z.
                values = np.moveaxis(values, 0, -1)
            elif values.shape[0] == 3 and values.shape[1:] == axis_shape[::-1]:
                # Legacy Matlab layout: component/z/y/x.
                values = np.transpose(values, (3, 2, 1, 0))
            elif values.shape[-1] != 3 or values.shape[:-1] != axis_shape:
                raise ValueError('Unsupported vector dump shape: {}'.format(
                    values.shape))
        else:
            values = np.asarray(fd_group['f0'])
            if values.shape == axis_shape[::-1]:
                # Legacy Matlab layout: z/y/x.
                values = np.transpose(values, (2, 1, 0))
            elif values.shape != axis_shape:
                raise ValueError('Unsupported scalar dump shape: {}'.format(
                    values.shape))
    return values, axes


def plot_plane(axis, x, y, values, title, colorbar_label):
    """Plot data stored in x/second-coordinate order."""
    image = axis.pcolormesh(x, y, np.squeeze(values).T, shading='auto')
    axis.set_aspect('equal')
    axis.set_xlabel('x (mm)')
    axis.set_ylabel(colorbar_label[0] + ' (mm)')
    axis.set_title(title)
    plt.colorbar(image, ax=axis, label=colorbar_label[1])


# Simulation controls -------------------------------------------------------
sim_path = os.path.abspath(os.environ.get(
    'OPENEMS_SIM_PATH', os.path.join(tempfile.gettempdir(), 'MRI_Loop_Coil')))
nr_ts = int(os.environ.get('OPENEMS_NR_TS', '30000'))
fixed_timesteps = env_flag('OPENEMS_FIXED_TIMESTEPS')
generate_only = env_flag('OPENEMS_GENERATE_ONLY')
post_proc_only = env_flag('OPENEMS_POST_ONLY')
skip_postprocess = env_flag('OPENEMS_SKIP_POSTPROCESS')
skip_plots = env_flag('OPENEMS_SKIP_PLOTS')
show_geometry = env_flag('OPENEMS_SHOW_GEOMETRY')

if generate_only and post_proc_only:
    raise ValueError('OPENEMS_GENERATE_ONLY and OPENEMS_POST_ONLY are exclusive')

print('Simulation path: {}'.format(sim_path))


# Model parameters ---------------------------------------------------------
unit = 1e-3                       # all geometry dimensions are in mm
f0 = 298e6                       # 7 T proton Larmor frequency
fc = 300e6                       # Gaussian-pulse 20 dB corner frequency

loop_length = 80
loop_width = 60
strip_width = 5
air_gap = strip_width / 3
loop_x = -130
gap_capacitance = 5.4e-12
port_resistance = 10

phantom_center = np.array([0.0, 0.0, 0.0])
phantom_radius = 100
phantom_epsilon = 76
phantom_kappa = 0.8               # S/m
phantom_density = 1000            # kg/m^3, needed for SAR

phantom_resolution = 4
air_margin = 150
air_resolution = C0 / (f0 + fc) / unit / 10


# FDTD and geometry --------------------------------------------------------
# EndCriteria=0 is the portable Python-API equivalent of --fixed-timesteps:
# it disables early energy-based termination while NrTS remains the hard cap.
FDTD = openEMS(NrTS=nr_ts,
               EndCriteria=0.0 if fixed_timesteps else 1e-4,
               CellConstantMaterial=False)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['MUR'] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

loop_metal = CSX.AddMetal('loop')
caps_y = CSX.AddLumpedElement(
    'caps_y', ny='y', caps=False, C=gap_capacitance)
caps_z = CSX.AddLumpedElement(
    'caps_z', ny='z', caps=False, C=gap_capacitance)


def add_loop_box(start, stop):
    loop_metal.AddBox(start, stop, priority=10)


# Horizontal y-directed strips at the bottom and top of the loop.
add_loop_box([loop_x, -loop_width / 2, -loop_length / 2],
             [loop_x, -air_gap / 2, -loop_length / 2 + strip_width])
add_loop_box([loop_x, -loop_width / 2, loop_length / 2],
             [loop_x, -air_gap / 2, loop_length / 2 - strip_width])
add_loop_box([loop_x, loop_width / 2, -loop_length / 2],
             [loop_x, air_gap / 2, -loop_length / 2 + strip_width])
add_loop_box([loop_x, loop_width / 2, loop_length / 2],
             [loop_x, air_gap / 2, loop_length / 2 - strip_width])

# Vertical z-directed strips at the left and right sides of the loop.
add_loop_box([loop_x, -loop_width / 2, -loop_length / 2 + strip_width],
             [loop_x, -loop_width / 2 + strip_width, -air_gap / 2])
add_loop_box([loop_x, -loop_width / 2, loop_length / 2 - strip_width],
             [loop_x, -loop_width / 2 + strip_width, air_gap / 2])
add_loop_box([loop_x, loop_width / 2, -loop_length / 2 + strip_width],
             [loop_x, loop_width / 2 - strip_width, -air_gap / 2])
add_loop_box([loop_x, loop_width / 2, loop_length / 2 - strip_width],
             [loop_x, loop_width / 2 - strip_width, air_gap / 2])

# Tuning capacitors in the left, right, and top gaps.
caps_z.AddBox(
    [loop_x, -loop_width / 2 + strip_width / 2 - air_gap / 2, -air_gap / 2],
    [loop_x, -loop_width / 2 + strip_width / 2 + air_gap / 2, air_gap / 2],
    priority=10)
caps_z.AddBox(
    [loop_x, loop_width / 2 - strip_width / 2 - air_gap / 2, -air_gap / 2],
    [loop_x, loop_width / 2 - strip_width / 2 + air_gap / 2, air_gap / 2],
    priority=10)
caps_y.AddBox(
    [loop_x, -air_gap / 2,
     loop_length / 2 - strip_width / 2 - air_gap / 2],
    [loop_x, air_gap / 2,
     loop_length / 2 - strip_width / 2 + air_gap / 2],
    priority=10)

# The fourth gap contains the source and its series feed resistance.
port = FDTD.AddLumpedPort(
    port_nr=1,
    R=port_resistance,
    start=[loop_x, -air_gap / 2,
           -loop_length / 2 + strip_width / 2 - air_gap / 2],
    stop=[loop_x, air_gap / 2,
          -loop_length / 2 + strip_width / 2 + air_gap / 2],
    p_dir='y',
    excite=True,
    priority=100)

# Homogeneous saline phantom; no external anatomical model is required.
saline = CSX.AddMaterial(
    'saline_phantom',
    epsilon=phantom_epsilon,
    kappa=phantom_kappa,
    density=phantom_density)
saline.AddSphere(priority=1, center=phantom_center, radius=phantom_radius)


# Mesh ---------------------------------------------------------------------
# Seed all loop and lumped-element box boundaries, equivalent to DetectEdges.
FDTD.AddEdges2Grid(dirs='all', properties=[loop_metal, caps_y, caps_z])

# Refine the entire phantom box and force a cell around x=0.
for direction in 'xyz':
    mesh.AddLine(direction, [-phantom_radius, phantom_radius])
mesh.AddLine('x', [loop_x, -phantom_resolution / 2,
                   phantom_resolution / 2])
mesh.SmoothMeshLines('all', phantom_resolution)

# Add and grade the surrounding free-space region.
for direction in 'xyz':
    lines = mesh.GetLines(direction)
    mesh.AddLine(direction, [lines[0] - air_margin, lines[-1] + air_margin])
mesh.SmoothMeshLines('all', air_resolution, ratio=1.5)

# Match the Matlab tutorial's ten-cell boundary padding.
add_boundary_padding(mesh, 10)

cell_count = np.prod([len(mesh.GetLines(d)) for d in 'xyz'])
print('Mesh: {:,} Yee cells'.format(cell_count))


# Frequency-domain H-field and SAR slices ---------------------------------
body_start = phantom_center - phantom_radius
body_stop = phantom_center + phantom_radius

H_xy = CSX.AddDump(
    'Hf_xy', dump_type=11, frequency=[f0], file_type=1, dump_mode=2)
H_xy.AddBox(body_start * [1, 1, 0], body_stop * [1, 1, 0])
SAR_xy = CSX.AddDump(
    'SAR_xy', dump_type=20, frequency=[f0], file_type=1, dump_mode=2)
SAR_xy.AddBox(body_start * [1, 1, 0], body_stop * [1, 1, 0])

H_xz = CSX.AddDump(
    'Hf_xz', dump_type=11, frequency=[f0], file_type=1, dump_mode=2)
H_xz.AddBox(body_start * [1, 0, 1], body_stop * [1, 0, 1])
SAR_xz = CSX.AddDump(
    'SAR_xz', dump_type=20, frequency=[f0], file_type=1, dump_mode=2)
SAR_xz.AddBox(body_start * [1, 0, 1], body_stop * [1, 0, 1])


# Generate, inspect, and run ------------------------------------------------
xml_file = os.path.join(sim_path, 'MRI_Loop_Coil.xml')
if post_proc_only:
    if not os.path.isdir(sim_path):
        raise FileNotFoundError('Simulation path does not exist: ' + sim_path)
else:
    os.makedirs(sim_path, exist_ok=True)
    FDTD.Write2XML(xml_file)
    print('Wrote model: {}'.format(xml_file))

    if show_geometry:
        os.system('{} "{}"'.format(AppCSXCAD_BIN, xml_file))

    if generate_only:
        raise SystemExit(0)

    run_options = {'cleanup': True, 'verbose': 1}
    engine = os.environ.get('OPENEMS_ENGINE')
    if engine:
        run_options['engine'] = engine
    if os.environ.get('OPENEMS_NUM_THREADS'):
        run_options['numThreads'] = int(os.environ['OPENEMS_NUM_THREADS'])
    if os.environ.get('OPENEMS_GPU_DEVICE'):
        run_options['gpu_device'] = int(os.environ['OPENEMS_GPU_DEVICE'])
    if os.environ.get('OPENEMS_GPU_KERNEL'):
        run_options['gpu_kernel'] = os.environ['OPENEMS_GPU_KERNEL']
    FDTD.Run(sim_path, **run_options)

if skip_postprocess:
    raise SystemExit(0)


# Port, SAR, and B1 post-processing ----------------------------------------
frequency = np.linspace(f0 - fc, f0 + fc, 501)
port.CalcPort(sim_path, frequency)
Zin = port.uf_tot / port.if_tot
s11 = port.uf_ref / port.uf_inc
accepted_power = np.interp(f0, frequency, port.P_acc)
if not np.isfinite(accepted_power) or accepted_power <= 0:
    raise RuntimeError('Accepted power at f0 is not positive; simulation may '
                       'be too short for frequency-domain post-processing')

f0_index = np.argmin(np.abs(frequency - f0))
print('At {:.3f} MHz: Zin={:.5g}, S11={:.3f} dB, Pacc={:.5g} W'.format(
    f0 / 1e6, Zin[f0_index],
    20 * np.log10(np.abs(s11[f0_index])), accepted_power))

sar_xy, sar_xy_axes = read_frequency_dump(
    os.path.join(sim_path, 'SAR_xy.h5'))
sar_xz, sar_xz_axes = read_frequency_dump(
    os.path.join(sim_path, 'SAR_xz.h5'))
H_xy, H_xy_axes = read_frequency_dump(
    os.path.join(sim_path, 'Hf_xy.h5'), vector=True)
H_xz, H_xz_axes = read_frequency_dump(
    os.path.join(sim_path, 'Hf_xz.h5'), vector=True)

sar_xy /= accepted_power
sar_xz /= accepted_power
B1p_xy = 0.5 * MUE0 * (H_xy[..., 0] + 1j * H_xy[..., 1])
B1m_xy = 0.5 * MUE0 * (H_xy[..., 0] - 1j * H_xy[..., 1])
B1p_xz = 0.5 * MUE0 * (H_xz[..., 0] + 1j * H_xz[..., 1])
B1m_xz = 0.5 * MUE0 * (H_xz[..., 0] - 1j * H_xz[..., 1])
B1p_xy /= np.sqrt(accepted_power)
B1m_xy /= np.sqrt(accepted_power)
B1p_xz /= np.sqrt(accepted_power)
B1m_xz /= np.sqrt(accepted_power)

print('Peak plotted SAR: {:.5g} W/kg per W accepted'.format(
    max(np.nanmax(sar_xy), np.nanmax(sar_xz))))
print('Peak plotted |B1+|: {:.5g} uT/sqrt(W)'.format(
    1e6 * max(np.nanmax(np.abs(B1p_xy)), np.nanmax(np.abs(B1p_xz)))))

if not skip_plots:
    fig, axes = plt.subplots(1, 2, num='S parameters and admittance',
                             figsize=(12, 5), tight_layout=True)
    axes[0].plot(frequency / 1e6, 20 * np.log10(np.abs(s11)), 'k-', lw=2)
    axes[0].set(xlabel='frequency (MHz)', ylabel='$S_{11}$ (dB)',
                title='reflection coefficient')
    axes[0].grid()
    axes[1].plot(frequency / 1e6, np.real(1 / Zin), 'k-', lw=2,
                 label='real')
    axes[1].plot(frequency / 1e6, np.imag(1 / Zin), 'r--', lw=2,
                 label='imaginary')
    axes[1].set(xlabel='frequency (MHz)', ylabel='admittance (S)',
                title='feed-port admittance')
    axes[1].grid()
    axes[1].legend()

    fig, axes = plt.subplots(1, 2, num='Local SAR', figsize=(12, 5),
                             tight_layout=True)
    plot_plane(axes[0], sar_xy_axes[0], sar_xy_axes[1], sar_xy,
               'axial local SAR', ('y', 'W/kg per W accepted'))
    plot_plane(axes[1], sar_xz_axes[0], sar_xz_axes[2], sar_xz,
               'sagittal local SAR', ('z', 'W/kg per W accepted'))

    fig, axes = plt.subplots(2, 2, num='B1 fields', figsize=(12, 10),
                             tight_layout=True)
    plot_plane(axes[0, 0], H_xy_axes[0], H_xy_axes[1],
               1e6 * np.abs(B1p_xy), 'axial $|B_1^+|$',
               ('y', 'uT/sqrt(W)'))
    plot_plane(axes[0, 1], H_xy_axes[0], H_xy_axes[1],
               1e6 * np.abs(B1m_xy), 'axial $|B_1^-|$',
               ('y', 'uT/sqrt(W)'))
    plot_plane(axes[1, 0], H_xz_axes[0], H_xz_axes[2],
               1e6 * np.abs(B1p_xz), 'sagittal $|B_1^+|$',
               ('z', 'uT/sqrt(W)'))
    plot_plane(axes[1, 1], H_xz_axes[0], H_xz_axes[2],
               1e6 * np.abs(B1m_xz), 'sagittal $|B_1^-|$',
               ('z', 'uT/sqrt(W)'))
    plt.show()
