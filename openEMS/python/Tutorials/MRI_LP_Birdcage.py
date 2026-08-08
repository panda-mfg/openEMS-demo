# -*- coding: utf-8 -*-
"""
Tutorials / 3 T MRI low-pass birdcage coil.

This is the Python counterpart of matlab/Tutorials/MRI_LP_Birdcage.m.
The eight-rung coil is driven in quadrature and loaded by a self-contained
spherical saline phantom, so no external voxel body model is required.

Tested with
 - Python 3.12
 - openEMS v0.37

Environment variables useful for batch runs:

OPENEMS_SIM_PATH
    Output directory (default: the system temporary directory).
OPENEMS_ENGINE
    Solver engine, for example multithreaded or gpu.
OPENEMS_NUM_THREADS / OPENEMS_GPU_DEVICE / OPENEMS_GPU_KERNEL
    Optional engine settings.
OPENEMS_NR_TS / OPENEMS_FIXED_TIMESTEPS
    Limit the timestep count and, optionally, force exactly that many steps.
OPENEMS_GENERATE_ONLY / OPENEMS_SETUP_ONLY / OPENEMS_POST_ONLY
    Write XML only, initialize the solver only, or process existing results.
OPENEMS_SKIP_POSTPROCESS / OPENEMS_SKIP_PLOTS
    Disable result processing or interactive figures for automated tests.
OPENEMS_SHOW_GEOMETRY / OPENEMS_DUMP_STATISTICS
    Open AppCSXCAD or write solver statistics.

(C) 2013-2026 Thorsten Liebig <thorsten.liebig@gmx.de>
Python translation (C) 2026 openEMS contributors
"""

import os
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

from CSXCAD import AppCSXCAD_BIN, ContinuousStructure
from CSXCAD.SmoothMeshLines import SmoothMeshLines
from openEMS import openEMS
from openEMS.physical_constants import C0, MUE0


def env_flag(name, default=False):
    """Read a conventional boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def add_axial_pml_padding(mesh, number_of_cells=10):
    """Extend both z faces by uniform cells, as Matlab AddPML does."""
    lines = np.unique(mesh.GetLines('z'))
    lower_delta = lines[1] - lines[0]
    upper_delta = lines[-1] - lines[-2]
    lower = lines[0] - lower_delta * np.arange(number_of_cells, 0, -1)
    upper = lines[-1] + upper_delta * np.arange(1, number_of_cells + 1)
    mesh.SetLines('z', np.r_[lower, lines, upper])


def read_frequency_dump(filename, vector=False):
    """Read native or legacy Cartesian/cylindrical dumps in mesh-axis order."""
    with h5py.File(filename, 'r') as h5_file:
        mesh_group = h5_file['/Mesh']
        if 'rho' in mesh_group:
            axis_names = ('rho', 'alpha', 'z')
        else:
            axis_names = ('x', 'y', 'z')
        axes = tuple(np.asarray(mesh_group[name]) for name in axis_names)
        axis_shape = tuple(len(axis) for axis in axes)
        fd_group = h5_file['/FieldData/FD']

        if vector:
            if 'f0' in fd_group:
                values = np.asarray(fd_group['f0'])
            else:
                values = np.asarray(fd_group['f0_real'])
                values = values + 1j * np.asarray(fd_group['f0_imag'])
            if values.shape[0] == 3 and values.shape[1:] == axis_shape:
                values = np.moveaxis(values, 0, -1)
            elif values.shape[0] == 3 and values.shape[1:] == axis_shape[::-1]:
                values = np.transpose(values, (3, 2, 1, 0))
            elif values.shape[-1] != 3 or values.shape[:-1] != axis_shape:
                raise ValueError('Unsupported vector dump shape: {}'.format(
                    values.shape))
        else:
            values = np.asarray(fd_group['f0'])
            if values.shape == axis_shape[::-1]:
                values = np.transpose(values, (2, 1, 0))
            elif values.shape != axis_shape:
                raise ValueError('Unsupported scalar dump shape: {}'.format(
                    values.shape))
    return values, axes


def close_periodic_alpha(alpha, *fields):
    """Append the first angular column at alpha+2*pi for seamless plots."""
    closed_alpha = np.r_[alpha, alpha[0] + 2 * np.pi]
    closed_fields = [
        np.concatenate((field, field[:, :1]), axis=1) for field in fields
    ]
    return closed_alpha, closed_fields


def sample_edges(samples, nonnegative=False):
    """Convert monotonically ordered sample centers to plotting cell edges."""
    samples = np.asarray(samples)
    midpoints = 0.5 * (samples[:-1] + samples[1:])
    edges = np.r_[
        samples[0] - (midpoints[0] - samples[0]),
        midpoints,
        samples[-1] + (samples[-1] - midpoints[-1])]
    if nonnegative:
        edges[0] = max(0, edges[0])
    return edges


def plot_cylindrical_plane(axis, radius, alpha, values, title, label):
    """Plot an r/alpha array on a Cartesian x/y plane."""
    radius_edges = sample_edges(radius, nonnegative=True)
    alpha_edges = sample_edges(alpha)
    radius_grid, alpha_grid = np.meshgrid(
        radius_edges, alpha_edges, indexing='ij')
    x_grid = radius_grid * np.cos(alpha_grid)
    y_grid = radius_grid * np.sin(alpha_grid)
    image = axis.pcolormesh(
        x_grid, y_grid, np.real(values), shading='flat')
    axis.set_aspect('equal')
    axis.set_xlabel('x (mm)')
    axis.set_ylabel('y (mm)')
    axis.set_title(title)
    plt.colorbar(image, ax=axis, label=label)


# Simulation controls -------------------------------------------------------
sim_path = os.path.abspath(os.environ.get(
    'OPENEMS_SIM_PATH',
    os.path.join(tempfile.gettempdir(), 'MRI_LP_Birdcage')))
nr_ts = int(os.environ.get('OPENEMS_NR_TS', '1000000000'))
fixed_timesteps = env_flag('OPENEMS_FIXED_TIMESTEPS')
generate_only = env_flag('OPENEMS_GENERATE_ONLY')
setup_only = env_flag('OPENEMS_SETUP_ONLY')
post_proc_only = env_flag('OPENEMS_POST_ONLY')
skip_postprocess = env_flag('OPENEMS_SKIP_POSTPROCESS')
skip_plots = env_flag('OPENEMS_SKIP_PLOTS')
show_geometry = env_flag('OPENEMS_SHOW_GEOMETRY')

if sum((generate_only, setup_only, post_proc_only)) > 1:
    raise ValueError('Generate-only, setup-only, and post-only are exclusive')

print('Simulation path: {}'.format(sim_path))


# Model parameters ---------------------------------------------------------
unit = 1e-3
f0 = 128e6
excitation_f0 = 75e6
excitation_fc = 75e6

bore_radius = 320
bore_length = 1600

number_of_rungs = 8
coil_radius = 120
strip_width = 10
port_width = strip_width / 2
port_length = strip_width / 2
coil_length = 250
capacitance = 2.6e-12

feed_positions = (1, 3)
feed_amplitudes = (1.0 + 0j, -1j)

phantom_center = np.array([0.0, 0.0, 0.0])
phantom_radius = 90
phantom_epsilon = 78
phantom_kappa = 0.6
phantom_density = 1000
body_mesh_resolution = 2.5

lambda_min = C0 / (excitation_f0 + excitation_fc)
radial_axial_resolution = min(15, lambda_min / 20 / unit)
# A tiny upward tolerance prevents the Python smoother from bisecting cells
# that are exactly one Matlab angular mesh step wide because of roundoff.
angular_resolution = body_mesh_resolution / coil_radius * (1 + 1e-12)


# Cylindrical multigrid FDTD setup -----------------------------------------
FDTD = openEMS(
    NrTS=nr_ts,
    CoordSystem=1,
    EndCriteria=0.0 if fixed_timesteps else 1e-4,
    MultiGrid=[10.0, 20.0],
    CellConstantMaterial=True)
FDTD.SetGaussExcite(excitation_f0, excitation_fc)
FDTD.SetBoundaryCond([0, 0, 0, 0, 3, 3])

CSX = ContinuousStructure(CoordSystem=1)
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)


# Birdcage geometry --------------------------------------------------------
metal = CSX.AddMetal('metal')
capacitors = CSX.AddLumpedElement(
    'caps', ny='z', caps=False, C=capacitance)

angular_strip_width = strip_width / coil_radius
angular_port_width = port_width / coil_radius
rung_pitch = 2 * np.pi / number_of_rungs
alpha_start = -np.pi - rung_pitch / 2
omega0 = 2 * np.pi * f0
period0 = 1 / f0

ports = []
feed_index = 0
alpha0 = alpha_start

for rung_number in range(1, number_of_rungs + 1):
    rung_alpha = alpha0 + rung_pitch / 2

    capacitors.AddBox(
        [coil_radius, rung_alpha - angular_port_width / 2, -port_length / 2],
        [coil_radius, rung_alpha + angular_port_width / 2, port_length / 2],
        priority=1)

    upper_gap_start = [
        coil_radius,
        rung_alpha - angular_port_width / 2,
        coil_length / 2 - strip_width / 2 - port_length]
    upper_gap_stop = [
        coil_radius,
        rung_alpha + angular_port_width / 2,
        coil_length / 2 - strip_width / 2]

    if rung_number in feed_positions:
        feed_amplitude = feed_amplitudes[feed_index]
        delay = -np.angle(feed_amplitude) / omega0
        if delay < 0:
            delay += period0
        port = FDTD.AddLumpedPort(
            port_nr=feed_index + 1,
            R=50,
            start=upper_gap_start,
            stop=upper_gap_stop,
            p_dir='z',
            excite=abs(feed_amplitude),
            priority=100,
            delay=delay)
        ports.append(port)
        feed_index += 1
        upper_rung_start_z = upper_gap_start[2]
    else:
        upper_rung_start_z = coil_length / 2

    metal.AddBox(
        [coil_radius, rung_alpha - angular_strip_width / 2,
         upper_rung_start_z],
        [coil_radius, rung_alpha + angular_strip_width / 2, port_length / 2],
        priority=1)
    metal.AddBox(
        [coil_radius, rung_alpha - angular_strip_width / 2, -coil_length / 2],
        [coil_radius, rung_alpha + angular_strip_width / 2, -port_length / 2],
        priority=1)

    mesh.AddLine('a', rung_alpha)
    alpha0 += rung_pitch

metal.AddBox(
    [coil_radius, alpha_start, -(coil_length - strip_width) / 2],
    [coil_radius, alpha_start + 2 * np.pi,
     -(coil_length + strip_width) / 2],
    priority=1)
metal.AddBox(
    [coil_radius, alpha_start, (coil_length - strip_width) / 2],
    [coil_radius, alpha_start + 2 * np.pi,
     (coil_length + strip_width) / 2],
    priority=1)


# Spherical saline phantom -------------------------------------------------
saline = CSX.AddMaterial(
    'saline_phantom',
    epsilon=phantom_epsilon,
    kappa=phantom_kappa,
    density=phantom_density)
saline.AddSphere(priority=1, center=phantom_center, radius=phantom_radius)


# Mesh ---------------------------------------------------------------------
FDTD.AddEdges2Grid(dirs='all', properties=[metal, capacitors])
for port in ports:
    FDTD.AddEdges2Grid(dirs='all', properties=port.port_props)

radial_seed = np.unique(np.r_[
    body_mesh_resolution * 1.5,
    phantom_radius,
    mesh.GetLines('r')])
radial_lines = SmoothMeshLines(
    radial_seed, body_mesh_resolution, ratio=1.5)
mesh.SetLines('r', np.unique(np.r_[0, radial_lines]))

axial_seed = np.unique(np.r_[
    phantom_center[2] - phantom_radius,
    phantom_center[2] + phantom_radius,
    mesh.GetLines('z')])
mesh.SetLines(
    'z', SmoothMeshLines(axial_seed, body_mesh_resolution, ratio=1.5))

mesh.AddLine('r', bore_radius)
mesh.AddLine('z', [-bore_length / 2, bore_length / 2])
mesh.SmoothMeshLines('r', radial_axial_resolution, ratio=1.5)
mesh.SmoothMeshLines('a', angular_resolution, ratio=1.5)
mesh.SmoothMeshLines('z', radial_axial_resolution, ratio=1.5)

alpha_lines = mesh.GetLines('a')
dump_start = [0, alpha_lines[0], -coil_length / 2]
dump_stop = [coil_radius, alpha_lines[-1], coil_length / 2]

E_field = CSX.AddDump(
    'Ef', file_type=1, dump_type=10, dump_mode=2, frequency=[f0])
E_field.AddBox(dump_start, dump_stop)
H_field = CSX.AddDump(
    'Hf', file_type=1, dump_type=11, dump_mode=2, frequency=[f0])
H_field.AddBox(dump_start, dump_stop)
SAR = CSX.AddDump(
    'SAR', file_type=1, dump_type=20, dump_mode=2, frequency=[f0])
SAR.AddBox(dump_start, dump_stop)

H_time = CSX.AddDump('Ht', file_type=1, dump_type=1, dump_mode=2)
H_time.AddBox(
    [0, alpha_lines[0], 0],
    [coil_radius, alpha_lines[-1], 0])

add_axial_pml_padding(mesh, 10)

cell_count = np.prod([len(mesh.GetLines(axis)) for axis in ('r', 'a', 'z')])
print('Mesh-line product: {:,}'.format(cell_count))


# Generate, inspect, and run ------------------------------------------------
xml_file = os.path.join(sim_path, 'BirdCage.xml')
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

    run_options = {
        'cleanup': True,
        'verbose': 1,
        'setup_only': setup_only,
    }
    engine = os.environ.get('OPENEMS_ENGINE')
    if engine:
        run_options['engine'] = engine
    if os.environ.get('OPENEMS_NUM_THREADS'):
        run_options['numThreads'] = int(os.environ['OPENEMS_NUM_THREADS'])
    if os.environ.get('OPENEMS_GPU_DEVICE'):
        run_options['gpu_device'] = int(os.environ['OPENEMS_GPU_DEVICE'])
    if os.environ.get('OPENEMS_GPU_KERNEL'):
        run_options['gpu_kernel'] = os.environ['OPENEMS_GPU_KERNEL']
    if env_flag('OPENEMS_DUMP_STATISTICS'):
        run_options['dump_statistics'] = True
    FDTD.Run(sim_path, **run_options)

if setup_only or skip_postprocess:
    raise SystemExit(0)


# Port, SAR, and B1 post-processing ----------------------------------------
frequency = np.linspace(
    excitation_f0 - excitation_fc,
    excitation_f0 + excitation_fc,
    201)
s11 = None
s22 = None
try:
    for port in ports:
        port.CalcPort(sim_path, frequency)
    s11 = ports[0].uf_ref / ports[0].uf_inc
    s22 = ports[1].uf_ref / ports[1].uf_inc
except (IndexError, ValueError) as error:
    print('Port spectra unavailable: {}. Increase OPENEMS_NR_TS for at '
          'least two recorded port samples.'.format(error))

if s11 is not None:
    f0_index = np.argmin(np.abs(frequency - f0))
    print('Nearest sample to {:.3f} MHz is {:.3f} MHz'.format(
        f0 / 1e6, frequency[f0_index] / 1e6))
    print('S11={:.3f} dB, S22={:.3f} dB'.format(
        20 * np.log10(np.abs(s11[f0_index])),
        20 * np.log10(np.abs(s22[f0_index]))))

sar, sar_axes = read_frequency_dump(os.path.join(sim_path, 'SAR.h5'))
H, H_axes = read_frequency_dump(
    os.path.join(sim_path, 'Hf.h5'), vector=True)

sar_z_index = np.argmin(np.abs(sar_axes[2]))
H_z_index = np.argmin(np.abs(H_axes[2]))
sar_xy = sar[:, :, sar_z_index]
H_xy = H[:, :, H_z_index, :]

alpha_grid = H_axes[1][None, :]
Br = H_xy[..., 0]
Ba = H_xy[..., 1]
Bx = MUE0 * (Br * np.cos(alpha_grid) - Ba * np.sin(alpha_grid))
By = MUE0 * (Br * np.sin(alpha_grid) + Ba * np.cos(alpha_grid))
B1p = 0.5 * (Bx + 1j * By)
B1m = 0.5 * (Bx - 1j * By)

sar_alpha, (sar_xy,) = close_periodic_alpha(sar_axes[1], sar_xy)
H_alpha, (B1p, B1m) = close_periodic_alpha(H_axes[1], B1p, B1m)

print('Peak axial SAR: {:.6g} W/kg'.format(np.nanmax(sar_xy)))
print('Peak axial |B1+|: {:.6g} uT'.format(
    1e6 * np.nanmax(np.abs(B1p))))
print('Peak axial |B1-|: {:.6g} uT'.format(
    1e6 * np.nanmax(np.abs(B1m))))

if not skip_plots:
    if s11 is not None:
        fig, axis = plt.subplots(
            num='Birdcage S parameters', tight_layout=True)
        axis.plot(frequency / 1e6, 20 * np.log10(np.abs(s11)),
                  'k-', lw=2, label='$S_{11}$')
        axis.plot(frequency / 1e6, 20 * np.log10(np.abs(s22)),
                  'r--', lw=2, label='$S_{22}$')
        axis.set_xlabel('frequency (MHz)')
        axis.set_ylabel('S parameter (dB)')
        axis.grid()
        axis.legend()

    fig, axis = plt.subplots(
        num='Birdcage local SAR', figsize=(6, 5), tight_layout=True)
    plot_cylindrical_plane(
        axis, sar_axes[0], sar_alpha, sar_xy,
        'axial local SAR', 'W/kg')

    max_b1 = max(np.nanmax(np.abs(B1p)), np.nanmax(np.abs(B1m)))
    fig, axes = plt.subplots(
        1, 2, num='Birdcage B1 fields', figsize=(12, 5),
        tight_layout=True)
    plot_cylindrical_plane(
        axes[0], H_axes[0], H_alpha, 1e6 * np.abs(B1p),
        'axial $|B_1^+|$', 'uT')
    plot_cylindrical_plane(
        axes[1], H_axes[0], H_alpha, 1e6 * np.abs(B1m),
        'axial $|B_1^-|$', 'uT')
    if max_b1 > 0:
        for axis in axes:
            for image in axis.collections:
                image.set_clim(0, 1e6 * max_b1)
    plt.show()
