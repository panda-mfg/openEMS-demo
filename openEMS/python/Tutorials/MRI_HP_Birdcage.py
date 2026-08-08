# -*- coding: utf-8 -*-
"""
Tutorials / 3 T MRI high-pass birdcage coil.

The sixteen-rung coil uses 16 tuning capacitors in each segmented end ring
(32 total) and is loaded by a spherical saline phantom. Normal operation
drives two upper-end-ring capacitor gaps in quadrature. Tuning mode drives
only port 1, terminates port 2 in 50 ohms, omits field dumps, and reports the
resonance nearest 128 MHz.

Useful environment variables:

OPENEMS_HP_CAP_PF
    End-ring capacitor value in pF (default: 22.9).
OPENEMS_CAP_Q
    Parallel-loss quality factor for every capacitor, referenced to 128 MHz
    (default: 600).
OPENEMS_FIELD_PORT / OPENEMS_SUPERPOSITION_PARTNER
    For stable exact quadrature, solve feed 1 and feed 2 separately. Set the
    first variable to 1 or 2 and, on the final run, point the second variable
    at the other feed's completed simulation directory.
OPENEMS_BODY_MESH_MM
    Phantom and nominal angular mesh resolution in mm (default: 2.5).
OPENEMS_B1_ROI_RADIUS_MM
    Radius of the central axial ROI (default: 80% of phantom radius).
OPENEMS_TUNING_ONLY
    Enable the single-port, no-field-dump tuning configuration.
OPENEMS_SIM_PATH
    Output directory.
OPENEMS_ENGINE / OPENEMS_NUM_THREADS / OPENEMS_GPU_DEVICE
    Solver engine controls.
OPENEMS_NR_TS / OPENEMS_FIXED_TIMESTEPS
    Maximum timestep count and fixed-step mode.
OPENEMS_GENERATE_ONLY / OPENEMS_POST_ONLY
    Generate XML only or post-process existing output.
OPENEMS_SKIP_POSTPROCESS / OPENEMS_SKIP_PLOTS / OPENEMS_SHOW_GEOMETRY
    Batch and visualization controls.

(C) 2026 openEMS contributors
"""

import json
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
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def add_axial_pml_padding(mesh, number_of_cells=10):
    lines = np.unique(mesh.GetLines('z'))
    lower_delta = lines[1] - lines[0]
    upper_delta = lines[-1] - lines[-2]
    lower = lines[0] - lower_delta * np.arange(number_of_cells, 0, -1)
    upper = lines[-1] + upper_delta * np.arange(1, number_of_cells + 1)
    mesh.SetLines('z', np.r_[lower, lines, upper])


def read_frequency_dump(filename, vector=False):
    with h5py.File(filename, 'r') as h5_file:
        mesh_group = h5_file['/Mesh']
        axis_names = ('rho', 'alpha', 'z')
        axes = tuple(np.asarray(mesh_group[name]) for name in axis_names)
        axis_shape = tuple(len(axis) for axis in axes)
        fd_group = h5_file['/FieldData/FD']
        if vector:
            if 'f0' in fd_group:
                values = np.asarray(fd_group['f0'])
            else:
                values = (
                    np.asarray(fd_group['f0_real'])
                    + 1j * np.asarray(fd_group['f0_imag']))
            if values.shape[0] == 3 and values.shape[1:] == axis_shape:
                values = np.moveaxis(values, 0, -1)
            elif values.shape[0] == 3 and values.shape[1:] == axis_shape[::-1]:
                values = np.transpose(values, (3, 2, 1, 0))
            elif values.shape[-1] != 3 or values.shape[:-1] != axis_shape:
                raise ValueError(
                    'Unsupported vector dump shape: {}'.format(values.shape))
        else:
            values = np.asarray(fd_group['f0'])
            if values.shape == axis_shape[::-1]:
                values = np.transpose(values, (2, 1, 0))
            elif values.shape != axis_shape:
                raise ValueError(
                    'Unsupported scalar dump shape: {}'.format(values.shape))
    return values, axes


def close_periodic_alpha(alpha, *fields):
    closed_alpha = np.r_[alpha, alpha[0] + 2 * np.pi]
    closed_fields = [
        np.concatenate((field, field[:, :1]), axis=1) for field in fields
    ]
    return closed_alpha, closed_fields


def interpolate_angular_field(alpha, field, target_alpha):
    """Linearly interpolate a radius-by-alpha field at one angle."""
    target_alpha = (
        (target_alpha - alpha[0]) % (2 * np.pi) + alpha[0])
    upper = np.searchsorted(alpha, target_alpha, side='right')
    upper = min(max(upper, 1), len(alpha) - 1)
    lower = upper - 1
    fraction = (
        (target_alpha - alpha[lower])
        / (alpha[upper] - alpha[lower]))
    return field[:, lower] + fraction * (field[:, upper] - field[:, lower])


def sample_edges(samples, nonnegative=False):
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


def weighted_quantile(values, weights, quantiles):
    """Return area-weighted quantiles of finite values."""
    values = np.asarray(values).ravel()
    weights = np.asarray(weights).ravel()
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[finite]
    weights = weights[finite]
    if values.size == 0:
        raise ValueError('ROI contains no finite, positively weighted samples')
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative = (cumulative - 0.5 * weights[order]) / cumulative[-1]
    return np.interp(quantiles, cumulative, values)


def weighted_field_statistics(values, weights):
    """Summarize an axial field magnitude using cylindrical cell areas."""
    values = np.asarray(values).ravel()
    weights = np.asarray(weights).ravel()
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[finite]
    weights = weights[finite]
    total_weight = np.sum(weights)
    mean = np.sum(weights * values) / total_weight
    standard_deviation = np.sqrt(
        np.sum(weights * (values - mean) ** 2) / total_weight)
    p05, median, p95 = weighted_quantile(
        values, weights, [0.05, 0.50, 0.95])
    uniformity = (
        100 * (1 - (p95 - p05) / (p95 + p05))
        if p95 + p05 > 0 else float('nan'))
    return {
        'mean_uT_per_sqrt_W': float(mean),
        'std_uT_per_sqrt_W': float(standard_deviation),
        'coefficient_of_variation_percent': float(
            100 * standard_deviation / mean if mean > 0 else np.nan),
        'minimum_uT_per_sqrt_W': float(np.min(values)),
        'p05_uT_per_sqrt_W': float(p05),
        'median_uT_per_sqrt_W': float(median),
        'p95_uT_per_sqrt_W': float(p95),
        'maximum_uT_per_sqrt_W': float(np.max(values)),
        'robust_uniformity_percent': float(uniformity),
    }


def resonance_from_reactance(frequency, impedance, target_frequency):
    """Return the nearest positive-resistance input-reactance zero."""
    reactance = np.imag(impedance)
    resistance = np.real(impedance)
    finite = np.isfinite(reactance) & np.isfinite(resistance)
    frequency = frequency[finite]
    reactance = reactance[finite]
    resistance = resistance[finite]
    crossings = []
    for index in range(len(reactance) - 1):
        x0 = reactance[index]
        x1 = reactance[index + 1]
        if x0 == 0 and resistance[index] > 0:
            crossings.append(frequency[index])
        elif x0 * x1 < 0:
            fraction = -x0 / (x1 - x0)
            crossing_resistance = (
                resistance[index]
                + fraction * (resistance[index + 1] - resistance[index]))
            if crossing_resistance > 0:
                crossings.append(
                    frequency[index]
                    + fraction * (frequency[index + 1] - frequency[index]))
    if crossings:
        return min(crossings, key=lambda value: abs(value - target_frequency))
    return frequency[np.argmin(np.abs(reactance))]


def _series_component_from_reactance(reactance, omega):
    if reactance > 0:
        return {'kind': 'inductor', 'value_si': float(reactance / omega)}
    return {
        'kind': 'capacitor',
        'value_si': float(-1 / (omega * reactance))}


def _shunt_component_from_susceptance(susceptance, omega):
    if susceptance > 0:
        return {'kind': 'capacitor', 'value_si': float(susceptance / omega)}
    return {
        'kind': 'inductor',
        'value_si': float(-1 / (omega * susceptance))}


def _series_component_impedance(component, frequency):
    omega = 2 * np.pi * np.asarray(frequency)
    with np.errstate(divide='ignore', invalid='ignore'):
        if component['kind'] == 'inductor':
            return 1j * omega * component['value_si']
        return 1 / (1j * omega * component['value_si'])


def _shunt_component_admittance(component, frequency):
    omega = 2 * np.pi * np.asarray(frequency)
    with np.errstate(divide='ignore', invalid='ignore'):
        if component['kind'] == 'capacitor':
            return 1j * omega * component['value_si']
        return 1 / (1j * omega * component['value_si'])


def apply_external_l_match(load_impedance, frequency, match):
    """Transform simulated coil impedance through an ideal external L-match."""
    series_impedance = _series_component_impedance(
        match['series_component'], frequency)
    shunt_admittance = _shunt_component_admittance(
        match['shunt_component'], frequency)
    with np.errstate(divide='ignore', invalid='ignore'):
        if match['topology'] == 'load_shunt_then_series':
            shunted_load = 1 / (1 / load_impedance + shunt_admittance)
            return series_impedance + shunted_load
        series_load = series_impedance + load_impedance
        return 1 / (1 / series_load + shunt_admittance)


def synthesize_external_l_match(
        load_impedance, reference_impedance, frequency):
    """Choose an ideal two-reactance L-match at one design frequency."""
    if reference_impedance <= 0 or frequency <= 0:
        raise ValueError('Matching impedance and frequency must be positive')
    if not np.isfinite(load_impedance) or np.real(load_impedance) <= 0:
        raise ValueError('Matching requires a finite positive-resistance load')

    omega = 2 * np.pi * frequency
    candidates = []

    def add_candidate(topology, series_reactance, shunt_susceptance):
        if abs(series_reactance) < 1e-12:
            return
        if abs(shunt_susceptance) < 1e-15:
            return
        series_component = _series_component_from_reactance(
            series_reactance, omega)
        shunt_component = _shunt_component_from_susceptance(
            shunt_susceptance, omega)
        candidates.append({
            'topology': topology,
            'series_component': series_component,
            'shunt_component': shunt_component,
            'series_reactance_ohm_at_target': float(series_reactance),
            'shunt_susceptance_s_at_target': float(shunt_susceptance),
            'reactive_stress_score': float(
                abs(series_reactance) / reference_impedance
                + abs(1 / shunt_susceptance) / reference_impedance),
        })

    load_admittance = 1 / load_impedance
    conductance = float(np.real(load_admittance))
    load_side_radicand = (
        conductance / reference_impedance - conductance ** 2)
    if conductance > 0 and load_side_radicand >= -1e-15:
        required_susceptance = np.sqrt(max(0.0, load_side_radicand))
        for sign in (-1, 1):
            total_susceptance = sign * required_susceptance
            shunt_susceptance = (
                total_susceptance - np.imag(load_admittance))
            shunted_load = 1 / (
                load_admittance + 1j * shunt_susceptance)
            add_candidate(
                'load_shunt_then_series',
                -np.imag(shunted_load),
                shunt_susceptance)

    load_resistance = float(np.real(load_impedance))
    source_side_radicand = (
        load_resistance * reference_impedance - load_resistance ** 2)
    if load_resistance > 0 and source_side_radicand >= -1e-12:
        required_reactance = np.sqrt(max(0.0, source_side_radicand))
        for sign in (-1, 1):
            total_reactance = sign * required_reactance
            series_reactance = (
                total_reactance - np.imag(load_impedance))
            shunt_susceptance = (
                total_reactance
                / (load_resistance * reference_impedance))
            add_candidate(
                'series_then_source_shunt',
                series_reactance,
                shunt_susceptance)

    if not candidates:
        raise ValueError(
            'No two-component external L-match exists for this load')

    for candidate in candidates:
        expected_impedance = apply_external_l_match(
            load_impedance, frequency, candidate)
        candidate['expected_z_real_ohm'] = float(np.real(expected_impedance))
        candidate['expected_z_imag_ohm'] = float(np.imag(expected_impedance))
        candidate['target_error_ohm'] = float(
            abs(expected_impedance - reference_impedance))

    # Prefer a conventional L/C pair, then the candidate with less reactive
    # stress. Both choices are analytically matched at the design frequency.
    return min(
        candidates,
        key=lambda candidate: (
            candidate['series_component']['kind']
            == candidate['shunt_component']['kind'],
            candidate['reactive_stress_score']))


def matching_component_text(component):
    if component['kind'] == 'inductor':
        return 'L={:.6g} nH'.format(component['value_si'] * 1e9)
    return 'C={:.6g} pF'.format(component['value_si'] * 1e12)

# Controls -----------------------------------------------------------------
sim_path = os.path.abspath(os.environ.get(
    'OPENEMS_SIM_PATH',
    os.path.join(tempfile.gettempdir(), 'MRI_HP_Birdcage')))
capacitance_pf = float(os.environ.get('OPENEMS_HP_CAP_PF', '22.9'))
capacitance = capacitance_pf * 1e-12
capacitor_q = float(os.environ.get('OPENEMS_CAP_Q', '600'))
if capacitor_q <= 0:
    raise ValueError('OPENEMS_CAP_Q must be positive')
tuning_only = env_flag('OPENEMS_TUNING_ONLY')
field_port = int(os.environ.get('OPENEMS_FIELD_PORT', '0'))
if field_port not in (0, 1, 2):
    raise ValueError('OPENEMS_FIELD_PORT must be 0, 1, or 2')
superposition_partner = os.environ.get('OPENEMS_SUPERPOSITION_PARTNER')
if superposition_partner:
    superposition_partner = os.path.abspath(superposition_partner)
    if field_port == 0:
        raise ValueError(
            'OPENEMS_SUPERPOSITION_PARTNER requires OPENEMS_FIELD_PORT=1 or 2')
nr_ts = int(os.environ.get('OPENEMS_NR_TS', '1000000000'))
fixed_timesteps = env_flag('OPENEMS_FIXED_TIMESTEPS')
timestep_factor = float(os.environ.get('OPENEMS_TIMESTEP_FACTOR', '1.0'))
if timestep_factor <= 0 or timestep_factor > 1:
    raise ValueError('OPENEMS_TIMESTEP_FACTOR must be in the interval (0, 1]')
generate_only = env_flag('OPENEMS_GENERATE_ONLY')
post_proc_only = env_flag('OPENEMS_POST_ONLY')
skip_postprocess = env_flag('OPENEMS_SKIP_POSTPROCESS')
skip_plots = env_flag('OPENEMS_SKIP_PLOTS')
show_geometry = env_flag('OPENEMS_SHOW_GEOMETRY')
multigrid_enabled = env_flag('OPENEMS_MULTIGRID', True)

if generate_only and post_proc_only:
    raise ValueError('OPENEMS_GENERATE_ONLY and OPENEMS_POST_ONLY are exclusive')

print('Simulation path: {}'.format(sim_path))
print('End-ring capacitor: {:.6g} pF'.format(capacitance_pf))
if tuning_only:
    mode_description = 'single-port tuning'
elif field_port:
    mode_description = 'single-port field solution {}'.format(field_port)
else:
    mode_description = 'simultaneous quadrature'
print('Mode: {}'.format(mode_description))
print('Capacitor parallel-loss Q at 128 MHz: {:.6g}'.format(capacitor_q))
feed_topology_description = (
    '50-ohm voltage-source port in parallel with end-ring tuning C||R')
input_plot_title = 'Direct-fed birdcage: before and after external matching'
print('Feed topology: {}'.format(feed_topology_description))

# Physical and geometric parameters ----------------------------------------
unit = 1e-3
target_frequency = 128e6
excitation_f0 = 75e6
excitation_fc = 75e6
end_ring_cap_parallel_r = (
    capacitor_q / (2 * np.pi * target_frequency * capacitance))

bore_radius = 320
bore_length = 1600
number_of_rungs = 16
coil_radius = 120
strip_width = 10
capacitor_width = strip_width / 2
coil_length = 250

# Feed upper-end-ring capacitor gaps 2 and 6. With 16 segments, these gaps
# are separated by four pitches, or 90 degrees. The ports span the same
# angular gaps as the tuning capacitors and therefore do not interrupt any
# rung.
feed_positions = (2, 6)
feed_ring_z = coil_length / 2
# For B1+ = (Bx + j*By)/2, port 2 must lag port 1 by 90 degrees.
# The -j amplitude below is implemented as a positive quarter-period delay.
if tuning_only or field_port == 1:
    feed_amplitudes = (1.0 + 0j, 0j)
elif field_port == 2:
    feed_amplitudes = (0j, 1.0 + 0j)
else:
    feed_amplitudes = (1.0 + 0j, -1j)

phantom_center = np.array([0.0, 0.0, 0.0])
phantom_radius = 90
phantom_epsilon = 78
phantom_kappa = 0.6
phantom_density = 1000
body_mesh_resolution = float(os.environ.get('OPENEMS_BODY_MESH_MM', '2.5'))
if body_mesh_resolution <= 0:
    raise ValueError('OPENEMS_BODY_MESH_MM must be positive')

lambda_min = C0 / (excitation_f0 + excitation_fc)
radial_axial_resolution = min(15, lambda_min / 20 / unit)
angular_resolution = body_mesh_resolution / coil_radius * (1 + 1e-12)


# Cylindrical multigrid solver ---------------------------------------------
fdtd_options = {
    'NrTS': nr_ts,
    'CoordSystem': 1,
    'EndCriteria': 0.0 if fixed_timesteps else 1e-4,
    'TimeStepFactor': timestep_factor,
    'CellConstantMaterial': True,
    'OverSampling': 100,
}
if multigrid_enabled:
    fdtd_options['MultiGrid'] = [10.0, 20.0]
FDTD = openEMS(**fdtd_options)
FDTD.SetGaussExcite(excitation_f0, excitation_fc)
FDTD.SetBoundaryCond([0, 0, 0, 0, 3, 3])

CSX = ContinuousStructure(CoordSystem=1)
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)


# High-pass birdcage geometry ----------------------------------------------
metal = CSX.AddMetal('metal')
end_ring_caps = CSX.AddLumpedElement(
    'end_ring_caps', ny='a', caps=False, C=capacitance,
    R=end_ring_cap_parallel_r)

rung_pitch = 2 * np.pi / number_of_rungs
alpha_start = -np.pi
alpha_stop = alpha_start + 2 * np.pi
angular_strip_width = strip_width / coil_radius
angular_capacitor_width = capacitor_width / coil_radius
omega0 = 2 * np.pi * target_frequency
period0 = 1 / target_frequency


def periodic_ranges(range_start, range_stop):
    """Split an angular interval only when it crosses the periodic seam."""
    if range_start < alpha_start:
        return [
            (alpha_start, range_stop),
            (range_start + 2 * np.pi, alpha_stop)]
    if range_stop > alpha_stop:
        return [
            (range_start, alpha_stop),
            (alpha_start, range_stop - 2 * np.pi)]
    return [(range_start, range_stop)]


ports = []
end_ring_capacitor_count = 0
capacitor_loss_resistor_count = 0
for rung_index in range(number_of_rungs):
    rung_number = rung_index + 1
    rung_alpha = alpha_start + rung_index * rung_pitch
    lower_start = -coil_length / 2 + strip_width / 2
    upper_stop = coil_length / 2 - strip_width / 2

    rung_start = rung_alpha - angular_strip_width / 2
    rung_stop = rung_alpha + angular_strip_width / 2
    for range_start, range_stop in periodic_ranges(rung_start, rung_stop):
        metal.AddBox(
            [coil_radius, range_start, lower_start],
            [coil_radius, range_stop, upper_stop],
            priority=1)
    mesh.AddLine('a', rung_alpha)

    # One C||R tuning element per segment in each end ring. A feed port uses
    # the same gap as its capacitor, so the voltage source is in parallel.
    cap_alpha = rung_alpha + rung_pitch / 2
    for ring_z in (-coil_length / 2, coil_length / 2):
        cap_start = [
            coil_radius,
            cap_alpha - angular_capacitor_width / 2,
            ring_z - strip_width / 2]
        cap_stop = [
            coil_radius,
            cap_alpha + angular_capacitor_width / 2,
            ring_z + strip_width / 2]
        end_ring_caps.AddBox(cap_start, cap_stop, priority=10)
        end_ring_capacitor_count += 1
        capacitor_loss_resistor_count += 1

        if ring_z == feed_ring_z and rung_number in feed_positions:
            feed_index = feed_positions.index(rung_number)
            feed_amplitude = feed_amplitudes[feed_index]
            delay = (
                -np.angle(feed_amplitude) / omega0
                if feed_amplitude else 0)
            if delay < 0:
                delay += period0

            port_start = cap_start
            port_stop = cap_stop

            port = FDTD.AddLumpedPort(
                port_nr=feed_index + 1,
                R=50,
                start=port_start,
                stop=port_stop,
                p_dir='a',
                excite=abs(feed_amplitude),
                priority=100,
                delay=delay)
            ports.append(port)

    # Conductive end-ring arcs leave one uniform capacitor gap per segment.

    lower_arc_start = (
        rung_alpha - rung_pitch / 2 + angular_capacitor_width / 2)
    lower_arc_stop = (
        rung_alpha + rung_pitch / 2 - angular_capacitor_width / 2)
    for range_start, range_stop in periodic_ranges(
            lower_arc_start, lower_arc_stop):
        metal.AddBox(
            [coil_radius, range_start,
             -coil_length / 2 - strip_width / 2],
            [coil_radius, range_stop,
             -coil_length / 2 + strip_width / 2],
            priority=1)

    upper_arc_start = (
        rung_alpha - rung_pitch / 2 + angular_capacitor_width / 2)
    upper_arc_stop = (
        rung_alpha + rung_pitch / 2 - angular_capacitor_width / 2)
    for range_start, range_stop in periodic_ranges(
            upper_arc_start, upper_arc_stop):
        metal.AddBox(
            [coil_radius, range_start,
             coil_length / 2 - strip_width / 2],
            [coil_radius, range_stop,
             coil_length / 2 + strip_width / 2],
            priority=1)

assert end_ring_capacitor_count == 2 * number_of_rungs
assert capacitor_loss_resistor_count == 2 * number_of_rungs
assert len(ports) == len(feed_positions)
print('End-ring tuning capacitances: {} per ring ({} total)'.format(
    number_of_rungs, end_ring_capacitor_count))
print('Parallel capacitor-loss resistors: {} total'.format(
    capacitor_loss_resistor_count))
print('Feed ports: upper end-ring gaps {} (90 degrees apart)'.format(
    feed_positions))


# Saline load ---------------------------------------------------------------
saline = CSX.AddMaterial(
    'saline_phantom',
    epsilon=phantom_epsilon,
    kappa=phantom_kappa,
    density=phantom_density)
saline.AddSphere(priority=1, center=phantom_center, radius=phantom_radius)


# Mesh ---------------------------------------------------------------------
geometry_mesh_properties = [metal, end_ring_caps]
FDTD.AddEdges2Grid(dirs='all', properties=geometry_mesh_properties)
for port in ports:
    FDTD.AddEdges2Grid(dirs='all', properties=port.port_props)

radial_seed = np.unique(np.r_[
    body_mesh_resolution * 1.5,
    phantom_radius,
    mesh.GetLines('r')])
mesh.SetLines(
    'r',
    np.unique(np.r_[
        0,
        SmoothMeshLines(
            radial_seed, body_mesh_resolution, ratio=1.5)]))

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

# Every cylindrical multigrid coarsening halves the periodic alpha cells.
# Add symmetric refinement lines until the outer-grid count is divisible by
# 2**len(MultiGrid)=4; using eight also keeps every active grid even-sized.
alpha_lines = np.asarray(mesh.GetLines('a'))
extra_alpha_lines = (-(len(alpha_lines) - 1)) % 8
if extra_alpha_lines:
    alpha_cell_indices = np.floor(
        (np.arange(extra_alpha_lines) + 0.5)
        * (len(alpha_lines) - 1) / extra_alpha_lines).astype(int)
    alpha_midpoints = 0.5 * (
        alpha_lines[alpha_cell_indices]
        + alpha_lines[alpha_cell_indices + 1])
    mesh.AddLine('a', alpha_midpoints)
    print('Added {} alpha refinement lines for cylindrical multigrid'.format(
        extra_alpha_lines))
assert (len(mesh.GetLines('a')) - 1) % 8 == 0
mesh.SmoothMeshLines('z', radial_axial_resolution, ratio=1.5)

# Put the isocenter at the center of an axial Yee cell instead of on a mesh
# boundary. H-field samples are cell-centered in z, so this guarantees an
# exact z=0 sample for the dedicated axial dump.
z_lines = np.asarray(mesh.GetLines('z'))
z_center = phantom_center[2]
z_lines = z_lines[np.abs(z_lines - z_center) > 1e-9]
z_lines = np.unique(np.r_[
    z_lines,
    z_center - body_mesh_resolution / 2,
    z_center + body_mesh_resolution / 2])
mesh.SetLines('z', z_lines)

if not tuning_only:
    alpha_lines = mesh.GetLines('a')
    dump_start = [0, alpha_lines[0], -coil_length / 2]
    dump_stop = [coil_radius, alpha_lines[-1], coil_length / 2]
    H_field = CSX.AddDump(
        'Hf', file_type=1, dump_type=11, dump_mode=2,
        frequency=[target_frequency])
    H_field.AddBox(dump_start, dump_stop)
    H_isocenter = CSX.AddDump(
        'Hf_isocenter', file_type=1, dump_type=11, dump_mode=2,
        frequency=[target_frequency])
    H_isocenter.AddBox(
        [0, alpha_lines[0], phantom_center[2]],
        [coil_radius, alpha_lines[-1], phantom_center[2]])
    SAR = CSX.AddDump(
        'SAR', file_type=1, dump_type=20, dump_mode=2,
        frequency=[target_frequency])
    SAR.AddBox(dump_start, dump_stop)

add_axial_pml_padding(mesh, 10)

mesh_product = np.prod(
    [len(mesh.GetLines(axis)) for axis in ('r', 'a', 'z')])
print('Mesh-line product: {:,}'.format(mesh_product))


# Generate and run ----------------------------------------------------------
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

    # Reload the serialized cylindrical model before setup. Newly created
    # Python box primitives can otherwise retain Cartesian input coordinates
    # in memory, while XML loading correctly applies the grid coordinate
    # system. Reloading also guarantees that the solver uses exactly the model
    # reviewed in AppCSXCAD.
    run_FDTD = openEMS()
    run_FDTD.ReadFromXML(xml_file)
    run_FDTD.SetTimeStepFactor(timestep_factor)

    run_options = {'cleanup': True, 'verbose': 1}
    engine = os.environ.get('OPENEMS_ENGINE')
    if engine:
        run_options['engine'] = engine
    if os.environ.get('OPENEMS_NUM_THREADS'):
        run_options['numThreads'] = int(os.environ['OPENEMS_NUM_THREADS'])
    if os.environ.get('OPENEMS_GPU_DEVICE'):
        run_options['gpu_device'] = int(os.environ['OPENEMS_GPU_DEVICE'])
    if env_flag('OPENEMS_DUMP_STATISTICS'):
        run_options['dump_statistics'] = True
    run_FDTD.Run(sim_path, **run_options)

if skip_postprocess:
    raise SystemExit(0)


# Port spectra and tuning metric -------------------------------------------
frequency = np.unique(np.r_[
    np.linspace(
        excitation_f0 - excitation_fc,
        excitation_f0 + excitation_fc,
        1001),
    target_frequency])
def capture_port_data(port):
    """Copy port spectra before CalcPort is reused for another run."""
    names = (
        'uf_tot', 'if_tot', 'uf_inc', 'if_inc', 'uf_ref', 'if_ref',
        'P_inc', 'P_ref', 'P_acc')
    return {name: np.array(getattr(port, name), copy=True) for name in names}


for port in ports:
    port.CalcPort(sim_path, frequency)
current_port_data = [capture_port_data(port) for port in ports]

superposition_ready = bool(field_port and superposition_partner)
if field_port and not superposition_ready:
    raise ValueError(
        'A field-port result must be post-processed with '
        'OPENEMS_SUPERPOSITION_PARTNER set to the other completed run')
if superposition_ready:
    if not os.path.isdir(superposition_partner):
        raise FileNotFoundError(
            'Superposition partner does not exist: ' + superposition_partner)
    for port in ports:
        port.CalcPort(superposition_partner, frequency)
    partner_port_data = [capture_port_data(port) for port in ports]
    if field_port == 1:
        port_1_data = current_port_data
        port_2_data = partner_port_data
        port_1_path = sim_path
        port_2_path = superposition_partner
    else:
        port_1_data = partner_port_data
        port_2_data = current_port_data
        port_1_path = superposition_partner
        port_2_path = sim_path
    passive_input_data = port_1_data[0]
    print('Exact quadrature superposition inputs: {} and {}'.format(
        port_1_path, port_2_path))
else:
    passive_input_data = current_port_data[0]

input_impedance = passive_input_data['uf_tot'] / passive_input_data['if_tot']
s11 = passive_input_data['uf_ref'] / passive_input_data['uf_inc']
search_mask = (frequency >= 10e6) & (frequency <= 148e6)
resonance = resonance_from_reactance(
    frequency[search_mask],
    input_impedance[search_mask],
    target_frequency)
resonance_index = np.argmin(np.abs(frequency - resonance))
tuning_result = {
    'capacitance_pf': capacitance_pf,
    'resonance_hz': float(resonance),
    'target_hz': target_frequency,
    'error_hz': float(resonance - target_frequency),
    's11_db': float(20 * np.log10(np.abs(s11[resonance_index]))),
    'zin_real_ohm': float(np.real(input_impedance[resonance_index])),
    'zin_imag_ohm': float(np.imag(input_impedance[resonance_index])),
}
print('Resonance: {:.6f} MHz (error {:+.3f} MHz)'.format(
    resonance / 1e6, (resonance - target_frequency) / 1e6))
print('At resonance: S11={:.3f} dB, Zin={:.4g}{:+.4g}j ohm'.format(
    tuning_result['s11_db'],
    tuning_result['zin_real_ohm'],
    tuning_result['zin_imag_ohm']))

with open(os.path.join(sim_path, 'tuning_result.json'), 'w') as result_file:
    json.dump(tuning_result, result_file, indent=2)

if tuning_only or superposition_ready:
    target_index = np.argmin(np.abs(frequency - target_frequency))
    target_impedance = input_impedance[target_index]
    reference_impedance = 50.0
    target_s11_before_db = float(20 * np.log10(
        max(abs(s11[target_index]), 1e-15)))

    external_match = synthesize_external_l_match(
        target_impedance, reference_impedance, target_frequency)
    matched_input_impedance = apply_external_l_match(
        input_impedance, frequency, external_match)
    matched_s11 = (
        (matched_input_impedance - reference_impedance)
        / (matched_input_impedance + reference_impedance))
    target_matched_impedance = matched_input_impedance[target_index]
    target_s11_after_db = float(20 * np.log10(
        max(abs(matched_s11[target_index]), 1e-15)))

    if external_match['topology'] == 'load_shunt_then_series':
        topology_description = (
            'coil -> shunt component across coil -> series component -> '
            '50-ohm source')
        shunt_location = 'directly across coil input'
    else:
        topology_description = (
            'coil -> series component -> shunt component at source -> '
            '50-ohm source')
        shunt_location = 'across 50-ohm source-side input'

    external_matching_result = {
        'model': 'direct-fed full-wave coil plus ideal postprocessed L-match',
        'physical_matching_enabled': False,
        'external_matching_postprocessed': True,
        'external_matching_assumptions': (
            'ideal lossless lumped components; coil spectrum from full-wave '
            'simulation is unchanged'),
        'feed_topology': feed_topology_description,
        'frequency_hz': target_frequency,
        'reference_impedance_ohm': reference_impedance,
        'end_ring_capacitance_pf': capacitance_pf,
        'capacitor_q_at_target_frequency': capacitor_q,
        'end_ring_cap_parallel_resistance_ohm': end_ring_cap_parallel_r,
        'external_match_topology': external_match['topology'],
        'external_match_topology_description': topology_description,
        'external_shunt_component_location': shunt_location,
        'external_series_component': external_match['series_component'],
        'external_shunt_component': external_match['shunt_component'],
        'series_reactance_ohm_at_target': (
            external_match['series_reactance_ohm_at_target']),
        'shunt_susceptance_s_at_target': (
            external_match['shunt_susceptance_s_at_target']),
        'before_matching': {
            'zin_real_ohm': float(np.real(target_impedance)),
            'zin_imag_ohm': float(np.imag(target_impedance)),
            's11_db': target_s11_before_db,
        },
        'after_ideal_external_matching': {
            'zin_real_ohm': float(np.real(target_matched_impedance)),
            'zin_imag_ohm': float(np.imag(target_matched_impedance)),
            's11_db': target_s11_after_db,
        },
    }
    external_result_json = os.path.join(
        sim_path, 'external_matching_result.json')
    with open(external_result_json, 'w') as matching_file:
        json.dump(external_matching_result, matching_file, indent=2)

    print('Direct-fed Zin at {:.3f} MHz: '
          '{:.4g}{:+.4g}j ohm, S11={:.3f} dB'.format(
              target_frequency / 1e6,
              np.real(target_impedance),
              np.imag(target_impedance),
              target_s11_before_db))
    print('Expected ideal external match: {}'.format(topology_description))
    print('  series {}'.format(matching_component_text(
        external_match['series_component'])))
    print('  shunt {} ({})'.format(
        matching_component_text(external_match['shunt_component']),
        shunt_location))
    print('Expected matched Zin: {:.4g}{:+.4g}j ohm, S11={:.3f} dB'.format(
        np.real(target_matched_impedance),
        np.imag(target_matched_impedance),
        target_s11_after_db))
    print('Saved external matching values: {}'.format(external_result_json))

    plot_mask = frequency >= 10e6
    plot_frequency = frequency[plot_mask] / 1e6
    s11_before_db = 20 * np.log10(
        np.maximum(np.abs(s11[plot_mask]), 1e-15))
    s11_after_db = 20 * np.log10(
        np.maximum(np.abs(matched_s11[plot_mask]), 1e-15))
    spectrum_csv = os.path.join(
        sim_path, 'S11_impedance_before_after.csv')
    np.savetxt(
        spectrum_csv,
        np.column_stack((
            frequency[plot_mask],
            s11_before_db,
            np.real(input_impedance[plot_mask]),
            np.imag(input_impedance[plot_mask]),
            s11_after_db,
            np.real(matched_input_impedance[plot_mask]),
            np.imag(matched_input_impedance[plot_mask]))),
        delimiter=',',
        header=(
            'frequency_hz,S11_before_db,Zin_before_real_ohm,'
            'Zin_before_imag_ohm,S11_after_db,Zin_after_real_ohm,'
            'Zin_after_imag_ohm'),
        comments='')
    print('Saved before/after S11 and impedance data: {}'.format(
        spectrum_csv))

    input_figure, input_axes = plt.subplots(
        2, 1, num='High-pass birdcage input and external match',
        figsize=(9, 8), tight_layout=True, sharex=True)
    input_axes[0].plot(
        plot_frequency, s11_before_db, color='0.45', label='before matching')
    input_axes[0].plot(
        plot_frequency, s11_after_db, 'g-', label='after ideal matching')
    input_axes[0].axhline(-10, color='0.5', lw=1, ls=':')
    input_axes[0].axvline(target_frequency / 1e6, color='r', ls='--')
    input_axes[0].set_ylabel('S11 (dB)')
    input_axes[0].set_ylim(-80, 5)
    input_axes[0].set_title(input_plot_title)
    input_axes[0].grid()
    input_axes[0].legend()

    input_axes[1].plot(
        plot_frequency, np.real(input_impedance[plot_mask]),
        color='0.35', ls='--', label='Re(Zin), before')
    input_axes[1].plot(
        plot_frequency, np.imag(input_impedance[plot_mask]),
        color='0.65', ls='--', label='Im(Zin), before')
    input_axes[1].plot(
        plot_frequency, np.real(matched_input_impedance[plot_mask]),
        'b-', label='Re(Zin), after')
    input_axes[1].plot(
        plot_frequency, np.imag(matched_input_impedance[plot_mask]),
        'm-', label='Im(Zin), after')
    input_axes[1].axhline(0, color='k', lw=1)
    input_axes[1].axhline(reference_impedance, color='0.5', lw=1, ls=':')
    input_axes[1].axvline(target_frequency / 1e6, color='r', ls='--')
    input_axes[1].set_xlabel('frequency (MHz)')
    input_axes[1].set_ylabel('Impedance (ohm)')
    input_axes[1].grid()
    input_axes[1].legend()
    input_png = os.path.join(
        sim_path, 'S11_impedance_before_after.png')
    input_figure.savefig(input_png, dpi=200, bbox_inches='tight')
    print('Saved before/after S11 and impedance plot: {}'.format(input_png))
    if skip_plots or superposition_ready:
        plt.close(input_figure)
    else:
        plt.show()
    if tuning_only:
        raise SystemExit(0)


# Quadrature B1 maps -------------------------------------------------------
target_index = np.argmin(np.abs(frequency - target_frequency))


def isocenter_field_path(path):
    filename = os.path.join(path, 'Hf_isocenter.h5')
    if not os.path.isfile(filename):
        filename = os.path.join(path, 'Hf.h5')
    return filename


if superposition_ready:
    # The two simulations have identical geometry, terminations, waveforms,
    # and direct parallel feeds; only the active source differs. Maxwell's
    # equations are linear, so complex field and port quantities superpose.
    H1, H_axes = read_frequency_dump(
        isocenter_field_path(port_1_path), vector=True)
    H2, H2_axes = read_frequency_dump(
        isocenter_field_path(port_2_path), vector=True)
    if H1.shape != H2.shape or any(
            not np.allclose(axis_1, axis_2)
            for axis_1, axis_2 in zip(H_axes, H2_axes)):
        raise ValueError('Field-port dump grids do not match')

    s22 = port_2_data[1]['uf_ref'] / port_2_data[1]['uf_inc']
    H_z_index = np.argmin(np.abs(H_axes[2] - phantom_center[2] * unit))
    alpha_grid = H_axes[1][None, :]
    polarity_candidates = []
    for candidate_phase in (-1j, 1j):
        candidate_H = H1 + candidate_phase * H2
        candidate_xy = candidate_H[:, :, H_z_index, :]
        candidate_Br = candidate_xy[..., 0]
        candidate_Ba = candidate_xy[..., 1]
        candidate_Bx = MUE0 * (
            candidate_Br * np.cos(alpha_grid)
            - candidate_Ba * np.sin(alpha_grid))
        candidate_By = MUE0 * (
            candidate_Br * np.sin(alpha_grid)
            + candidate_Ba * np.cos(alpha_grid))
        candidate_B1p = 0.5 * (candidate_Bx + 1j * candidate_By)
        candidate_B1m = 0.5 * (candidate_Bx - 1j * candidate_By)
        phantom_samples = H_axes[0] <= phantom_radius * unit + 1e-15
        polarity_score = (
            np.mean(np.abs(candidate_B1p[phantom_samples]))
            / max(np.mean(np.abs(candidate_B1m[phantom_samples])), 1e-30))

        candidate_u_tot = [
            port_1_data[index]['uf_tot']
            + candidate_phase * port_2_data[index]['uf_tot']
            for index in range(len(ports))]
        candidate_i_tot = [
            port_1_data[index]['if_tot']
            + candidate_phase * port_2_data[index]['if_tot']
            for index in range(len(ports))]
        candidate_power = float(sum(
            0.5 * np.real(
                candidate_u_tot[index][target_index]
                * np.conj(candidate_i_tot[index][target_index]))
            for index in range(len(ports))))
        if np.isfinite(candidate_power) and candidate_power > 0:
            polarity_candidates.append((
                polarity_score, candidate_phase, candidate_H,
                candidate_u_tot, candidate_i_tot, candidate_power))

    if not polarity_candidates:
        raise ValueError('No quadrature polarity produced positive accepted power')
    (polarity_score, quadrature_phase, H, combined_u_tot, combined_i_tot,
     accepted_power) = max(polarity_candidates, key=lambda item: item[0])
    combined_u_inc = [
        port_1_data[index]['uf_inc']
        + quadrature_phase * port_2_data[index]['uf_inc']
        for index in range(len(ports))]
    combined_u_ref = [
        port_1_data[index]['uf_ref']
        + quadrature_phase * port_2_data[index]['uf_ref']
        for index in range(len(ports))]
    active_reflection = [
        combined_u_ref[index] / combined_u_inc[index]
        for index in range(len(ports))]
    current_ratio = (
        combined_i_tot[1][target_index] / combined_i_tot[0][target_index])
    quadrature_method = 'exact linear superposition of two single-port runs'
    sar_available = False
    print('Quadrature polarity check selected port-2 phase {:+.1f} deg '
          '(mean |B1+|/|B1-| score {:.6g})'.format(
              np.angle(quadrature_phase, deg=True), polarity_score))
else:
    s22 = (
        current_port_data[1]['uf_ref'] / current_port_data[1]['uf_inc'])
    H, H_axes = read_frequency_dump(
        isocenter_field_path(sim_path), vector=True)
    accepted_power = float(sum(
        data['P_acc'][target_index] for data in current_port_data))
    current_ratio = (
        current_port_data[1]['if_tot'][target_index]
        / current_port_data[0]['if_tot'][target_index])
    active_reflection = [s11, s22]
    quadrature_phase = -1j
    quadrature_method = 'simultaneous two-port excitation'
    sar, sar_axes = read_frequency_dump(os.path.join(sim_path, 'SAR.h5'))
    sar_z_index = np.argmin(
        np.abs(sar_axes[2] - phantom_center[2] * unit))
    sar_xy = sar[:, :, sar_z_index] / accepted_power
    sar_available = True

if not np.isfinite(accepted_power) or accepted_power <= 0:
    raise ValueError('Total accepted power must be positive for B1 scaling')
print('At {:.3f} MHz: passive S11={:.3f} dB, passive S22={:.3f} dB'.format(
    frequency[target_index] / 1e6,
    20 * np.log10(np.abs(s11[target_index])),
    20 * np.log10(np.abs(s22[target_index]))))
print('Active reflection: port 1={:.3f} dB, port 2={:.3f} dB'.format(
    20 * np.log10(max(abs(active_reflection[0][target_index]), 1e-15)),
    20 * np.log10(max(abs(active_reflection[1][target_index]), 1e-15))))
print('Accepted power at {:.3f} MHz: {:.6g} W'.format(
    frequency[target_index] / 1e6, accepted_power))
print('Port-2/port-1 current: magnitude={:.4f}, phase={:+.3f} deg'.format(
    abs(current_ratio), np.angle(current_ratio, deg=True)))

H_z_index = np.argmin(np.abs(H_axes[2] - phantom_center[2] * unit))
H_xy = H[:, :, H_z_index, :]
isocenter_z_mm = H_axes[2][H_z_index] / unit
print('B1 axial slice: z={:.6g} mm'.format(isocenter_z_mm))

alpha_grid = H_axes[1][None, :]
Br = H_xy[..., 0]
Ba = H_xy[..., 1]
Bx = MUE0 * (Br * np.cos(alpha_grid) - Ba * np.sin(alpha_grid))
By = MUE0 * (Br * np.sin(alpha_grid) + Ba * np.cos(alpha_grid))
power_scale = np.sqrt(accepted_power)
B1p = 0.5 * (Bx + 1j * By) / power_scale
B1m = 0.5 * (Bx - 1j * By) / power_scale

# Area-weighted central circular ROI on the unclosed cylindrical field grid.
roi_radius_mm = float(os.environ.get(
    'OPENEMS_B1_ROI_RADIUS_MM', str(0.8 * phantom_radius)))
if roi_radius_mm <= 0 or roi_radius_mm > phantom_radius:
    raise ValueError(
        'OPENEMS_B1_ROI_RADIUS_MM must be within the saline phantom')
radius_edges = sample_edges(H_axes[0], nonnegative=True)
alpha_edges = sample_edges(H_axes[1])
cell_areas = (
    0.5 * (radius_edges[1:] ** 2 - radius_edges[:-1] ** 2)[:, None]
    * np.diff(alpha_edges)[None, :])
roi_mask = np.broadcast_to(
    H_axes[0][:, None] <= roi_radius_mm * unit + 1e-15,
    B1p.shape)
roi_weights = np.where(roi_mask, cell_areas, 0.0)
b1p_roi_statistics = weighted_field_statistics(
    1e6 * np.abs(B1p), roi_weights)
b1m_roi_statistics = weighted_field_statistics(
    1e6 * np.abs(B1m), roi_weights)
b1_roi_result = {
    'frequency_hz': target_frequency,
    'quadrature_method': quadrature_method,
    'quadrature_port_2_phase_deg': float(
        np.angle(quadrature_phase, deg=True)),
    'field_solution_paths': (
        [port_1_path, port_2_path] if superposition_ready else [sim_path]),
    'physical_lc_matching_enabled': False,
    'feed_topology': feed_topology_description,
    'external_matching_postprocessed': True,
    'external_matching_result_file': 'external_matching_result.json',
    'end_ring_capacitance_pf': capacitance_pf,
    'capacitor_q_at_target_frequency': capacitor_q,
    'end_ring_cap_parallel_resistance_ohm': end_ring_cap_parallel_r,
    'capacitor_q_reference_frequency_hz': target_frequency,
    'axial_slice_z_mm': float(isocenter_z_mm),
    'roi_definition': 'central circular ROI, radius <= specified value',
    'roi_radius_mm': roi_radius_mm,
    'roi_area_mm2': float(np.sum(roi_weights) / unit ** 2),
    'power_normalization': 'total accepted power at both 50-ohm input ports',
    'accepted_input_power_w': accepted_power,
    'port_1_s11_db': float(20 * np.log10(
        max(abs(s11[target_index]), 1e-15))),
    'port_2_s22_db': float(20 * np.log10(
        max(abs(s22[target_index]), 1e-15))),
    'port_1_active_reflection_db': float(20 * np.log10(
        max(abs(active_reflection[0][target_index]), 1e-15))),
    'port_2_active_reflection_db': float(20 * np.log10(
        max(abs(active_reflection[1][target_index]), 1e-15))),
    'port_2_to_port_1_current_magnitude': float(abs(current_ratio)),
    'port_2_to_port_1_current_phase_deg': float(
        np.angle(current_ratio, deg=True)),
    'B1_plus': b1p_roi_statistics,
    'B1_minus': b1m_roi_statistics,
    'mean_B1_minus_to_B1_plus_ratio': float(
        b1m_roi_statistics['mean_uT_per_sqrt_W']
        / b1p_roi_statistics['mean_uT_per_sqrt_W']),
}
roi_json = os.path.join(sim_path, 'B1_ROI_statistics.json')
with open(roi_json, 'w') as roi_file:
    json.dump(b1_roi_result, roi_file, indent=2)
print('Area-weighted B1 ROI: radius={:.3f} mm, area={:.3f} mm^2'.format(
    roi_radius_mm, b1_roi_result['roi_area_mm2']))
print('  B1+ mean={:.6g}, std={:.6g} uT/sqrt(W), CV={:.3f}%, '
      'p05/p50/p95={:.6g}/{:.6g}/{:.6g}'.format(
          b1p_roi_statistics['mean_uT_per_sqrt_W'],
          b1p_roi_statistics['std_uT_per_sqrt_W'],
          b1p_roi_statistics['coefficient_of_variation_percent'],
          b1p_roi_statistics['p05_uT_per_sqrt_W'],
          b1p_roi_statistics['median_uT_per_sqrt_W'],
          b1p_roi_statistics['p95_uT_per_sqrt_W']))
print('  B1- mean={:.6g} uT/sqrt(W), mean B1-/B1+={:.6g}'.format(
    b1m_roi_statistics['mean_uT_per_sqrt_W'],
    b1_roi_result['mean_B1_minus_to_B1_plus_ratio']))
print('Saved B1 ROI statistics: {}'.format(roi_json))

if sar_available:
    sar_alpha, (sar_xy,) = close_periodic_alpha(sar_axes[1], sar_xy)
    print('Peak axial SAR: {:.6g} W/kg per accepted W'.format(
        np.nanmax(sar_xy)))
H_alpha, (B1p, B1m) = close_periodic_alpha(H_axes[1], B1p, B1m)
print('Peak axial |B1+|: {:.6g} uT/sqrt(W)'.format(
    1e6 * np.nanmax(np.abs(B1p))))
print('Peak axial |B1-|: {:.6g} uT/sqrt(W)'.format(
    1e6 * np.nanmax(np.abs(B1m))))

# Always save dedicated B1+ and B1- maps, including batch runs.
phantom_mask = H_axes[0] <= (phantom_radius + 1e-9) * unit
b1_plot_radius = H_axes[0][phantom_mask] / unit
b1p_plot = B1p[phantom_mask]
b1m_plot = B1m[phantom_mask]

b1_plus_figure, b1_plus_axis = plt.subplots(
    num='High-pass birdcage B1+ field',
    figsize=(6, 5), tight_layout=True)
plot_cylindrical_plane(
    b1_plus_axis, b1_plot_radius, H_alpha, 1e6 * np.abs(b1p_plot),
    'axial $|B_1^+|/\\sqrt{{P_{{in}}}}$ at z={:.3f} mm'.format(
        isocenter_z_mm),
    'uT/sqrt(W)')
b1_plus_axis.add_patch(plt.Circle(
    (0, 0), roi_radius_mm, fill=False, color='w', ls='--', lw=1.2))
b1_plus_png = os.path.join(sim_path, 'B1_plus.png')
b1_plus_figure.savefig(b1_plus_png, dpi=200, bbox_inches='tight')
print('Saved B1+ field map: {}'.format(b1_plus_png))

b1_minus_figure, b1_minus_axis = plt.subplots(
    num='High-pass birdcage B1- field',
    figsize=(6, 5), tight_layout=True)
plot_cylindrical_plane(
    b1_minus_axis, b1_plot_radius, H_alpha, 1e6 * np.abs(b1m_plot),
    'axial $|B_1^-|/\\sqrt{{P_{{in}}}}$ at z={:.3f} mm'.format(
        isocenter_z_mm),
    'uT/sqrt(W)')
b1_minus_axis.add_patch(plt.Circle(
    (0, 0), roi_radius_mm, fill=False, color='w', ls='--', lw=1.2))
b1_minus_png = os.path.join(sim_path, 'B1_minus.png')
b1_minus_figure.savefig(b1_minus_png, dpi=200, bbox_inches='tight')
print('Saved B1- field map: {}'.format(b1_minus_png))

# Extract y=0 through the axial isocenter: alpha=0 on +x and alpha=pi on -x.
b1p_positive_x = interpolate_angular_field(H_alpha, B1p, 0)
b1p_negative_x = interpolate_angular_field(H_alpha, B1p, np.pi)
x_line_mm = np.r_[
    -H_axes[0][::-1] / unit,
    H_axes[0] / unit]
b1p_x_line = 1e6 * np.abs(np.r_[
    b1p_negative_x[::-1],
    b1p_positive_x])
phantom_line_mask = np.abs(x_line_mm) <= phantom_radius
x_line_mm = x_line_mm[phantom_line_mask]
b1p_x_line = b1p_x_line[phantom_line_mask]

b1_line_figure, b1_line_axis = plt.subplots(
    num='High-pass birdcage B1+ x profile',
    figsize=(7, 4.5), tight_layout=True)
b1_line_axis.plot(x_line_mm, b1p_x_line, 'b-', lw=2)
b1_line_axis.axvline(0, color='k', lw=1, ls='--')
b1_line_axis.set_xlim(-phantom_radius, phantom_radius)
b1_line_axis.set_xlabel('x (mm), y=0')
b1_line_axis.set_ylabel(r'$|B_1^+|/\sqrt{P_{in}}$ ($\mu$T/$\sqrt{W}$)')
b1_line_axis.set_title(
    '$|B_1^+|$ x profile at z={:.3f} mm'.format(isocenter_z_mm))
b1_line_axis.grid()
b1_line_png = os.path.join(sim_path, 'B1_plus_x_line.png')
b1_line_figure.savefig(b1_line_png, dpi=200, bbox_inches='tight')
print('Saved B1+ x-direction line plot: {}'.format(b1_line_png))

if not skip_plots:
    fig, axes = plt.subplots(
        1, 2, num='High-pass birdcage B1 fields',
        figsize=(12, 5), tight_layout=True)
    plot_cylindrical_plane(
        axes[0], H_axes[0] / unit, H_alpha, 1e6 * np.abs(B1p),
        'axial $|B_1^+|$', 'uT/sqrt(W)')
    plot_cylindrical_plane(
        axes[1], H_axes[0] / unit, H_alpha, 1e6 * np.abs(B1m),
        'axial $|B_1^-|$', 'uT/sqrt(W)')
    if sar_available:
        fig, axis = plt.subplots(
            num='High-pass birdcage SAR', figsize=(6, 5), tight_layout=True)
        plot_cylindrical_plane(
            axis, sar_axes[0] / unit, sar_alpha, sar_xy,
            'axial local SAR', 'W/kg per accepted W')
    plt.show()
else:
    plt.close(b1_plus_figure)
    plt.close(b1_minus_figure)
    plt.close(b1_line_figure)
