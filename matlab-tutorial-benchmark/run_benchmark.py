#!/usr/bin/env python3
"""Run every runnable Matlab/Octave tutorial on CPU and GPU and compare outputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, '/home/shanda/openEMS-gpu/local/python-packages')
sys.path.append('/usr/lib/python3/dist-packages')

import h5py
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


ROOT = Path('/home/shanda/openEMS-gpu/matlab-tutorial-benchmark')
SOURCE = Path('/home/shanda/openEMS-gpu/openEMS-Project/openEMS/matlab/Tutorials')
HARNESS = ROOT / 'harness'
OPENEMS_MATLAB = Path('/home/shanda/openEMS-gpu/local/share/openEMS/matlab')
CSXCAD_MATLAB = Path('/home/shanda/openEMS-gpu/local/share/CSXCAD/matlab')
CTB = Path('/home/shanda/openEMS-gpu/openEMS-Project/CTB')
CPU_BINARY = Path('/tmp/openems-cpu-optimization/openEMS')
GPU_BINARY = Path('/tmp/openems-gpu-full/openEMS')

TUTORIALS = [
    'Bent_Patch_Antenna',
    'CRLH_Extraction',
    'CRLH_LeakyWaveAnt',
    'Circ_Waveguide',
    'Conical_Horn_Antenna',
    'CylindricalWave_CC',
    'Dipole_SAR',
    'Helical_Antenna',
    'Horn_Antenna',
    'MRI_LP_Birdcage',
    'MRI_Loop_Coil',
    'MSL_NotchFilter',
    'Parallel_Plate_Waveguide',
    'Patch_Antenna_Phased_Array',
    'RCS_Sphere',
    'RadarUWBTutorial',
    'Rect_Waveguide',
    'Simple_Patch_Antenna',
    'StripLine2MSL',
]
HELPERS = ['CreateCRLH.m', 'Patch_Antenna_Array.m']
SKIP_COMPARE_NAMES = {
    'benchmark_meta.txt',
    'openEMS_run_stats.txt',
    'openEMS_stats.txt',
}
FLOAT_PATTERN = re.compile(
    rb'(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?(?![A-Za-z_])'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--gpu-kernel', default='auto')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--only', action='append', choices=TUTORIALS)
    parser.add_argument('--reuse', action='store_true')
    return parser.parse_args()


def prepare_case(tutorial: str, engine: str) -> tuple[Path, Path]:
    case_dir = ROOT / 'runs' / engine / tutorial
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    for name in HELPERS + [f'{tutorial}.m']:
        source = SOURCE / name
        if source.exists():
            text = source.read_text(encoding='utf-8')
            if name == 'Helical_Antenna.m':
                text = text.replace('post_proc_only = 1;', 'post_proc_only = 0;', 1)
            (case_dir / name).write_text(text, encoding='utf-8')
    return case_dir, case_dir / f'{tutorial}.m'


def read_meta(case_dir: Path) -> dict[str, Any]:
    metas = sorted(case_dir.rglob('benchmark_meta.txt'))
    if not metas:
        return {}
    result: dict[str, Any] = {'path': str(metas[0].relative_to(case_dir))}
    for line in metas[0].read_text(errors='replace').splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            result[key] = value
    if 'wall_seconds' in result:
        result['wall_seconds'] = float(result['wall_seconds'])
    if 'exit_status' in result:
        result['exit_status'] = int(result['exit_status'])
    return result


def classify_failure(log_text: str, timed_out: bool) -> str:
    if timed_out:
        return 'timeout'
    missing_markers = (
        'Ella_26y_V2_1mm',
        'DB_h5_20120711_SEMCADv14.8.h5',
        'No such file or directory',
        'file not found',
    )
    if any(marker.lower() in log_text.lower() for marker in missing_markers):
        return 'missing_external_data'
    if 'openEMSBenchmark:SolverFailed' in log_text or 'openEMS exited with status' in log_text:
        return 'solver_failure'
    if 'NoSimulation' in log_text:
        return 'no_solver_invocation'
    return 'tutorial_generation_failure'


def run_case(tutorial: str, engine: str, args: argparse.Namespace) -> dict[str, Any]:
    case_dir, tutorial_file = prepare_case(tutorial, engine)
    binary = CPU_BINARY if engine == 'cpu' else GPU_BINARY
    env = os.environ.copy()
    env.update({
        'OPENEMS_BENCHMARK_ENGINE': engine,
        'OPENEMS_BENCHMARK_BINARY': str(binary),
        'OPENEMS_BENCHMARK_STEPS': str(args.steps),
        'OPENEMS_BENCHMARK_THREADS': str(args.threads),
        'OPENEMS_BENCHMARK_GPU_DEVICE': '0',
        'OPENEMS_BENCHMARK_GPU_KERNEL': args.gpu_kernel,
        'OPENEMS_BENCHMARK_TUTORIAL': str(tutorial_file),
        'OPENEMS_BENCHMARK_MATLAB_PATH': str(OPENEMS_MATLAB),
        'OPENEMS_BENCHMARK_CSXCAD_PATH': str(CSXCAD_MATLAB),
        'OPENEMS_BENCHMARK_CTB_PATH': str(CTB),
        'LD_LIBRARY_PATH': f'/tmp/openems-gpu-full:/home/shanda/openEMS-gpu/local/lib',
        'GNUTERM': 'unknown',
    })
    command = [
        '/usr/bin/octave-cli', '--quiet', '--no-history', '--eval',
        f"addpath('{HARNESS}', '-begin'); run_tutorial();",
    ]
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=case_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
        )
        returncode = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        output = (exc.stdout or '')
        if isinstance(output, bytes):
            output = output.decode(errors='replace')
        output += f'\n[benchmark] timeout after {args.timeout} seconds\n'
    process_seconds = time.perf_counter() - started
    (case_dir / 'octave.log').write_text(output, encoding='utf-8')
    meta = read_meta(case_dir)
    passed = returncode == 0 and bool(meta) and meta.get('exit_status') == 0
    result = {
        'tutorial': tutorial,
        'engine': engine,
        'status': 'pass' if passed else 'fail',
        'failure_kind': None if passed else classify_failure(output, timed_out),
        'returncode': returncode,
        'process_seconds': process_seconds,
        'solver': meta,
        'log': str((case_dir / 'octave.log').relative_to(ROOT)),
    }
    (case_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def result_files(case_dir: Path) -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    for path in case_dir.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(case_dir)
        if path.name in SKIP_COMPARE_NAMES or path.name in {'octave.log', 'result.json'}:
            continue
        if rel.parts and rel.parts[0] == '__pycache__':
            continue
        if path.suffix.lower() == '.m':
            continue
        files[rel] = path
    return files


def hdf5_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, 'r') as handle:
        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                arrays[name] = np.asarray(obj[...])
        handle.visititems(visit)
    return arrays


def vtk_arrays(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == '.vtk':
        reader = vtk.vtkGenericDataObjectReader()
    else:
        reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutput()
    if output is None:
        raise RuntimeError(f'VTK could not read {path}')

    arrays: dict[str, np.ndarray] = {}
    for group_name, getter_name in (
        ('point', 'GetPointData'),
        ('cell', 'GetCellData'),
        ('field', 'GetFieldData'),
    ):
        if not hasattr(output, getter_name):
            continue
        attributes = getattr(output, getter_name)()
        if attributes is None:
            continue
        for index in range(attributes.GetNumberOfArrays()):
            array = attributes.GetArray(index)
            if array is None:
                continue
            name = array.GetName() or f'array_{index}'
            arrays[f'{group_name}/{name}'] = vtk_to_numpy(array)
    return arrays


def numeric_metrics(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, Any]:
    if cpu.shape != gpu.shape:
        return {'status': 'shape_mismatch', 'cpu_shape': cpu.shape, 'gpu_shape': gpu.shape}
    cpu = np.asarray(cpu, dtype=np.complex128)
    gpu = np.asarray(gpu, dtype=np.complex128)
    finite = np.isfinite(cpu) & np.isfinite(gpu)
    if not np.all(finite):
        same_nonfinite = np.array_equal(np.isnan(cpu), np.isnan(gpu)) and np.array_equal(np.isinf(cpu), np.isinf(gpu))
        if not same_nonfinite:
            return {'status': 'nonfinite_mismatch'}
        cpu = cpu[finite]
        gpu = gpu[finite]
    if cpu.size == 0:
        return {'status': 'pass', 'count': 0, 'max_abs': 0.0, 'rel_l2': 0.0, 'scale': 0.0}
    delta = np.abs(gpu - cpu)
    max_abs = float(np.max(delta))
    scale = float(max(np.max(np.abs(cpu)), np.max(np.abs(gpu))))
    denominator = float(np.linalg.norm(cpu.ravel()))
    rel_l2 = float(np.linalg.norm((gpu - cpu).ravel()) / max(denominator, 1e-30))
    passed = max_abs <= 5e-7 + 1e-4 * scale
    return {
        'status': 'pass' if passed else 'mismatch',
        'count': int(cpu.size),
        'max_abs': max_abs,
        'rel_l2': rel_l2,
        'scale': scale,
    }


def compare_file(cpu_path: Path, gpu_path: Path) -> list[dict[str, Any]]:
    rel_name = cpu_path.name
    if h5py.is_hdf5(cpu_path) and h5py.is_hdf5(gpu_path):
        cpu_arrays = hdf5_arrays(cpu_path)
        gpu_arrays = hdf5_arrays(gpu_path)
        rows: list[dict[str, Any]] = []
        for dataset in sorted(set(cpu_arrays) | set(gpu_arrays)):
            if dataset not in cpu_arrays or dataset not in gpu_arrays:
                rows.append({'component': f'{rel_name}:{dataset}', 'status': 'missing_dataset'})
            else:
                row = numeric_metrics(cpu_arrays[dataset], gpu_arrays[dataset])
                row['component'] = f'{rel_name}:{dataset}'
                rows.append(row)
        return rows

    if cpu_path.suffix.lower() in {'.vtk', '.vti', '.vtr', '.vts', '.vtp', '.vtu'}:
        cpu_arrays = vtk_arrays(cpu_path)
        gpu_arrays = vtk_arrays(gpu_path)
        phase_magnitudes = None
        if '_arg.' in cpu_path.name:
            cpu_magnitude_path = cpu_path.with_name(cpu_path.name.replace('_arg.', '_abs.'))
            gpu_magnitude_path = gpu_path.with_name(gpu_path.name.replace('_arg.', '_abs.'))
            if cpu_magnitude_path.exists() and gpu_magnitude_path.exists():
                phase_magnitudes = (
                    vtk_arrays(cpu_magnitude_path),
                    vtk_arrays(gpu_magnitude_path),
                )
        rows = []
        for dataset in sorted(set(cpu_arrays) | set(gpu_arrays)):
            if dataset not in cpu_arrays or dataset not in gpu_arrays:
                rows.append({'component': f'{rel_name}:{dataset}', 'status': 'missing_dataset'})
            elif phase_magnitudes and dataset in phase_magnitudes[0] and dataset in phase_magnitudes[1]:
                cpu_phase = cpu_arrays[dataset]
                gpu_phase = gpu_arrays[dataset]
                strength = np.maximum(
                    np.abs(phase_magnitudes[0][dataset]),
                    np.abs(phase_magnitudes[1][dataset]),
                )
                mask = strength > np.max(strength) * 1e-6
                circular_delta = np.abs(np.angle(np.exp(1j * (gpu_phase - cpu_phase))))
                selected = circular_delta[mask]
                max_circular = float(np.max(selected)) if selected.size else 0.0
                rms_circular = float(np.sqrt(np.mean(selected ** 2))) if selected.size else 0.0
                rows.append({
                    'component': f'{rel_name}:{dataset}',
                    'status': 'pass' if max_circular <= 1e-2 else 'phase_mismatch',
                    'count': int(selected.size),
                    'max_abs': max_circular,
                    'rel_l2': rms_circular,
                    'scale': float(np.max(strength)),
                    'comparison': 'circular_phase_above_1e-6_peak_magnitude',
                })
            else:
                row = numeric_metrics(cpu_arrays[dataset], gpu_arrays[dataset])
                row['component'] = f'{rel_name}:{dataset}'
                rows.append(row)
        return rows

    cpu_bytes = cpu_path.read_bytes()
    gpu_bytes = gpu_path.read_bytes()
    if cpu_bytes == gpu_bytes:
        return [{'component': rel_name, 'status': 'pass', 'exact': True}]
    if b'mode_purity' in cpu_bytes and b'mode_purity' in gpu_bytes:
        try:
            cpu_table = np.loadtxt(io.StringIO(cpu_bytes.decode('utf-8')), comments='%', ndmin=2)
            gpu_table = np.loadtxt(io.StringIO(gpu_bytes.decode('utf-8')), comments='%', ndmin=2)
        except (UnicodeDecodeError, ValueError):
            cpu_table = np.empty((0, 0))
            gpu_table = np.empty((0, 0))
        if cpu_table.shape == gpu_table.shape and cpu_table.shape[1] >= 3:
            rows = []
            for column, label in ((0, 'time'), (1, 'signal')):
                row = numeric_metrics(cpu_table[:, column], gpu_table[:, column])
                row['component'] = f'{rel_name}:{label}'
                rows.append(row)
            signal_scale = max(
                float(np.max(np.abs(cpu_table[:, 1]))),
                float(np.max(np.abs(gpu_table[:, 1]))),
            )
            meaningful = np.maximum(
                np.abs(cpu_table[:, 1]), np.abs(gpu_table[:, 1])
            ) > max(1e-20, signal_scale * 1e-9)
            row = numeric_metrics(cpu_table[meaningful, 2], gpu_table[meaningful, 2])
            row['component'] = f'{rel_name}:mode_purity_when_signal_meaningful'
            rows.append(row)
            return rows
    cpu_data = b'\n'.join(
        line for line in cpu_bytes.splitlines()
        if not line.lstrip().startswith((b'%', b'#'))
    )
    gpu_data = b'\n'.join(
        line for line in gpu_bytes.splitlines()
        if not line.lstrip().startswith((b'%', b'#'))
    )
    cpu_numbers = np.asarray([float(item) for item in FLOAT_PATTERN.findall(cpu_data)])
    gpu_numbers = np.asarray([float(item) for item in FLOAT_PATTERN.findall(gpu_data)])
    if cpu_numbers.size and gpu_numbers.size:
        row = numeric_metrics(cpu_numbers, gpu_numbers)
        row['component'] = rel_name
        return [row]
    return [{
        'component': rel_name,
        'status': 'binary_mismatch',
        'cpu_sha256': hashlib.sha256(cpu_bytes).hexdigest(),
        'gpu_sha256': hashlib.sha256(gpu_bytes).hexdigest(),
    }]


def compare_tutorial(tutorial: str) -> dict[str, Any]:
    cpu_dir = ROOT / 'runs' / 'cpu' / tutorial
    gpu_dir = ROOT / 'runs' / 'gpu' / tutorial
    cpu_files = result_files(cpu_dir)
    gpu_files = result_files(gpu_dir)
    details: list[dict[str, Any]] = []
    missing: list[str] = []
    for rel in sorted(set(cpu_files) | set(gpu_files)):
        if rel not in cpu_files or rel not in gpu_files:
            missing.append(str(rel))
            continue
        for row in compare_file(cpu_files[rel], gpu_files[rel]):
            row['file'] = str(rel)
            details.append(row)
    failures = [row for row in details if row['status'] != 'pass']
    numeric = [row for row in details if 'max_abs' in row]
    return {
        'tutorial': tutorial,
        'status': 'pass' if not missing and not failures else 'mismatch',
        'common_files': len(set(cpu_files) & set(gpu_files)),
        'missing_files': missing,
        'compared_components': len(details),
        'failed_components': len(failures),
        'worst_max_abs': max((row['max_abs'] for row in numeric), default=0.0),
        'worst_rel_l2': max((row['rel_l2'] for row in numeric), default=0.0),
        'details': details,
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        '# Matlab/Octave tutorial CPU/GPU benchmark',
        '',
        f"- Fixed timesteps: {summary['configuration']['steps']}",
        f"- CPU: multithreaded, {summary['configuration']['threads']} threads",
        f"- GPU: device 0, kernel selector `{summary['configuration']['gpu_kernel']}`",
        '- Scope: solver execution and raw solver artifacts; interactive and tutorial post-processing is skipped.',
        '',
        '| Tutorial | CPU | GPU | CPU solver s | GPU solver s | Speedup | Parity | Notes |',
        '|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    by_key = {(row['tutorial'], row['engine']): row for row in summary['runs']}
    parity = {row['tutorial']: row for row in summary['comparisons']}
    for tutorial in summary['configuration']['tutorials']:
        cpu = by_key[(tutorial, 'cpu')]
        gpu = by_key[(tutorial, 'gpu')]
        cpu_time = cpu.get('solver', {}).get('wall_seconds')
        gpu_time = gpu.get('solver', {}).get('wall_seconds')
        speedup = cpu_time / gpu_time if cpu_time and gpu_time else None
        comparison = parity.get(tutorial)
        notes = []
        if cpu['failure_kind']:
            notes.append(f"CPU: {cpu['failure_kind']}")
        if gpu['failure_kind']:
            notes.append(f"GPU: {gpu['failure_kind']}")
        def fmt(value: float | None) -> str:
            return f'{value:.3f}' if value is not None else '—'
        lines.append(
            f"| {tutorial} | {cpu['status']} | {gpu['status']} | {fmt(cpu_time)} | "
            f"{fmt(gpu_time)} | {fmt(speedup)} | {comparison['status'] if comparison else '—'} | "
            f"{'; '.join(notes)} |"
        )
    lines.extend([
        '',
        'Parity tolerance: `max_abs <= 5e-7 + 1e-4 * max_signal` for numeric datasets.',
        'Mode purity is compared only where its associated signal exceeds numerical noise; phase dumps use circular distance where magnitude exceeds 1e-6 of peak.',
        'The full machine-readable metrics and per-component comparisons are in `summary.json`.',
        '',
    ])
    (ROOT / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    args = parse_args()
    tutorials = args.only or TUTORIALS
    ROOT.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for engine in ('cpu', 'gpu'):
        for index, tutorial in enumerate(tutorials, 1):
            print(f'[{engine} {index}/{len(tutorials)}] {tutorial}', flush=True)
            result_path = ROOT / 'runs' / engine / tutorial / 'result.json'
            if args.reuse and result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                result = run_case(tutorial, engine, args)
            runs.append(result)
            solver_time = result.get('solver', {}).get('wall_seconds')
            suffix = f' solver={solver_time:.3f}s' if solver_time is not None else ''
            print(f"  {result['status']}{suffix}", flush=True)

    run_map = {(row['tutorial'], row['engine']): row for row in runs}
    comparisons = []
    for tutorial in tutorials:
        if run_map[(tutorial, 'cpu')]['status'] == 'pass' and run_map[(tutorial, 'gpu')]['status'] == 'pass':
            comparisons.append(compare_tutorial(tutorial))

    summary = {
        'configuration': {
            'steps': args.steps,
            'threads': args.threads,
            'gpu_kernel': args.gpu_kernel,
            'timeout_seconds': args.timeout,
            'tutorials': tutorials,
            'helper_files': HELPERS,
        },
        'runs': runs,
        'comparisons': comparisons,
    }
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_markdown(summary)
    passed_cpu = sum(row['status'] == 'pass' and row['engine'] == 'cpu' for row in runs)
    passed_gpu = sum(row['status'] == 'pass' and row['engine'] == 'gpu' for row in runs)
    passed_parity = sum(row['status'] == 'pass' for row in comparisons)
    print(f'CPU pass {passed_cpu}/{len(tutorials)}; GPU pass {passed_gpu}/{len(tutorials)}; parity pass {passed_parity}/{len(comparisons)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
