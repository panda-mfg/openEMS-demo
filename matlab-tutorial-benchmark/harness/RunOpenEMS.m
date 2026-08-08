function RunOpenEMS(Sim_Path, Sim_File, opts, Settings)
% Deterministic CPU/GPU benchmark launcher used only by the tutorial harness.

if nargin < 2
    error('openEMSBenchmark:InvalidArguments', ...
          'RunOpenEMS requires a simulation path and XML filename');
end
if nargin < 3
    opts = '';
end

engine = getenv('OPENEMS_BENCHMARK_ENGINE');
binary = getenv('OPENEMS_BENCHMARK_BINARY');
steps_text = getenv('OPENEMS_BENCHMARK_STEPS');
threads = getenv('OPENEMS_BENCHMARK_THREADS');
gpu_device = getenv('OPENEMS_BENCHMARK_GPU_DEVICE');
gpu_kernel = getenv('OPENEMS_BENCHMARK_GPU_KERNEL');

if isempty(engine) || isempty(binary) || isempty(steps_text)
    error('openEMSBenchmark:MissingEnvironment', ...
          'benchmark engine, binary, and timestep environment variables are required');
end
if isempty(threads)
    threads = '8';
end
if isempty(gpu_device)
    gpu_device = '0';
end
if isempty(gpu_kernel)
    gpu_kernel = 'auto';
end

xml_path = fullfile(Sim_Path, Sim_File);
harness_path = fileparts(mfilename('fullpath'));
patcher = fullfile(harness_path, 'patch_openems_xml.py');
patch_command = sprintf('python3 "%s" "%s" %s', patcher, xml_path, steps_text);
[patch_status, patch_output] = system(patch_command);
if patch_status ~= 0
    fprintf(2, '%s', patch_output);
    error('openEMSBenchmark:XmlPatchFailed', ...
          'failed to apply fixed timestep controls to %s', xml_path);
end

if strcmp(engine, 'cpu')
    engine_options = sprintf('--engine=multithreaded --numThreads=%s', threads);
elseif strcmp(engine, 'gpu')
    engine_options = sprintf('--engine=gpu --gpu-device=%s --gpu-kernel=%s', ...
                             gpu_device, gpu_kernel);
else
    error('openEMSBenchmark:InvalidEngine', 'unknown engine: %s', engine);
end

save_path = pwd;
cd(Sim_Path);
cleanup = onCleanup(@() cd(save_path));
command = sprintf('"%s" "%s" %s --fixed-timesteps --legacyHDF5Dumps --dump-statistics %s', ...
                  binary, Sim_File, engine_options, opts);
fprintf('[benchmark] command: %s\n', command);
timer = tic;
[status, output] = system(command);
elapsed = toc(timer);
fprintf('%s', output);

meta = fopen('benchmark_meta.txt', 'w');
if meta >= 0
    fprintf(meta, 'engine=%s\n', engine);
    fprintf(meta, 'binary=%s\n', binary);
    fprintf(meta, 'timesteps=%s\n', steps_text);
    fprintf(meta, 'wall_seconds=%.9f\n', elapsed);
    fprintf(meta, 'exit_status=%d\n', status);
    fclose(meta);
end

if status ~= 0
    error('openEMSBenchmark:SolverFailed', ...
          'openEMS exited with status %d for %s', status, Sim_File);
end

error('openEMSBenchmark:SimulationComplete', ...
      'solver completed; benchmark intentionally skipped tutorial post-processing');
end
