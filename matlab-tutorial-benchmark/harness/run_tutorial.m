function run_tutorial()
% Run one isolated tutorial and translate the expected benchmark stop to success.

harness_path = fileparts(mfilename('fullpath'));
tutorial_file = getenv('OPENEMS_BENCHMARK_TUTORIAL');
work_path = fileparts(tutorial_file);
openems_matlab = getenv('OPENEMS_BENCHMARK_MATLAB_PATH');
csxcad_matlab = getenv('OPENEMS_BENCHMARK_CSXCAD_PATH');
ctb_path = getenv('OPENEMS_BENCHMARK_CTB_PATH');

addpath(openems_matlab, '-end');
addpath(csxcad_matlab, '-end');
if ~isempty(ctb_path)
    addpath(ctb_path, '-end');
end
addpath(work_path, '-end');
addpath(harness_path, '-begin');
set(0, 'defaultfigurevisible', 'off');

fprintf('[benchmark] octave=%s\n', OCTAVE_VERSION);
fprintf('[benchmark] tutorial=%s\n', tutorial_file);
try
    run(tutorial_file);
    error('openEMSBenchmark:NoSimulation', ...
          'tutorial returned without invoking RunOpenEMS');
catch err
    if strcmp(err.identifier, 'openEMSBenchmark:SimulationComplete')
        fprintf('[benchmark] RESULT=PASS: %s\n', err.message);
        return;
    end
    fprintf(2, '[benchmark] RESULT=FAIL id=%s message=%s\n', ...
            err.identifier, err.message);
    for index = 1:numel(err.stack)
        fprintf(2, '  at %s:%d\n', err.stack(index).file, err.stack(index).line);
    end
    exit(2);
end
end
