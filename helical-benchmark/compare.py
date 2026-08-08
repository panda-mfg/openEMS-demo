import numpy as np

cpu = np.load('cpu/metrics.npz')
gpu = np.load('gpu/metrics.npz')
freq = cpu['freq']
cpu_s11 = cpu['s11']
gpu_s11 = gpu['s11']
cpu_min = np.argmin(np.abs(cpu_s11))
gpu_min = np.argmin(np.abs(gpu_s11))

for name in ('Dmax_dB', 'Prad', 'efficiency', 'theta_HPBW'):
    print(name, float(cpu[name]), float(gpu[name]))
print('cpu_s11_min_hz_db', freq[cpu_min], 20*np.log10(abs(cpu_s11[cpu_min])))
print('gpu_s11_min_hz_db', freq[gpu_min], 20*np.log10(abs(gpu_s11[gpu_min])))
print('s11_max_abs', np.max(np.abs(cpu_s11-gpu_s11)))
print('s11_rms_abs', np.sqrt(np.mean(np.abs(cpu_s11-gpu_s11)**2)))
print('zin_max_abs_ohm', np.max(np.abs(cpu['Zin']-gpu['Zin'])))
print('prad_relative_percent', 100*abs(cpu['Prad']-gpu['Prad'])/abs(cpu['Prad']))
print('dmax_delta_db', float(gpu['Dmax_dB']-cpu['Dmax_dB']))
print('throughput_speedup', 167.06/43.53)
print('fdtd_time_speedup', 65.47/20.09)
print('wall_speedup', 133.42/93.57)
