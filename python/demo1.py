# DEMO 1
# Linear scan with three scatterers.
# Using a Gaussian analytic beam profile.
import sys
sys.path.append('.')
from pyrfsim import RfSimulator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import gausspulse
import numpy as np

# Create and configure tracker
sim = RfSimulator("fixed")
sim.set_parameter("verbose", "1")
sim.set_print_debug(True)

# Set general simulation parameters
sim.set_parameter("sound_speed", "1540.0")
sim.set_parameter("num_cpu_cores", "all")

# Set scatterers
num_scatterers = 16
scatterers_data = np.zeros((num_scatterers, 4), dtype='float32')
scatterers_data[:,2] = np.linspace(0.01, 0.16, num_scatterers)
scatterers_data[:,3] = np.ones((num_scatterers,))
sim.set_fixed_scatterers(scatterers_data)

# Define excitation signal
fs = 50e6
ts = 1.0/fs
fc = 2.5e6
tc = 1.0/fc
t_vector = np.arange(-16*tc, 16*tc, ts)
bw = 0.5
plt.figure(1)
samples = np.array(gausspulse(t_vector, bw=bw, fc=fc), dtype='float32')
center_index = int(len(t_vector)/2) 
plt.plot(t_vector, samples);
plt.title('Excitation signal')
plt.xlabel('Time [s]')
plt.ylabel('Exitation')
plt.show()
sim.set_excitation(samples, center_index, fs)

# Define a scan sequence
num_lines = 12
origins = np.zeros((num_lines, 3), dtype='float32')
origins[:,0] = np.linspace(-0.04, 0.04, num_lines)


x_axis = np.array([1.0, 0.0, 0.0])
y_axis = np.array([0.0, 1.0, 0.0])
z_axis = np.array([0.0, 0.0, 1.0])
num_lines = origins.shape[0]
directions = np.array(np.tile(z_axis, (num_lines, 1)), dtype='float32')
length = 0.20
lateral_dirs = np.array(np.tile(x_axis, (num_lines, 1)), dtype='float32')
timestamps = np.zeros((num_lines,), dtype='float32')
sim.set_scan_sequence(origins, directions, length, lateral_dirs, timestamps)


# Set the beam profile
sigma_lateral = 1e-3
sigma_elevational = 1e-3
sim.set_analytical_beam_profile(sigma_lateral, sigma_elevational)

# Do the simulation : result is IQ data
rf_lines = sim.simulate_lines()

# extract the envelopes
rf_lines = np.real(abs(rf_lines))

# Env.detection and log-compression.
plt.figure(2);
dyn_range = 50;
gain = 30;
#img = rf_to_image(rfLines', gain, dyn_range);
plt.imshow(rf_lines, interpolation='nearest', cmap=cm.Greys_r, aspect='auto');
plt.show()
