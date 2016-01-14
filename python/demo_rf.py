import numpy as np
from pyrfsim import RfSimulator
import argparse
from scipy.signal import gausspulse
import h5py
import matplotlib.pyplot as plt

description="""
    Simulate using fixed scatterers from a HDF5 file.
    Scan type is a linear scan in the XZ plane.
    
    Demonstrates how to process the simulated IQ data to
    obtain real-valued RF data. This is achieved by using
    a demodulation frequency equal to zero and discarding
    the imaginary part of the IQ data.
    
    Draws both B-mode image and an "RF image" obtained by
    plotting selected actual RF lines vertically (without
    log-compression)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--h5_file", help="Hdf5 file with [fixed] scatterers", default="../generated_phantoms/carotid_plaque.h5")
    parser.add_argument("--x0", help="Left scan width", type=float, default=-0.08)
    parser.add_argument("--x1", help="Right scan width", type=float, default=0.076)
    parser.add_argument("--num_lines", type=int, help="Total number of B-mode lines", default=512)
    parser.add_argument("--tx_decimation", type=int, default=16, help="Decimation when drawing RF lines")
    parser.add_argument("--device_no", help="GPU device no to use", type=int, default=0)
    parser.add_argument("--no_gpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

if args.no_gpu:
    sim = RfSimulator("fixed")
else:
    sim = RfSimulator("gpu_fixed")
    sim.set_parameter("gpu_device", "%d"%args.device_no)

with h5py.File(args.h5_file, "r") as f:
    scatterers_data = f["data"].value
sim.set_fixed_scatterers(scatterers_data)
print "The number of scatterers is %d" % scatterers_data.shape[0]

# configure simulation parameters
sim.set_parameter("sound_speed", "1540.0")
sim.set_parameter("phase_delay", "on")
sim.set_parameter("verbose", "0")

# avoid ugly part on top
sim.set_parameter("use_arc_projection", "off")

# use no radial decimation since we want to draw nice
# smooth RF lines
sim.set_parameter("radial_decimation", "1")

# configure the RF excitation : zero demodulation freq
fs = 100e6
ts = 1.0/fs
fc = 2.1e6
tc = 1.0/fc
t_vector = np.arange(-16*tc, 16*tc, ts)
bw = 0.2
samples = np.array(gausspulse(t_vector, bw=bw, fc=fc), dtype="float32")
center_index = int(len(t_vector)/2)
demod_freq = 0.0
sim.set_excitation(samples, center_index, fs, demod_freq)

# define the scan sequence
start_depth = -0.003
origins = np.zeros((args.num_lines, 3), dtype="float32")
origins[:,0] = np.linspace(args.x0, args.x1, args.num_lines)
origins[:,2] = np.ones((args.num_lines,))*start_depth 
x_axis = np.array([1.0, 0.0, 0.0])
z_axis = np.array([0.0, 0.0, 1.0])
directions = np.array(np.tile(z_axis, (args.num_lines, 1)), dtype="float32")
length = 0.067
lateral_dirs = np.array(np.tile(x_axis, (args.num_lines, 1)), dtype="float32")
timestamps = np.zeros((args.num_lines,), dtype="float32")
sim.set_scan_sequence(origins, directions, length, lateral_dirs, timestamps)

# configure the beam profile
sim.set_analytical_beam_profile(1e-3, 1e-3)

iq_lines = sim.simulate_lines()

# Discard the imaginary component to get RF data
rf_lines = np.real(iq_lines[:,::args.tx_decimation])

# Normalize so that all RF-samples are within [-1, 1]
rf_lines = rf_lines/np.max(abs(rf_lines.flatten()))   

# Make nice RF-line image
num_samples, num_beams = rf_lines.shape
print "Number of samples: %d" % num_samples
print "Number of RF lines: %d" % num_beams
rf_width = 0.70*(args.x1-args.x0)/num_beams
common_times = 1e6*np.array(range(num_samples))/fs
for beam_no,x_pos in enumerate(np.linspace(args.x0, args.x1, num_beams)):
    scaled_samples = rf_lines[:, beam_no]*rf_width
    plt.plot(scaled_samples + x_pos, common_times, c="blue", linewidth=0.5)
plt.xlabel("x position [m]")
plt.ylabel("Time [us]")
plt.gca().invert_yaxis()
plt.xlim(args.x0*1.03, args.x1*1.03)
    
# B-mode image
gain = 1
dyn_range = 40
plt.figure()    
bmode = np.array(abs(iq_lines), dtype="float32")
normalize_factor = np.max(bmode.flatten())
bmode = 20*np.log10(gain*bmode/normalize_factor)
bmode = 255.0*(bmode+dyn_range)/dyn_range
# clamp to [0, 255]
bmode[bmode < 0]     = 0.0
bmode[bmode > 255.0] = 255.0
y_min = start_depth
y_max = start_depth+length
plt.imshow(bmode, extent=[args.x0, args.x1, y_max, y_min], cmap=plt.get_cmap("gray"))

plt.show()
    
    
