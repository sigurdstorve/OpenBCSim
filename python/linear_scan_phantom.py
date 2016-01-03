import numpy as np
from pyrfsim import RfSimulator
import argparse
from scipy.signal import gausspulse
from time import time
import h5py

description="""
    Simulate using scatterers from hdf file.
    Scan type is a linear scan in the XZ plane.
    
    This script is also useful for measuring
    the simulation time over a number of equal
    runs.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Hdf5 file with scatterers")
    parser.add_argument("--x0", help="Left scan width", type=float, default=-1e-2)
    parser.add_argument("--x1", help="Right scan width", type=float, default=1e-2)
    parser.add_argument("--num_lines", type=int, default=192)
    parser.add_argument("--num_frames", help="Each frame is equal, but can be used to test performance", type=int, default=1)
    parser.add_argument("--visualize", help="Visualize the middle RF line", action="store_true")
    parser.add_argument("--save_pdf", help="Save .pdf image", action="store_true")
    parser.add_argument("--device_no", help="GPU device no to use", type=int, default=0)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--save_simdata_file", help="Export simulated data in a format that can be loaded in the GUI app.", type=str, default=None)
    args = parser.parse_args()

if args.use_gpu:
    sim = RfSimulator("gpu_fixed")
    sim.set_parameter("gpu_device", "%d"%args.device_no)
else:
    sim = RfSimulator("fixed")


sim.set_parameter("verbose", "0")

with h5py.File(args.h5_file, "r") as f:
    scatterers_data = f["data"].value
sim.set_fixed_scatterers(scatterers_data)
print "The number of scatterers is %d" % scatterers_data.shape[0]

# configure simulation parameters
sim.set_parameter("sound_speed", "1540.0")

# configure the RF excitation
fs = 100e6
ts = 1.0/fs
fc = 2.5e6
tc = 1.0/fc
t_vector = np.arange(-16*tc, 16*tc, ts)
bw = 0.2
samples = np.array(gausspulse(t_vector, bw=bw, fc=fc), dtype='float32')
center_index = int(len(t_vector)/2) 
sim.set_excitation(samples, center_index, fs)

# define the scan sequence
origins = np.zeros((args.num_lines, 3), dtype='float32')
origins[:,0] = np.linspace(args.x0, args.x1, args.num_lines)
x_axis = np.array([1.0, 0.0, 0.0])
z_axis = np.array([0.0, 0.0, 1.0])
directions = np.array(np.tile(z_axis, (args.num_lines, 1)), dtype='float32')
length = 0.12
lateral_dirs = np.array(np.tile(x_axis, (args.num_lines, 1)), dtype='float32')
timestamps = np.zeros((args.num_lines,), dtype='float32')
sim.set_scan_sequence(origins, directions, length, lateral_dirs, timestamps)

# configure the beam profile
sim.set_analytical_beam_profile(1e-3, 1e-3)

frame_sim_times = []
for frame_no in range(args.num_frames):
    start_time = time()
    rf_lines = sim.simulate_lines()
    frame_sim_times.append(time()-start_time)

# get envelope of IQ data
rf_lines = np.real(abs(rf_lines))    
    
if args.save_simdata_file != None:
    with h5py.File(args.save_simdata_file, "w") as f:
        f["sim_data"] = np.array(rf_lines, dtype='float32')
    print "Simulation output written to %s" % args.save_simdata_file
    
print 'Simulation time: %f +- %f s  (N=%d)' % (np.mean(frame_sim_times), np.std(frame_sim_times), args.num_frames)    

if args.save_pdf or args.visualize:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    num_samples, num_lines = rf_lines.shape
    plt.figure(1)
    plt.plot(rf_lines[:, num_lines/2])
    if args.save_pdf: plt.savefig("frame1-out.pdf")
    plt.figure(2)
    plt.imshow(rf_lines, aspect='auto')
    if args.save_pdf: plt.savefig("frame2-out.pdf")
if args.visualize:
    plt.show()
if args.save_pdf:
    print 'Image written to disk.'
    
    
