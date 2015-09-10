import numpy as np
from pyrfsim import RfSimulator
import argparse
from scipy.signal import gausspulse
from time import time

description="""
    Demo program showing how to use the fixed-scatterer GPU
    implementation from Python.
    
    Also useful to measure the running time of the GPU
    implementations.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num_scatterers", type=int, default=1000000)
    parser.add_argument("--num_lines", type=int, default=192)
    parser.add_argument("--num_frames", help="Each frame is equal, but can be used to test performance", type=int, default=1)
    parser.add_argument("--visualize", help="Visualize the middle RF line", action="store_true")
    parser.add_argument("--save_pdf", help="Save .pdf image", action="store_true")
    args = parser.parse_args()

sim = RfSimulator("gpu_fixed")



sim.set_verbose(False)

# configure scatterers (in a 3D cube)
x0 = -0.04; x1 = 0.04
y0 = -0.04; y1 = 0.04
z0 =  0.02; z1 = 0.10
scatterers_data = np.empty((args.num_scatterers, 4), dtype='float32')
scatterers_data[:,0] = np.random.uniform(low=x0, high=x1, size=(args.num_scatterers,))
scatterers_data[:,1] = np.random.uniform(low=y0, high=y1, size=(args.num_scatterers,))
scatterers_data[:,2] = np.random.uniform(low=z0, high=z1, size=(args.num_scatterers,))
scatterers_data[:,3] = np.random.uniform(low=0.0, high=1.0, size=(args.num_scatterers,))
sim.set_fixed_scatterers(scatterers_data)

# configure simulation parameters
sim.set_parameters(1540.0)

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
origins[:,0] = np.linspace(x0, x1, args.num_lines)
x_axis = np.array([1.0, 0.0, 0.0])
z_axis = np.array([0.0, 0.0, 1.0])
directions = np.array(np.tile(z_axis, (args.num_lines, 1)), dtype='float32')
length = 0.12
lateral_dirs = np.array(np.tile(x_axis, (args.num_lines, 1)), dtype='float32')
timestamps = np.zeros((args.num_lines,), dtype='float32')
sim.set_scan_sequence(origins, directions, length, lateral_dirs, timestamps)

# configure the beam profile
sim.set_analytical_beam_profile(1e-3, 1e-3)

start_time = time()
for frame_no in range(args.num_frames):
    rf_lines = sim.simulate_lines()
    print 'Simulated frame %d' % frame_no
end_time = time()
elapsed_time = end_time-start_time
print '\n=== Summary ==='
print 'Number of point-scatterers was %d' % args.num_scatterers
print 'Used %f seconds in total.' % elapsed_time
print 'Time pr. frame: %f [ms]' % (1000.0*elapsed_time/args.num_frames)
print 'Time pr. RF line: %f [ms]' % (1000.0*elapsed_time/(args.num_frames*args.num_lines))
    
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
    
    
