import argparse
from pyrfsim import RfSimulator
import numpy as np
import h5py
from scipy.signal import hilbert, gausspulse
from time import time
import matplotlib.pyplot as plt

description="""
    Plot performance graphs for GPU fixed-scatterer algorithm.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--line_length", help="Length of scanline", type=float, default=0.1)
    parser.add_argument("--fs", help="Sampling frequency [Hz]", type=float, default=50e6)
    parser.add_argument("--fc", help="Pulse center frequency [Hz] (also demod. freq.)", type=float, default=5.0e6)
    parser.add_argument("--bw", help="Pulse fractional bandwidth", type=float, default=0.2)
    parser.add_argument("--sigma_lateral", help="Lateral beamwidth", type=float, default=0.5e-3)
    parser.add_argument("--sigma_elevational", help="Elevational beamwidth", type=float, default=1e-3)
    parser.add_argument("--num_lines", help="Number of IQ lines in scansequence", type=int, default=128)
    parser.add_argument("--x0", help="Scanseq width in meters (left end)", type=float, default=-0.03)
    parser.add_argument("--x1", help="Scanseq width in meters (right end)", type=float, default=0.03)
    args = parser.parse_args()
        
    # create and configure simulator
    sim = RfSimulator("gpu")
    sim.set_parameter("verbose", "0")
    sim.set_print_debug(False)
    sim.set_parameter("sound_speed", "1540.0")
    sim.set_parameter("radial_decimation", "15")
    sim.set_parameter("phase_delay", "on")
        
    # define excitation signal
    t_vector = np.arange(-16/args.fc, 16/args.fc, 1.0/args.fs)
    samples = np.array(gausspulse(t_vector, bw=args.bw, fc=args.fc), dtype="float32")
    center_index = int(len(t_vector)/2) 
    demod_freq = args.fc
    sim.set_excitation(samples, center_index, args.fs, demod_freq)
            
    # create linear scan sequence
    origins      = np.empty((args.num_lines, 3), dtype="float32")
    directions   = np.empty((args.num_lines, 3), dtype="float32")
    lateral_dirs = np.empty((args.num_lines, 3), dtype="float32")
    
    for beam_no in range(args.num_lines):
        x = args.x0 + float(beam_no)*(args.x1-args.x0)/(args.num_lines-1)
        origins[beam_no, :]      = [x, 0.0, 0.0]
        directions[beam_no, :]   = [0.0, 0.0, 1.0]
        lateral_dirs[beam_no, :] = [1.0, 0.0, 0.0]
    timestamps = np.zeros(shape=(args.num_lines,), dtype="float32")
    sim.set_scan_sequence(origins, directions, args.line_length, lateral_dirs, timestamps)

    # set analytical beam profile
    sim.set_analytical_beam_profile(args.sigma_lateral, args.sigma_elevational)

    powers_of_two = np.linspace(12, 23, 31)
    all_nums = [int(2**k) for k in powers_of_two]
    
    plt.figure()
    plt.ion()
    plt.show()
    
    sim_times = []
    sim_nums  = []
    for num_scatterers in all_nums:
        
        # configure with random scatterers
        data = np.empty((num_scatterers, 4), dtype="float32")
        data[:, 0] = np.random.uniform(low=args.x0, high=args.x1, size=(num_scatterers,))
        data[:, 1] = np.random.uniform(low=args.x0, high=args.x1, size=(num_scatterers,))
        data[:, 2] = np.random.uniform(low=0.0, high=args.line_length+1e-2, size=(num_scatterers,))
        data[:, 3] = np.random.uniform(low=-1.0, high=1.0, size=(num_scatterers,))
        sim.clear_fixed_scatterers()
        sim.add_fixed_scatterers(data)
        
        # do the simulation
        start_time = time()
        iq_lines = sim.simulate_lines()
        end_time = time()
        elapsed_time = end_time-start_time
        print "Simulation took %f sec @ %d scatterers." % (elapsed_time, num_scatterers)
        
        # store time per IQ-line
        sim_times.append(elapsed_time/args.num_lines)
        sim_nums.append(num_scatterers)
        
        plt.clf()
        plt.plot(sim_nums, sim_times)
        plt.xlabel("Number of scatterers")
        plt.ylabel("Simulation time")
        plt.draw()
    plt.ioff()
    
    sim_nums = np.array(sim_nums)
    sim_times = np.array(sim_times)
    sim_times2 = sim_times/sim_nums
    plt.figure()
    plt.plot(sim_nums, sim_times2)
    plt.xlabel("Number of scatterers")
    plt.ylabel("Sim. time per line per scatterer")
    
    last_value = sim_times2[-1] 
    plt.axhline(last_value, linestyle="--", label="%e sec" % last_value)
    plt.legend()
    
    plt.show()