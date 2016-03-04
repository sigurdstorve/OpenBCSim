import numpy as np
from pyrfsim import RfSimulator
import argparse
from scipy.signal import gausspulse
from time import time
import h5py
import matplotlib.pyplot as plt

description="""
    Compare GPU and CPU for a linear scan in the XZ plane
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Hdf5 file with scatterers")
    parser.add_argument("--x0", help="Left scan width", type=float, default=-1e-2)
    parser.add_argument("--x1", help="Right scan width", type=float, default=1e-2)
    parser.add_argument("--num_lines", type=int, default=16)
    parser.add_argument("--device_no", help="GPU device no to use", type=int, default=0)
    parser.add_argument("--line_length", help="Length of scanline", type=float, default=0.12)
    args = parser.parse_args()

    sim_cpu = RfSimulator("cpu")
    sim_gpu = RfSimulator("gpu")
    sim_gpu.set_parameter("gpu_device", "%d"%args.device_no)
    
    with h5py.File(args.h5_file, "r") as f:
        # load the fixed dataset if exists
        if "data" in f:
            scatterers_data = f["data"].value
            print "Configuring %d fixed scatterers" % scatterers_data.shape[0]
            sim_cpu.add_fixed_scatterers(scatterers_data)
            sim_gpu.add_fixed_scatterers(scatterers_data)
        
        # load the spline dataset if exists
        if "spline_degree" in f:
            amplitudes     = f["amplitudes"].value
            control_points = f["control_points"].value
            knot_vector    = f["knot_vector"].value    
            spline_degree  = f["spline_degree"].value
            print "Configuring %d spline scatterers" % control_points.shape[0]
            sim_cpu.add_spline_scatterers(spline_degree, knot_vector, control_points, amplitudes)
            sim_gpu.add_spline_scatterers(spline_degree, knot_vector, control_points, amplitudes)
    
    # configure simulation parameters
    sim_cpu.set_parameter("verbose", "0");            sim_gpu.set_parameter("verbose", "0")
    sim_cpu.set_parameter("sound_speed", "1540.0");   sim_gpu.set_parameter("sound_speed", "1540.0")
    sim_cpu.set_parameter("radial_decimation", "10"); sim_gpu.set_parameter("radial_decimation", "10")
    sim_cpu.set_parameter("phase_delay", "on");       sim_gpu.set_parameter("phase_delay", "on")

    # configure the RF excitation
    fs = 50e6
    ts = 1.0/fs
    fc = 2.5e6
    tc = 1.0/fc
    t_vector = np.arange(-16*tc, 16*tc, ts)
    bw = 0.2
    samples = np.array(gausspulse(t_vector, bw=bw, fc=fc), dtype="float32")
    center_index = int(len(t_vector)/2) 
    sim_cpu.set_excitation(samples, center_index, fs, fc)
    sim_gpu.set_excitation(samples, center_index, fs, fc)

    # define the scan sequence
    origins = np.zeros((args.num_lines, 3), dtype="float32")
    origins[:,0] = np.linspace(args.x0, args.x1, args.num_lines)
    x_axis = np.array([1.0, 0.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    directions = np.array(np.tile(z_axis, (args.num_lines, 1)), dtype="float32")
    lateral_dirs = np.array(np.tile(x_axis, (args.num_lines, 1)), dtype="float32")
    timestamps = np.zeros((args.num_lines,), dtype="float32")
    sim_cpu.set_scan_sequence(origins, directions, args.line_length, lateral_dirs, timestamps)
    sim_gpu.set_scan_sequence(origins, directions, args.line_length, lateral_dirs, timestamps)

    # configure the beam profile
    sim_cpu.set_analytical_beam_profile(1e-3, 1e-3)
    sim_gpu.set_analytical_beam_profile(1e-3, 1e-3)

    print "Simulating on CPU..."
    rf_lines_cpu = sim_cpu.simulate_lines()
    print "Simulating on GPU..."
    rf_lines_gpu = sim_gpu.simulate_lines()
    
    num_samples, num_lines = rf_lines_cpu.shape
    plt.figure()
    plt.ion()
    plt.show()
    for line_no in range(num_lines):
        plt.clf()
        plt.subplot(2,2,1)
        real_cpu = np.real(rf_lines_cpu[:, line_no])
        real_gpu = np.real(rf_lines_gpu[:, line_no])
        real_diff = real_cpu-real_gpu
        plt.plot(real_cpu, label="CPU")
        plt.plot(real_gpu, label="GPU")
        plt.title("Real part: CPU vs. GPU")
        plt.legend()
        
        plt.subplot(2,2,2)
        plt.plot(real_diff)
        plt.title("Difference")
        
        plt.subplot(2,2,3)
        imag_cpu = np.imag(rf_lines_cpu[:, line_no])
        imag_gpu = np.imag(rf_lines_gpu[:, line_no])
        imag_diff = imag_cpu-imag_gpu
        plt.plot(imag_cpu, label="CPU")
        plt.plot(imag_gpu, label="GPU")
        plt.title("Imaginary part: CPU vs. GPU")
        plt.legend()
        
        plt.subplot(2,2,4)
        plt.plot(imag_diff)
        plt.title("Difference")
        
        plt.draw()
        if raw_input("Press enter") == "q": exit()
        
        