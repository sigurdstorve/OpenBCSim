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

def do_simulation(args):
    if args.use_gpu:
        sim = RfSimulator("gpu")
        sim.set_parameter("gpu_device", "%d"%args.device_no)
    else:
        sim = RfSimulator("cpu")

    sim.set_parameter("verbose", "0")

    with h5py.File(args.h5_file, "r") as f:
        scatterers_data = f["data"].value
    sim.add_fixed_scatterers(scatterers_data)
    print "The number of scatterers is %d" % scatterers_data.shape[0]

    # configure simulation parameters
    sim.set_parameter("sound_speed", "1540.0")
    sim.set_parameter("radial_decimation", "10")
    sim.set_parameter("phase_delay", "on")
    sim.set_parameter("noise_amplitude", "%f" % args.noise_ampl)

    # configure the RF excitation
    fs = 80e6
    ts = 1.0/fs
    fc = 5.0e6
    tc = 1.0/fc
    t_vector = np.arange(-16*tc, 16*tc, ts)
    bw = 0.3
    samples = np.array(gausspulse(t_vector, bw=bw, fc=fc), dtype="float32")
    center_index = int(len(t_vector)/2) 
    sim.set_excitation(samples, center_index, fs, fc)

    # define the scan sequence
    origins = np.zeros((args.num_lines, 3), dtype="float32")
    origins[:,0] = np.linspace(args.x0, args.x1, args.num_lines)
    x_axis = np.array([1.0, 0.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    directions = np.array(np.tile(z_axis, (args.num_lines, 1)), dtype="float32")
    length = 0.06
    lateral_dirs = np.array(np.tile(x_axis, (args.num_lines, 1)), dtype="float32")
    timestamps = np.zeros((args.num_lines,), dtype="float32")
    sim.set_scan_sequence(origins, directions, length, lateral_dirs, timestamps)

    # configure the beam profile
    sim.set_analytical_beam_profile(1e-3, 1e-3)

    frame_sim_times = []
    for frame_no in range(args.num_frames):
        start_time = time()
        iq_lines = sim.simulate_lines()
        frame_sim_times.append(time()-start_time)
        
    if args.save_simdata_file != "":
        with h5py.File(args.save_simdata_file, "w") as f:
            f["sim_data_real"] = np.array(np.real(iq_lines), dtype="float32")
            f["sim_data_imag"] = np.array(np.imag(iq_lines), dtype="float32")
        print "Simulation output written to %s" % args.save_simdata_file
        
    print "Simulation time: %f +- %f s  (N=%d)" % (np.mean(frame_sim_times), np.std(frame_sim_times), args.num_frames)    

    if args.pdf_file != "" and not args.visualize:
        import matplotlib as mpl
        mpl.use("Agg")
    if args.pdf_file != "" or args.visualize:
        import matplotlib.pyplot as plt
        num_samples, num_lines = iq_lines.shape
        plt.figure(1, figsize=(18,9))
        plt.subplot(1,2,1)
        plt.plot(np.real(iq_lines[:, num_lines/2]))
        plt.title("Middle RF line")
        plt.subplot(1,2,2)
        plt.imshow(np.real(abs(iq_lines)), aspect="auto", interpolation="nearest")
        if args.pdf_file != "":
            plt.savefig(args.pdf_file)
            print "Image written to disk."
    if args.visualize:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Hdf5 file with scatterers")
    parser.add_argument("--x0", help="Left scan width", type=float, default=-1e-2)
    parser.add_argument("--x1", help="Right scan width", type=float, default=1e-2)
    parser.add_argument("--num_lines", type=int, default=192)
    parser.add_argument("--num_frames", help="Each frame is equal, but can be used to test performance", type=int, default=1)
    parser.add_argument("--visualize", help="Visualize the middle RF line", action="store_true")
    parser.add_argument("--pdf_file", help="Name of pdf file to save, if not empty", default="")
    parser.add_argument("--device_no", help="GPU device no to use", type=int, default=0)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--save_simdata_file", help="Export simulated data in a format that can be loaded in the GUI app.", type=str, default="")
    parser.add_argument("--noise_ampl", help="Simulator noise", type=float, default=0)
    args = parser.parse_args()
    
    do_simulation(args)
