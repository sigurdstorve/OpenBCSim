import argparse
from pyrfsim import RfSimulator
import numpy as np
import h5py
from scipy.signal import gausspulse
from time import time
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("../phantom_scripts")
import bsplines

description="""
    Example script for demonstrating that the use of the spline-
    based simulation algorithm on the GPU is faster than the
    fixed-scatterer GPU algorithm for a dynamic M-mode scan.
    
    With the spline-based algorithm it is not neccessary to update
    all scatterers at every timestep.
    
    The same M-mode image is simulated twice:
    1) Using the spline-based simulator and a large scan sequence
    containing all beams.
    2) With the fixed-scatterer algorithm by simulating one line,
    updating all scatterers, simulating next line, updating, etc.
    (The scatterer datasets are precomputed, only data transfer
    and simulation is included in the total elapsed time.)
    
    The reason that the spline-based algorithm is much faster is
    that it avoids large memory copies from host memory to GPU
    memory at every time step.
"""

def create_fixed_datasets(args, control_points, amplitudes, spline_degree, knot_vector, timestamps):
    """
    Create a vector of fixed-scatterer datasets by rendering a spline
    phantom at many timestamps.
    """
    num_cs = control_points.shape[1]
    fixed_scatterers = []
    for time in timestamps:
        print "Pre-computing fixed scatterers for timestep %f" % time
        p = np.zeros_like(control_points[:,0,:])
        for j in range(num_cs):
            p += bsplines.B(j, spline_degree, time, knot_vector)*control_points[:,j,:]
        num_scatterers, num_comp = p.shape
        assert num_comp==3
        scatterers = np.empty((num_scatterers, 4), dtype="float32")
        scatterers[:,:3] = p
        scatterers[:, 3] = amplitudes
        fixed_scatterers.append(scatterers)
    
    return fixed_scatterers

def run_spline_simulation(sim_spline):
    """ Returns IQ lines and simulation time. """

    start_time = time()
    iq_lines = sim_spline.simulate_lines()
    end_time = time()
    elapsed_time = end_time-start_time

    return iq_lines, elapsed_time
    
def run_fixed_simulation(sim_fixed, origin, direction, lateral_dir, line_length, timestamps, fixed_scatterers):
    """ Returns IQ lines """
    res = []
    origins      = np.empty((1, 3), dtype="float32")
    directions   = np.empty((1, 3), dtype="float32")
    lateral_dirs = np.empty((1, 3), dtype="float32")
    origins[0,:]      = origin
    directions[0,:]   = direction
    lateral_dirs[0,:] = lateral_dir
    
    start_time = time()
    for cur_timestamp,cur_scatterers in zip(timestamps,fixed_scatterers):
        timestamps = np.array([cur_timestamp], dtype="float32")
        sim_fixed.clear_fixed_scatterers()
        sim_fixed.add_fixed_scatterers(cur_scatterers)
        sim_fixed.set_scan_sequence(origins, directions, line_length, lateral_dirs, timestamps)
        res.append( sim_fixed.simulate_lines() )
    end_time = time()
    elapsed_time = end_time-start_time
    
    iq_lines = np.squeeze(np.array(res, dtype="complex64").transpose())
    return iq_lines, elapsed_time

def make_mmode_image(iq_lines, gain=0.6, dyn_range=30):
    print "iq_lines has shape %s" % str(iq_lines.shape)
    mmode_img = np.array(abs(iq_lines), dtype="float32")
    normalize_factor = np.max(mmode_img.flatten())
    mmode_img = 20*np.log10(gain*(mmode_img+1e-6)/normalize_factor)
    mmode_img = 255.0*(mmode_img+dyn_range)/dyn_range
    # clamp to [0, 255]
    mmode_img[mmode_img < 0]     = 0.0
    mmode_img[mmode_img > 255.0] = 255.0
    
    plt.imshow(mmode_img, aspect="auto", cmap=plt.get_cmap("Greys_r"))
    plt.clim(0.0, 255.0)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scatterer_file", help="Spline scatterer dataset", default="../generated_phantoms/lv_spline_model.h5")
    parser.add_argument("--line_length", help="Length of M-mode beam", type=float, default=0.1)
    parser.add_argument("--prf", help="Pulse repetition frequency [Hz]", type=float, default=1200)
    parser.add_argument("--num_beams_total", help="Number of M-mode beams", type=int, default=1000)
    parser.add_argument("--fs", help="Sampling frequency [Hz]", type=float, default=50e6)
    parser.add_argument("--fc", help="Pulse center frequency [Hz]", type=float, default=2.5e6)
    parser.add_argument("--bw", help="Pulse fractional bandwidth", type=float, default=0.2)
    parser.add_argument("--sigma_lateral", help="Lateral beamwidth", type=float, default=0.5e-3)
    parser.add_argument("--sigma_elevational", help="Elevational beamwidth", type=float, default=1e-3)
    parser.add_argument("--start_time", help="Start time of simulation", type=float, default=0.0)
    parser.add_argument("--end_time", help="Will reset to start time after this", type=float, default=0.999)
    parser.add_argument("--xz_tilt_angle", help="Control M-mode beam direction", type=float, default=0.0)
    args = parser.parse_args()
    
    c0 = 1540.0
    prt = 1.0/args.prf
    
    direction = np.array([np.sin(args.xz_tilt_angle), 0.0, np.cos(args.xz_tilt_angle)])
    origin = np.array([0.0, 0.0, -0.01])
    
    # load spline scatterers
    with h5py.File(args.scatterer_file, "r") as f:
        control_points = f["control_points"].value
        amplitudes     = f["amplitudes"].value
        knot_vector    = f["knot_vector"].value
        spline_degree  = f["spline_degree"].value
    num_cs = control_points.shape[1]
    
    # create beam times (w/wrapping)
    cur_time = args.start_time
    timestamps = []
    for i in range(args.num_beams_total):
        timestamps.append(cur_time)
        cur_time += prt
        if cur_time >= args.end_time: cur_time = args.start_time
    timestamps = np.array(timestamps, dtype="float32")
        
    # precompute fixed-scatterer datasets
    fixed_scatterers = create_fixed_datasets(args, control_points, amplitudes, spline_degree, knot_vector, timestamps)
    
    # create two simulator instances - one for spline-only and one fixed-only
    sim_fixed  = RfSimulator("gpu")
    sim_spline = RfSimulator("gpu")
    sim_fixed.set_parameter("verbose", "0");            sim_spline.set_parameter("verbose", "0")
    sim_fixed.set_print_debug(False);                   sim_spline.set_print_debug(False)
    sim_fixed.set_parameter("sound_speed", "%f" % c0);  sim_spline.set_parameter("sound_speed", "%f" % c0)
    sim_fixed.set_parameter("phase_delay", "on");       sim_spline.set_parameter("phase_delay", "on")
    sim_fixed.set_parameter("radial_decimation", "5");  sim_spline.set_parameter("radial_decimation", "5")

    # define excitation signal
    t_vector = np.arange(-16/args.fc, 16/args.fc, 1.0/args.fs)
    samples = np.array(gausspulse(t_vector, bw=args.bw, fc=args.fc), dtype="float32")
    center_index = int(len(t_vector)/2) 
    demod_freq = args.fc
    sim_fixed.set_excitation(samples, center_index, args.fs, demod_freq)
    sim_spline.set_excitation(samples, center_index, args.fs, demod_freq)    
        
    # create big scan sequence with all M-mode beams (for the spline algorithm)
    origins      = np.empty((args.num_beams_total, 3), dtype="float32")
    directions   = np.empty((args.num_beams_total, 3), dtype="float32")
    lateral_dirs = np.empty((args.num_beams_total, 3), dtype="float32")
    y_axis       = np.array([0.0, 1.0, 0.0])
    lateral_dir  = np.cross(y_axis, direction)
    for beam_no in range(args.num_beams_total):
        origins[beam_no, :]      = origin
        directions[beam_no, :]   = direction
        lateral_dirs[beam_no, :] = lateral_dir
    
    # set the beam profile
    sim_fixed.set_analytical_beam_profile(args.sigma_lateral, args.sigma_elevational)
    sim_spline.set_analytical_beam_profile(args.sigma_lateral, args.sigma_elevational)
    
    # configure spline simulator
    sim_spline.add_spline_scatterers(spline_degree, knot_vector, control_points, amplitudes)
    sim_spline.set_scan_sequence(origins, directions, args.line_length, lateral_dirs, timestamps)

    # fixed
    iq_lines_fixed, sim_time_fixed = run_fixed_simulation(sim_fixed, origin, direction, lateral_dir,
                                                          args.line_length, timestamps, fixed_scatterers)
    plt.figure(1)
    make_mmode_image(iq_lines_fixed)
    plt.title("M-Mode produced with the fixed algorithm : %f sec" % sim_time_fixed)
    
    # spline
    iq_lines_spline, sim_time_spline = run_spline_simulation(sim_spline)
    plt.figure(2)
    make_mmode_image(iq_lines_spline)
    plt.title("M-Mode produced with the spline algorithm : %f sec" % sim_time_spline)
        
    plt.show()