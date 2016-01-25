import argparse
from pyrfsim import RfSimulator
import numpy as np
import h5py
from scipy.signal import gausspulse
from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

description="""
    Example for simulating a tissue Doppler frame
    using the autocorrelation method. The scan type
    is a sector scan.
    
    Will simulate a number of frames equal to the
    packet size and draw color-coded velocities.
    The Doppler power is used for thresholding.
"""

def simulate_doppler_frame(args, timestamp, sim, origin):
    """
    Create scan sequence for one frame where all RF-lines have the same
    timestamp.
    """
    print "Timestamp is %f" % timestamp
    origins      = np.empty((args.num_lines, 3), dtype="float32")
    directions   = np.empty((args.num_lines, 3), dtype="float32")
    lateral_dirs = np.empty((args.num_lines, 3), dtype="float32")
    timestamps   = np.ones((args.num_lines,), dtype="float32")*timestamp
    beam_angles = np.linspace(-0.5*args.width, 0.5*args.width, args.num_lines)
    for beam_no in range(args.num_lines):
        theta = beam_angles[beam_no]
        y_axis = np.array([0.0, 1.0, 0.0])
        direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
        origins[beam_no, :]      = origin
        directions[beam_no, :]   = direction
        lateral_dirs[beam_no, :] = np.cross(direction, y_axis)
    sim.set_scan_sequence(origins, directions, args.line_length, lateral_dirs, timestamps)

    iq_lines = sim.simulate_lines()
    return iq_lines
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scatterer_file", help="Scatterer dataset (spline)", default="../generated_phantoms/contracting_cylinder_spline.h5")
    parser.add_argument("--line_length", help="Length of Doppler beams", type=float, default=0.1)
    parser.add_argument("--prf", help="Pulse repetition frequency [Hz]", type=int, default=2000)
    parser.add_argument("--packet_size", help="Packet size", type=int, default=3)
    parser.add_argument("--fs", help="Sampling frequency [Hz]", type=float, default=60e6)
    parser.add_argument("--fc", help="Pulse center frequency [Hz]", type=float, default=5.0e6)
    parser.add_argument("--bw", help="Pulse fractional bandwidth", type=float, default=0.1)
    parser.add_argument("--sigma_lateral", help="Lateral beamwidth", type=float, default=0.5e-3)
    parser.add_argument("--sigma_elevational", help="Elevational beamwidth", type=float, default=1e-3)
    parser.add_argument("--rad_decimation", help="Radial decimation factor", type=int, default=20)
    parser.add_argument("--width", help="Sector width in radians", type=float, default=1.1)
    parser.add_argument("--num_lines", help="Number of image lines", type=int, default=60)
    parser.add_argument("--sim_time", help="Scan timestamp", type=float, default=0.5)
    parser.add_argument("--normalized_power_threshold", type=float, default=0.0001)
    args = parser.parse_args()
    
    c0 = 1540.0
    prt = 1.0/args.prf
    origin = np.array([0.0, 0.0, -5e-3])  # beam origin
    
    with h5py.File(args.scatterer_file, "r") as f:
        control_points = f["control_points"].value
        amplitudes     = f["amplitudes"].value
        knot_vector    = f["knot_vector"].value
        spline_degree  = f["spline_degree"].value
    num_cs = control_points.shape[1]
    print "Loaded spline phantom with %d control points" % num_cs
    
    # create and configure fixed tracker - use type 1 since
    # all lines in a frame will have the same timestamp
    sim = RfSimulator("gpu_spline1")
    sim.set_parameter("verbose", "0")
    sim.set_print_debug(False)
    sim.set_parameter("sound_speed", "%f" % c0)
    sim.set_parameter("radial_decimation", "%d"%args.rad_decimation)
    sim.set_parameter("phase_delay", "on")

    # define excitation signal
    t_vector = np.arange(-16/args.fc, 16/args.fc, 1.0/args.fs)
    samples = np.array(gausspulse(t_vector, bw=args.bw, fc=args.fc), dtype="float32")
    center_index = int(len(t_vector)/2)
    demod_freq = args.fc
    sim.set_excitation(samples, center_index, args.fs, demod_freq)

    # configure analytical beam profile
    sim.set_analytical_beam_profile(args.sigma_lateral, args.sigma_elevational)

    # set spline scatterers
    sim.set_spline_scatterers(spline_degree, knot_vector, control_points, amplitudes)
    
    # simulate one packet
    iq_frames = []
    for i in range(args.packet_size):
        frame = simulate_doppler_frame(args, args.sim_time+i*prt, sim, origin)
        iq_frames.append(frame)
        
    # compute autocorrelation frame at lag 0 (for power)
    power_frame = np.zeros(iq_frames[0].shape, dtype="float32")
    for i in range(args.packet_size):
        power_frame += abs(iq_frames[i])**2
        
    # compute autocorrelation frame at lag 1 (for velocity)
    acf_frame = np.zeros(iq_frames[0].shape, dtype="complex64")
    for i in range(args.packet_size-1):
        acf_frame += np.conj(iq_frames[i])*iq_frames[i+1]
    v_nyq = c0*args.prf/(4.0*args.fc)
    velocity_frame = np.angle(acf_frame)*v_nyq/np.pi

    # user thresholded power frame as mask frame
    power_frame = power_frame/np.max(power_frame.flatten())
    mask_frame = np.array(power_frame)
    inds = power_frame > args.normalized_power_threshold
    mask_frame[inds] = 1.0
    mask_frame[np.logical_not(inds)] = 0.0
    
    velocity_frame = velocity_frame*mask_frame
    num_samples, num_beams = velocity_frame.shape
    
    thetas = np.linspace(-0.5*args.width, 0.5*args.width, num_beams)
    ranges = np.linspace(0.0, args.line_length, num_samples)
    tt,rr = np.meshgrid(thetas, ranges)
    xx = rr*np.sin(tt)
    yy = rr*np.cos(tt)
    
    # visualize the results
    plt.figure(1)
    plt.pcolor(xx, yy, mask_frame, cmap="Greys_r")
    plt.colorbar()
    plt.title("Mask frame from power thresholding")
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()

    plt.figure(2)
    plt.pcolor(xx, yy, velocity_frame, cmap="seismic", vmin=-v_nyq, vmax=v_nyq)
    plt.colorbar()
    plt.title("Time: %f, VNyquist is %f m/s" % (args.sim_time, v_nyq)) 
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    
    plt.figure(3)
    gain = 1.0
    dyn_range = 60
    normalize_factor = np.max(power_frame.flatten())
    log_power = 20*np.log10(gain*power_frame/normalize_factor)
    log_power = 255.0*(log_power+dyn_range)/dyn_range
    # clamp to [0, 255]
    log_power[log_power < 0]     = 0.0
    log_power[log_power > 255.0] = 255.0
    plt.pcolor(xx, yy, log_power, cmap="Greys_r", vmin=0.0, vmax=255.0)
    plt.title("Log-power image")
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    
    plt.show()
    