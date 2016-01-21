import sys
import argparse
from pyrfsim import RfSimulator
import numpy as np
import h5py
from scipy.signal import hilbert, gausspulse
from time import time
import math

description=\
"""
    An example script for performing a PW Doppler scan.
    
    The line starts in the origin and is directed along
    the positive z-axis.
    The scanning is performed on a user-supplied spline-
    phantom file.
    
    The use of CPU or GPU can be controlled by using the
    flag "use_gpu".
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scatterer_file", help="Scatterer dataset (spline)")
    parser.add_argument("--line_length", help="Length of Doppler beam", type=float, default=0.1)
    parser.add_argument("--sample_pos", help="Normalized sample pos in [0,1] along the beam.", type=float, default=0.5)
    parser.add_argument("--prf", help="Pulse repetition frequency [Hz]", type=int, default=5000)
    parser.add_argument("--num_beams", help="Number of Doppler beams", type=int, default=5000)
    parser.add_argument("--fs", help="Sampling frequency [Hz]", type=float, default=100e6)
    parser.add_argument("--store_audio", help="Store a .wav audio file", action="store_true")
    parser.add_argument("--fc", help="Pulse center frequency [Hz] (also demod. freq.)", type=float, default=5.0e6)
    parser.add_argument("--bw", help="Pulse fractional bandwidth", type=float, default=0.2)
    parser.add_argument("--sigma_lateral", help="Lateral beamwidth", type=float, default=0.5e-3)
    parser.add_argument("--sigma_elevational", help="Elevational beamwidth", type=float, default=1e-3)
    parser.add_argument("--use_gpu", help="Perform simulations using the gpu_spline2 algorithm", action="store_true")
    parser.add_argument("--save_pdf", help="Save pdf figures", action="store_true")
    parser.add_argument("--visualize", help="Interactive figures", action="store_true")
    args = parser.parse_args()
    
    c0 = 1540.0
    
    # Create and configure
    if args.use_gpu:
        print "Using GPU"
        sim = RfSimulator("gpu_spline2")
    else:
        print "Using CPU"
        sim = RfSimulator("spline")
        sim.set_parameter("num_cpu_cores", "all")
    sim.set_parameter("verbose", "0")
    sim.set_print_debug(False)
    sim.set_parameter("sound_speed", "%f" % c0)
    sim.set_parameter("radial_decimation", "30")

    # Enable phase-delays for smooth curves
    sim.set_parameter("phase_delay", "on")
    
    # Set spline scatterers
    with h5py.File(args.scatterer_file, "r") as f:
        control_points = np.array(f["control_points"].value, dtype="float32")
        amplitudes     = np.array(f["amplitudes"].value, dtype="float32")
        knot_vector    = np.array(f["knot_vector"].value, dtype="float32")
        spline_degree  = int(f["spline_degree"].value)
        
    sim.set_spline_scatterers(spline_degree, knot_vector, control_points, amplitudes)
    print "Number of scatterers: %d" % control_points.shape[0]
    
    # Define excitation signal
    t_vector = np.arange(-16/args.fc, 16/args.fc, 1.0/args.fs)
    samples = np.array(gausspulse(t_vector, bw=args.bw, fc=args.fc), dtype="float32")
    center_index = int(len(t_vector)/2) 
    demod_freq = args.fc
    sim.set_excitation(samples, center_index, args.fs, demod_freq)
    if args.save_pdf:
        import matplotlib as mpl
        # HACK needed for saving plots on a Linux server with
        # no display configured.
        mpl.use("Agg")
    
    if args.save_pdf or args.visualize:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        plt.figure()
        plt.plot(t_vector, samples)
        plt.title("Excitation signal")
    if args.save_pdf:
        plt.savefig("figure1.pdf")
    if args.visualize:
        plt.show()
        
    # Create big scan sequence
    origins      = np.empty((args.num_beams, 3), dtype="float32")
    directions   = np.empty((args.num_beams, 3), dtype="float32")
    lateral_dirs = np.empty((args.num_beams, 3), dtype="float32")
    
    for beam_no in range(args.num_beams):
        origins[beam_no, :]      = [0.0, 0.0, 0.0]
        directions[beam_no, :]   = [0.0, 0.0, 1.0]
        lateral_dirs[beam_no, :] = [1.0, 0.0, 0.0]
    timestamps = np.array(range(args.num_beams), dtype="float32")/args.prf
    sim.set_scan_sequence(origins, directions, args.line_length, lateral_dirs, timestamps)

    # Set the beam profile
    sim.set_analytical_beam_profile(args.sigma_lateral, args.sigma_elevational)

    # Do the simulation
    print "Simulating IQ data"
    start_time = time()
    iq_lines = sim.simulate_lines()
    end_time = time()
    print "Simulation took %f sec" % (end_time-start_time)
    
    # get slow-time samples
    num_samples = iq_lines.shape[0]
    sample_idx = int(args.sample_pos*(num_samples-1))
    print "Sample index is %d" % sample_idx

    slowtime_samples = iq_lines[sample_idx, :]
    
    if args.visualize or args.save_pdf:
        plt.figure()
        plt.plot(timestamps, slowtime_samples.real)
        plt.title("Slowtime samples")
        plt.xlabel("Slow time")
    if args.save_pdf:
        plt.savefig("figure2.pdf")
    
    fft_len = 256
    nd = 5
    num_spectrums = int( (len(slowtime_samples)-fft_len)/nd )
    
    pixels = np.empty( (fft_len, num_spectrums))
    hann_win = np.hanning(fft_len)
    for spect_no in range(num_spectrums):
        i0 = spect_no*nd
        temp = slowtime_samples[i0:(i0+fft_len)]*hann_win
        temp_fft = np.fft.fftshift(np.fft.fft(temp))
        temp_fft = temp_fft[::-1] # flip y axis
        pixels[:, spect_no] = abs(temp_fft)
    
    # Normalize to [0, 1]
    max_value = np.max(abs(pixels.flatten()))
    pixels = pixels / max_value
    db_range = 60
    
    # log-compress
    pixels = 127*30*np.log10(pixels)/db_range + 127
    # var np.log10(pixels) frr
    pixels[pixels < 0] = 0
    
    # spectrum freq. limits
    min_freq = -0.5*args.prf
    max_freq = 0.5*args.prf
    
    # use Doppler equation to map freq. to speed
    min_vel = min_freq*c0/(2.0*args.fc)
    max_vel = max_freq*c0/(2.0*args.fc)
    
    if args.visualize or args.save_pdf:
        plt.figure()
        plt.imshow(pixels, cmap=cm.Greys_r, aspect="auto", extent=[timestamps[0], timestamps[-1], min_vel, max_vel])
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.axhline(0.0, linestyle="--", c="w", linewidth=2)
        #plt.colorbar()
    if args.save_pdf:
        plt.savefig("figure3.pdf")
    
    if args.visualize:
        plt.show()
    
    if args.store_audio:
        from scipy.io.wavfile import write
        from scipy.interpolate import interp1d
        fs_wav = 44100
        f = interp1d(timestamps, np.real(slowtime_samples))
        new_times = np.arange(timestamps[0], timestamps[-1], 1.0/fs_wav)
        new_samples = f(new_times)
        
        new_samples = new_samples / np.max(abs(new_samples))
        new_samples = new_samples*32000
        new_samples = np.array(new_samples, dtype="int16")
        
        audio_file = "pw_audio.wav"
        write(audio_file, fs_wav, new_samples)
        print "Audio written to %s" % audio_file
    
 
