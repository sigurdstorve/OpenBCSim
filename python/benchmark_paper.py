import numpy as np
from pyrfsim import RfSimulator
import argparse
from scipy.signal import gausspulse
from time import time
import h5py
import sys
sys.path.append("../phantom_scripts")
import rotation3d
import os

description="""
    Benchmark performance on a collection of the generated
    phantoms. Will compare different configurations of 
    CPU- and GPU algorithms.
"""
x_axis = np.array([1.0, 0.0, 0.0], dtype="float32")
y_axis = np.array([0.0, 1.0, 0.0], dtype="float32")
z_axis = np.array([0.0, 0.0, 1.0], dtype="float32")

def configure_linear_scan(sim, t_min, t_max, num_lines, x_min=-0.04, x_max=0.04, length=0.05):
    """ Linear scan with random times for all scanline.  """
    origins = np.zeros((num_lines, 3), dtype="float32")
    origins[:,0] = np.linspace(x_min, x_max, num_lines)
    directions = np.array(np.tile(z_axis, (num_lines, 1)), dtype="float32")
    lateral_dirs = np.array(np.tile(x_axis, (num_lines, 1)), dtype="float32")
    timestamps = np.array(np.random.uniform(low=t_min, high=t_max, size=(num_lines,)), dtype="float32")
    sim.set_scan_sequence(origins, directions, length, lateral_dirs, timestamps)

def configure_sector_scan(sim, t_min, t_max, num_lines, width=0.8, length=0.12):
    """ Sector scan with random times for all scanlines. """
    origins = np.zeros((num_lines, 3), dtype="float32")
    directions = np.empty((num_lines, 3), dtype="float32")
    lateral_dirs = np.empty((num_lines, 3), dtype="float32")
    for beam_no, angle in enumerate(np.linspace(-0.5*width, 0.5*width, num_lines)):
        M = rotation3d.rot_mat_y(angle)
        directions[beam_no, :]   = M.dot(z_axis)
        lateral_dirs[beam_no, :] = M.dot(x_axis)
    timestamps = np.array(np.random.uniform(low=t_min, high=t_max, size=(num_lines,)), dtype="float32")
    sim.set_scan_sequence(origins, directions, length, lateral_dirs, timestamps)

def reconfigure_scatterers(sim, h5_phantom):
    """ Configures scatterers and returns total number. """
    sim.clear_fixed_scatterers()
    sim.clear_spline_scatterers()
    total_num = 0
    with h5py.File(h5_phantom, "r") as f:
        if "data" in f:
            scatterers_data = f["data"].value
            sim.add_fixed_scatterers(scatterers_data)
            total_num += scatterers_data.shape[0]
        if "control_points" in f:
            amplitudes     = f["amplitudes"].value
            control_points = f["control_points"].value
            knot_vector    = f["knot_vector"].value
            spline_degree  = int(f["spline_degree"].value)
            sim.add_spline_scatterers(spline_degree, knot_vector, control_points, amplitudes)
            total_num += control_points.shape[0]
    return total_num

def simulate_with_timing(sim, num_rep, png_file=None):
    """ Returns list simulation times per frame (rep), and optionally save PNG."""

    frame_times = []
    for rep_no in range(num_rep):
        start_time = time()
        iq_lines = sim.simulate_lines()
        frame_times.append(time() - start_time)
        
    if png_file != None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        img = np.real(abs(iq_lines))
        plt.imshow(img, cmap=plt.get_cmap("Greys_r"), aspect="auto", interpolation="nearest")
        plt.savefig(png_file)

    return frame_times
    
def ns_per_scatterer(frame_times, num_scatterers, num_lines):
    """ Return mean, std sim time in ns per scatterer per line """
    ns_per_scatterer = map(lambda time_sec: 1e9*time_sec/(num_lines*num_scatterers), frame_times)
    ns_mean = np.mean(ns_per_scatterer)
    ns_std  = np.std(ns_per_scatterer)
    return ns_mean, ns_std
 
class ResultBuffer(object):
    def __init__(self):
        self.lines = []
        
    def add_msg(self, msg):
        self.lines.append("%s\n"%msg)
        print "*** %s" % msg
    
    def write(self, out_file):
        with open(out_file, "w") as f:
            for l in self.lines:
                f.write(l)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("res_file", help="Text file to write results")
    parser.add_argument("--num_lines", type=int, default=192)
    parser.add_argument("--num_rep", help="For statistics on timing", type=int, default=10)
    parser.add_argument("--save_pdf", help="Save .pdf images", action="store_true")
    parser.add_argument("--device_no", help="GPU device no. to use", type=int, default=0)
    parser.add_argument("--num_cpu_cores", type=int, default=8)
    parser.add_argument("--fs", help="Sampling freq [Hz]", type=float, default=50e6)
    parser.add_argument("--fc", help="Pulse center freq [Hz]", type=float, default=2.5e6)
    parser.add_argument("--bw", help="Pulse bandwidth", type=float, default=0.1)
    parser.add_argument("--phantom_folder", default="../generated_phantoms")
    parser.add_argument("--arc_proj", choices=["on", "off"], default="on")
    parser.add_argument("--phase_delay", choices=["on", "off"], default="on")
    parser.add_argument("--enable_cpu", help="Also time CPU impl (slow)", action="store_true")
    parser.add_argument("--lut_file", help="Use lookup table beam profile", default="")
    args = parser.parse_args()

    res_buffer = ResultBuffer()
    
    sim_types = ["gpu"]
    if args.enable_cpu:
        sim_types.append("cpu")
    
    for sim_type in sim_types:
        sim = RfSimulator(sim_type)
        device_name = ""
        if sim_type == "gpu":
            sim.set_parameter("gpu_device", "%d" % args.device_no)
            device_name = sim.get_parameter("cur_device_name")
        elif sim_type == "cpu":
            sim.set_parameter("num_cpu_cores", "%d" % args.num_cpu_cores)
        res_buffer.add_msg("=== SIMULATION RESULTS WITH %s %s ===" % (sim_type.upper(), device_name))
        sim.set_parameter("verbose", "0")
        sim.set_parameter("sound_speed", "1540.0")
        sim.set_parameter("radial_decimation", "30")
        sim.set_parameter("use_arc_projection", args.arc_proj)
        res_buffer.add_msg("Arc projection: %s" % args.arc_proj)
        sim.set_parameter("phase_delay", args.phase_delay)
        res_buffer.add_msg("Complex phase delay: %s" % args.phase_delay)
            
        # configure the RF excitation
        ts = 1.0/args.fs
        tc = 1.0/args.fc
        t_vector = np.arange(-16*tc, 16*tc, ts)
        samples = np.array(gausspulse(t_vector, bw=args.bw, fc=args.fc), dtype="float32")
        center_index = int(len(t_vector)/2)
        f_demod = args.fc
        sim.set_excitation(samples, center_index, args.fs, f_demod)
    
        # configure the beam profile
        if args.lut_file != "":
            with h5py.File(args.lut_file, "r") as lut_f:
                samples    = np.array(lut_f["beam_profile"].value, dtype="float32")
                r_ext = lut_f["rad_extent"].value
                e_ext = lut_f["ele_extent"].value
                l_ext = lut_f["lat_extent"].value
            sim.set_lut_beam_profile(float(r_ext[0]), float(r_ext[1]),\
                                     float(l_ext[0]), float(l_ext[1]),\
                                     float(e_ext[0]), float(e_ext[1]), samples)
            res_buffer.add_msg("using lookup-table: %s" % args.lut_file)
        else:
            sim.set_analytical_beam_profile(1e-3, 1e-3)
            res_buffer.add_msg("using analytic beam profile")

        res_buffer.add_msg("CASE 1: Linear scan of plaque phantom")
        num_scatterers = reconfigure_scatterers(sim, os.path.join(args.phantom_folder, "carotid_plaque.h5"))
        assert num_scatterers==sim.get_total_num_scatterers()
        configure_linear_scan(sim, 0.0, 1.0, args.num_lines)
        rep_times = simulate_with_timing(sim, args.num_rep, "benchmark_carotid_plaque_%s.png"%sim_type)
        ns_mean, ns_std = ns_per_scatterer(rep_times, num_scatterers, args.num_lines)
        res_buffer.add_msg("   Number of scatterers: %d" % num_scatterers)
        res_buffer.add_msg("   %3.3f +- %3.3f nanosec per scatterer per line [N=%d]" % (ns_mean, ns_std, len(rep_times)))
        res_buffer.add_msg("   (%s)" % str(rep_times))
        
        res_buffer.add_msg("CASE 2: Tissue with flow phantom")
        num_scatterers = reconfigure_scatterers(sim, os.path.join(args.phantom_folder, "tissue_with_parabolic_flow.h5"))
        assert num_scatterers==sim.get_total_num_scatterers()
        configure_linear_scan(sim, 0.0, 1.0, args.num_lines)
        rep_times = simulate_with_timing(sim, args.num_rep, "benchmark_tissue_with_parabolic_flow_%s.png"%sim_type)
        ns_mean, ns_std = ns_per_scatterer(rep_times, num_scatterers, args.num_lines)
        res_buffer.add_msg("   Number of scatterers: %d" % num_scatterers)
        res_buffer.add_msg("   %3.3f +- %3.3f nanosec per scatterer per line [N=%d]" % (ns_mean, ns_std, len(rep_times)))
        res_buffer.add_msg("   (%s)" % str(rep_times))

        res_buffer.add_msg("CASE 3: Contracting cylinder")
        num_scatterers = reconfigure_scatterers(sim, os.path.join(args.phantom_folder, "contracting_cylinder_spline.h5"))
        assert num_scatterers==sim.get_total_num_scatterers()
        configure_sector_scan(sim, 0.0, 0.99, args.num_lines)
        rep_times = simulate_with_timing(sim, args.num_rep, "benchmark_contracting_cylinder_spline_%s.png"%sim_type)
        ns_mean, ns_std = ns_per_scatterer(rep_times, num_scatterers, args.num_lines)
        res_buffer.add_msg("   Number of scatterers: %d" % num_scatterers)
        res_buffer.add_msg("   %3.3f +- %3.3f nanosec per scatterer per line [N=%d]" % (ns_mean, ns_std, len(rep_times)))
        res_buffer.add_msg("   (%s)" % str(rep_times))

        res_buffer.add_msg("CASE 4: Left ventricle phantom")
        num_scatterers = reconfigure_scatterers(sim, os.path.join(args.phantom_folder, "lv_spline_model.h5"))
        assert num_scatterers==sim.get_total_num_scatterers()
        configure_sector_scan(sim, 0.0, 0.99, args.num_lines)
        rep_times = simulate_with_timing(sim, args.num_rep, "benchmark_lv_spline_model_%s.png"%sim_type)
        ns_mean, ns_std = ns_per_scatterer(rep_times, num_scatterers, args.num_lines)
        res_buffer.add_msg("   Number of scatterers: %d" % num_scatterers)
        res_buffer.add_msg("   %3.3f +- %3.3f nanosec per scatterer per line [N=%d]" % (ns_mean, ns_std, len(rep_times)))
        res_buffer.add_msg("   (%s)" % str(rep_times))

        res_buffer.add_msg("CASE 5: Rotating cube")
        num_scatterers = reconfigure_scatterers(sim, os.path.join(args.phantom_folder, "rot_cube.h5"))
        assert num_scatterers==sim.get_total_num_scatterers()
        configure_sector_scan(sim, 0.0, 0.99, args.num_lines)
        rep_times = simulate_with_timing(sim, args.num_rep, "benchmark_rot_cube_%s.png"%sim_type)
        ns_mean, ns_std = ns_per_scatterer(rep_times, num_scatterers, args.num_lines)
        res_buffer.add_msg("   Number of scatterers: %d" % num_scatterers)
        res_buffer.add_msg("   %3.3f +- %3.3f nanosec per scatterer per line [N=%d]" % (ns_mean, ns_std, len(rep_times)))
        res_buffer.add_msg("   (%s)" % str(rep_times))
        
        res_buffer.add_msg("CASE 6: Simple linear scatterers")
        num_scatterers = reconfigure_scatterers(sim, os.path.join(args.phantom_folder, "simple.h5"))
        assert num_scatterers==sim.get_total_num_scatterers()
        configure_sector_scan(sim, 0.0, 0.99, args.num_lines)
        rep_times = simulate_with_timing(sim, args.num_rep, "benchmark_simple_%s.png"%sim_type)
        ns_mean, ns_std = ns_per_scatterer(rep_times, num_scatterers, args.num_lines)
        res_buffer.add_msg("   Number of scatterers: %d" % num_scatterers)
        res_buffer.add_msg("   %3.3f +- %3.3f nanosec per scatterer per line [N=%d]" % (ns_mean, ns_std, len(rep_times)))
        res_buffer.add_msg("   (%s)" % str(rep_times))
        
    res_buffer.write(args.res_file)
    
