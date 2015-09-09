import sys
import bsplines
import h5py
import argparse
import numpy as np
import random

description = """
    Generate a hdf5 file with scatterers on spline form for a box moving
    harmonically up and down along the z-axis.
"""

def create_phantom(args):    
    knot_vector = bsplines.uniform_regular_knot_vector(args.num_control_points,
                                                       args.spline_degree,
                                                       t0=args.t_start,
                                                       t1=args.t_end)
    knot_vector = np.array(knot_vector, dtype='float32')
    
    knot_avgs = bsplines.control_points(args.spline_degree, knot_vector)
    
    nodes = np.empty( (args.num_scatterers, args.num_control_points, 4), dtype='float32')        
    for scatterer_i in range(args.num_scatterers):
        print 'Scatterer %d of %d' % (scatterer_i, args.num_scatterers)
        # generate random start point
        x0 = random.uniform(args.x_min, args.x_max)
        y0 = random.uniform(args.y_min, args.y_max)
        z0 = args.z0 + random.uniform(-0.5, 0.5)*args.thickness
        scatterer_amplitude=random.uniform(0.0, 1.0)       
        for control_pt_i, t_star in enumerate(knot_avgs):
            x = x0
            y = y0
            z = z0 + args.ampl*np.cos(2*np.pi*args.freq*t_star)
            nodes[scatterer_i, control_pt_i, :] = [x, y, z, scatterer_amplitude]                     
    
    with h5py.File(args.h5_file, 'w') as f:
        f["nodes"] = nodes
        f["spline_degree"] = args.spline_degree
        f["file_format_version"] = "1"
        f["knot_vector"] = knot_vector

    print 'Scatterer dataset written to %s' % args.h5_file
    print 'knot vector: %s' % knot_vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('h5_file', help='Name of output scatterer datafile')
    parser.add_argument('--x_min', help='Minimum x value for scatterer distribution [m]', type=float, default=-0.025)
    parser.add_argument('--x_max', help='Maximum x value for scatterer distribution [m]', type=float, default=0.025)
    parser.add_argument('--y_min', help='Minimum y value for scatterer distribution [m]', type=float, default=-0.025) 
    parser.add_argument('--y_max', help='Maximum y value for scatterer distribution [m]', type=float, default=0.025)
    parser.add_argument('--thickness', help='Thickness of box in z-direction [m]', type=float, default=0.05)
    parser.add_argument('--z0', help='Mean z value for midpoint of box', type=float, default=8e-2)
    parser.add_argument('--ampl', help='Amplitude of oscillation [m]', type=float, default=1e-2)
    parser.add_argument('--freq', help='Frequency of oscillation [Hz]', type=float, default=1.3)
    parser.add_argument('--num_scatterers', help='Number of random scatterers inside box', type=int, default=100000)
    parser.add_argument('--num_control_points', help='Number of spline control points.', type=int, default=10)
    parser.add_argument('--t_start', help='Start time [sec]', type=float, default=0.0)
    parser.add_argument('--t_end', help='End time [sec]', type=float, default=1.0)
    parser.add_argument('--spline_degree', help='The spline degree used in approximation', type=int, default=3)
    args = parser.parse_args()
    
    create_phantom(args)