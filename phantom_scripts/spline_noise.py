import numpy as np
import argparse
import h5py
import bsplines

description = """\
    Create noise as random spline paths.
"""

def create_phantom(args):
    nodes = np.empty((args.num_scatterers, args.num_cs, 4), dtype='float32')
    nodes[:,:,0] = np.random.uniform(size=(args.num_scatterers, args.num_cs), low=args.x_min, high=args.x_max)
    nodes[:,:,1] = np.random.uniform(size=(args.num_scatterers, args.num_cs), low=args.y_min, high=args.y_max)
    nodes[:,:,2] = np.random.uniform(size=(args.num_scatterers, args.num_cs), low=args.z_min, high=args.z_max)
    nodes[:,:,3] = np.random.uniform(size=(args.num_scatterers, args.num_cs), low=0.0, high=1.0)
    with h5py.File(args.h5_file, 'w') as f:
        f["nodes"] = nodes
        f["spline_degree"] = args.spline_degree
        f["knot_vector"] = np.array(bsplines.uniform_regular_knot_vector(args.num_cs, args.spline_degree, t0=0.0, t1=1.001), dtype='float32')
    print 'Dataset written to %s' % args.h5_file
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('h5_file', help='Name of scatterer h5 file')
    parser.add_argument('--num_scatterers', help='Number of point scatterers', type=int, default=100000)
    parser.add_argument('--x_min', help='bbox', type=float, default=-0.04) 
    parser.add_argument('--x_max', help='bbox', type=float, default=0.04)     
    parser.add_argument('--y_min', help='bbox', type=float, default=-0.04) 
    parser.add_argument('--y_max', help='bbox', type=float, default=0.04)     
    parser.add_argument('--z_min', help='bbox', type=float, default=0.01) 
    parser.add_argument('--z_max', help='bbox', type=float, default=0.08)
    parser.add_argument('--spline_degree', help='Spline degree to use in approximation', type=int, default=3)
    parser.add_argument('--num_cs', help='Number of control points to use in spline approximation', type=int, default=20)
    args = parser.parse_args()
    
    create_phantom(args)
