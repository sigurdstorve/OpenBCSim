import bsplines
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
from phantom_common import load_scale_function

description="""\
    Create spline cylinder phantom parallel to the z-axis with one of its circular faces
    in origin which is compressed either harmonically or according to some stretching
    function loaded from a Hdf5 file.
"""

def create_phantom(args):

    # Load scaling function
    start_time, end_time, scale_fn = load_scale_function(args.h5_scale)
    print 'Loaded scale function on [%f, %f] sec.' % (start_time, end_time)
    
    # Generate scatterers in a box
    xs = np.random.uniform(low=-args.r0, high=args.r0, size=(args.num_scatterers,))
    ys = np.random.uniform(low=-args.r0, high=args.r0, size=(args.num_scatterers,))
    zs = np.random.uniform(low=0.0, high=args.z0, size=(args.num_scatterers,))
    pts = zip(xs, ys, zs)

    # Discard scatterers outside of cylinder
    pts = filter(lambda (x,y,z): x**2+y**2 <= args.r0**2, pts)
    xs, ys, zs = map(np.array, zip(*pts))
    num_scatterers = len(xs)
    
    # Create random amplitudes
    amplitudes = np.random.uniform(low=0.0, high=1.0, size=(num_scatterers,))  
    
    # Create knot vector
    knots = bsplines.uniform_regular_knot_vector(args.num_control_points, args.spline_degree,
                                                 t0=start_time, t1=end_time)
    knot_avgs = bsplines.control_points(args.spline_degree, knots)
    
    control_points = np.empty((num_scatterers, args.num_control_points, 3), dtype='float32')

    for c_i, t_star in enumerate(knot_avgs):
        alpha = scale_fn(t_star)
        control_points[:, c_i, 0] = xs/np.sqrt(alpha)
        control_points[:, c_i, 1] = ys/np.sqrt(alpha)
        control_points[:, c_i, 2] = zs*alpha

    with h5py.File(args.h5_out, 'w') as f:
        f["spline_degree"] = args.spline_degree
        f["control_points"] = control_points
        f["amplitudes"] = np.array(amplitudes, dtype="float32")
        f["knot_vector"] = np.array(knots, dtype='float32')    
    print 'Spline scatterer dataset written to %s' % args.h5_out
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('h5_out', help='Name of hdf5 file with spline scatterers data')
    parser.add_argument('h5_scale', help='Hdf5 with scaling signal used when deforming scatterers')
    parser.add_argument('--r0', help='Radius of initial cylinder [m]', type=float, default=1e-2)
    parser.add_argument('--z0', help='Depth of initial cylinder', type=float, default=0.12)
    parser.add_argument('--num_scatterers', help='Num scatterers in initial box', type=int, default=10000)
    parser.add_argument('--num_control_points', help='Number of spline control points for each scatterer', type=int, default=10)
    parser.add_argument('--spline_degree', help='Spline degree in approximation', type=int, default=3)
    args = parser.parse_args()
    
    create_phantom(args)
