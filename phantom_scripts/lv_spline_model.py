import bsplines
from lv_model_3d import ThickCappedZEllipsoid
from utils import *
import h5py
import argparse
from phantom_common import load_scale_function

description="""
    Create a spline model for a synthetic 3D LV myocardium which
    contracts cyclically either according to a spline function,
    with period of one second or according to a custom scaling
    signal loaded from a Hdf5 file.
 
    Apex is in the origin.
"""

def default_scale_function(t, ampl=1.0):
    """ Scaling of myocardium as function of time. Period is 1"""
    t = t % 1.0
    assert 0 <= t <= 1.0
    degree = 2
    control_points = [1.0, 1.0, 1.0-0.5*ampl, 1.0-0.5*ampl, 1.0-0.75*ampl, 1.0-0.75*ampl, 1.0, 1.0]
    knots = [0.0,0.0,0.0,0.1,0.25,0.4,0.6,0.9,1.0,1.0,1.001]
    value = bsplines.render_spline(degree, knots, control_points, [1.0-t])
    return value
    
def create_phantom(args):
    lv_model = ThickCappedZEllipsoid(args.x_min,
                                     args.x_max,
                                     args.y_min,
                                     args.y_max,
                                     args.z_min,
                                     args.z_max,
                                     args.thickness,
                                     args.z_ratio)

    # Create the LV scatterers in box and remove the ones not
    # inside of thick myocardium. 
    xs,ys,zs = generate_random_scatterers_in_box(args.x_min,
                                                 args.x_max,
                                                 args.y_min,
                                                 args.y_max,
                                                 args.z_min,
                                                 args.z_max,
                                                 args.num_scatterers_in_box,
                                                 args.thickness)
    
    xs,ys,zs = remove_points_outside_of_interior(xs, ys, zs, lv_model)
    _as = np.random.uniform(low=0.0, high=args.lv_max_amplitude, size=(len(xs),))
    assert(len(ys) == len(xs) and len(zs) == len(xs))
    num_scatterers = len(xs)
    
    print 'Total number of scatterers: %d' % num_scatterers

    # Decide which scaling function to use
    if args.scale_h5_file != None:
        start_time, end_time, scale_fn = load_scale_function(args.scale_h5_file)
        print 'Loaded scaling function on [%f, %f]' % (start_time, end_time)
        print 'Hacking args.t0 and args.t1'
        args.t0 = start_time
        args.t1 = end_time
    else:
        scale_fn = lambda t: default_scale_function(t, ampl=args.motion_ampl)
    
    
    ts = np.linspace(args.t0, args.t1, 1000)

    # knot vector for the approximation
    knot_vector = bsplines.uniform_regular_knot_vector(args.num_cs, args.spline_degree, t0=args.t0, t1=args.t1)
    knot_vector = np.array(knot_vector, dtype='float32')
    knot_avgs = bsplines.control_points(args.spline_degree, knot_vector)

    nodes = np.zeros( (num_scatterers, args.num_cs, 4), dtype='float32')
    for cs_i, t_star in enumerate(knot_avgs):
        print 'Computing control points for knot average %d of %d' % (cs_i+1, args.num_cs)

        s = scale_fn(t_star)

        # Compute control point position. Amplitude is unchanged.
        nodes[:,cs_i,0] = s*np.array(xs)
        nodes[:,cs_i,1] = s*np.array(ys)
        nodes[:,cs_i,2] = s*np.array(zs)
        nodes[:,cs_i,3] = _as 

    
    # Write results to disk
    with h5py.File(args.h5_file) as f:
        f['spline_degree'] = args.spline_degree
        f['knot_vector'] = knot_vector
        f['nodes'] = nodes
    
    print 'Spline scatterer dataset written to %s' % args.h5_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Name of hdf5 file to write scatterers data")
    parser.add_argument("--thickness", help="Thickness of myocardium", type=float, default=8e-3)
    parser.add_argument("--z_ratio", help="Ratio in [0,1] of where to cap ellipsoid", type=float, default=0.7)
    parser.add_argument("--x_min", help="Extent of ellipsoid in x-direction [m]", type=float, default=-0.02)
    parser.add_argument("--x_max", help="Extent of ellipsoid in x-direction [m]", type=float, default=0.02)
    parser.add_argument("--y_min", help="Extent of ellipsoid in y-direction [m]", type=float, default=-0.02)
    parser.add_argument("--y_max", help="Extent of ellipsoid in y-direction [m]", type=float, default=0.02)
    parser.add_argument("--z_min", help="Extent of ellipsoid in z-direction [m]", type=float, default=0.008)
    parser.add_argument("--z_max", help="Extent of ellipsoid in z-direction [m]", type=float, default=0.09)
    parser.add_argument("--num_scatterers_in_box", help="Number of scatterers in box around myocardium", type=int, default=400000)
    parser.add_argument("--motion_ampl", help="Amplitude of contraction", type=float, default=0.25)
    parser.add_argument("--t0", help="Starting time [s]", type=float, default=0.0)
    parser.add_argument("--t1", help="Ending time [s]", type=float, default=1.0)
    parser.add_argument("--spline_degree", help="Degree used in spline approximation", type=int, default=2)
    parser.add_argument("--num_cs", help="Number of control points used in spline approximation", type=int, default=10)
    parser.add_argument("--scale_h5_file", help="Hdf5 file with scaling trace", type=str, default=None)
    parser.add_argument("--lv_max_amplitude", help="Maximum scatterer amplitude", type=float, default=1.0)
    
    args = parser.parse_args()
    create_phantom(args)