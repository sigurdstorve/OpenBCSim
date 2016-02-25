import numpy as np
import argparse
import h5py
import bsplines

description="""
    Phantom with both stationary tissue and dynamic flow. The flow
    scatterers are splines and the velocity profile can be controlled
    by adjusting the peak velocity (in the center of the vessel) and
    an exponent, where a value of two gives parabolic, while a high
    value approximates constant flow.
    
    NOTE: Currently, more than the required number of control points
    are used. In theory, it should be enough with two control points
    as the motion of each scatterer is in a straight line with 
    constant velocity. This will be fixed in the future.
"""

def create_fixed_scatterers(args, h5_f):
    common_size = (args.num_tissue_scatterers,)
    xs = np.random.uniform(low=-0.5*args.tissue_length, high=0.5*args.tissue_length, size=common_size)
    ys = np.random.uniform(low=-0.5*args.box_dim, high=0.5*args.box_dim, size=common_size)
    zs = np.random.uniform(low=0.0, high=args.box_dim, size=common_size)
    ampls = np.random.uniform(low=-1.0, high=1.0, size=common_size)
    
    keep_inds = (zs-0.5*args.box_dim)**2 + ys**2 >= args.radius**2
    xs = xs[keep_inds]
    ys = ys[keep_inds]
    zs = zs[keep_inds]
    ampls = ampls[keep_inds]
    
    data = np.stack([xs, ys, zs, ampls], axis=-1)
    print "Final number of tissue scatterers: %d " % data.shape[0]
    
    h5_f["data"] = np.array(data)
    
def create_spline_scatterers(args, h5_f):
    num_scatterers = args.num_flow_scatterers
    spline_degree = 2

    knot_vector = bsplines.uniform_regular_knot_vector(args.num_cs, spline_degree, t0=args.t0, t1=args.t1+0.001)
    knot_vector = np.array(knot_vector, dtype="float32")
    knot_avgs = bsplines.control_points(spline_degree, knot_vector)

    total_t = args.t1-args.t0

    # create random scatterers in a cylinder centered around the x-axis
    common_size = (num_scatterers,)
    ys = np.random.uniform(low=-args.radius, high=args.radius, size=common_size)
    zs = np.random.uniform(low=-args.radius, high=args.radius, size=common_size)
    rs = np.sqrt(ys**2 + zs**2)
    keep_inds = rs <= args.radius
    ys = ys[keep_inds]
    zs = zs[keep_inds] 
    rs = rs[keep_inds]

    num_scatterers = len(rs)
    print "Number of flow scatterers after first filtering: %d" % num_scatterers
    # currenly only support for t0=0
    assert(abs(args.t0) < 1e-6)
    
    x_min = -0.5*args.tissue_length - args.peak_velocity*total_t
    x_max = 0.5*args.tissue_length
    print "x components in [%f, %f]" % (x_min, x_max)
    xs = np.random.uniform(low=x_min, high=x_max, size=(num_scatterers,))

    # compute velocity along x-axis for all scatterers
    #    (1-(r/R)**K), where K controls the shape
    # this is a function of the radius
    velocities = args.peak_velocity*(1.0-(rs/args.radius)**args.exponent)

    # compute the x-coordinate of each scatterer at the end time
    end_xs = xs + args.t1*velocities

    # ...and remove those who never will enter the tissue region
    keep_inds = (end_xs >= -0.5*args.tissue_length)
    xs = xs[keep_inds]
    ys = ys[keep_inds]
    zs = zs[keep_inds]
    velocities = velocities[keep_inds]
    
    num_scatterers = len(xs)
    ampls = np.random.uniform(low=-1.0, high=1.0, size=(num_scatterers,))*args.flow_ampl_factor
    
    control_points = np.zeros( (num_scatterers, args.num_cs, 3), dtype="float32")
    for cs_i, t_star in enumerate(knot_avgs):
        print "Computing control points for knot average %d of %d" % (cs_i+1, args.num_cs)

        # Compute control point position. Amplitude is unchanged.
        control_points[:, cs_i, 0] = np.array(xs) + t_star*velocities
        control_points[:, cs_i, 1] = np.array(ys)
        
        # shift down along positive z-axis
        control_points[:, cs_i, 2] = np.array(zs) + 0.5*args.box_dim

    # Write results to disk
    with h5py.File(args.h5_file) as f:
        f["spline_degree"]  = spline_degree
        f["knot_vector"]    = knot_vector
        f["control_points"] = control_points
        f["amplitudes"]     = np.array(ampls, dtype="float32")

    print "Final number of flow scatterers: %d" % num_scatterers
def create_phantom(args):
    with h5py.File(args.h5_file, "w") as f:
        create_fixed_scatterers(args, f)
        create_spline_scatterers(args, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Output file")
    parser.add_argument("--num_tissue_scatterers", help="Number in initial box", type=int, default=1000000)
    parser.add_argument("--num_flow_scatterers", help="Number of dynamic flow scatterers", type=int, default=1000000)
    parser.add_argument("--box_dim", help="yz dimensions of box [m]", type=float, default=0.03)
    parser.add_argument("--radius", help="Tube radius [m]", type=float, default=0.008)
    parser.add_argument("--tissue_length", type=float, default=8e-2)
    parser.add_argument("--num_cs", help="Number of control points for each spline", type=int, default=6)
    parser.add_argument("--flow_ampl_factor", help="Flow ampltiude scale factor", type=float, default=0.3)
    parser.add_argument("--peak_velocity", help="Peak velocity of flow [m/s]", type=float, default=15e-2)
    parser.add_argument("--t0", help="Start time [s]", type=float, default=0.0)
    parser.add_argument("--t1", help="End time [s]", type=float, default=1.0)
    parser.add_argument("--exponent", help="Controls the shape of the velocity profile (2 gives parabolic)", type=float, default=2)
    args = parser.parse_args()
    
    create_phantom(args)