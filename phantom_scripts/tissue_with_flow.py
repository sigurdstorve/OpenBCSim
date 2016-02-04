import numpy as np
import argparse
import h5py
import bsplines

description="""
    Phantom which models tissue with fixed scatterers
    and simple flow using dynamic spline scatterers.
    Spline degreee is two.
"""

def create_fixed_scatterers(args, h5_f):
    common_size = (args.num_tissue_scatterers,)
    xs = np.random.uniform(low=-0.5*args.x_length, high=0.5*args.x_length, size=common_size)
    ys = np.random.uniform(low=-0.5*args.box_dim, high=0.5*args.box_dim, size=common_size)
    zs = np.random.uniform(low=0.0, high=args.box_dim, size=common_size)
    ampls = np.random.uniform(low=-1.0, high=1.0, size=common_size)
    
    keep_inds = (zs-0.5*args.box_dim)**2 + ys**2 >= args.radius**2
    xs = xs[keep_inds]
    ys = ys[keep_inds]
    zs = zs[keep_inds]
    ampls = ampls[keep_inds]
    
    data = np.stack([xs, ys, zs, ampls], axis=-1)
    print data.shape
    
    h5_f["data"] = np.array(data)
    
def create_spline_scatterers(args, h5_f):
    num_scatterers = args.num_flow_scatterers
    spline_degree = 2

    knot_vector = bsplines.uniform_regular_knot_vector(args.num_cs, spline_degree, t0=0.0, t1=1.001)
    knot_vector = np.array(knot_vector, dtype="float32")
    knot_avgs = bsplines.control_points(spline_degree, knot_vector)

    L = args.x_length
    # starting positions uniformly distributed in [-1.5L, 0.5L]
    xs = np.random.uniform(low=-1.5*L, high=0.5*L, size=(num_scatterers))
    
    # Assuming t=0...1
    #   function of time: x(t) = (1-t)*x0 + t*(x0+L) = x0 - t*x0 + t*x0 + t*L
    #                          = x0 + t*L

    common_size = (num_scatterers,)
    ys = np.random.uniform(low=-args.radius, high=args.radius, size=common_size)
    zs = np.random.uniform(low=-args.radius, high=args.radius, size=common_size)
    ampls = np.random.uniform(low=-1.0, high=1.0, size=common_size)*args.flow_ampl_factor
    
    keep_inds = ys**2 + zs**2 < args.radius**2
    xs = xs[keep_inds]
    ys = ys[keep_inds]
    zs = zs[keep_inds] + 0.5*args.box_dim
    ampls = ampls[keep_inds]
    
    num_scatterers = len(xs)
    control_points = np.zeros( (num_scatterers, args.num_cs, 3), dtype="float32")
    for cs_i, t_star in enumerate(knot_avgs):
        print "Computing control points for knot average %d of %d" % (cs_i+1, args.num_cs)

        # Compute control point position. Amplitude is unchanged.
        control_points[:, cs_i, 0] = np.array(xs) + t_star*L
        control_points[:, cs_i, 1] = np.array(ys)
        control_points[:, cs_i, 2] = np.array(zs)

    # Write results to disk
    with h5py.File(args.h5_file) as f:
        f["spline_degree"]  = spline_degree
        f["knot_vector"]    = knot_vector
        f["control_points"] = control_points
        f["amplitudes"]     = np.array(ampls, dtype="float32")

def create_phantom(args):
    with h5py.File(args.h5_file, "w") as f:
        create_fixed_scatterers(args, f)
        create_spline_scatterers(args, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Output file")
    parser.add_argument("--num_tissue_scatterers", help="Number in initial box", type=int, default=1000000)
    parser.add_argument("--num_flow_scatterers", help="Number of dynamic flow scatterers", type=int, default=200000)
    parser.add_argument("--box_dim", help="yz dimensions of box [m]", type=float, default=0.03)
    parser.add_argument("--radius", help="Tube radius [m]", type=float, default=0.008)
    parser.add_argument("--x_length", type=float, default=8e-2)
    parser.add_argument("--num_cs", help="Number of control points for each spline", type=int, default=6)
    parser.add_argument("--flow_ampl_factor", help="Flow ampltiude scale factor", type=float, default=0.3)
    args = parser.parse_args()
    
    create_phantom(args)