import numpy as np
import argparse
import h5py
import bsplines
import matplotlib.pyplot as plt

description="""
    Spinning disk phantom in the xz plane.
"""

def create_phantom(args):
    xs = np.random.uniform(low=-args.radius, high=args.radius, size=(args.num_scatterers,))
    ys = np.zeros(shape=(args.num_scatterers,))
    zs = np.random.uniform(low=-args.radius, high=args.radius, size=(args.num_scatterers,))
    ampls = np.random.uniform(low=-1.0, high=1.0, size=(args.num_scatterers,))
    
    keep_inds = xs**2+zs**2 <= args.radius**2
    xs = xs[keep_inds]
    ys = ys[keep_inds]
    zs = zs[keep_inds]
    ampls = ampls[keep_inds]
    points = np.array(np.vstack([xs, ys, zs]), dtype="float32")
    num_scatterers = points.shape[1]
    print "After filtering: %d scatterers." % num_scatterers
    
    # the total number of control points should be num_cs,
    # num_cs - degree different control points are needed.
    control_points = np.empty((num_scatterers, args.num_cs, 3), dtype="float32")
    for cs_idx,ry in enumerate(np.linspace(0.0, 2*np.pi, args.num_cs-args.degree, endpoint=False)):
        rot_mat = np.array([[np.cos(ry),  0.0, np.sin(ry)],
                            [0.0,         1.0, 0.0       ],
                            [-np.sin(ry), 0.0, np.cos(ry)]], dtype="float32")
        control_points[:, cs_idx, :] = rot_mat.dot(points).transpose()
    
    # repeat the degree first control points for all splines
    control_points[:, args.num_cs-args.degree:, :] = control_points[:, 0:args.degree, :]
    
    temp0 = -float(args.degree)/(args.num_cs-args.degree)
    temp1 = float(args.num_cs)/(args.num_cs-args.degree)
    num_knots = args.num_cs+args.degree+1
    knots = np.linspace(temp0, temp1, num_knots)*args.period

    # shift all control points for all splines in z-direction
    control_points[:, :, 2] += args.z0
    
    with h5py.File(args.h5_file, "w") as f:
        f["amplitudes"]     = np.array(ampls, dtype="float32")
        f["control_points"] = np.array(control_points, dtype="float32")
        f["knot_vector"]    = np.array(knots, dtype="float32")
        f["spline_degree"]  = args.degree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Output hdf5 file")
    parser.add_argument("--degree", help="Spline degree", type=int, default=2)
    parser.add_argument("--num_cs", help="Number of spline control points", type=int, default=10)
    parser.add_argument("--period", help="Evaluation interval is [0, period] (rot.speed)", type=float, default=1.0)
    parser.add_argument("--num_scatterers", type=int, default=20000)
    parser.add_argument("--z0", help="Z component of center point", type=float, default=0.025)
    parser.add_argument("--radius", help="Radius of the disk", type=float, default=2e-2)
    args = parser.parse_args()
    
    create_phantom(args)
    