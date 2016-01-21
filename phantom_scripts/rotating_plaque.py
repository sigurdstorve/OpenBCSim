import numpy as np
import h5py
import argparse
import bsplines

description="""\
    Artery short-axis in xz-plane with a plaque ball located
    at the border and rotating with constant angular velocity.
"""

def create_phantom(args):
    xs = np.array(np.random.uniform(size=(args.num_scatterers,), low=args.x_min, high=args.x_max))
    ys = np.zeros((args.num_scatterers,))
    zs = np.array(np.random.uniform(size=(args.num_scatterers,), low=args.z_min, high=args.z_max))
    ampls = np.array(np.random.uniform(size=(args.num_scatterers,), low=0.0, high=1.0))
    
    # indices to keep
    outside_inds = (zs-args.z0)**2 + xs**2 >= args.radius**2
    ampls[np.logical_not(outside_inds)] *= args.inside_ampl    
    
    num_tissue_scatterers = len(xs)
    
    plaque_xs = np.array(np.random.uniform(size=(args.num_plaque_scatterers,),
                         low=-args.plaque_radius, high=args.plaque_radius))
    plaque_ys = np.zeros((args.num_plaque_scatterers,))
    plaque_zs = np.array(np.random.uniform(size=(args.num_plaque_scatterers,),
                         low=-args.plaque_radius, high=args.plaque_radius))
    plaque_ampls = np.array(np.random.uniform(size=(args.num_plaque_scatterers,), low=0.0, high=1.0))
    
    plaque_ampls[plaque_xs**2 + plaque_zs**2 >= args.plaque_radius**2] = 0.0
    
    num_scatterers = num_tissue_scatterers + args.num_plaque_scatterers
    print 'Total number of scatterers: %d' % num_scatterers
    
    # knot vector for the approximation
    knot_vector = bsplines.uniform_regular_knot_vector(args.num_cs, args.spline_degree, t0=0.0, t1=1.0)
    knot_vector = np.array(knot_vector, dtype='float32')
    knot_avgs = bsplines.control_points(args.spline_degree, knot_vector)

    control_points = np.zeros( (num_scatterers, args.num_cs, 3), dtype='float32')
    amplitudes = np.zeros( (num_scatterers,), dtype="float32")
    
    for i in range(args.num_cs):
        theta = i*np.pi*2/args.num_cs
        control_points[:num_tissue_scatterers,i,0] = xs
        control_points[:num_tissue_scatterers,i,1] = ys
        control_points[:num_tissue_scatterers,i,2] = zs

        control_points[num_tissue_scatterers:,i,0] = plaque_xs + args.radius*np.sin(theta)
        control_points[num_tissue_scatterers:,i,1] = plaque_ys
        control_points[num_tissue_scatterers:,i,2] = plaque_zs + args.z0 + args.radius*np.cos(theta)

        amplitudes[num_tissue_scatterers:] = plaque_ampls
        amplitudes[:num_tissue_scatterers] = ampls
    
    with h5py.File(args.h5_file, 'w') as f:
        f["control_points"] = control_points
        f["amplitudes"] = amplitudes
        f["spline_degree"] = args.spline_degree
        f["knot_vector"] = knot_vector
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file", help="Output file")
    parser.add_argument("--x_min", help="Bounding box [m]", type=float, default=-0.015)
    parser.add_argument("--x_max", help="Bounding box [m]", type=float, default=0.015)
    parser.add_argument("--z_min", help="Bounding box [m]", type=float, default=0.0)
    parser.add_argument("--z_max", help="Bounding box [m]", type=float, default=0.03)
    parser.add_argument("--num_scatterers", type=int, default=100000)
    parser.add_argument("--z0", help="Artery center depth [m]", type=float, default=0.015)
    parser.add_argument("--radius", help="Vessel radius [m]", type=float, default=5e-3)
    parser.add_argument("--num_cs", help="Number of spline control points", type=int, default=10)
    parser.add_argument("--spline_degree", type=int, default=3)
    parser.add_argument("--plaque_radius", help="Radius of plaque ball [m]", type=float, default=1.6e-3)
    parser.add_argument("--num_plaque_scatterers", help="Number of scatterers in plaque ball", type=int, default=1000)
    parser.add_argument("--inside_ampl", help="Amplitude scaling factor for inside scatterers", type=float, default=0.02)
    args = parser.parse_args()
    
    create_phantom(args)
    