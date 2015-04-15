import numpy as np
import h5py
import matplotlib.pyplot
from rotation3d import general_rot_mat
import argparse
import bsplines

description = """\
    Script to generate a spline scatterer phantom of
    a rotating box.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('h5_file', help='Name of scatterer file')
    parser.add_argument('--x_min', help='bbox [m]', type=float, default=-0.03)
    parser.add_argument('--x_max', help='bbox [m]', type=float, default=0.03)
    parser.add_argument('--y_min', help='bbox [m]', type=float, default=-0.03)
    parser.add_argument('--y_max', help='bbox [m]', type=float, default=0.03)
    parser.add_argument('--z_min', help='bbox [m]', type=float, default=-0.03)
    parser.add_argument('--z_max', help='bbox [m]', type=float, default=0.03)
    parser.add_argument('--z0', help='Z coordinate of box center', type=float, default=0.06)
    parser.add_argument('--num_cs', help='Number of control points in spline approximation', type=int, default=20)
    parser.add_argument('--spline_degree', help='Spline degree', type=int, default=3)
    parser.add_argument('--t0', help='Start time', type=float, default=0.0)
    parser.add_argument('--t1', help='End time', type=float, default=1.0)
    parser.add_argument('--num_scatterers', help='Num scatterers', type=int, default=100000)
    parser.add_argument('--x_angular_velocity', type=float, default=3.14)
    parser.add_argument('--y_angular_velocity', type=float, default=1.0)
    parser.add_argument('--z_angular_velocity', type=float, default=0.4)
    args = parser.parse_args()
    
    xs = np.random.uniform(size=(args.num_scatterers,), low=args.x_min, high=args.x_max)
    ys = np.random.uniform(size=(args.num_scatterers,), low=args.y_min, high=args.y_max)
    zs = np.random.uniform(size=(args.num_scatterers,), low=args.z_min, high=args.z_max)
    as_ = np.random.uniform(size=(args.num_scatterers,), low=0.0, high=1.0)
    
    scatterers = np.empty( (3, args.num_scatterers))
    scatterers[0, :] = xs
    scatterers[1, :] = ys
    scatterers[2, :] = zs
    
    nodes = np.empty( (args.num_scatterers, args.num_cs, 4), dtype='float32')
    
    knots = bsplines.uniform_regular_knot_vector(args.num_cs, args.spline_degree, t0=args.t0, t1=args.t1)
    knot_avgs = bsplines.control_points(args.spline_degree, knots)

    
    for control_point_i, t_star in enumerate(knot_avgs):    
        print 't=%f' % t_star
        x_angle = args.x_angular_velocity*t_star
        y_angle = args.y_angular_velocity*t_star
        z_angle = args.z_angular_velocity*t_star
        rot_mat = general_rot_mat(x_angle, y_angle, z_angle)
        
        temp = np.dot(rot_mat, scatterers).transpose()
        temp[:, 2] += args.z0
        nodes[:, control_point_i, 0:3] = temp
        nodes[:, control_point_i, 3] = as_
        
    if False:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatterer_i = 0
        xs_vis = nodes[scatterer_i, :, 0]
        ys_vis = nodes[scatterer_i, :, 1]
        zs_vis = nodes[scatterer_i, :, 2]
        ax.plot(xs_vis, ys_vis, zs_vis)
        plt.show() 
   
    with h5py.File(args.h5_file, 'w') as f:
        f["spline_degree"] = args.spline_degree
        f["nodes"] = nodes
        f["knot_vector"] = knots
           
 
