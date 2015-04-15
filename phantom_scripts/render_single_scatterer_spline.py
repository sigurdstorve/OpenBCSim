import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
import bsplines
import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_file')
    parser.add_argument('--num_splines', help='Number of random scatterers to render', type=int, default=1)
    parser.add_argument('--spline_idx', type=int, default=None)
    parser.add_argument('--show_dots', action='store_true')
    args = parser.parse_args()
    
    with h5py.File(args.h5_file, 'r') as f:
        nodes =         f["nodes"].value
        knot_vector =   f["knot_vector"].value
        spline_degree = f["spline_degree"].value

    inds = np.random.uniform(size=(args.num_splines,), low=0, high=nodes.shape[0])
    inds = [int(i) for i in inds]
        
    ts = np.linspace(knot_vector[0], knot_vector[-1]-0.0001, 1000)
    
    fig = plt.figure()
    fig.add_subplot(111, projection='3d')
    for scatterer_no in inds:
        if args.spline_idx != None: scatterer_no = args.spline_idx
        print scatterer_no
        xs_cs = nodes[scatterer_no, :, 0]
        ys_cs = nodes[scatterer_no, :, 1]
        zs_cs = nodes[scatterer_no, :, 2]
        control_points = []
        for i in range(len(xs_cs)):
            control_points.append( np.array([xs_cs[i], ys_cs[i], zs_cs[i]]) )
        ps = bsplines.render_spline(spline_degree, knot_vector, control_points, ts)
        xs,ys,zs = zip(*ps)
        
        # Render control grid
        plt.plot(xs_cs, ys_cs, zs_cs, marker='o', c='k', alpha=0.6)
        
        # Render spline
        plt.plot(xs, ys, zs, linewidth=2, c='b')
        
        # Render trajectory
        # Render som points [scatterers]
        num_dots = 12
        t0 = 0.8
        t1 = 0.9
        ts = np.linspace(t0, t1, num_dots)
        dot_alphas = np.linspace(0.0, 1.0, num_dots)
        for dot_no in range(num_dots):
            dot_alpha = dot_alphas[dot_no]
            t = ts[dot_no]
            pts = bsplines.render_spline(spline_degree, knot_vector, control_points, [t])
            xs_vis, ys_vis, zs_vis = zip(*pts)
            plt.gca().plot(xs_vis, ys_vis, zs_vis, marker='o', linestyle='', alpha=dot_alpha, c='r', markersize=10)
        
    plt.show()
    