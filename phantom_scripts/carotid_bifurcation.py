import numpy as np
import argparse
import h5py
from copy import deepcopy

description="""\
    Script for creating synthetic carotid artery phantoms.
"""

class PlaqueBall:
    def __init__(self, x0, r, theta, center_r, z0):
        """
        Representation of a 3D plaque ball.
        x0:         X-coordinate of ball center
        r:          Radius of plaque ball
        theta:      Orientation in the YZ-plane of ball center
        center_r:   Radius of ball center
        z0:         Z-coordinate of start of center_r
        """
        self.x0       = x0
        self.r        = r
        self.theta    = theta
        self.center_r = center_r
        self.z0       = z0

class CarotidArteryBifurcartionPhantom:
    def __init__(self, args):
        
        # Generate scatterers in a box.
        common_size = (args.num_scatterers,)
        xs = np.random.uniform(size=common_size, low=args.x_min, high=args.x_max)
        ys = np.random.uniform(size=common_size, low=args.y_min, high=args.y_max)
        zs = np.random.uniform(size=common_size, low=args.z_min, high=args.z_max)
        ampls = np.random.uniform(size=common_size, low=0.0, high=1.0)
        
        self.args = args
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.ampls = ampls
    
        # List of list of indices of scatterers that are inside the lumen
        # All lists are to be combined by or'ing them together
        self.interior_inds = []
        
        # List of plaque balls
        self.plaque_balls = {"common":[],
                             "upper":[],
                             "lower":[]}
        
    def get_scatterers_data(self):
        """
        Returns all scatterers.
        """
        xs    = self.xs
        ys    = self.ys
        zs    = self.zs
        ampls = self.ampls
        args  = self.args
        
        # Recompute all indices
        self._process()
        
        outside_inds = self._get_final_indices()
        inside_inds = np.logical_not(outside_inds)
        
        final_ampls = np.array(ampls)
        final_ampls[inside_inds] *= args.lumen_ampl
        
        return xs, ys, zs, final_ampls
    
    def add_plaque_ball(self, plaque_ball, phantom_part):
        """
        Add a new plaque ball to one of the three parts of
        the phantom.
        """
        plaque_balls = self.plaque_balls
        
        if not plaque_balls.has_key(phantom_part):
            raise RuntimeError("Illegal model part")
        
        plaque_balls[phantom_part].append(plaque_ball)
    
    def add_symmetric_plaque_balls(self, plaque_ball, phantom_part, num_balls=10):
        """
        Repeat a plaque ball for many theta angles, which is
        useful to make a constricted part on the artery.
        """
        for i in range(num_balls):
            new_ball = deepcopy(plaque_ball)
            new_ball.theta = i*2*np.pi/num_balls
            self.add_plaque_ball(new_ball, phantom_part)
    
    def _process(self):
        plaque_balls = self.plaque_balls
        
        self._process_common_artery(plaque_balls["common"])
        self._process_bifurcated_arteries()
    
    def _process_common_artery(self, plaque_balls):
        """
        Finds and stores the index of all scatterers that
        are inside of the common artery.
        """
        args          = self.args
        xs            = self.xs
        ys            = self.ys
        zs            = self.zs
        interior_inds = self.interior_inds
        
        temp_inds = np.logical_and(xs >= args.x_min, xs <= args.common_x_max)
        temp_zs = zs-args.z0
        temp_inds = np.logical_and(temp_inds, (temp_zs**2 + ys**2) <= args.large_r**2)

        # Take any plaque balls into account 
        for plaque_ball in plaque_balls:
            plaque_interior_inds = self._get_plaque_interior_inds(plaque_ball,
                                                                  xs,
                                                                  ys,
                                                                  temp_zs)
            
            # The scatterers inside of plaque ball should be excluded from the lumen
            # scatterers.
            temp_inds = np.logical_and(temp_inds, np.logical_not(plaque_interior_inds))
            print len(xs[plaque_interior_inds])

        interior_inds.append(temp_inds)
    
    def _process_bifurcated_arteries(self):
        args         = self.args
        plaque_balls = self.plaque_balls
        
        self._process_bifurcation(args.theta,  plaque_balls["upper"])
        self._process_bifurcation(-args.theta, plaque_balls["lower"])
    
    def _process_bifurcation(self, angle, plaque_balls):
        """
        Finds and stores the index of all scatterers that
        are inside one of the bifurcated arteries.
        """
        args          = self.args
        interior_inds = self.interior_inds
        xs            = self.xs
        ys            = self.ys
        zs            = self.zs
        
        temp_xs = xs*np.cos(angle) - (zs-args.z0)*np.sin(angle)
        temp_zs = xs*np.sin(angle) + (zs-args.z0)*np.cos(angle)
        temp_inds = np.logical_and(temp_xs >= 0.0, temp_xs <= args.x_max)
        temp_inds = np.logical_and(temp_inds, (temp_zs**2 + ys**2) <= args.small_r**2) 
        
        # Take any plaque balls into account 
        for plaque_ball in plaque_balls:
            plaque_interior_inds = self._get_plaque_interior_inds(plaque_ball,
                                                                  temp_xs,
                                                                  ys,
                                                                  temp_zs)
            
            # The scatterers inside of plaque ball should be excluded from the lumen
            # scatterers.
            temp_inds = np.logical_and(temp_inds, np.logical_not(plaque_interior_inds))
            
        interior_inds.append(temp_inds)
    
    def _get_plaque_interior_inds(self, plaque_ball, transformed_xs, transformed_ys, transformed_zs):
        """
        Computes the indices of the interior of a plaque ball which
        is associated with one of the threr model parts.
        """
        
        temp_y = plaque_ball.center_r*np.cos(plaque_ball.theta)
        temp_z = plaque_ball.center_r*np.sin(plaque_ball.theta)
        plaque_interior_inds = (transformed_xs-plaque_ball.x0)**2 + (transformed_ys-temp_y)**2 + (transformed_zs-temp_z)**2 <= plaque_ball.r**2
        return plaque_interior_inds
                
    def _get_final_indices(self):
        """
        Returns a list of indices of the scatterers that
        are outside of the artery lumen.
        """
        args          = self.args
        interior_inds = self.interior_inds
        
        # First determine indices of scatterers inside of lumen
        temp_inds = np.array([False]*args.num_scatterers)

        for inds in interior_inds:
            temp_inds = np.logical_or(temp_inds, inds)
                
        # Then find the complement (which is indices of scatterers outside
        # of the lumen)
        final_inds = np.logical_not(temp_inds)

        return final_inds

def create_phantom(args):
    carotid_phantom = CarotidArteryBifurcartionPhantom(args)

    if args.enable_plaque:
        plaque_ball = PlaqueBall(0.5*args.x_max, 5e-3, np.pi/2, args.small_r + 2.5e-3, args.z0)
        carotid_phantom.add_symmetric_plaque_balls(plaque_ball, "lower")

        plaque_ball = PlaqueBall(0.75*args.x_max, 5e-3, np.pi/2, args.small_r + 1.5e-3, args.z0)
        carotid_phantom.add_plaque_ball(plaque_ball, "upper")

        plaque_ball = PlaqueBall(0.5*args.x_min, 7e-3, np.pi/2, args.small_r+5e-3, args.z0)
        carotid_phantom.add_symmetric_plaque_balls(plaque_ball, "common")

    
    # Get the scatterers data
    xs, ys, zs, ampls = carotid_phantom.get_scatterers_data()
    final_num_scatterers = len(xs)
        
    if args.visualize:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs)
        ax.set_xlim(args.x_min, args.x_max)
        ax.set_ylim(args.y_min, args.y_max)
        ax.set_zlim(args.z_min, args.z_max)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.show()
    
    # Write to .h5 file
    scatterers = np.empty((final_num_scatterers, 4), dtype='float32')
    scatterers[:,0] = xs
    scatterers[:,1] = ys
    scatterers[:,2] = zs
    scatterers[:,3] = ampls
    with h5py.File(args.h5_file, 'w') as f:
        f["data"] = scatterers
    print 'Wrote %d scatterers to %s' % (final_num_scatterers, args.h5_file)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('h5_file', help="Name of fixed-scatterer hdf5 file")
    parser.add_argument("--z0", help="Depth of center [m]", type=float, default=0.025)
    parser.add_argument("--x_min", help="Bounding box for scatterers [m]", type=float, default=-0.08)
    parser.add_argument("--x_max", help="Bounding box for scatterers [m]", type=float, default=0.08)
    parser.add_argument("--y_min", help="Bounding box for scatterers [m]", type=float, default=-0.03)
    parser.add_argument("--y_max", help="Bounding box for scatterers [m]", type=float, default=0.03)
    parser.add_argument("--z_min", help="Bounding box for scatterers [m]", type=float, default=0.0)
    parser.add_argument("--z_max", help="Bounding box for scatterers [m]", type=float, default=0.05)
    parser.add_argument("--num_scatterers", help="Number of random scatterers in bounding box", type=int, default=5000000)
    parser.add_argument("--small_r", help="Radius after bifurcation", type=float, default=5e-3)
    parser.add_argument("--large_r", help="Radius before bifurcation", type=float, default=8.2e-3)
    parser.add_argument("--common_x_max", type=float, default=13e-3)
    parser.add_argument("--theta", help="Angle of bifurcating arteries [radians]", type=float, default=np.pi*10/180.0)
    parser.add_argument("--visualize", help="Render the artery scatterers", action="store_true")
    parser.add_argument("--lumen_ampl", help="Scaling factor for scatterers inside lumen", type=float, default=0.0)
    parser.add_argument("--enable_plaque", action="store_true")
    args = parser.parse_args()
    
    create_phantom(args)
