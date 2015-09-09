import argparse
import numpy as np
import matplotlib.pyplot as plt
from splined_loader import load_spline_points
from scipy.interpolate import interp1d
import h5py
from utils import point_inside_polygon

description="""
    Code for generating a scatterer phantom of a
    realistic tube-like structure such as an
    artery by specifying the cross-sectional shape
    at selected points and interpolating between.
    
    The x-axis is (arbitrarily) defined as the long-axis
    of the phantom.
"""

class InterpolatedTube:
    """
    Interpolate between samples of the crossection of a
    tube. Useful for simulating e.g. a realistic artery.
    """
    def __init__(self, spline_files, parameter_values,
                 scale=1.0, crossection_pts=100, spline_limits=(0.0, 1.0),
                 interp_kind='linear'):
        """
        spline_files:       List of .txt files which can be loaded as a spline.
        parameter_values:   Parametric position of each spline cross-section.
        scale:              Common scale factor for all splines.
        crossection_pts:    Number of points to use when rendering the crossectional
                            spline curves.
        spline_limits:      Parametric limits to use when evaluating splines.
        interp_kind:        Interpolation type - forwarded to scipy.interp1d
        """
        if len(spline_files) != len(parameter_values):
            raise RuntimeError("number of spline files and parameter values must match")
        
        crossections = []
        t0, t1 = spline_limits
        curve_ts = np.linspace(t0, t1, crossection_pts)
        for spline_file in spline_files:
            curve_pts = load_spline_points(spline_file, curve_ts)*scale
            crossections.append(curve_pts)

        f = interp1d(parameter_values, crossections, axis=0, kind=interp_kind)
        
        self.crossection_function = f
        self.crossections         = crossections
        self.parameter_values     = parameter_values
    
    def get_limits(self):
        """
        Returns common bounding rectangle for all curves
        in cross-sectional plane the cross-sectional planes.
        Returns [x_min, x_max, y_min, y_max]
        """
        x_min_values = []; x_max_values = []
        y_min_values = []; y_max_values = []
        for i,curve_pts in enumerate(self.crossections):
            x_min, y_min = np.min(curve_pts, axis=0)
            x_max, y_max = np.max(curve_pts, axis=0)
            x_min_values.append(x_min); x_max_values.append(x_max)
            y_min_values.append(y_min); y_max_values.append(y_max)
        x_min = np.min(x_min_values)
        x_max = np.max(x_max_values)
        y_min = np.min(y_min_values)
        y_max = np.max(y_max_values)
        return [x_min, x_max, y_min, y_max]
        
    def evaluate_curve(self, par_value):
        """
        Returns an interpolated crossectional curve.
        """
        par_values = self.parameter_values
        
        f = self.crossection_function
        if par_value < par_values[0] or par_value > par_values[-1]:
            raise RuntimeError("value outside of data range")
        return f(par_value)

def create_phantom(args):

    # assuming constant distance between cross-sectional shapes.
    num_splines = len(args.spline_files)
    parameter_values = np.linspace(args.x_min, args.x_max, num_splines)

    artery_model = InterpolatedTube(args.spline_files,
                                    parameter_values,
                                    scale=args.scale,
                                    interp_kind='cubic')
        
    # the cross-sectional curves are assumed to be in the yz-plane,
    # while the long axis is the x-axis.
    y_min, y_max, z_min, z_max = artery_model.get_limits()
    print 'Scatterer extent: x=%f..%f, y=%f...%f, z=%f..%f'\
            % (args.x_min, args.x_max, y_min, y_max, z_min, z_max)
    
    y_length = y_max-y_min
    z_length = z_max-z_min
    y_extra = args.space_factor*y_length
    z_extra = args.space_factor*z_length
    
    # create random scatterers
    xs  = np.random.uniform(low=args.x_min, high=args.x_max, size=(args.num_scatterers,))
    ys  = np.random.uniform(low=y_min-y_extra, high=y_max+y_extra, size=(args.num_scatterers,))
    zs  = np.random.uniform(low=z_min-z_extra, high=z_max+y_extra, size=(args.num_scatterers,))
    _as = np.random.uniform(low=-1.0, high=1.0, size=(args.num_scatterers,))
    
    for scatterer_no in range(args.num_scatterers):
        x = xs[scatterer_no]
        y = ys[scatterer_no]
        z = zs[scatterer_no]
        
        # evaluate interpolated curve for this x
        curve_pts = artery_model.evaluate_curve(x)

        if scatterer_no % 20000 == 0:
            print 'Processed scatterer %d of %d' % (scatterer_no, args.num_scatterers)
        
        # convert 2d array to list of (x,y)
        polygon = [(curve_pts[i,0],curve_pts[i,1]) for i in range(curve_pts.shape[0])]

        is_inside = point_inside_polygon(y, z, polygon)
        k = args.outside_factor
        if is_inside:
            k = args.inside_factor
        _as[scatterer_no] *= k
        
    
    data = np.empty((args.num_scatterers, 4), dtype='float32')
    data[:, 0] = xs
    data[:, 1] = ys
    data[:, 2] = zs
    data[:, 3] = _as
    
    with h5py.File(args.h5_out, 'w') as f:
        f["data"] = data
    print 'Data written to %s' % args.h5_out
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('x_min', type=float, help='Min x-value (left end of tube)')
    parser.add_argument('x_max', type=float, help='Max x-value (right end of tube)')
    parser.add_argument('h5_out', help='Output HDF5 file name')
    parser.add_argument('spline_files', type=str, nargs='+', help='The cross-sectional splines')
    parser.add_argument('--scale', help='Isotropic curve scale factor', type=float)
    parser.add_argument('--num_scatterers', help='Number of scatterers', type=int, default=1000000)
    parser.add_argument('--inside_factor', help='Interior scatterer amplitude', type=float, default=0.1)
    parser.add_argument('--outside_factor', help='Exterior scatterer amplitude', type=float, default=1.0)
    parser.add_argument('--space_factor', help='Padding outside of crossections.', type=float, default=0.5)
    args = parser.parse_args()
    
    create_phantom(args)
    
