import numpy as np
import bsplines

### Code to load and render a spline saved with "splined".

def load_spline(filename):
    def get_str(in_file, prefix):
        line = in_file.readline()
        prefix_key = '%s:' % prefix
        if not prefix_key in line: raise RuntimeError("Unable to find key %s" % prefix)
        return line.strip(prefix_key).strip()
    def read_float(in_file, prefix):
        return float(get_str(in_file, prefix))
    def read_int(in_file, prefix):
        return int(get_str(in_file, prefix))
    def read_vector(in_file, prefix):
        return [float(t) for t in get_str(in_file, prefix).split()]
    
    spline_data = {}
    with open(filename, 'r') as f:
        spline_data['version'] = read_float(f, 'version')
        spline_data['degree']  = read_int(f, 'degree')
        spline_data['knots']   = read_vector(f, 'knots')
        spline_data['x']       = read_vector(f, 'x')
        spline_data['y']       = read_vector(f, 'y')
        spline_data['t0']      = read_float(f, 't0')
        spline_data['t1']      = read_float(f, 't1')
    spline_data['cs'] = np.array(zip(spline_data['x'], spline_data['y']))
    return spline_data

def evaluate_spline(spline_data, ts):
    cs     = spline_data['cs']
    num_cs = cs.shape[0]
    points = []
    degree = spline_data['degree']
    knots  = spline_data['knots']
    for t in ts:
        point = np.zeros_like(cs[0,:])
        for j in range(num_cs):
            point += bsplines.B(j, degree, t, knots)*cs[j,:]
        points.append(point)
    return np.array(points)

def load_spline_points(spline_file, parameter_values):
    return evaluate_spline(load_spline(spline_file), parameter_values)
    
