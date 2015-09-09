import h5py
from scipy.interpolate import interp1d
import numpy as np
    
def load_scale_function(h5_file):
    """
    Constructs a one-variable function object from a
    Hdf5 containing sampled data for a scaling signal.
    
    Returns t_min, t_max, scale_fn
    
    t_min:      Start time
    t_max:      End time
    scale_fn:   Scaling function taking time as input
    """
    with h5py.File(h5_file, 'r') as f:
        times   = f["times"].value
        factors = f["factors"].value
        
    from scipy.interpolate import interp1d
    scale_fn = interp1d(times, factors)
        
    t_min = np.min(times)
    t_max = np.max(times)
    
    return t_min, t_max, scale_fn
    