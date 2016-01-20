import numpy as np
import argparse
import h5py

description="""
    Script for generating a gaussian lookup-table.
    
    It should yield similar results as the analytical
    beam profile.
"""

if __name__ == '__main__':
    sigma_lat = 1e-3
    sigma_ele = 1e-3
    num_rad_samples = 128
    num_lat_samples = 64
    num_ele_samples = 63
    r_min = 0.0; r_max = 0.12
    l_min = -3e-3; l_max = 3e-3
    e_min = -3e-3; e_max = 3e-3
    
    h5_file = 'gaussian_lut.h5'
    
    r_mesh, l_mesh, e_mesh = np.meshgrid(np.linspace(r_min, r_max, num_rad_samples),
                                         np.linspace(l_min, l_max, num_lat_samples),
                                         np.linspace(e_min, e_max, num_ele_samples),
                                         indexing='ij')
    
    beam_profile = np.exp( -(l_mesh**2/sigma_lat**2 + e_mesh**2/sigma_ele**2) )

    # probably a nicer way to do this...
    for rad_idx, r_norm in enumerate(np.linspace(0.0, 1.0, num_rad_samples)):
        w = 1.0/(2*r_norm+0.1)
        print 'Radial index: %d: w=%f' % (rad_idx, w)
        beam_profile[rad_idx, :, :] *= w
        
    print beam_profile[0,:,:]
    print beam_profile[-1,:,:]
    
    with h5py.File(h5_file, 'w') as f:
        f["beam_profile"] = np.array(beam_profile, dtype='float32')
        f["rad_extent"]   = np.array([r_min, r_max], dtype='float32')
        f["lat_extent"]   = np.array([l_min, l_max], dtype='float32')
        f["ele_extent"]   = np.array([e_min, e_max], dtype='float32')
        
    if True:
        import matplotlib.pyplot as plt
        rad_idx = num_rad_samples/2
        plt.imshow(beam_profile[rad_idx,:,:])
        plt.show()
    