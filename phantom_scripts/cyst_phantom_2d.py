import numpy as np
import h5py
import argparse

description="""
    Create a phantom to replicate the running time
    experiment in the COLE-paper by Gao et al.
    This script creates a fixed-scatterers dataset.
    Amplitudes are uniformly distributed in [1, 1].
"""

def create_phantom(args):
    x_min = -34.5e-3
    x_max = 34.5e-3
    z_min = 20e-3
    z_max = 90e-3
    area = (x_max-x_min)*(z_max-z_min)
    num_scatterers = int(args.density*area*1e6)
    print 'Number of scatterers is %d' % num_scatterers
    
    xs = np.random.uniform(low=x_min, high=x_max, size=(num_scatterers,))
    ys = np.zeros((num_scatterers,))
    zs = np.random.uniform(low=z_min, high=z_max, size=(num_scatterers,))
    ampls = np.random.uniform(low=-1.0, high=1.0, size=(num_scatterers,))
    
    # scale the amplitudes in the three cystic regions
    def make_cystic_region(z0, cyst_radius):
        ampls[xs**2 + ys**2 + (zs-z0)**2 <= cyst_radius**2] *= args.cyst_scale
    for params in [(40e-3, 10e-3), (60e-3, 5e-3), (80e-3, 2.5e-3)]:
        make_cystic_region(*params)
    # same cystic amplitude outside of x=[-10mm,10mm]
    ampls[xs < -10e-3] *= args.cyst_scale
    ampls[xs >  10e-3] *= args.cyst_scale
    
    with h5py.File(args.h5_file, 'w') as f:
        f["data"] = np.empty((num_scatterers, 4), dtype='float32')
        f["data"][:, 0] = xs
        f["data"][:, 1] = ys
        f["data"][:, 2] = zs
        f["data"][:, 3] = ampls
    print 'Scatterers written to %s' % args.h5_file
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('h5_file', help='Output file')
    parser.add_argument('--density', help='Scatterers pr. mm^2', type=float, default=500.0)
    parser.add_argument('--cyst_scale', help='Cyst amplitudes scale factor', type=float, default=0.3)
    args = parser.parse_args()

    create_phantom(args)