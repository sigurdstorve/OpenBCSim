import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

description="""
    Create a few point scatterers along the positive z-axis.
    This is useful to test lookup-table beam profiles.
"""

def create_phantom(args):
    data = np.ones((args.num_scatterers,4), dtype="float32")
    data[:,0] = np.zeros((args.num_scatterers,))
    data[:,1] = np.zeros((args.num_scatterers,))
    data[:,2] = np.linspace(args.z0, args.z1, args.num_scatterers)
    
    with h5py.File(args.h5_file, "w") as f:
        f["data"] = data
    print "Dataset written to %s" % args.h5_file
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Name of scatterer hdf5 file")
    parser.add_argument("--num_scatterers", type=int, default=12)
    parser.add_argument("--z0", type=float, default=0.0)
    parser.add_argument("--z1", type=float, default=0.12)
    args = parser.parse_args()
    
    create_phantom(args)
    