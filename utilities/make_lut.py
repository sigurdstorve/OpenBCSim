import numpy as np
import argparse
import h5py

description="""
    Script for generating a gaussian lookup-table.
    
    It should yield similar results as the analytical
    beam profile.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Output beam profile")
    parser.add_argument("--rad_min", type=float, default=0.0)
    parser.add_argument("--rad_max", type=float, default=0.15)
    parser.add_argument("--ele_min", type=float, default=-2e-2)
    parser.add_argument("--ele_max", type=float, default=2e-2)
    parser.add_argument("--lat_min", type=float, default=-2e-2)
    parser.add_argument("--lat_max", type=float, default=2e-2)
    parser.add_argument("--num_samples_rad", type=int, default=128)
    parser.add_argument("--num_samples_lat", type=int, default=32)
    parser.add_argument("--num_samples_ele", type=int, default=32)
    parser.add_argument("--r_min", help="Radius at start depth", type=float, default=5e-3)
    parser.add_argument("--r_max", help="Radius at end depth", type=float, default=13e-3)
    args = parser.parse_args()
    
    data_dims = (args.num_samples_rad, args.num_samples_lat, args.num_samples_ele)
    data = np.empty(data_dims, dtype="float32")
    for lat_i in range(args.num_samples_lat):
        print "%2.1f%% complete..." % (100.0*lat_i/args.num_samples_lat)
        for ele_i in range(args.num_samples_ele):
            for rad_i in range(args.num_samples_rad):
                x = args.lat_min + lat_i*(args.lat_max-args.lat_min)/(args.num_samples_lat-1)
                y = args.ele_min + ele_i*(args.ele_max-args.ele_min)/(args.num_samples_ele-1)
                r = args.r_min + rad_i*(args.r_max-args.r_min)/(args.num_samples_rad-1) # lat/ele radius
                data[rad_i, lat_i, ele_i] = np.exp(-(x**2+y**2)/r**2)
    
    with h5py.File(args.h5_file, 'w') as f:
        f["beam_profile"] = data
        f["rad_extent"]   = np.array([args.rad_min, args.rad_max], dtype="float32")
        f["lat_extent"]   = np.array([args.lat_min, args.lat_max], dtype="float32")
        f["ele_extent"]   = np.array([args.ele_min, args.ele_max], dtype="float32")
    print "Beam profile written to %s" % args.h5_file
    