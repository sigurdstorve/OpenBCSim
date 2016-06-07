import numpy as np
import h5py
import argparse

description="""
    Merge partial LUT simulations into a single, complete
    LUT with no NaN's.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_out", help="Output file")
    parser.add_argument("h5_in", nargs="+", help="Input files")
    args = parser.parse_args()
    
    # read all partial simulations
    beam_profiles = []
    for h5_in in args.h5_in:
        print "Processing %s..." % h5_in
        with h5py.File(h5_in, "r") as f:
            beam_profiles.append(f["beam_profile"].value)
            ele_extent = f["ele_extent"].value
            lat_extent = f["lat_extent"].value
            rad_extent = f["rad_extent"].value

    beam_profile = sum(beam_profiles)
    min_value = np.min(beam_profile.flatten())
    max_value = np.max(beam_profile.flatten())
    beam_profile = (beam_profile-min_value)/(max_value-min_value)
            
    with h5py.File(args.h5_out, "w") as f:
        f["beam_profile"] = beam_profile
        f["ele_extent"] = ele_extent
        f["lat_extent"] = lat_extent
        f["rad_extent"] = rad_extent
    
    