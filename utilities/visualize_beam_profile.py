import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

description="""
    Plot simulated beam profile.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file")
    args = parser.parse_args()
    
    with h5py.File(args.h5_file, "r") as f:
        beam_profile = f["beam_profile"].value
        ele_extent = f["ele_extent"].value
        lat_extent = f["lat_extent"].value
        rad_extent = f["rad_extent"].value
    
    print "Radial extent: %s" % str(rad_extent)
    print "Lateral extent: %s" % str(lat_extent)
    print "Elevational extent: %s" % str(ele_extent)        
    
    print "Min sample value: %e" % np.min(beam_profile.flatten())
    print "Max sample value: %e" % np.max(beam_profile.flatten())
    
    rad_dim = 0
    num_rad_samples = beam_profile.shape[rad_dim]
    
    plt.figure()
    plt.ion()
    plt.show()
    img_extent = [ele_extent[0], ele_extent[1], lat_extent[0], lat_extent[1]] # riktig?
    for r_idx in range(num_rad_samples):
        plt.clf()
        plt.imshow(beam_profile[r_idx,:,:], extent=img_extent, cmap="Greys", aspect="auto", interpolation="nearest")
        plt.draw()
        plt.title("Radial index %d of %d" % (r_idx+1, num_rad_samples))
        plt.colorbar()
        raw_input("Press enter")
    