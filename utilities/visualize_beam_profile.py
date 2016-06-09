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
    parser.add_argument("--normalize", action="store_true", help="normalize in [0, 1]")
    parser.add_argument("--interp", default="nearest", help="Matplotlib interpolation type")
    parser.add_argument("--dyn_range", help="Dynamic range for plotting", type=int, default=70)
    parser.add_argument("--ele_extent", help="Override elevational extent for plotting", type=float, nargs="+")
    parser.add_argument("--lat_extent", help="Override lateral extent for plotting", type=float, nargs="+")
    parser.add_argument("--rad_extent", help="Override radial extent for plotting", type=float, nargs="+")
    args = parser.parse_args()
    
    with h5py.File(args.h5_file, "r") as f:
        beam_profile = f["beam_profile"].value
        ele_extent = f["ele_extent"].value
        lat_extent = f["lat_extent"].value
        rad_extent = f["rad_extent"].value
    
    if args.ele_extent != None:
        assert len(args.ele_extent) == 2
        ele_extent = args.ele_extent
    if args.lat_extent != None:
        assert len(args.lat_extent) == 2
        lat_extent = args.lat_extent
    if args.rad_extent != None:
        assert len(args.rad_extent) == 2
        rad_extent = args.rad_extent
    
    print "Radial extent: %s m" % str(rad_extent)
    print "Lateral extent: %s m" % str(lat_extent)
    print "Elevational extent: %s m" % str(ele_extent)        
    
    min_value = np.min(beam_profile.flatten())
    max_value = np.max(beam_profile.flatten())
    print "Min sample value: %e" % min_value
    print "Max sample value: %e" % max_value
    
    rad_dim = 0
    lat_dim = 1
    ele_dim = 2
    num_rad_samples = beam_profile.shape[rad_dim]
    num_lat_samples = beam_profile.shape[lat_dim]
    num_ele_samples = beam_profile.shape[ele_dim]
    
    if args.normalize:
        beam_profile = (beam_profile-min_value)/(max_value-min_value)
        
    plt.figure()
    plt.ion()
    plt.show()
    
    # radial-lateral [assumes symmetric lateral/elevational extents]
    ele_dim = num_ele_samples/2
    rad_lat_slice = beam_profile[:, :, ele_dim]
    plt.imshow(20*np.log10(rad_lat_slice.transpose()), extent=[rad_extent[0]*1000, rad_extent[1]*1000, lat_extent[0]*1000, lat_extent[1]*1000], interpolation=args.interp)
    plt.title("Radial-Lateral")
    plt.xlabel("millimeters")
    plt.ylabel("millimeters")
    plt.clim(0.0, -args.dyn_range)
    plt.colorbar()
    plt.gca().set_aspect("equal")
    
    # radial-elevational [assumes symmetric lateral/elevational extents]
    plt.figure()
    lat_dim = num_lat_samples/2
    rad_ele_slice = beam_profile[:, lat_dim, :]
    plt.imshow(20*np.log10(rad_ele_slice.transpose()), extent=[rad_extent[0]*1000, rad_extent[1]*1000, ele_extent[0]*1000, ele_extent[1]*1000], interpolation=args.interp)
    plt.title("Radial-Elevational")
    plt.xlabel("millimeters")
    plt.ylabel("millimeters")
    plt.clim(0.0, -args.dyn_range)
    plt.colorbar()
    plt.gca().set_aspect("equal")

    plt.figure()
    # short-axis plots for all radial distances
    img_extent = [ele_extent[0]*1000, ele_extent[1]*1000, lat_extent[0]*1000, lat_extent[1]*1000]
    for r_idx in range(num_rad_samples):
        plt.clf()
        plt.imshow(beam_profile[r_idx,:,:], extent=img_extent, cmap="Greys_r", aspect="auto", interpolation=args.interp)
        plt.draw()
        plt.title("Lat-Ele: Radial index %d of %d" % (r_idx+1, num_rad_samples))
        if args.normalize:
            plt.clim(0.0, 1.0)
        plt.xlabel("millimeters")
        plt.ylabel("millimeters")
        plt.colorbar()
        plt.gca().set_aspect("equal")
        raw_input("Press enter")
