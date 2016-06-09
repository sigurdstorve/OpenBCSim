import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
from matplotlib.ticker import FormatStrFormatter

description="""
    Plot simulated beam profile.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file")
    parser.add_argument("--normalize", action="store_true", help="normalize in [0, 1]")
    parser.add_argument("--interp", default="nearest", help="Matplotlib interpolation type")
    parser.add_argument("--dyn_range", help="Dynamic range for plotting", type=int, default=70)
    parser.add_argument("--ele_extent", help="Elevational extent to show in plot [mm]", type=float, nargs="+")
    parser.add_argument("--lat_extent", help="Lateral extent to show in plot [mm]", type=float, nargs="+")
    parser.add_argument("--rad_extent", help="Radial extent to show in plot [mm]", type=float, nargs="+")
    parser.add_argument("--cmap", help="Specify custom colormap name", default="Greys_r")
    parser.add_argument("--db_scale", help="Plot in decibels", action="store_true")
    parser.add_argument("--save_pdf", help="Save PDF figures", action="store_true")
    args = parser.parse_args()
    
    with h5py.File(args.h5_file, "r") as f:
        beam_profile = f["beam_profile"].value
        ele_extent = f["ele_extent"].value
        lat_extent = f["lat_extent"].value
        rad_extent = f["rad_extent"].value
    
    # sanitize plotting extents (if any)
    if args.ele_extent != None: assert len(args.ele_extent) == 2
    if args.lat_extent != None: assert len(args.lat_extent) == 2
    if args.rad_extent != None: assert len(args.rad_extent) == 2

    print "Dataset geometry:"
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
    
    db_ticks = np.linspace(0.0, -args.dyn_range, 5)
    if args.db_scale:
        trans_function = lambda x: 20.0*np.log10(x)
    else:
        trans_function = lambda x: x
        
    # radial-lateral [assumes symmetric lateral/elevational extents]
    ele_dim = num_ele_samples/2
    rad_lat_slice = beam_profile[:, :, ele_dim]
    plt.imshow(trans_function(rad_lat_slice.transpose()), extent=[rad_extent[0]*1000, rad_extent[1]*1000, lat_extent[0]*1000, lat_extent[1]*1000], interpolation=args.interp, cmap=args.cmap)
    plt.title("Radial-Lateral")
    plt.xlabel("millimeters")
    plt.ylabel("millimeters")
    cb = plt.colorbar(format=FormatStrFormatter("%2.0f"))
    if args.db_scale:
        plt.clim(0.0, -args.dyn_range)
        cb.set_label("dB")
        cb.set_ticks(db_ticks)
    else:
        plt.clim(0.0, 1.0)
    plt.gca().set_aspect("equal")
    if args.rad_extent != None: plt.xlim(*args.rad_extent)
    if args.lat_extent != None: plt.ylim(*args.lat_extent)
    if args.save_pdf:
        plt.savefig("beam_profile_rad_lat.pdf", bbox_inches="tight", pad_inches=0.1, dpi=300)
    
    # radial-elevational [assumes symmetric lateral/elevational extents]
    plt.figure()
    lat_dim = num_lat_samples/2
    rad_ele_slice = beam_profile[:, lat_dim, :]
    plt.imshow(trans_function(rad_ele_slice.transpose()), extent=[rad_extent[0]*1000, rad_extent[1]*1000, ele_extent[0]*1000, ele_extent[1]*1000], interpolation=args.interp, cmap=args.cmap)
    plt.title("Radial-Elevational")
    plt.xlabel("millimeters")
    plt.ylabel("millimeters")
    cb = plt.colorbar(format=FormatStrFormatter("%2.0f"))
    if args.db_scale:
        plt.clim(0.0, -args.dyn_range)
        cb.set_label("dB")
        cb.set_ticks(db_ticks)
    else:
        plt.clim(0.0, 1.0)
    plt.gca().set_aspect("equal")
    if args.rad_extent != None: plt.xlim(*args.rad_extent)
    if args.ele_extent != None: plt.ylim(*args.ele_extent)
    if args.save_pdf:
        plt.savefig("beam_profile_rad_ele.pdf", bbox_inches="tight", pad_inches=0.1, dpi=300)
    
    plt.figure()
    # short-axis plots for all radial distances
    img_extent = [ele_extent[0]*1000, ele_extent[1]*1000, lat_extent[0]*1000, lat_extent[1]*1000]
    for r_idx in range(num_rad_samples):
        plt.clf()
        plt.imshow(trans_function(beam_profile[r_idx,:,:]), extent=img_extent, aspect="auto", interpolation=args.interp, cmap=args.cmap)
        plt.draw()
        plt.title("Lat-Ele: Radial index %d of %d" % (r_idx+1, num_rad_samples))
        if args.normalize:
            plt.clim(0.0, 1.0)
        plt.xlabel("Elevation [millimeters]")
        plt.ylabel("Lateral [millimeters]")
        if args.db_scale:
            plt.clim(0.0, -args.dyn_range)
        else:
            plt.clim(0.0, 1.0)
        plt.colorbar()
        plt.gca().set_aspect("equal")
        if args.ele_extent != None: plt.xlim(*args.ele_extent)
        if args.lat_extent != None: plt.ylim(*args.lat_extent)
        raw_input("Press enter")
