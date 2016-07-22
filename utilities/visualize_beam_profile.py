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
    parser.add_argument("--pdf_dpi", help="DPI for image in PDF", type=float, default=300.0)
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
    
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    plt.ion()
    plt.show()
    
    if args.db_scale:
        vmin = -args.dyn_range
        vmax = 0.0
        cb_ticks = np.linspace(0.0, -args.dyn_range, 5)
        trans_function = lambda x: 20.0*np.log10(x)
    else:
        vmin = 0.0
        vmax = 1.0
        cb_ticks = np.linspace(0.0, 1.0, 5)
        trans_function = lambda x: x
        
    # radial-lateral [assumes symmetric lateral/elevational extents]
    ele_dim = num_ele_samples/2
    rad_lat_slice = beam_profile[:, :, ele_dim]
    im0 = ax0.imshow(trans_function(rad_lat_slice.transpose()), extent=[rad_extent[0]*1000, rad_extent[1]*1000, lat_extent[0]*1000, lat_extent[1]*1000],
                     interpolation=args.interp, cmap=args.cmap, vmin=vmin, vmax=vmax)
    ax0.set_ylabel("Lateral [mm]")
    ax0.set_aspect("equal")
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.yaxis.set_ticks_position('left')
    ax0.xaxis.set_ticks_position('bottom')
    if args.rad_extent != None: ax0.set_xlim(*args.rad_extent)
    if args.lat_extent != None: ax0.set_ylim(*args.lat_extent)
    
    # radial-elevational [assumes symmetric lateral/elevational extents]
    lat_dim = num_lat_samples/2
    rad_ele_slice = beam_profile[:, lat_dim, :]
    im1 = ax1.imshow(trans_function(rad_ele_slice.transpose()), extent=[rad_extent[0]*1000, rad_extent[1]*1000, ele_extent[0]*1000, ele_extent[1]*1000],
                     interpolation=args.interp, cmap=args.cmap, vmin=vmin, vmax=vmax)
    ax1.set_ylabel("Elevational [mm]")
    ax1.set_aspect("equal")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    if args.rad_extent != None: plt.xlim(*args.rad_extent)
    if args.ele_extent != None: plt.ylim(*args.ele_extent)

    fig.subplots_adjust(hspace=0.5)
    plt.suptitle("Pulse-echo sensitivity")
    ax1.set_xlabel("Radial distance [mm]")
    fig.subplots_adjust(right=0.93, top=0.88)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im1, format=FormatStrFormatter("%2.0f"), cax=cbar_ax, ticks=cb_ticks)
    cb.set_clim(-args.dyn_range, 0.0)    
    cb.draw_all() # needed for some reason
    if args.db_scale:
        cb.set_label("dB")
    cb.outline.set_visible(False)

    if args.save_pdf:
        plt.savefig("beam_profile_pulse_echo.pdf", bbox_inches="tight", pad_inches=0.01, dpi=args.pdf_dpi)
    
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
