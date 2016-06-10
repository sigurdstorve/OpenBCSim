import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import h5py

description="""
    Interactive development script for testing out
    various log-compression curves.
    Can load IQ-data from a HDF5 file and visualize
    input and output.
"""

def clamp(v, low=0.0, high=255.0):
    """ Clamp a scalar or array to [0.0, 255.0]. """
    res = v
    if isinstance(res, np.ndarray):
        res[res < low]  = low
        res[res > high] = high
    else:
        if res < low: res = low
        if res > high: res = high
    return res

# This defines the non-linear gain curve (compression curve)
# f:[0.0, 1.0] -> [0.0, 255.0] 
def formula(in_value, dyn_range, reject):
    db_value = 20.0*np.log10(in_value+1e-12)
    db_value = 255.0*(db_value - reject) / dyn_range
    return clamp(db_value)

def formula_OLD(in_value, dyn_range, reject):
    temp = 20.0*np.log10(reject*in_value)
    temp = (255.0/dyn_range)*(temp + dyn_range)
    return clamp(temp)
    
init_dyn_range = 40.0
init_reject = -30

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--h5_iq", help="Hdf5 file with IQ data", default=None)
    parser.add_argument("--interp_type", help="Interpolation type", default="bilinear")
    args = parser.parse_args()
    
    # input image data assumed to be normalized in [0.0, 1.0]
    if args.h5_iq == None:
        # generate random values
        img_data = np.random.uniform(low=0.0, high=1.0, size=((300, 300)))
    else:
        with h5py.File(args.h5_iq) as f_in:
            iq_data = f_in["iq_real"].value + 1.0J*f_in["iq_imag"].value
        frame_no = 0
        iq_frame = iq_data[frame_no, :, :]
        env_frame = np.real(abs(iq_frame))
        min_val = np.min(env_frame.flatten())
        max_val = np.max(env_frame.flatten())
        print "Min. value: %f" % min_val
        print "Max. value: %f" % max_val
        img_data = (env_frame-min_val) / (max_val-min_val)
        img_data = img_data.transpose()
        
    
    # (1) gain curve (2) input img (3) output img
    fig, [ax0, ax1, ax2] = plt.subplots(nrows=1, ncols=3)
    plt.subplots_adjust(bottom=0.25)

    xs = np.linspace(0.0, 1.0, 300)
    curve_in,  = ax0.plot(xs, 255*xs, lw=2.0, color="black")
    curve_out, = ax0.plot(xs, formula(xs, init_dyn_range, init_reject), lw=2.0, color="red")
    ax0.set_xlabel("Input value")
    ax1.set_ylabel("Output value")
    ax0.set_xlim(-0.05, 1.05)
    ax0.set_ylim(-4, 260)

    im_in  = ax1.imshow(img_data, aspect="auto", interpolation=args.interp_type, cmap="Greys_r", vmin=0.0, vmax=1.0)
    fig.colorbar(im_in, label="Input")
    im_out = ax2.imshow(formula(img_data, init_dyn_range, init_reject), aspect="auto", interpolation=args.interp_type, cmap="Greys_r", vmin=0.0, vmax=255.0) 
    fig.colorbar(im_out, label="Output")

    ax_dyn_range = plt.axes([0.25, 0.05, 0.65, 0.03])
    ax_reject    = plt.axes([0.25, 0.10, 0.65, 0.03])
        
    def refresh(dyn_range, reject):
        img_out = formula(img_data, dyn_range, reject)
        im_out.set_array(img_out)
        
        curve_out.set_ydata(formula(xs, dyn_range, reject))
        fig.canvas.draw_idle()
        
    slider_dyn_range = Slider(ax_dyn_range, "DynRange", 1.0, 150.0, valinit=init_dyn_range)
    slider_reject = Slider(ax_reject, "Reject", -120.0, 20.0, valinit=init_reject)

    # callbacks
    callback_fn = lambda value: refresh(slider_dyn_range.val, slider_reject.val)
    slider_dyn_range.on_changed(callback_fn)
    slider_reject.on_changed(callback_fn)

    fig.set_size_inches(18, 8, forward=True)
    plt.show()
    