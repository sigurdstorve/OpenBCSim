import numpy as np
import h5py
import argparse

description="""
    Script for making dummy data to test the replay
    functionality in the GUI app.
    
    Store real- and imaginary part of IQ data in
    separate arrays since it is easier to read back
    from C++.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_file", help="Name of output hdf5 file")
    parser.add_argument("--num_lines", help="Number of image IQ lines", type=int, default=128)
    parser.add_argument("--num_samples", help="Number of radial samples", type=int, default=600)
    parser.add_argument("--num_frames", help="Number of frames to simulate", type=int, default=1)
    args = parser.parse_args()

    with h5py.File(args.h5_file, "w") as f:
        low = -1.0
        high = 1.0
        if args.num_frames == 1:
            size = (args.num_samples, args.num_lines)
            f["sim_data_real"] = np.array(np.random.uniform(low=low, high=high, size=size), dtype="float32")
            f["sim_data_imag"] = np.array(np.random.uniform(low=low, high=high, size=size), dtype="float32")
        else:
            size = (args.num_frames, args.num_samples, args.num_lines)
            f["sim_data_real"] = np.array(np.random.uniform(low=low, high=high, size=size), dtype="float32")
            f["sim_data_imag"] = np.array(np.random.uniform(low=low, high=high, size=size), dtype="float32")
            