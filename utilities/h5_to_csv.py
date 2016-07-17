import h5py
import argparse

description="""
    Convert arrays (of equal length) from HDF5 file to
    a CSV file.
    
    Datasets must have the same length.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_in", help="Input HDF5 file")
    parser.add_argument("csv_out", help="Output CSV file")
    parser.add_argument("dset_names", help="Name of datasets to copy from HDF5 file", nargs="+")
    parser.add_argument("--csv_delimiter", help="Delimiter symbol in CSV file", default=";")
    args = parser.parse_args()
    
    with h5py.File(args.h5_in) as f_in:
        with open(args.csv_out, "w") as f_out:
            for t in [args.dset_names,]+zip(*map(lambda dset_name: f_in[dset_name].value, args.dset_names)):
                f_out.write(args.csv_delimiter.join(["%s" % str(v) for v in t]) + "\n")
    