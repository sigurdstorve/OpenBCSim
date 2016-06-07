import argparse
import os

description="""
    Quick-and-dirty way of simulating a beam profile by using an arbitrary
    number of processed. The principle of operation is to divide the discrete
    simulation grid into N equal sizes (outer index). 
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("h5_base", help="Base part of HDF5 files (or final output if only one CPU) [without .h5]")
    parser.add_argument("matlab_exe", help="Path to Matlab executable")
    parser.add_argument("matlab_script", help="Name of .m simulation script")
    parser.add_argument("--x_min", type=float, default=-2e-2)
    parser.add_argument("--x_max", type=float, default=2e-2)
    parser.add_argument("--num_x", type=int, default=8)
    parser.add_argument("--y_min", type=float, default=-2e-2)
    parser.add_argument("--y_max", type=float, default=2e-2)
    parser.add_argument("--num_y", type=int, default=8)
    parser.add_argument("--z_min", type=float, default=1e-3)
    parser.add_argument("--z_max", type=float, default=160e-3)
    parser.add_argument("--num_z", type=float, default=32)
    parser.add_argument("--num_processes", help="Number of processes to split (=num CPUs to use)", type=int, default=1)
    args = parser.parse_args()
    
    # create Matlab code for creating geometry struct.
    struct_str = "struct('x_min',%e,'x_max',%e,'num_x',%d,'y_min',%e,'y_max',%e,'num_y',%d,'z_min',%e,'z_max',%e,'num_z',%d)"\
                % (args.x_min,args.x_max,args.num_x,args.y_min,args.y_max,args.num_y,args.z_min,args.z_max,args.num_z)

    for job_no in range(args.num_processes):
        cur_h5_out = "%s_%d_of_%d.h5" % (args.h5_base, job_no+1, args.num_processes)
        cmd = "%s -nosplash -nodesktop -r \"%s(%s, '%s', [%d,%d]);exit;\" " % (args.matlab_exe, args.matlab_script, struct_str, cur_h5_out, args.num_processes, job_no+1)
        print "=== JOB %d ===" % job_no
        print cmd
        os.system(cmd)
        