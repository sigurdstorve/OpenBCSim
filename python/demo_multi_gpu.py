import threading
import time
from linear_scan_phantom import do_simulation
import argparse

class Dummy:
    pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file", help="Phantom to scan")
    parser.add_argument("num_devices", help="Number of GPUs to use", type=int)
    parser.add_argument("--save_pdf", action="store_true")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # spawn one thread for each GPU    
    threads = []
    for device_no in range(args.num_devices):
        # set up common parameters for each GPU
        params = Dummy()
        params.h5_file = args.h5_file
        params.x0 = -3e-2
        params.x1 = 3e-2
        params.num_lines = 512
        params.num_frames = 1
        params.visualize = False
        params.use_gpu = True
        params.save_simdata_file = ""
        params.noise_ampl = 0

        # specific parameters
        if args.save_pdf:
            params.pdf_file = "Res_GPU_device_%d.pdf" % device_no
        else:
            params.pdf_file = ""
        params.device_no = device_no

        t = threading.Thread(target=do_simulation, args=(params,))
        t.start()
        threads.append(t)

    print "Waiting for all threads to finish...",
    for thread in threads:
        thread.join()
    print "Done."
    end_time = time.time()
    elapsed_time = end_time-start_time
    print "Total time elapsed: %f sec." % elapsed_time
    print "Time per device: %f sec" % (elapsed_time/args.num_devices)
