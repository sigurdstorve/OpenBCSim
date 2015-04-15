import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

# Create Cylinder phantom parallel to the z-axis with one of its circular faces
# in origin which is compressed harmonically.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', help='Frames per second', default=120, type=int)
    parser.add_argument('h5_out', help='Name of hdf5 file with scatterers data')
    parser.add_argument('--r0', help='Radius of cylinder [m]', type=float, default=1e-2)
    parser.add_argument('--num_scatterers', help='Num scatterers in initial box', type=int, default=10000)
    args = parser.parse_args()

    # Radius of cylinder
    #r0 = 30e-3
    r0 = args.r0
    # Depth of cylinder
    z0 = 0.12
    num_scatterers = args.num_scatterers

    # Generate scatterers in a box
    xs = np.random.uniform(low=-r0, high=r0, size=(num_scatterers,))
    ys = np.random.uniform(low=-r0, high=r0, size=(num_scatterers,))
    zs = np.random.uniform(low=0.0, high=z0, size=(num_scatterers,))
    pts = zip(xs, ys, zs)

    # Discard scatterers outside of cylinder
    pts = filter(lambda (x,y,z): x**2+y**2 <= r0**2, pts)
    xs, ys, zs = map(np.array, zip(*pts))
    num_scatterers = len(xs)
    
    # Create random amplitudes
    _as = np.random.uniform(low=0.0, high=1.0, size=(num_scatterers,))  
    
    # The harmonic compression motion (alpha)
    dt = 1.0/args.fps
    motion_f = 0.8  # [Hz]
    motion_a = 0.1  # How much shortening/stretching
    timestamps = np.arange(0.0, 1.0/motion_f, dt)
    alphas = 1 + motion_a*np.cos(2*np.pi*motion_f*timestamps)

    h5_out = h5py.File(args.h5_out, 'w')
    num_steps = len(timestamps)
    scatterers_data = np.empty((num_scatterers, 4, num_steps), dtype='float32')
    for step_no,alpha in enumerate(alphas):
        print 'alpha = %f (%d of %d) ...' % (alpha, step_no, len(alphas))
        
        # Compute updated scatterer positions (old slow)
        #rs = np.sqrt(xs**2 + ys**2)
        #thetas = np.arctan2(ys, xs)
        #rs_trans = rs/np.sqrt(alpha)
        #xs_trans = rs_trans*np.cos(thetas)
        #ys_trans = rs_trans*np.sin(thetas)
        #zs_trans = zs*alpha
        
        # Store positions for this timestep
        #scatterers_data[:,0,step_no] = xs_trans
        #scatterers_data[:,1,step_no] = ys_trans
        #scatterers_data[:,2,step_no] = zs_trans
        #scatterers_data[:,3,step_no] = _as 

        scatterers_data[:,0,step_no] = xs/np.sqrt(alpha)
        scatterers_data[:,1,step_no] = ys/np.sqrt(alpha)
        scatterers_data[:,2,step_no] = zs*alpha
        scatterers_data[:,3,step_no] = _as 
    #_scatterers_data = h5_out.create_dataset("scatterers", (num_scatterers, 4, num_steps), dtype='float32')
    h5_out["scatterers"] = scatterers_data
    
    print 'Number of frames: %d' % len(alphas)
    h5_out["timestamps"] = timestamps
    h5_out.close()
    print 'Simulated scatterers written to %s' % args.h5_out
