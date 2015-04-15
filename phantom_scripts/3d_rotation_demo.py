import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotation3d import general_rot_mat

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    p_original = np.empty( (3, 8) )
    p_original[:, 0] = [-1,-1,-1]
    p_original[:, 1] = [1,-1,-1]
    p_original[:, 2] = [1,1,-1]
    p_original[:, 3] = [-1,1,-1]
    p_original[:, 4] = [-1,-1,1]
    p_original[:, 5] = [1,-1,1]
    p_original[:, 6] = [1,1,1]
    p_original[:, 7] = [-1,1,1]
    
    rot_x = 0.0
    rot_y = 0.0
    rot_z = 0.0
    plt.ion()
    plt.show()
    dx = 0.02
    dy = 0.05
    dz = 0.1
    while True:
        ax.cla()    
        rot_mat = general_rot_mat(rot_x, rot_y, rot_z)
        p = np.dot(rot_mat, p_original)
        
        plt.plot(p[0,:], p[1,:], p[2,:], linestyle='-', marker='o', c='k')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        

        rot_x += dx
        rot_y += dy
        rot_z += dz

        plt.draw()
