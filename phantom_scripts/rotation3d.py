import numpy as np

def rot_mat_x(t):
    """ Rotation of t radians around x-axis. """
    return np.array([[1.0, 0.0, 0.0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t), np.cos(t)]])
def rot_mat_y(t):
    """ Rotation of t radians around y-axis. """
    return np.array([[np.cos(t), 0.0, np.sin(t)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(t), 0.0, np.cos(t)]])

def rot_mat_z(t):
    """ Rotation of t radians around z-axis. """
    return np.array([[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t), np.cos(t), 0.0],
                     [0.0, 0.0, 1.0]])

def general_rot_mat(alpha, beta, gamma):
    r = rot_mat_x(alpha).dot(rot_mat_y(beta)).dot(rot_mat_z(gamma))
    return r

