import numpy as np


def inversion():
    inv_matrix = -np.identity(3)
    return inv_matrix


def rotation_matrix(axis, angle):

    axis = np.array(axis) / np.linalg.norm(axis)  # normalize axis

    angle = np.deg2rad(angle)

    cos_term = 1 - np.cos(angle)
    rot_matrix = [[axis[0]**2*cos_term + np.cos(angle),              axis[0]*axis[1]*cos_term - axis[2]*np.sin(angle), axis[0]*axis[2]*cos_term + axis[1]*np.sin(angle)],
                  [axis[1]*axis[0]*cos_term + axis[2]*np.sin(angle), axis[1]**2*cos_term + np.cos(angle),              axis[1]*axis[2]*cos_term - axis[0]*np.sin(angle)],
                  [axis[2]*axis[0]*cos_term - axis[1]*np.sin(angle), axis[1]*axis[2]*cos_term + axis[0]*np.sin(angle), axis[2]**2*cos_term + np.cos(angle)]]

    return rot_matrix


def reflection(reflection_axis):
    uax = np.dot(reflection_axis, reflection_axis)

    return np.identity(3) - 2*np.outer(reflection_axis, reflection_axis)/uax

