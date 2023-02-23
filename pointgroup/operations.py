import numpy as np


def inversion():
    inv_matrix = -np.identity(3)
    return inv_matrix


def rotation_matrix(axis, angle):
    a = angle * np.pi / 180
    cos = np.cos(a)
    sin = np.sin(a)
    u = axis / np.linalg.norm(axis)

    # Rotation matrix
    r = np.zeros((3, 3))
    r[0, 0] = cos + (u[0] ** 2) * (1 - cos)
    r[0, 1] = u[0] * u[1] * (1 - cos) - u[2] * sin
    r[0, 2] = u[0] * u[2] * (1 - cos) + u[1] * sin
    r[1, 0] = u[1] * u[0] * (1 - cos) + u[2] * sin
    r[1, 1] = cos + (u[1] ** 2) * (1 - cos)
    r[1, 2] = u[1] * u[2] * (1 - cos) - u[0] * sin
    r[2, 0] = u[2] * u[0] * (1 - cos) - u[1] * sin
    r[2, 1] = u[2] * u[1] * (1 - cos) + u[0] * sin
    r[2, 2] = cos + (u[2] ** 2) * (1 - cos)

    return r


def reflection(reflection_axis):
    uax = np.dot(reflection_axis, reflection_axis)

    return np.identity(3) - 2*np.outer(reflection_axis, reflection_axis)/uax

