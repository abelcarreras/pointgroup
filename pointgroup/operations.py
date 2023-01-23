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


def reflection(normal):
    n = normal / np.linalg.norm(normal)

    ref = np.zeros((3, 3))

    # Reflexion matrix
    ref[0, 0] = 1 - 2 * n[0] ** 2
    ref[0, 1] = 2 * n[0] * n[1]
    ref[0, 2] = 2 * n[0] * n[2]
    ref[1, 0] = ref[0, 1]
    ref[1, 1] = 1 - 2 * n[1] ** 2
    ref[1, 2] = 2 * n[1] * n[2]
    ref[2, 0] = ref[0, 2]
    ref[2, 1] = ref[1, 2]
    ref[2, 2] = 1 - 2 * n[2] ** 2

    return ref
