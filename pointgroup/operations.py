import numpy as np
from pointgroup.tools import magic_formula, rotation_matrix


class Inversion:

    def get_matrix(self):
        return -np.identity(3)


class Rotation:
    def __init__(self, axis, order=1):
        self._order = order

        self._axis = np.array(axis)

    def get_matrix(self):

        return rotation_matrix(self._axis, 2*np.pi / self._order)


class ImproperRotation:
    def __init__(self, axis, order=1):
        self._order = order

        self._axis = np.array(axis)

    def get_matrix(self):

        rot_matrix = rotation_matrix(self._axis, 2*np.pi / self._order)

        uax = np.dot(self._axis, self._axis)
        refl_matrix = np.identity(3) - 2 * np.outer(self._axis, self._axis) / uax

        return np.dot(rot_matrix, refl_matrix.T)


class Reflection:
    def __init__(self, axis):

        norm = np.linalg.norm(axis)
        assert abs(norm) > 1e-8
        self._axis = np.array(axis) / norm   # normalize axis

    def get_matrix(self):
        uax = np.dot(self._axis, self._axis)

        return np.identity(3) - 2*np.outer(self._axis, self._axis)/uax

