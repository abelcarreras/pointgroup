__version__ = '0.1'

import itertools
import numpy as np
from pointgroup.operations import inversion, rotation_matrix, reflection
from pointgroup import tools


class PointGroup:
    """
    Point group main class
    """

    def __init__(self, positions, symbols, tolerance_eig=0.01, tolerance_op=0.02):
        self._tolerance_eig = tolerance_eig
        self._tolerance_op = tolerance_op
        self._symbols = symbols
        self._max_order = 0
        self._cent_coord = np.array(positions) - tools.get_center_mass(self._symbols, np.array(positions))

        # determine inertia tensor
        inertia_tensor = tools.get_inertia_tensor(self._symbols, self._cent_coord)
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

        self._cent_coord = np.dot(self._cent_coord, eigenvectors)
        eigenvectors = np.dot(eigenvectors, eigenvectors.T)

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors.T

        self.schoenflies_symbol = ''
        val = self._eigenvalues
        if any([i < self._tolerance_eig for i in val]):
            # Linear groups
            self._lineal()
        elif (abs(val[0] - val[1]) < self._tolerance_eig and
              abs(val[0] - val[2]) < self._tolerance_eig):
            # Spherical group
            self._spherical()
        elif (abs(val[0] - val[1]) > self._tolerance_eig and
              abs(val[1] - val[2]) > self._tolerance_eig and
              abs(val[0] - val[2]) > self._tolerance_eig):
            # Asymmetric group
            self._asymmetric()
        else:
            # Symmetric group
            self._symmetric()

    def get_point_group(self):
        """
        get the point symmetry group symbol

        :return: the point symmetry group symbol
        """
        return self.schoenflies_symbol

    def get_standard_coordinates(self):
        """
        get the coordinates centered in the center of mass and oriented along principal axis of inertia

        :return: the coordinates
        """
        return self._cent_coord.tolist()

    def get_principal_axis_of_inertia(self):
        """
        get the principal axis of inertia in rows in increasing order of momenta of inertia

        :return: the principal axis of inertia
        """
        return self._eigenvectors.tolist()

    def get_principal_moments_of_inertia(self):
        """
        get the principal momenta of inertia in increasing order

        :return: list of momenta of inertia
        """
        return self._eigenvalues.tolist()

    # internal methods
    def _lineal(self):

        # set orientation
        idx = np.argmin(self._eigenvalues)
        main_axis = self._eigenvectors[idx]
        p_axis = tools.get_perpendicular(main_axis)
        self._set_orientation(main_axis, p_axis)

        if self._check_op(inversion()):
            self.schoenflies_symbol = 'Dinfh'
        else:
            self.schoenflies_symbol = 'Cinfv'

    def _asymmetric(self):

        n_c2 = 0
        main_axis = self._eigenvectors[0]
        for axis in self._eigenvectors:
            c2 = rotation_matrix(axis, 180)
            if self._check_op(c2):
                n_c2 += 1
                main_axis = axis

        self._max_order = 2

        if n_c2 == 0:
            self._no_rot_axis()
        elif n_c2 == 1:
            self._cyclic(main_axis, 2)
        else:
            self._dihedral(main_axis, 2)

    def _symmetric(self):

        if abs(self._eigenvalues[0] - self._eigenvalues[1]) < self._tolerance_eig:
            idx = 2
        elif abs(self._eigenvalues[0] - self._eigenvalues[2]) < self._tolerance_eig:
            idx = 1
        else:
            idx = 0

        main_axis = self._eigenvectors[idx, :]

        n_order = self._get_axis_rot_order(main_axis, n_max=9)
        self._max_order = n_order

        p_axis = tools.get_perpendicular(main_axis)
        for angle in np.arange(0, 180, 2):
            axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
            c2 = rotation_matrix(axis, 360 / 2)
            if self._check_op(c2):
                self._dihedral(main_axis, n_order)
                self._set_orientation(main_axis, axis)
                return

        self._cyclic(main_axis, n_order)

    def _spherical(self):
        """
        Handle spherical groups (T, O, I)

        :return:
        """

        def get_axis_list(step=2):
            for ev in np.identity(3):
                yield ev
            for yaw in np.arange(0, 180, step):
                for pitch in np.arange(0, 180, step):
                    x = np.cos(yaw) * np.cos(pitch)
                    y = np.sin(yaw) * np.cos(pitch)
                    z = np.sin(pitch)
                    yield [x, y, z]

        main_axis = None
        for axis in get_axis_list():
            c5 = rotation_matrix(axis, 360 / 5)
            c4 = rotation_matrix(axis, 360 / 4)
            c3 = rotation_matrix(axis, 360 / 3)

            if self._check_op(c5):
                self.schoenflies_symbol = "I"
                main_axis = axis
                self._max_order = 5
                break
            elif self._check_op(c4):
                self.schoenflies_symbol = "O"
                main_axis = axis
                self._max_order = 4
                break
            elif self._check_op(c3):
                self.schoenflies_symbol = "T"
                main_axis = axis
                self._max_order = 3

        if main_axis is None:
            raise Exception('Error in spherical group')

        # I or Ih
        if self.schoenflies_symbol == 'I':
            if self._check_op(inversion()):
                self.schoenflies_symbol += 'h'

            # set molecule orientation in I
            p_axis = tools.get_perpendicular(main_axis)
            for angle in np.arange(0, 180, 2):
                axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
                c2 = rotation_matrix(axis, 360 / 2)
                if self._check_op(c2):
                    p_axis = axis
                    break

            self._set_orientation(main_axis, p_axis)

        # O or Oh
        if self.schoenflies_symbol == 'O':
            if self._check_op(inversion()):
                self.schoenflies_symbol += 'h'

            # set molecule orientation in O
            p_axis = tools.get_perpendicular(main_axis)
            for angle in np.arange(0, 90, 2):
                axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
                c4 = rotation_matrix(axis, 360 / 4)
                if self._check_op(c4):
                    p_axis = axis
                    break

            self._set_orientation(main_axis, p_axis)

        # T or Td, Th
        if self.schoenflies_symbol == 'T':
            p_axis = tools.get_perpendicular(main_axis)

            # set molecule orientation in T
            def determine_orientation():
                t_axis = np.dot(rotation_matrix(p_axis, np.rad2deg(np.arccos(-1/3))/2), main_axis)
                for angle in np.arange(0, 180, 2):
                    axis = np.dot(t_axis, rotation_matrix(main_axis, angle))

                    c2 = rotation_matrix(axis, 360 / 2)
                    if self._check_op(c2):
                        t_axis = np.dot(rotation_matrix(p_axis, 90), main_axis)
                        axis = np.dot(t_axis, rotation_matrix(main_axis, angle))
                        self._set_orientation(main_axis, axis)
                        return

                raise Exception('Error orientation T')

            determine_orientation()

            if self._check_op(inversion()):
                self.schoenflies_symbol += 'h'

            t_axis = np.dot(rotation_matrix(p_axis, 90), main_axis)
            for angle in np.arange(0, 180, 2):
                axis = np.dot(t_axis, rotation_matrix(main_axis, angle))

                if self._check_op(reflection(axis)):
                    self.schoenflies_symbol += 'd'
                    return

    def _no_rot_axis(self):
        for i, vector in enumerate(self._eigenvectors):
            if self._check_op(reflection(vector)):
                self.schoenflies_symbol = 'Cs'
                p_axis = tools.get_perpendicular(vector)
                self._set_orientation(vector, p_axis)
                break
            else:
                self._set_orientation(self._eigenvectors[0], self._eigenvectors[1])
                if self._check_op(inversion()):
                    self.schoenflies_symbol = 'Ci'
                else:
                    self.schoenflies_symbol = 'C1'

    def _cyclic(self, main_axis=0, order=0):

        self.schoenflies_symbol = "C{}".format(order)

        if self._check_op(reflection(main_axis)):
            self.schoenflies_symbol += 'h'
            return

        p_axis = tools.get_perpendicular(main_axis)
        for angle in np.arange(0, 180, 2):
            axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
            if self._check_op(reflection(axis)):
                self.schoenflies_symbol += 'v'
                self._set_orientation(main_axis, axis)
                break

        Cn = rotation_matrix(main_axis, 360 / (2*order))
        Sn = np.dot(Cn, reflection(main_axis))
        if self._check_op(Sn):
            self.schoenflies_symbol = "S{}".format(2*order)
            return

    def _dihedral(self, main_axis, order):

        self.schoenflies_symbol = "D{}".format(order)

        if self._check_op(reflection(main_axis)):
            self.schoenflies_symbol += 'h'
            return

        p_axis = tools.get_perpendicular(main_axis)
        for angle in np.arange(0, 180, 2):
            axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
            if self._check_op(reflection(axis)):
                self.schoenflies_symbol += 'd'
                break

    def _get_axis_rot_order(self, axis, n_max):
        """
        Get rotation order for a given axis

        :param axis: the axis
        :param n_max: maximum order to scan
        :return: order
        """

        order = 0
        for i in list(range(2, n_max)):
            Cn = rotation_matrix(axis, 360 / i)
            if self._check_op(Cn):
                order = i
        return order

    def _check_op(self, sym_matrix):
        """
        check if operation exists

        :param sym_matrix: operation matrix
        :return: True or False
        """

        def find_atom_in_list(atom, tol):
            atom_ind = []
            difference = [np.sqrt(np.sum(x)) for x in np.square(self._cent_coord - atom)]
            for idx, dif in enumerate(difference < np.sqrt(tol)):
                if dif:
                    atom_ind.append(idx)
            return atom_ind

        for idx, atom in enumerate(self._cent_coord):
            atom_op = np.dot(sym_matrix, atom)
            ind = find_atom_in_list(atom_op, self._tolerance_op)
            if not (len(ind) == 1 and self._symbols[ind[0]] == self._symbols[idx]):
                return False
        return True

    def _set_orientation(self, main_axis, p_axis):
        """
        set molecular orientation along main_axis (x) and p_axis (y).

        :param main_axis: principal orientation axis (must be unitary)
        :param p_axis: secondary axis perpendicular to principal (must be unitary)
        :return:
        """
        orientation = np.array([main_axis, p_axis, np.cross(main_axis, p_axis)])
        self._cent_coord = np.dot(self._cent_coord, orientation.T)
        self._eigenvalues = np.dot(self._eigenvalues, orientation.T)
