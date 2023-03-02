__version__ = '0.2'
import numpy as np
from pointgroup.operations import inversion, rotation_matrix, reflection
from pointgroup import tools


class PointGroup:
    """
    Point group main class
    """

    def __init__(self, positions, symbols, tolerance_eig=1e-3, tolerance_ang=5):
        self._tolerance_eig = tolerance_eig
        self._tolerance_ang = tolerance_ang  # in degrees
        self._symbols = symbols
        self._cent_coord = np.array(positions) - tools.get_center_mass(self._symbols, np.array(positions))
        self._ref_orientation = np.identity(3)

        # determine inertia tensor
        inertia_tensor = tools.get_inertia_tensor(self._symbols, self._cent_coord)
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors.T

        # initialize variables
        self._schoenflies_symbol = ''
        self._max_order = 1

        eig_degeneracy = tools.get_degeneracy(self._eigenvalues, self._tolerance_eig)

        # Linear groups
        if np.min(abs(self._eigenvalues)) < self._tolerance_eig:
            self._lineal()

        # Asymmetric group
        elif eig_degeneracy == 1:
            self._asymmetric()

        # Symmetric group
        elif eig_degeneracy == 2:
            self._symmetric()

        # Spherical group
        elif eig_degeneracy == 3:
            self._spherical()

        else:
            raise Exception('Group type error')

    def get_point_group(self):
        """
        get the point symmetry group symbol

        :return: the point symmetry group symbol
        """
        return self._schoenflies_symbol

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
            self._schoenflies_symbol = 'Dinfh'
        else:
            self._schoenflies_symbol = 'Cinfv'

    def _asymmetric(self):

        self._set_orientation(self._eigenvectors[0], self._eigenvectors[1])

        n_axis_c2 = 0
        main_axis = [1, 0, 0]
        for axis in np.identity(3):
            c2 = rotation_matrix(axis, 180)
            if self._check_op(c2, flex=0.9):
                n_axis_c2 += 1
                main_axis = axis

        self._max_order = 2

        if n_axis_c2 == 0:
            self._max_order = 0
            self._no_rot_axis()
        elif n_axis_c2 == 1:
            self._cyclic(main_axis)
        else:
            self._dihedral(main_axis)

    def _symmetric(self):

        idx = tools.get_non_degenerated(self._eigenvalues, self._tolerance_eig)
        main_axis = self._eigenvectors[idx]

        self._max_order = self._get_axis_rot_order(main_axis, n_max=9)

        p_axis = tools.get_perpendicular(main_axis)
        for angle in np.arange(0, 360/self._max_order, self._tolerance_ang):
            axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
            c2 = rotation_matrix(axis, 360 / 2)
            if self._check_op(c2, flex=1.5):
                self._dihedral(main_axis)
                self._set_orientation(main_axis, axis)
                return

        self._cyclic(main_axis)

    def _spherical(self):
        """
        Handle spherical groups (T, O, I)

        :return:
        """

        def get_axis_list():
            for ev in np.identity(3):
                yield ev
            for yaw in np.arange(0, 360, self._tolerance_ang):
                for pitch in np.arange(0, 180, self._tolerance_ang):
                    x = np.cos(yaw) * np.cos(pitch)
                    y = np.sin(yaw) * np.cos(pitch)
                    z = np.sin(pitch)
                    yield np.array([x, y, z])

        main_axis = None
        while main_axis is None:
            for axis in get_axis_list():
                c5 = rotation_matrix(axis, 360 / 5)
                c4 = rotation_matrix(axis, 360 / 4)
                c3 = rotation_matrix(axis, 360 / 3)

                if self._check_op(c5, flex=0.9):
                    self._schoenflies_symbol = "I"
                    main_axis = axis
                    self._max_order = 5
                    break
                elif self._check_op(c4, flex=0.9):
                    self._schoenflies_symbol = "O"
                    main_axis = axis
                    self._max_order = 4
                    break
                elif self._check_op(c3, flex=0.9):
                    self._schoenflies_symbol = "T"
                    main_axis = axis
                    self._max_order = 3

            if main_axis is None:
                self._tolerance_ang *= 1.1

        # I or Ih
        if self._schoenflies_symbol == 'I':
            if self._check_op(inversion(), flex=1.5):
                self._schoenflies_symbol += 'h'

            # set molecule orientation in I
            p_axis = tools.get_perpendicular(main_axis)
            for angle in np.arange(0, 360 / self._max_order, self._tolerance_ang):
                axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
                c2 = rotation_matrix(axis, 360 / 2)
                if self._check_op(c2):
                    p_axis = axis
                    break

            self._set_orientation(main_axis, p_axis)

        # O or Oh
        if self._schoenflies_symbol == 'O':
            if self._check_op(inversion(), flex=1.5):
                self._schoenflies_symbol += 'h'

            # set molecule orientation in O
            p_axis = tools.get_perpendicular(main_axis)
            for angle in np.arange(0, 360 / self._max_order, self._tolerance_ang):
                axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
                c4 = rotation_matrix(axis, 360 / 4)
                if self._check_op(c4, flex=3.0):
                    p_axis = axis
                    break

            self._set_orientation(main_axis, p_axis)

        # T or Td, Th
        if self._schoenflies_symbol == 'T':
            p_axis = tools.get_perpendicular(main_axis)

            # set molecule orientation in T
            def determine_orientation():
                t_axis = np.dot(rotation_matrix(p_axis, np.rad2deg(np.arccos(-1/3))/2), main_axis)
                for angle in np.arange(0, 360 / self._max_order, self._tolerance_ang):
                    axis = np.dot(t_axis, rotation_matrix(main_axis, angle))

                    c2 = rotation_matrix(axis, 360 / 2)
                    if self._check_op(c2, flex=1.5):
                        t_axis = np.dot(rotation_matrix(p_axis, 90), main_axis)
                        axis = np.dot(t_axis, rotation_matrix(main_axis, angle))
                        self._set_orientation(main_axis, axis)
                        return [1, 0, 0]

                raise Exception('Error orientation T group')

            main_axis = determine_orientation()

            if self._check_op(inversion(), flex=3.0):
                self._schoenflies_symbol += 'h'
                return

            if self._check_op(reflection([0, 0, 1]), flex=3.0):
                self._schoenflies_symbol += 'd'
                return

    def _no_rot_axis(self):
        for i, vector in enumerate(np.identity(3)):
            if self._check_op(reflection(vector), flex=1.5):
                self._schoenflies_symbol = 'Cs'
                p_axis = tools.get_perpendicular(vector)
                self._set_orientation(vector, p_axis)
                break
            else:
                self._set_orientation(self._eigenvectors[0], self._eigenvectors[1])
                if self._check_op(inversion(), flex=1.5):
                    self._schoenflies_symbol = 'Ci'
                else:
                    self._schoenflies_symbol = 'C1'

    def _cyclic(self, main_axis):

        self._schoenflies_symbol = "C{}".format(self._max_order)

        if self._check_op(reflection(main_axis)):
            self._schoenflies_symbol += 'h'
            return

        p_axis = tools.get_perpendicular(main_axis)
        for angle in np.arange(0, 360 / self._max_order, self._tolerance_ang):
            axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
            if self._check_op(reflection(axis)):
                self._schoenflies_symbol += 'v'
                self._set_orientation(main_axis, axis)
                break

        Cn = rotation_matrix(main_axis, 360 / (2*self._max_order))
        Sn = np.dot(Cn, reflection(main_axis))
        if self._check_op(Sn):
            self._schoenflies_symbol = "S{}".format(2 * self._max_order)
            return

        p_axis = tools.get_perpendicular(main_axis)
        self._set_orientation(main_axis, p_axis)

    def _dihedral(self, main_axis):

        self._schoenflies_symbol = "D{}".format(self._max_order)

        if self._check_op(reflection(main_axis)):
            self._schoenflies_symbol += 'h'
            return

        p_axis = tools.get_perpendicular(main_axis)
        for angle in np.arange(0, 360/self._max_order, self._tolerance_ang):
            axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
            if self._check_op(reflection(axis)):
                self._schoenflies_symbol += 'd'
                return

    def _get_axis_rot_order(self, axis, n_max):
        """
        Get rotation order for a given axis

        :param axis: the axis
        :param n_max: maximum order to scan
        :return: order
        """

        order = 1
        for i in range(2, n_max):
            Cn = rotation_matrix(axis, 360 / i)
            if self._check_op(Cn, flex=0.9):
                order = i
        return order

    def _check_op(self, sym_matrix, print_data=False, flex=1.1):
        """
        check if operation exists

        :param sym_matrix: operation matrix
        :return: True or False
        """

        for idx, atom_coord in enumerate(self._cent_coord):

            atom_op = np.dot(sym_matrix, atom_coord)

            radii = np.linalg.norm(self._cent_coord, axis=1)
            radii = (radii + np.linalg.norm(atom_op)) / 2
            radii = np.clip(radii, 1e-3, None)

            difference_rad = (np.abs(np.linalg.norm(self._cent_coord, axis=1) - np.linalg.norm(atom_op))) / radii

            norm = np.array([np.linalg.norm(atom_op)*np.linalg.norm(c) for c in self._cent_coord])

            difference_ang = []
            for c, n in zip(self._cent_coord, norm):
                if n < 1e-3:
                    difference_ang.append(0.0)
                else:
                    difference_ang.append(np.arccos(np.clip(np.dot(c, atom_op)/n, -1.0, 1.0)))

            def check_diff(diff, diff2):
                for idx_2, (d1, d2) in enumerate(zip(diff, diff2)):
                    d_t = np.linalg.norm([d1, d2])
                    if (self._symbols[idx_2] == self._symbols[idx] and d_t < np.deg2rad(self._tolerance_ang)*flex):
                        return True
                    # if (self._symbols[idx_2] == self._symbols[idx] and d1 < np.deg2rad(self._tolerance_ang)*flex and d2 < np.deg2rad(self._tolerance_ang)*flex):
                        return True

                return False

            if not check_diff(difference_ang, difference_rad):
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
        self._ref_orientation = np.dot(self._ref_orientation, orientation.T)
