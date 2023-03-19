__version__ = '0.3'
import numpy as np
from pointgroup.operations import Inversion, Rotation, ImproperRotation, Reflection, rotation_matrix
from pointgroup import tools


def abs_to_rad(error, coord):
    coord = np.array(coord)
    return error / np.clip(np.linalg.norm(coord, axis=1), error, None)


def angle_between_vector_matrix(vector, coord, tolerance=1e-5):
    norm_coor = np.linalg.norm(coord, axis=1)
    norm_op_coor = np.linalg.norm(vector)

    angles = []
    for v, n in zip(np.dot(vector, coord.T), norm_coor*norm_op_coor):
        if n < tolerance:
            angles.append(0)
        else:
            angles.append(np.arccos(np.clip( v/n, -1.0, 1.0)))
    return np.array(angles)


def radius_diff_in_radiants(vector, coord, tolerance=1e-5):
    norm_coor = np.linalg.norm(coord, axis=1)
    norm_op_coor = np.linalg.norm(vector)

    average_radii = np.clip((norm_coor + norm_op_coor) / 2, tolerance, None)
    return np.abs(norm_coor - norm_op_coor) / average_radii


class PointGroup:
    """
    Point group main class
    """

    def __init__(self, positions, symbols, tolerance_eig=1e-2, tolerance_ang=5):
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

        if self._check_op(Inversion()):
            self._schoenflies_symbol = 'Dinfh'
        else:
            self._schoenflies_symbol = 'Cinfv'

    def _asymmetric(self):

        self._set_orientation(self._eigenvectors[0], self._eigenvectors[1])

        n_axis_c2 = 0
        main_axis = [1, 0, 0]
        for axis in np.identity(3):
            c2 = Rotation(axis, order=2)
            if self._check_op(c2):
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
        for angle in np.arange(0, 360/self._max_order+self._tolerance_ang, self._tolerance_ang*2):
            axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
            c2 = Rotation(axis, order=2)
            if self._check_op(c2):
                self._dihedral(main_axis)
                self._set_orientation(main_axis, axis)
                return

        self._cyclic(main_axis)

    def _spherical(self):
        """
        Handle spherical groups (T, O, I)

        :return:
        """

        from pointgroup.grid import get_cubed_sphere_grid_points

        main_axis = None
        while main_axis is None:
            for axis in get_cubed_sphere_grid_points(np.deg2rad(self._tolerance_ang)*2):
                c5 = Rotation(axis, order=5)
                c4 = Rotation(axis, order=4)
                c3 = Rotation(axis, order=3)

                if self._check_op(c5):
                    self._schoenflies_symbol = "I"
                    main_axis = axis
                    self._max_order = 5
                    break
                elif self._check_op(c4):
                    self._schoenflies_symbol = "O"
                    main_axis = axis
                    self._max_order = 4
                    break
                elif self._check_op(c3):
                    self._schoenflies_symbol = "T"
                    main_axis = axis
                    self._max_order = 3

            if main_axis is None:
                print('increase tolerance')
                self._tolerance_ang *= 1.01

        p_axis_base = tools.get_perpendicular(main_axis)

        # I or Ih
        if self._schoenflies_symbol == 'I':
            def determine_orientation_I(main_axis):
                r_matrix = rotation_matrix(p_axis_base, np.arcsin((np.sqrt(5)+1)/(2*np.sqrt(3))))
                axis = np.dot(main_axis, r_matrix.T)

                # set molecule orientation in I
                for angle in np.arange(0, 360 / self._max_order+self._tolerance_ang, self._tolerance_ang):
                    rot_matrix = rotation_matrix(main_axis, angle)

                    c5_axis = np.dot(axis, rot_matrix.T)
                    c5 = Rotation(c5_axis, order=5)

                    if self._check_op(c5, tol_factor=1.2):
                        t_axis = np.dot(main_axis, rotation_matrix(p_axis_base, 90).T)
                        return np.dot(t_axis, rot_matrix.T)

                raise Exception('Error orientation I group')

            p_axis = determine_orientation_I(main_axis)
            self._set_orientation(main_axis, p_axis)

            if self._check_op(Inversion()):
                self._schoenflies_symbol += 'h'

        # O or Oh
        if self._schoenflies_symbol == 'O':

            # set molecule orientation in O
            def determine_orientation_O(main_axis):
                r_matrix = rotation_matrix(p_axis_base, 90)
                axis = np.dot(main_axis, r_matrix.T)

                for angle in np.arange(0, 360 / self._max_order+self._tolerance_ang, self._tolerance_ang):
                    rot_matrix = rotation_matrix(main_axis, angle)

                    c4_axis = np.dot(axis, rot_matrix.T)
                    c4 = Rotation(c4_axis, order=4)

                    if self._check_op(c4, tol_factor=1.2):
                        return axis

                raise Exception('Error orientation O group')

            p_axis = determine_orientation_O(main_axis)
            self._set_orientation(main_axis, p_axis)

            if self._check_op(Inversion()):
                self._schoenflies_symbol += 'h'

        # T or Td, Th
        if self._schoenflies_symbol == 'T':

            # set molecule orientation in T
            def determine_orientation_T(main_axis):
                r_matrix = rotation_matrix(p_axis_base, -np.rad2deg(np.arccos(-1/3)))
                axis = np.dot(main_axis, r_matrix.T)

                for angle in np.arange(0, 360 / self._max_order + self._tolerance_ang, self._tolerance_ang):
                    rot_matrix = rotation_matrix(main_axis, angle)

                    c3_axis = np.dot(axis, rot_matrix.T)
                    c3 = Rotation(c3_axis, order=3)

                    if self._check_op(c3, tol_factor=1.1):
                        t_axis = np.dot(main_axis, rotation_matrix(p_axis_base, 90).T)
                        return np.dot(t_axis, rot_matrix.T)

                raise Exception('Error orientation T group')

            p_axis = determine_orientation_T(main_axis)
            self._set_orientation(main_axis, p_axis)

            if self._check_op(Inversion()):
                self._schoenflies_symbol += 'h'
                return

            if self._check_op(Reflection([0, 0, 1]), tol_factor=np.sqrt(3) * 1.1):
                self._schoenflies_symbol += 'd'
                return

    def _no_rot_axis(self):
        for i, vector in enumerate(np.identity(3)):
            if self._check_op(Reflection(vector)):
                self._schoenflies_symbol = 'Cs'
                p_axis = tools.get_perpendicular(vector)
                self._set_orientation(vector, p_axis)
                break
            else:
                self._set_orientation(self._eigenvectors[0], self._eigenvectors[1])
                if self._check_op(Inversion()):
                    self._schoenflies_symbol = 'Ci'
                else:
                    self._schoenflies_symbol = 'C1'

    def _cyclic(self, main_axis):

        self._schoenflies_symbol = "C{}".format(self._max_order)

        if self._check_op(Reflection(main_axis), tol_factor=0.0):
            self._schoenflies_symbol += 'h'
            return

        p_axis = tools.get_perpendicular(main_axis)
        for angle in np.arange(0, 360 / self._max_order+self._tolerance_ang, self._tolerance_ang):
            axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
            if self._check_op(Reflection(axis), tol_factor=1 / tools.magic_formula(2)):
                self._schoenflies_symbol += 'v'
                self._set_orientation(main_axis, axis)
                break

        Sn = ImproperRotation(main_axis, order=2*self._max_order)
        if self._check_op(Sn):
            self._schoenflies_symbol = "S{}".format(2 * self._max_order)
            return

        p_axis = tools.get_perpendicular(main_axis)
        self._set_orientation(main_axis, p_axis)

    def _dihedral(self, main_axis):

        self._schoenflies_symbol = "D{}".format(self._max_order)

        if self._check_op(Reflection(main_axis), tol_factor=0.0):
            self._schoenflies_symbol += 'h'
            return

        p_axis = tools.get_perpendicular(main_axis)
        for angle in np.arange(0, 360/self._max_order+self._tolerance_ang, self._tolerance_ang):
            axis = np.dot(p_axis, rotation_matrix(main_axis, angle))
            if self._check_op(Reflection(axis), tol_factor=1 / tools.magic_formula(2)):
                self._schoenflies_symbol += 'd'
                return

    def _get_axis_rot_order(self, axis, n_max):
        """
        Get rotation order for a given axis

        :param axis: the axis
        :param n_max: maximum order to scan
        :return: order
        """

        for i in range(n_max, 1, -1):
            Cn = Rotation(axis, order=i)
            if self._check_op(Cn):
                return i
        return 1

    def _check_op(self, operation, print_data=False, tol_factor=1.0):
        """
        check if operation exists

        :param operation: operation orbject
        :return: True or False
        """
        sym_matrix = operation.get_matrix()
        tol_factor = operation.associated_error() * tol_factor
        error_abs_rad = abs_to_rad(self._tolerance_eig, coord=self._cent_coord)
        tolerance = np.deg2rad(self._tolerance_ang)

        op_coordinates = np.dot(self._cent_coord, sym_matrix.T)
        for idx, op_coord in enumerate(op_coordinates):

            difference_rad = radius_diff_in_radiants(op_coord, self._cent_coord, self._tolerance_eig)
            difference_ang = angle_between_vector_matrix(op_coord, self._cent_coord, self._tolerance_eig)

            def check_diff(diff, diff2):
                for idx_2, (d1, d2) in enumerate(zip(diff, diff2)):
                    if self._symbols[idx_2] != self._symbols[idx]:
                        continue
                    # d_r = np.linalg.norm([d1, d2])
                    tolerance_total = tolerance * tol_factor + error_abs_rad[idx_2]
                    if d1 < tolerance_total and d2 < tolerance_total:
                        return True
                return False

            if not check_diff(difference_ang, difference_rad):
                return False

            if print_data:
                print('Continue', idx)

        if print_data:
            print('Found!')
        return True

    def _set_orientation(self, main_axis, p_axis):
        """
        set molecular orientation along main_axis (x) and p_axis (y).

        :param main_axis: principal orientation axis (must be unitary)
        :param p_axis: secondary axis perpendicular to principal (must be unitary)
        :return:
        """

        assert np.linalg.norm(main_axis) > 1e-1
        assert np.linalg.norm(p_axis) > 1e-1

        orientation = np.array([main_axis, p_axis, np.cross(main_axis, p_axis)])
        self._cent_coord = np.dot(self._cent_coord, orientation.T)
        self._ref_orientation = np.dot(self._ref_orientation, orientation.T)
