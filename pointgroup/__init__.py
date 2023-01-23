__version__ = '0.1'

import itertools
import numpy as np
from pointgroup.operations import inversion, rotation_matrix, reflection
from pointgroup import sym_tools


class PointGroup:

    def __init__(self, positions, symbols, tolerance=0.01, tol=0.3):
        self.tolerance = tolerance
        self._tol = tol
        self._symbols = symbols
        self.schoenflies_symbol = ''
        self.sym_op = []
        self.rot_sym = []
        self._max_order = ''
        self._cent_coord = np.array(positions) - sym_tools.get_center_mass(self._symbols, np.array(positions))
        self.set_eigen_values_vectors()

    def get_schoenflies_symbol(self):
        if self.schoenflies_symbol == '':
            val = self._eigenvalues
            if any([i < self.tolerance for i in val]):
                # print('Linear molecule found')
                self._lineal()
            elif abs(val[0] - val[1]) < self.tolerance and abs(val[0] - val[2]) < self.tolerance:
                # print('Spherical molecule found')
                self._spherical()
            elif abs(val[0] - val[1]) > self.tolerance and abs(val[1] - val[2]) > self.tolerance \
                    and abs(val[0] - val[2]) > self.tolerance:
                # print('Asymmetric molecule found')
                self._asymmetric()
            else:
                # print('Symmetric molecule found')
                self._symmetric()

        return self.schoenflies_symbol

    # def which_schoenflies_symbol(self):
    #     print('The point group of the molecule is: ', self.get_schoenflies_symbol())

    def set_eigen_values_vectors(self):
        I = sym_tools.get_inertia_tensor(self._symbols, self._cent_coord)
        eigenvalues, eigenvectors = np.linalg.eigh(I)
        self._cent_coord = np.dot(self._cent_coord, eigenvectors)
        eigenvectors = np.dot(eigenvectors, eigenvectors.T)
        # I = sym_tools.get_inertia_tensor(self._symbols, self._cent_coord)
        # eigenvalues, eigenvectors = np.linalg.eigh(I)
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors.T

    def _lineal(self):

        if self.check_op(inversion()):
            self.schoenflies_symbol = 'Dinfh'
        else:
            self.schoenflies_symbol = 'Cinfh'

    def _asymmetric(self):

        for axis in self._eigenvectors:
            c2 = rotation_matrix(axis, 180)
            if self.check_op(c2):
                self.sym_op.append(c2)
                self.rot_sym.append((axis, 2))

        if len(self.sym_op) == 0:
            self._no_rot_axis()
        elif len(self.sym_op) == 1:
            self._cyclic()
        else:
            self._dihedral()

    def _symmetric(self):

        if abs(self._eigenvalues[0] - self._eigenvalues[1]) < self.tolerance:
            idx = 2
        elif abs(self._eigenvalues[0] - self._eigenvalues[2]) < self.tolerance:
            idx = 1
        else:
            idx = 0

        main_axis = self._eigenvectors[idx, :]
        self.check_order_rot(main_axis)
        self.rot_sym.append((main_axis, self._max_order))

        # Obtain all the c2 vectors perpendicular to main axis
        for ida, axis in enumerate(self._eigenvectors):
            if ida != idx:
                p_axis = axis
                break
        for angle in np.linspace(0, 360, 144, endpoint=False):
            axis = np.matmul(np.array(rotation_matrix(main_axis, angle)), np.array(p_axis))
            c2 = rotation_matrix(axis, 360 / 2)
            if self.check_op(c2):
                self.sym_op.append(c2)
                self.rot_sym.append((axis / np.linalg.norm(axis), 2))

        self.check_consistency()
        if len(self.rot_sym) >= 2:
            self._dihedral()
        elif len(self.rot_sym) == 1:
            self._cyclic()
        else:
            self._no_rot_axis()

    def _spherical(self):

        # I or Ih
        for axis in self._eigenvectors:
            c5 = rotation_matrix(axis, 360 / 5)
            if self.check_op(c5):
                self.sym_op.append(c5)
                self.rot_sym.append((axis, 5))
                self.schoenflies_symbol = "I"
                self._max_order = 5
        if self.schoenflies_symbol and self.check_op(inversion()):
            self.schoenflies_symbol += 'h'

        # O or Oh
        if not self.schoenflies_symbol:
            for axis in self._eigenvectors:
                c4 = rotation_matrix(axis, 360 / 4)
                if self.check_op(c4):
                    self.sym_op.append(c4)
                    self.rot_sym.append((axis, 4))
                    self.schoenflies_symbol = "O"
                    self._max_order = 4
            if self.schoenflies_symbol and self.check_op(inversion()):
                self.schoenflies_symbol += 'h'

        # Obtain all the c2 vectors and matrices
        for angle1 in np.linspace(0, 360, 72, endpoint=False):
            axis1 = np.matmul(np.array(rotation_matrix([1, 0, 0], angle1)), np.array([0, 0, 1]))
            for angle2 in np.linspace(0, 180, 72, endpoint=False):
                axis2 = np.matmul(np.array(rotation_matrix([0, 1, 0], angle2)), axis1)
                c2 = rotation_matrix(axis2, 360 / 2)
                if self.check_op(c2):
                    self.sym_op.append(c2)
                    self.rot_sym.append((axis2 / np.linalg.norm(axis2), 2))

        if self.rot_sym:
            self.check_consistency()

        # T or Td, Th
        m_type = ''
        if not self.schoenflies_symbol:
            # Obtain all the c3 vectors and matrices
            for angle1 in np.linspace(0, 360, 72, endpoint=False):
                axis1 = np.matmul(np.array(rotation_matrix([1, 0, 0], angle1)), np.array([0, 0, 1]))
                for angle2 in np.linspace(0, 180, 72, endpoint=False):
                    axis2 = np.matmul(np.array(rotation_matrix([0, 1, 0], angle2)), axis1)
                    c3 = rotation_matrix(axis2, 360 / 3)
                    if self.check_op(c3):
                        self.sym_op.append(c3)
                        self.rot_sym.append((axis2 / np.linalg.norm(axis2), 3))
            self.schoenflies_symbol = "T"
            self._max_order = 3
            self.check_consistency()
            for axis, order in self.rot_sym:
                if order == 3:
                    m_type = self.mirror_plane(axis)
                    break
            if m_type == '':
                if self.check_op(inversion()):
                    self.rot_sym.append(([0, 0, 0], 'i'))
                    self.sym_op.append(inversion())
                    self.schoenflies_symbol += 'h'
                # else:
                #     self.schoenflies_symbol += 'd'
            else:
                self.schoenflies_symbol += 'd'
            self._improper_rot()

    def _no_rot_axis(self):
        for vector in self._eigenvectors:
            if self.check_op(reflection(vector)):
                self.schoenflies_symbol = 'Cs'
                self.rot_sym.append((vector, 'Sh'))
                self.sym_op.append(reflection(vector))
                break
            else:
                if self.check_op(inversion()):
                    self.schoenflies_symbol = 'Ci'
                    self.rot_sym.append((vector, 'i'))
                    self.sym_op.append(inversion())
                else:
                    self.schoenflies_symbol = 'C1'

    def _cyclic(self):
        main_axis, order = max(self.rot_sym, key=lambda v: v[1])
        self.schoenflies_symbol = "C{}".format(order)
        m_type = self.mirror_plane(main_axis)
        if m_type != '':
            self.schoenflies_symbol += m_type
        else:
            self._improper_rot()
            for vector, order in self.rot_sym:
                if 'S' in str(order):
                    self.schoenflies_symbol = order

    def _dihedral(self):
        main_axis, order = max(self.rot_sym, key=lambda v: v[1])
        self.schoenflies_symbol = "D{}".format(order)
        m_type = self.mirror_plane(main_axis)
        if m_type != '':
            self.schoenflies_symbol += m_type
        self.check_consistency()

    def mirror_plane(self, main_axis):

        m_type = ''
        if self.check_op(reflection(main_axis)):
            self.rot_sym.append((main_axis, 'Sh'))
            self.sym_op.append(reflection(main_axis))
            m_type = 'h'
        else:
            # Sv planes #
            p_axis = np.random.randn(3)
            p_axis -= np.dot(p_axis, main_axis)*main_axis
            p_axis /= np.linalg.norm(p_axis)
            for angle in np.linspace(0, 360, 144, endpoint=False):
                axis = np.matmul(np.array(rotation_matrix(main_axis, angle)), np.array(p_axis))
                possible_mirror = reflection(axis)
                if self.check_op(possible_mirror):
                    m_type = 'v'
                    self.rot_sym.append((axis, 'Sv'))
                    self.sym_op.append(possible_mirror)

            # Sd planes #
            c2_axes, gt_axes = [], []
            for axis, order in self.rot_sym:
                if order == 2:
                    c2_axes.append(axis)
                if order == self._max_order:
                    gt_axes.append(axis)

            for c2_1, c2_2 in itertools.combinations(c2_axes, 2):
                c2_vector = c2_1 + c2_2
                if all(c2_vector < 1e-8):
                    continue
                else:
                    c2_vector /= np.linalg.norm(c2_vector)
                    for gt_vector in gt_axes:
                        p_axis = np.cross(c2_vector, gt_vector)
                        possible_mirror = reflection(p_axis)
                        if self.check_op(reflection(p_axis)):
                            m_type = 'd'
                            self.rot_sym.append((p_axis / np.linalg.norm(p_axis), 'Sd'))
                            self.sym_op.append(possible_mirror)

        return m_type

    def _improper_rot(self):

        for axis, order in self.rot_sym:
            if order == 2:
                ref_mat = reflection(axis)
                for n in range(20, 0, -2):
                    Cn = rotation_matrix(axis, 360 / n)
                    Sn = np.dot(Cn, ref_mat)
                    if self.check_op(Sn):
                        self.rot_sym.append((axis, 'S' + str(n)))
                        self.sym_op.append(Sn)
                        break

    def check_order_rot(self, axis):

        order = 0
        for i in list(range(2, 9)):
            Cn = rotation_matrix(axis, 360 / i)
            if self.check_op(Cn):
                order = i
                self.sym_op.append(Cn)
        self._max_order = order

    def check_consistency(self):

        non_repeated_vectors, matrix_op = [], []
        non_repeated_vectors.append(self.rot_sym[0])
        matrix_op.append(self.sym_op[0])
        for idx, element in enumerate(self.rot_sym):
            non_repeated = True
            for vector, order in non_repeated_vectors:
                diff = abs(element[0] - vector)
                add = abs(element[0] + vector)
                if (all(diff < self._tol) or all(add < self._tol)) \
                        and element[1] == order:
                    non_repeated = False
            if non_repeated:
                non_repeated_vectors.append((element[0], element[1]))
                matrix_op.append(self.sym_op[idx])

        self.rot_sym = non_repeated_vectors
        self.sym_op = matrix_op

    def check_op(self, sym_matrix, tol=1e-02):

        def find_atom_in_list(atom, tol):
            atom_ind = []
            difference = [np.sqrt(np.sum(x)) for x in np.square(self._cent_coord - atom)]
            for idx, dif in enumerate(difference < np.sqrt(tol)):
                if dif:
                    atom_ind.append(idx)
            return atom_ind

        for idx, atom in enumerate(self._cent_coord):
            atom_op = np.dot(sym_matrix, atom)
            ind = find_atom_in_list(atom_op, tol)
            if not (len(ind) == 1 and self._symbols[ind[0]] == self._symbols[idx]):
                return False
        return True

    def get_point_group(self):
        return self.get_schoenflies_symbol()

    def get_rot_vector(self):
        self.rot_sym.insert(0, ([0, 0, 0], 'E'))
        return self.rot_sym

    def get_sym_matrix(self):
        self.sym_op.insert(0, np.identity(3))
        return self.sym_op

    def get_centered_coordinates(self):
        return self._cent_coord
