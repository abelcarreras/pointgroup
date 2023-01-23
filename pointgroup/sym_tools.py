import warnings

import numpy as np
from pointgroup.element_data import element_mass


def get_mass(symbols):
    mass_vector = []
    for symbol in symbols:
        try:
            mass_vector.append(element_mass(symbol))
        except KeyError as e:
            warnings.warn('Atomic mass of element {} not found, using 1 u'.format(e))
            mass_vector.append(1.0)
    return mass_vector


def get_center_mass(symbols, coordinates):
    mass_vector = get_mass(symbols)
    cbye = [np.dot(mass_vector[i], coordinates[i]) for i in range(len(symbols))]
    r = np.sum(cbye, axis=0)
    r = r / np.sum(mass_vector)
    return r


def get_inertia_tensor(elements, coord):
    mass_vector = get_mass(elements)
    total_inertia = 0

    inertia_xx = np.dot(mass_vector, (np.square(coord[:, 1]) + np.square(coord[:, 2])))
    inertia_yy = np.dot(mass_vector, (np.square(coord[:, 0]) + np.square(coord[:, 2])))
    inertia_zz = np.dot(mass_vector, (np.square(coord[:, 0]) + np.square(coord[:, 1])))
    inertia_xy = -(np.dot(mass_vector, np.multiply(coord[:, 0], coord[:, 1])))
    inertia_yz = -(np.dot(mass_vector, np.multiply(coord[:, 1], coord[:, 2])))
    inertia_xz = -(np.dot(mass_vector, np.multiply(coord[:, 0], coord[:, 2])))

    inertia_matrix = ([[inertia_xx, inertia_xy, inertia_xz],
                       [inertia_xy, inertia_yy, inertia_yz],
                       [inertia_xz, inertia_yz, inertia_zz]])

    for idx, atom in enumerate(coord):
        total_inertia += mass_vector[idx] * np.dot(atom, atom)

    inertia_matrix /= total_inertia
    return inertia_matrix
