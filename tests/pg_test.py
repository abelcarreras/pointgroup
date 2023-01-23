import unittest
import numpy as np
import os
from pointgroup import PointGroup


dir_path = os.path.dirname(os.path.realpath(__file__))


class ModesTest(unittest.TestCase):
    longMessage = True


def make_test_function(coordinates, symbols, label):
    def test(self):

        pg = PointGroup(positions=coordinates, symbols=symbols)
        self.assertEqual(pg.get_point_group(), label)

    return test


with open('sym_molecules.xyz') as f:
    xyz_files = f.readlines()

i = 0
while True:
    try:
        na = int(xyz_files[i])
    except IndexError:
        break

    coordinates_block = xyz_files[i+2: i+2+na]

    coordinates = np.array([c.split()[1:] for c in coordinates_block], dtype=float)
    symbols = [c.split()[0] for c in coordinates_block]
    label = xyz_files[i+1].strip()

    print(label, PointGroup(coordinates, symbols).get_point_group())

    test_func = make_test_function(coordinates, symbols, label)
    setattr(ModesTest, 'test_{0}'.format(label), test_func)
    del test_func

    i += na + 2
