.. highlight:: rst

Get started
===========

PointGroup is a simple library based in a single class that is initialized from a set of
atomic positions and atomic symbols.

Determine the point symmetry group
----------------------------------

To determine the point symmetry group, first initialize a **PointGroup** object and then
access to **get_point_group()** method. This returns a string containing the label of the determined point group
for the given molecule. The following example shows the determination of the octahedral symmetry of SF\ :sub:`6` molecule:

.. code-block:: python

    from pointgroup import PointGroup


    pg = PointGroup(positions=[[ 0.000000,  0.000000,  0.000000],
                               [ 0.000000,  0.000000,  1.561000],
                               [ 0.000000,  1.561000,  0.000000],
                               [ 0.000000,  0.000000, -1.561000],
                               [ 0.000000, -1.561000,  0.000000],
                               [ 1.561000,  0.000000,  0.000000],
                               [-1.561000,  0.000000,  0.000000]],
                    symbols=['S', 'F', 'F', 'F', 'F', 'F', 'F'])

    print('Point group: ', pg.get_point_group())

