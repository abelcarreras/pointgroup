[![Test & upload to PyPI](https://github.com/abelcarreras/pointgroup/actions/workflows/python-publish.yml/badge.svg)](https://github.com/abelcarreras/pointgroup/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/pointgroup.svg)](https://badge.fury.io/py/pointgroup)

PointGroup
==========
Small utility to determine the point symmetry group of molecular geometries

Features
--------
- Get point symmetry group label
- Pure python implementation

Supported point groups
----------------------
- Linear: `Cinfv`, `Dinfh`
- Low symmetry: `C1`, `Cs`, `Ci`
- Cyclic: `Cn`, `Cnv`, `Cnh`, and improper `S2n`
- Dihedral: `Dn`, `Dnh`, `Dnd`
- Tetrahedral: `T`, `Td`, `Th`
- Octahedral: `O`, `Oh`
- Icosahedral: `I`, `Ih`

Finite cyclic and dihedral rotation orders are searched up to `n = 9`. This includes
`C2` through `C9`, their `v` and `h` variants, `D2` through `D9`, their `h` and `d`
variants, and improper rotations through `S18`.

Requisites
----------
- numpy


Simple example
--------------
```python
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
```

Acknowledgments
---------------
This utility adapts a code originally written by Efrem Bernuz

Contact info
------------
Abel Carreras  
abelcarreras83@gmail.com

Donostia International Physics Center (DIPC)  
Donostia-San Sebastian (Spain)
