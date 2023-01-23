
PointGroup
==========
Small utility to determine the point symmetry group of molecular geometries

Features
--------
- Pure python implementation

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

Acknowledges
------------
This utility adapts a code originally written by Efrem Bernuz

Contact info
------------
Abel Carreras  
abelcarreras83@gmail.com

Donostia International Physics Center (DIPC)  
Donostia-San Sebastian (Spain)
