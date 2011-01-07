# This script should be run from ipython. It puts all the modules in a 
# convenient namespace. 

import main
import lattice
import preprocess
import normals
import grid
import opencl

reload(main)
reload(lattice)
reload(preprocess)
reload(normals)
reload(grid)
reload(opencl)

# Go around three times to cache all the opencl stuff
for i in range(10): main.init_stage1()

from main import *
