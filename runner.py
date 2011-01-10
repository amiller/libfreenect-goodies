# This script should be run from ipython. It puts all the modules in a 
# convenient namespace. 

import main
import lattice
import preprocess
import normals
import grid
import opencl
import carve
import bgthread

reload(main)
reload(lattice)
reload(preprocess)
reload(normals)
reload(grid)
reload(opencl)
reload(carve)
reload(bgthread)

from pylab import *
from main import *

def show_normals():
  global nwL,nwR
  nwL,nwR = opencl.get_normals()
  nL,wL = nwL[:,:,:3],nwL[:,:,3]
  nR,wR = nwR[:,:,:3],nwR[:,:,3]
  figure(1); clf(); imshow(nL/2+.5)
  figure(2); clf(); imshow(nR/2+.5)

# Go around three times to cache all the opencl stuff
for i in range(10): main.init_stage1()

from main import *
