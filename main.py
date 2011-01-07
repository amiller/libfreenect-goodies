# This is the system running module. It will deal with loading stored data,
# interacting with the camera keeping tracking of the initialization states.

import freenect
import normals
import numpy as np
import scipy
import cv
import calibkinect
import lattice
import preprocess  
import grid
import opencl

# Wait on all the computes
WAIT_COMPUTE=True
SHOW_LATTICEPOINTS=False
SECOND_KINECT=True

# Duplo block sizes
LH = 0.0180
LW = 0.0160*2

# Jenga Block sizes (needs to be remeasured)
#LH = 0.0150
#LW = 0.0200

preprocess.load('test')

def showimagergb(name, data):
  image = cv.CreateImageHeader((data.shape[1], data.shape[0]),
                             cv.IPL_DEPTH_8U,
                             3)
  cv.SetData(image, data.tostring(), data.dtype.itemsize * data.shape[1] * data.shape[2])
  cv.ShowImage(name, image)

def grab():
  global depth, rgb
  global depth2, rgb2
  (depth,_),rgb = freenect.sync_get_depth(0), None
  if SECOND_KINECT: 
    (depth2,_),rgb = freenect.sync_get_depth(1), None
  #(depth ,_),(rgb ,_) = freenect.sync_get_depth(0), freenect.sync_get_video(0)
  #(depth2,_),(rgb2,_) = freenect.sync_get_depth(1), freenect.sync_get_video(1)

# Grab a blank frame!
def init_stage0():
  grab()
  preprocess.find_plane(depth)

# Grab a frame, assume we already have the table found
def init_stage1():
  import pylab
  grab()
  global mask, rect, r0, n, w, cc
  mask,rect = preprocess.threshold_and_mask(depth)
  (l,t),(r,b) = rect
  normals.normals_opencl(depth.astype('f'), np.array(mask[t:b,l:r]), rect, 6)
  #r0,_ = normals.mean_shift_optimize(n, w, r0, rect)
  #r0 = normals.flatrot_numpy(tableplane)
  global modelmat
  if modelmat is None:  
    global mat
    mat = np.zeros((3,4),'f')
    mat[:3,:3] = normals.flatrot_opencl(preprocess.tableplane,rect,noshow=True)
    nw = opencl.get_normals(rect=rect)
    n,w = nw[:,:,:3],nw[:,:,3]
    modelmat = lattice.lattice2(n,w,depth,rgb,mat,preprocess.tableplane,rect,init_t=True)
    #p_modelmat = lattice.lattice2_opencl(depth,rgb,mat,preprocess.tableplane,rect,init_t=True)
    
  else:
    mat = normals.flatrot_opencl(preprocess.tableplane,rect,modelmat,noshow=True)
    mr = np.dot(mat, np.linalg.inv(modelmat[:3,:3])) # find the residual rotation
    ut = np.dot(mr, modelmat[:3,3])                  # apply the residual to update translation
    modelmat[:3,:3] = mat[:3,:3]
    modelmat[:3, 3] = ut
    #modelmat = lattice.lattice2(n,w,depth,rgb,modelmat,preprocess.tableplane,rect,init_t=False)
    modelmat = lattice.lattice2_opencl(depth,rgb,modelmat,preprocess.tableplane,rect,init_t=False)

  #lattice.show_projections(rotpts,cc,w,rotn)
  grid.add_votes(lattice.XYZ, lattice.dXYZ, lattice.cXYZ)
  #grid.carve_background(depth, lattice.XYZ)

  #showimagergb('rgb',rgb[::2,::2,::-1].clip(0,255/2)*1)
  #showimagergb('depth',np.dstack(3*[depth[::2,::2]/8]).astype(np.uint8))
  #normals.show_normals(depth.astype('f'),rect, 5)
  
  pylab.waitforbuttonpress(0.001)
  
  #figure(1);
  #clf()
  #imshow(depth[t:b,l:r])
  
r0 = np.array([0,0,0])
modelmat = None
grid.initialize()
def go():
  global r0,modelmat
  r0 = np.array([0,0,0])
  modelmat = None
  grid.initialize()
  while 1:
    init_stage1()
  
if __name__ == "__main__":
  pass
