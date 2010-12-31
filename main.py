import freenect
import normals
import numpy as np
import scipy
import cv
import calibkinect
import lattice
import preprocess  

preprocess.load('test')

def showimagergb(name, data):
  image = cv.CreateImageHeader((data.shape[1], data.shape[0]),
                             cv.IPL_DEPTH_8U,
                             3)
  cv.SetData(image, data.tostring(), data.dtype.itemsize * data.shape[1] * data.shape[2])
  cv.ShowImage(name, image)

def grab():
  global depth, rgb
  (depth,_),(rgb,_) = freenect.sync_get_depth(), freenect.sync_get_video()

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
  n,w = normals.normals_opencl(depth.astype('f'), rect, 6)
  w *= mask[t:b,l:r]
  #r0,_ = normals.mean_shift_optimize(n, w, r0, rect)
  #r0 = normals.flatrot_numpy(tableplane)
  mat = normals.flatrot_opencl(n,w,preprocess.tableplane,rect,noshow=True)
  #mx,my,mz = lattice.lattice2(n,w,depth,mat,tableplane,rect)
  
  v,u = np.mgrid[t:b,l:r]
  x,y,z = normals.project(depth[v,u], u.astype(np.float32), v.astype(np.float32))
  rotpts = normals.apply_rot(mat, np.dstack((x,y,z)))
  rotn = normals.apply_rot(mat, n)
  cc = np.array(lattice.color_axis(*np.rollaxis(rotn,2),w=w))
  
  lattice.lattice2(n,w,depth,rgb,mat,preprocess.tableplane,rect)
  #lattice.show_projections(rotpts,cc,w,rotn)

  

  showimagergb('rgb',rgb[::2,::2,::-1].clip(0,255/2)*2)
  #normals.show_normals(depth.astype('f'),rect, 5)
  pylab.waitforbuttonpress(0.001)
  #figure(1);
  #clf()
  #imshow(depth[t:b,l:r])
  
r0 = np.array([0,0,0])
def go():
  global r0
  r0 = np.array([0,0,0])
  while 1:
    init_stage1()
  
if __name__ == "__main__":
  pass
