# This is the system running module. It will deal with loading stored data,
# interacting with the camera keeping tracking of the initialization states.

import freenect
import normals
import numpy as np
import scipy
import cv
import calibkinect
import lattice
if not 'preprocess' in globals():
  import preprocess
  preprocess.load('test')
import grid
import opencl



# Wait on all the computes
WAIT_COMPUTE=False
SHOW_LATTICEPOINTS=False
SECOND_KINECT=True

# Duplo block sizes
LH = 0.0180
LW = 0.0160

# Jenga Block sizes (needs to be remeasured)
#LH = 0.0150
#LW = 0.0200


def showimagergb(name, data):
  image = cv.CreateImageHeader((data.shape[1], data.shape[0]),
                             cv.IPL_DEPTH_8U,
                             3)
  cv.SetData(image, data.tostring(), data.dtype.itemsize * data.shape[1] * data.shape[2])
  cv.ShowImage(name, image)

def grab():
  global depthL, rgbL, depthR, rgbR
  (depthL,_),rgbL = freenect.sync_get_depth(0), None
  if SECOND_KINECT: 
    (depthR,_),rgbR = freenect.sync_get_depth(1), None
  else:
    (depthR,_),rgbR = (np.array([[]]),None),None

# Grab a blank frame!
def init_stage0():
  grab()
  preprocess.find_plane(depth)
  
def calibrate_orientation():
  import expmap
  global modelmat
  modelmat = None
  init_stage1()
  global nL,wL,nR,wR
  nwL,nwR = opencl.get_normals()
  nL,wL = nwL[:,:,:3],nwL[:,:,3]
  nR,wR = nwR[:,:,:3],nwR[:,:,3]
  #nL,wL = normals.normals_numpy(depthL.astype('f'),rectL)
  #nR,wR = normals.normals_numpy(depthR.astype('f'),rectR)
  axesL = normals.flatrot_numpy(nL,wL,preprocess.bgL['tableplane'])
  axesR = normals.flatrot_numpy(nR,wR,preprocess.bgR['tableplane'])
  mL,mR = np.eye(4), np.eye(4)
  mL[:3,:3] = expmap.axis2rot(axesL)
  mR[:3,:3] = expmap.axis2rot(axesR)
  
  # Tinker with the rotation here
  mR[:3,:3] = np.dot(expmap.axis2rot(np.array([0,-np.pi,0])),mR[:3,:3])
  
  mmatL = lattice.lattice2(nL,wL,depthL,rgbL,mL[:3,:4],preprocess.bgL['tableplane'],rectL,init_t=True)
  mmatR = lattice.lattice2(nR,wR,depthR,rgbR,mR[:3,:4],preprocess.bgR['tableplane'],rectR,init_t=True)
  
  #Tinker with the rotation here
  mL[:3,:4] = mmatL
  mR[:3,:4] = mmatR
  mR[0,3] -= 8*LW
  
  mmatR = lattice.lattice2(nR,wR,depthR,rgbR,mR[:3,:4],preprocess.bgR['tableplane'],rectR)
  #mmatL = lattice.lattice2(nL,wL,depthL,rgbL,mL[:3,:4],preprocess.bgL['tableplane'],rectL,init_t=True)

  calib_fix = np.dot(np.linalg.inv(mL),mR)  
  return calib_fix

  

# Grab a frame, assume we already have the table found
def init_stage1():
  import pylab
  grab()
  
  def from_rect(m,rect):
    (l,t),(r,b) = rect
    return m[t:b,l:r]
  
  global maskL, rectL
  global maskR, rectR
  
  if 1:
    (maskL,rectL) = preprocess.threshold_and_mask(depthL,preprocess.bgL)
    opencl.set_rect(rectL,((0,0),(0,0)))
    opencl.load_mask(np.array(from_rect(maskL,rectL)),'LEFT')    
    dL = from_rect(depthL,rectL).astype('f')
    filtL = scipy.ndimage.uniform_filter(dL,6)   
    opencl.load_filt(filtL,'LEFT')                    
    opencl.compute_normals('LEFT')
    
    (maskR,rectR) = preprocess.threshold_and_mask(depthR,preprocess.bgR)
    opencl.set_rect(rectL,rectR) 
    opencl.load_mask(np.array(from_rect(maskR,rectR)),'RIGHT')
    dR = from_rect(depthR,rectR).astype('f')
    filtR = scipy.ndimage.uniform_filter(dR,6)
    opencl.load_filt(filtR,'RIGHT')                    
    opencl.compute_normals('RIGHT').wait()
  else: 
    (maskL,rectL) = preprocess.threshold_and_mask(depthL,preprocess.bgL)
    (maskR,rectR) = preprocess.threshold_and_mask(depthR,preprocess.bgR)
    opencl.set_rect(rectL,rectR)
    normals.normals_opencl2(from_rect(depthL,rectL).astype('f'), 
                   np.array(from_rect(maskL,rectL)), rectL, 
                            from_rect(depthR,rectR).astype('f'),
                   np.array(from_rect(maskR,rectR)), rectR, 6)
  
  global modelmat
  if modelmat is None:  
    global mat
    mat = np.zeros((3,4),'f')
    mat[:3,:3] = normals.flatrot_opencl(preprocess.bgL['tableplane'],noshow=False)
    nwL,nwR = opencl.get_normals()
    n,w = nwL[:,:,:3],nwL[:,:,3]
    modelmat = lattice.lattice2(n,w,depthL,rgbL,mat,preprocess.bgL['tableplane'],rectL,init_t=True)
    #p_modelmat = lattice.lattice2_opencl(depth,rgb,mat,preprocess.tableplane,rect,init_t=True)
    
  else:
    mat = normals.flatrot_opencl(preprocess.bgL['tableplane'],modelmat,noshow=True)
    mr = np.dot(mat, np.linalg.inv(modelmat[:3,:3])) # find the residual rotation
    ut = np.dot(mr, modelmat[:3,3])                  # apply the residual to update translation
    modelmat[:3,:3] = mat[:3,:3]
    modelmat[:3, 3] = ut
    #modelmat = lattice.lattice2(n,w,depth,rgb,modelmat,preprocess.tableplane,rect,init_t=False)
    modelmat = lattice.lattice2_opencl(modelmat,preprocess.bgL['tableplane'],init_t=False)

  #lattice.show_projections(rotpts,cc,w,rotn)
  #grid.add_votes(lattice.XYZ, lattice.dXYZ, lattice.cXYZ)
  #grid.carve_background(depth, lattice.XYZ)

  #showimagergb('rgb',rgb[::2,::2,::-1].clip(0,255/2)*1)
  #showimagergb('depth',np.dstack(3*[depth[::2,::2]/8]).astype(np.uint8))
  #normals.show_normals(depth.astype('f'),rect, 5)
  
  #pylab.waitforbuttonpress(0.001)
  
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
