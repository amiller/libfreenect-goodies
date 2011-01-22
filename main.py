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
  #preprocess.load('test1')
  preprocess.load('testlab')
import grid
import opencl
import carve
import bgthread


# Wait on all the computes
WAIT_COMPUTE=False
SHOW_LATTICEPOINTS=False
SECOND_KINECT=True

# Duplo block sizes
LH = 0.0180
LW = 0.0152*1

# Jenga Block sizes (needs to be remeasured)
#LH = 0.0150
#LW = 0.0200

def take_keyframe():
  HR = carve.carve_background(depthR, LW, LH, grid.bounds,
    np.dot(modelmat, preprocess.RIGHT2LEFT), calibkinect.xyz_matrix())
  HL = carve.carve_background(depthL, LW, LH, grid.bounds,
    modelmat, calibkinect.xyz_matrix())
  
  xyzf = opencl.get_modelxyz()
  occH,vacH = carve.add_votes(xyzf, LW, LH, grid.bounds, lattice.meanx, lattice.meanz)
  
  grid.vote_grid = np.maximum(occH,grid.vote_grid)
  grid.carve_grid = np.maximum(np.maximum(grid.carve_grid, vacH), (HR+HL)*30)
  grid.carve_grid *= (occH<30)
  grid.refresh()
  

def showimagergb(name, data):
  image = cv.CreateImageHeader((data.shape[1], data.shape[0]),
                             cv.IPL_DEPTH_8U,
                             3)
  cv.SetData(image, data.tostring(), data.dtype.itemsize * data.shape[1] * data.shape[2])
  cv.ShowImage(name, image)

def grab():
  global depthL, rgbL, depthR, rgbR
  (depthL,_),(rgbL,_) = freenect.sync_get_depth(0), (None,None)#freenect.sync_get_video(0)
  if SECOND_KINECT: 
    (depthR,_),(rgbR,_) = freenect.sync_get_depth(1), (None,None)
  else:
    (depthR,_),(rgbR,_) = (np.array([[]]),None),(None,None)

# Grab a blank frame!
def init_stage0():
  grab()
  preprocess.find_plane(depth)
  
def calibrate_orientation(side='L'):
  import expmap
  global modelmat
  modelmat = None
  try:
    init_stage1()
  except:
    pass
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
  #mR[:3,:3] = np.dot(expmap.axis2rot(np.array([0,-np.pi,0])),mR[:3,:3])
  #mR[:3,:3] = np.dot(expmap.axis2rot(np.array([0,0,0])),mR[:3,:3])
  mR[:3,:3] = np.dot(expmap.axis2rot(np.array([0,-np.pi/2,0])),mR[:3,:3])

  
  mmatL = lattice.lattice2(nL,wL,depthL,rgbL,mL[:4,:4],preprocess.bgL['tableplane'],rectL,init_t=True)
  mmatR = lattice.lattice2(nR,wR,depthR,rgbR,mR[:4,:4],preprocess.bgR['tableplane'],rectR,init_t=True)
  
  #Tinker with the rotation here
  mL[:4,:4] = mmatL
  mR[:4,:4] = mmatR
  mR[0,3] -= 7*LW
  #mR[0,3] += 14*LW
  
  if side=='L':
    mmatL = lattice.lattice2(nL,wL,depthL,rgbL,mL[:4,:4],preprocess.bgL['tableplane'],rectL,init_t=True)
  else:
    mmatR = lattice.lattice2(nR,wR,depthR,rgbR,mR[:4,:4],preprocess.bgR['tableplane'],rectR)

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
  
  try:
    (maskL,rectL) = preprocess.threshold_and_mask(depthL,preprocess.bgL)
    (maskR,rectR) = preprocess.threshold_and_mask(depthR,preprocess.bgR)
  except:
    bgthread.found_keyframe = False
    grid.initialize()
  opencl.set_rect(rectL,rectR)
  normals.normals_opencl2(from_rect(depthL,rectL).astype('f'), 
                 np.array(from_rect(maskL,rectL)), rectL, 
                          from_rect(depthR,rectR).astype('f'),
                 np.array(from_rect(maskR,rectR)), rectR, 6)
  
  global modelmat
  if modelmat is None or not bgthread.found_keyframe:
    grid.initialize()  
    global mat
    mat = np.eye(4).astype('f')
    mat[:3,:3] = normals.flatrot_opencl(preprocess.bgL['tableplane'],noshow=True)
    nwL,nwR = opencl.get_normals()
    n,w = nwL[:,:,:3],nwL[:,:,3]
    #modelmat = lattice.lattice2(n,w,depthL,rgbL,mat,preprocess.bgL['tableplane'],rectL,init_t=True)
    modelmat = lattice.lattice2_opencl(mat,preprocess.bgL['tableplane'],init_t=True)
    #grid.add_votes_opencl(lattice.meanx,lattice.meanz)
    
  else:
    mat = normals.flatrot_opencl(preprocess.bgL['tableplane'],modelmat[:3,:4],noshow=True)
    mr = np.dot(mat, np.linalg.inv(modelmat[:3,:3])) # find the residual rotation
    ut = np.dot(mr, modelmat[:3,3])                  # apply the residual to update translation
    modelmat[:3,:3] = mat[:3,:3]
    modelmat[:3, 3] = ut
    #modelmat = lattice.lattice2(n,w,depth,rgb,modelmat,preprocess.tableplane,rect,init_t=False)
    modelmat = lattice.lattice2_opencl(modelmat,preprocess.bgL['tableplane'],init_t=False)
    
  if experiment_running:
    experiment_mat[experiment_counter] += [modelmat]
    
  xyzf = opencl.get_modelxyz()
  grid.add_votes_opencl(lattice.meanx, lattice.meanz)    

  bgupdate = bgthread.get_update()
  if bgupdate:
    grid.__dict__.update(bgupdate)
    #grid.fix_guide()
    
  #print 'sending frame'
  if 0:
    bgthread.update_track(xyzf, LW, LH, grid.bounds, lattice.meanx, lattice.meanz, 
            grid.vote_grid, grid.carve_grid,
            grid.keyvote_grid, grid.keycarve_grid)
  if 1:
    bgthread.update_keyframe(xyzf, LW, LH, grid.bounds, lattice.meanx, lattice.meanz,
            grid.keyvote_grid, grid.keycarve_grid, 
            depthL, depthR,
            modelmat, np.dot(modelmat, preprocess.RIGHT2LEFT), calibkinect.xyz_matrix(),
            maskL, maskR, rectL, rectR,
            rgbL, rgbR)  
  else:
    bgthread.found_keyframe = True    
    grid.vote_grid = grid.occH
    grid.carve_grid = grid.vacH
    grid.refresh()


  #lattice.show_projections(rotpts,cc,w,rotn)

  #grid.add_votes(lattice.XYZ, lattice.dXYZ, lattice.cXYZ)
  #grid.carve_background(depth, lattice.XYZ)

  #showimagergb('rgbL',rgbL[::2,::2,::-1].clip(0,255/2)*2)
  #showimagergb('rgbR',rgbR[::2,::2,::-1].clip(0,255/2)*2)
  #showimagergb('depth',np.dstack(3*[depth[::2,::2]/8]).astype(np.uint8))
  #normals.show_normals(depth.astype('f'),rect, 5)
  
  grid.window.lookat = preprocess.bgL['tablemean']
  grid.window.upvec = preprocess.bgL['tableplane'][:3]
  grid.window.Refresh()
  pylab.waitforbuttonpress(0.001)
  

modelmat = None
grid.initialize()

def resume():
  bgthread.reset()
  while 1:
    init_stage1()
    
@grid.window.eventx
def EVT_CHAR(e):
  char = e.GetKeyCode()
  global experiment_running, experiment_counter, experiment_mat
  if char == ord('q'):
    experiment_mat += [[]]
    experiment_running = True
    print 'starting experiment step %d' % experiment_counter
  if char == ord('p'):
    experiment_running = False
    print 'saved experiment step %d [%d]' % (experiment_counter, len(experiment_mat[-1]))
    experiment_counter += 1
    
def exp_save(name):
  for i in range(len(experiment_mat)):
    filename = 'experiment/%s_%d' % (name,i)
    np.save(filename, np.array(experiment_mat[i]))
  print 'ok'

def go():
  global experiment_running, experiment_counter, experiment_mat
  experiment_running = False
  experiment_counter = 0
  experiment_mat = []
  
  global modelmat

  modelmat = None
  bgthread.reset()
  grid.initialize()
  resume()
  
if __name__ == "__main__":
  pass
