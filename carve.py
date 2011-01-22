# Everything in here is meant to be called from either the main thread or the 
# background process! Make sure not to save ay state!

import numpy as np

def grid_vertices(grid,factor=1):
  """
  Given a boolean voxel grid, produce a list of vertices and indices 
  for drawing quads or line strips in opengl
  """
  q = [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
       [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
       [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
       [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
       [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
       [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]
  
  normal = [np.cross(np.subtract(qz[0],qz[1]),np.subtract(qz[0],qz[2])) for qz in q]
  
  blocks = np.array(grid.nonzero()).transpose().reshape(-1,1,3)
  q = np.array(q).reshape(1,-1,3)
  vertices = (q + blocks).reshape(-1,3)
  normals = np.tile(normal, (len(blocks),4)).reshape(-1,3)*factor
  line_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,1,2,2,3,3,0]
  quad_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,2,3]
  
  return vertices, normals, line_inds, quad_inds


def project(X, Y, Z, mat):
  x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
  y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
  z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
  w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
  w = 1/w
  return x*w, y*w, z*w
  
def image_bounds(KKL, LW, LH, bounds):
  """
  Find the ROI corresponding to the imaged grid.
  """
  # Project each of the 8 points in the bounds
  import itertools

  p = np.array(list(itertools.product(*zip(*bounds)))) * [LW,LH,LW]
  pz = np.hstack((p, 8*[[1]]))
  impts = np.dot(np.linalg.inv(KKL), pz.transpose())
  mx = impts[0] / impts[3]
  my = impts[1] / impts[3]
  pts = np.floor(np.array(((mx.min(), my.min()),(mx.max(),my.max())))).astype('i4')
  pts = np.minimum(pts,[[640,480],[640,480]])
  pts = np.maximum(pts,[[0,0,],[0,0]])
  return pts

def add_votes(XYZw, LW, LH, bounds, meanx, meanz):

  XYZ = XYZw[:,:3]
  cx,cz = np.rollaxis(np.frombuffer(np.array(XYZw[:,3]).data, dtype='i2').reshape(-1,2),1)
  
  diff = np.vstack((0.5*cx,0*cx,0.5*cz,)).transpose()

  occvotes = (XYZ - [meanx,0,meanz]) / [LW,LH,LW] + diff
  vacvotes = occvotes - diff - diff
  weights = np.abs(cx)+np.abs(cz)
  
  bins = [np.arange(bounds[0][i],bounds[1][i]+1) for i in range(3)]

  occH,_ = np.histogramdd(occvotes, bins, weights=weights)
  vacH,_ = np.histogramdd(vacvotes, bins, weights=weights)
  
  return occH, vacH


def carve_background(depth, LH, LW, bounds, modelmat, KKL):
  """
  We can carve out points in the grid as long as the points match the background
  and they fit in the grid. We have to randomly sample distances. 
  #TODO We can probably also randomly sample points themselves.
  #TODO Use OpenGL to draw a mask of the grid.
  """

  (l,t),(r,b) = image_bounds(np.dot(modelmat,KKL), LW, LH, bounds)
  
  DEC = 2
  ITER = 8
  
  depth = depth[t:b:DEC,l:r:DEC].astype('f')
  v,u = np.mgrid[t:b:DEC, l:r:DEC].astype('f')
  mask = depth<2047

  global bgm
  bgm = np.vstack([_.flatten() for _ in project(u,v,depth, np.dot(modelmat, KKL).astype('f'))])

  orig = modelmat[:3,3].reshape(3,1).astype('f')
  
  vec = orig - bgm;
  vec /= np.sqrt(np.sum(vec*vec,0))
  bgm += vec * LW * 2.5
  bins = [np.arange(bounds[0][i],bounds[1][i]+1) for i in range(3)]
  
  H = None
  for i in range(ITER):
    alpha = np.random.rand(bgm.shape[1]).astype('f')**5

    xyz = ((alpha) * orig + (1-alpha) * bgm).transpose() / [LW,LH,LW]
  
    m = np.all((xyz%1 > 0.3) & (xyz%1 < 0.7),1)
 
    H_,_ = np.histogramdd(xyz, bins, weights=m)
    H = H_ if H is None else H + H_
      
  return H