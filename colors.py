import scipy
import numpy as np
import calibkinect 

u,v = np.mgrid[:480,:640]

def project(depth, u, v):
  X,Y,Z = u,v,depth
  mat = np.dot(calibkinect.uv_matrix(), calibkinect.xyz_matrix())
  x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
  y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
  w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
  w = 1/w
  return x*w, y*w

colormap = [
  [0,1,0],
  [0,0,1],
  [1,1,0],
  [1,0,0],
]
  
def project_colors(depth, rgb, rect):
  (l,t),(r,b) = rect
  v,u = np.mgrid[t:b,l:r].astype('f')
  global uv
  # Project the duv matrix into U,V rgb coordinates using rgb_matrix() and xyz_matrix()
  uv = project(depth[t:b,l:r], u, v)[::-1]
  
  return [scipy.ndimage.map_coordinates(rgb[:,:,i].astype('f'), uv) for i in range(3)]
  
def choose_colors(R,G,B):
  c1 = np.argmax((R,G,B),0)-1
  c2 = (B*10>G)+2
  c = [c1<0]*c2 + [c1>=0]*c1
  return c.squeeze()