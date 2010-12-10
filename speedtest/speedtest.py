import numpy as np
import speedup_cython

import ctypes
speedup_ctypes = ctypes.cdll.LoadLibrary('speedup_ctypes.so')

import sys
sys.path += ['..']
import normals


rgb, depth = [x[1].astype(np.float32) for x in np.load('../data/block2.npz').items()]
v,u = np.mgrid[:480,:640]
#v,u = np.mgrid[:100,:100]
from IPython.Shell import IPShellEmbed
ip = IPShellEmbed(user_ns=globals()).IP.ipmagic





def gradient_ctypes(depth):
  dx = np.array(depth)
  dy = np.array(depth)
  speedup_ctypes.gradient(depth.ctypes.data, dx.ctypes.data, dy.ctypes.data, depth.shape[0], depth.shape[1])
  return dx,dy
  
def gradient_np(depth):
  dx = (np.roll(depth,-1,1) - np.roll(depth,1,1))/2
  dy = (np.roll(depth,-1,0) - np.roll(depth,1,0))/2
  return dx, dy

def gradient_cython(depth):
  dx,dy = speedup_cython.gradient(depth)
  
def gradient_test():
  print 'dx, dy'

  ip('timeit gradient_ctypes(depth)')
  ip('timeit gradient_np(depth)')
  ip('timeit gradient_cython(depth)')
  
gradient_test()




def fast_normal(x,y,z):
  # From "Motion estimation from noisy outdoor scenes"
  from scipy.ndimage.filters import uniform_filter
  filt = 8
  x,y,z = [uniform_filter(x, filt) for x in x,y,z]

  # Compute the xx,xy,yx,yz moments
  moments = np.zeros((3,3,x.shape[0],x.shape[1]))
  sums = np.zeros((3,x.shape[0],x.shape[1]))
  covs = np.zeros((3,3,x.shape[0],x.shape[1]))
  xyz = [x,y,z]



def normal_test():
  print 'normal_compute:'
  ip('timeit normals.fast_normals(depth[v,u],u.astype(np.float32),v.astype(np.float32))')
#normal_test()


def project_test():
  print 'native_'
  ip('timeit project(depth[v,u],v.astype(np.float32),u.astype(np.float32))')
  print 'project'
  ip('timeit speedup.project(depth[v,u],v.astype(np.float32),u.astype(np.float32))')

  #n1 = project(depth[v,u],v.astype(np.float32),u.astype(np.float32))
  #n2 = speedup.project(depth[v,u],v.astype(np.float32),u.astype(np.float32))

    