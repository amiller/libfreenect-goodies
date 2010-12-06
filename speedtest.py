import numpy as np
import timeit
#import pyximport; pyximport.install()
import speedup

rgb, depth = [x[1].astype(np.float32) for x in np.load('data/block2.npz').items()]
#v,u = np.mgrid[:480,:640]
v,u = np.mgrid[:100,:100]
from IPython.Shell import IPShellEmbed
ip = IPShellEmbed(user_ns=globals()).IP.ipmagic

from scipy.weave import converters

def project(depth, u, v):
  Z = -1.0 / (-0.0030711*depth + 3.3309495)
  X = -(u - 320.0) * Z / 590.0
  Y = (v - 240.0) * Z / 590.0
  return X,Y,Z
  
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

  
def normal_compute(x,y,z):
  from scipy.ndimage.filters import uniform_filter

  # Compute the xx,xy,yx,yz moments
  moments = np.zeros((3,3,x.shape[0],x.shape[1]))
  sums = np.zeros((3,x.shape[0],x.shape[1]))
  covs = np.zeros((3,3,x.shape[0],x.shape[1]))
  xyz = [x,y,z]
  filt = 8

  for i in range(3):
    sums[i] = uniform_filter(xyz[i], filt)
  for i in range(3):
    for j in range(i,3):
      m = uniform_filter(xyz[i] * xyz[j], filt)
      moments[i,j,:,:] = moments[j,i,:,:] = m
      covs[i,j,:,:] = covs[j,i,:,:] = m - sums[i] * sums[j]

  normals = np.zeros((x.shape[0],x.shape[1],3))
  weights = np.zeros((x.shape[0],x.shape[1]))
  for m in range(x.shape[0]):
    for n in range(x.shape[1]):
      # Find the normal vector
      w,v = np.linalg.eig(covs[:,:,m,n])
      ids = np.argsort(np.real(w)) # Find the index of the minimum eigenvalue
      #normals[m,n,:] = np.cross(v[:,ids[2]], v[:,ids[1]])
      normals[m,n,:] = v[:,ids[0]]
      if normals[m,n,:][2] < 0: normals[m,n,:] *= -1
      ww = w*w
      weights[m,n] = 1.0 - np.max(ww[ids[0]]/ww[ids[1]], ww[ids[0]]/ww[ids[2]])

  return normals, np.power(weights,40)

X,Y,Z = project(depth[v,u],v.astype(np.float32),u.astype(np.float32))
  
#print 'native_'
ip('timeit project(depth[v,u],v.astype(np.float32),u.astype(np.float32))')
#print 'project'
ip('timeit speedup.project(depth[v,u],v.astype(np.float32),u.astype(np.float32))')

#n1 = project(depth[v,u],v.astype(np.float32),u.astype(np.float32))
#n2 = speedup.project(depth[v,u],v.astype(np.float32),u.astype(np.float32))

#print 'normal_compute numpy:'
#ip('timeit normal_compute(X,Y,Z)')
