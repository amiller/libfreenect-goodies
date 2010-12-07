cimport cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
  double sqrt(double x)
  double cos(double x)
  double sin(double x)
  double arccos(double x)

def gradient(np.ndarray[np.float32_t,ndim=2,mode='c'] depth):
  h,w = depth.shape[0], depth.shape[1]
  cdef np.ndarray[np.float32_t,ndim=2,mode='c'] dx = np.zeros((h,w),np.float32)
  cdef np.ndarray[np.float32_t,ndim=2,mode='c'] dy = np.zeros((h,w),np.float32)

  for i from 1 < i < h-1:
    for j from 1 < j < w-1:
      dy[i,j] = (depth[i+1,j]-depth[i-1,j])/2
      dx[i,j] = (depth[i,j+1]-depth[i,j-1])/2  
  return dx, dy



