cimport cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
  double sqrt(double x)
  double cos(double x)
  double sin(double x)
  double arccos(double x)

@cython.boundscheck(False)
def project(np.ndarray[np.float32_t,ndim=2,mode='c'] depth,  
              np.ndarray[np.float32_t,ndim=2,mode='c'] u, 
              np.ndarray[np.float32_t,ndim=2,mode='c'] v):
            
  h,w = depth.shape[0], depth.shape[1]
  cdef np.ndarray[np.float32_t,ndim=2,mode='c'] X = np.zeros((h,w),np.float32)
  cdef np.ndarray[np.float32_t,ndim=2,mode='c'] Y = np.zeros((h,w),np.float32)
  cdef np.ndarray[np.float32_t,ndim=2,mode='c'] Z = np.zeros((h,w),np.float32)
  for i from 0 < i < h:
    for j from 0 < j < w:
      Z[i,j] = -1.0 / (-0.0030711*depth[i,j] + 3.3309495)
      X[i,j] = -(u[i,j] - 320.0) * Z[i,j] / 590.0
      Y[i,j] = (v[i,j] - 240.0) * Z[i,j] / 590.0

  return X,Y,Z

@cython.boundscheck(False)
def gradient(np.ndarray[np.float32_t,ndim=2,mode='c'] depth):

  h,w = depth.shape[0], depth.shape[1]
  cdef np.ndarray[np.float32_t,ndim=2,mode='c'] dx = np.zeros((h,w),np.float32)
  cdef np.ndarray[np.float32_t,ndim=2,mode='c'] dy = np.zeros((h,w),np.float32)

  for i from 1 < i < h-1:
    for j from 0 < j < w:
      dy[i,j] = (depth[i+1,j]-depth[i-1,j])/2
  for i from 0 < i < h:
    for j from 1 < j < w-1:
      dx[i,j] = (depth[i,j+1]-depth[i,j-1])/2
      
  return dx, dy


def eigs(np.ndarray[np.float32_t,ndim=4,mode='c'] cov):
  h,w = cov.shape[2], cov.shape[3]
  cdef float a_, b_, c_, d_, e_, f_, g_, h_, i_
  cdef float a, b, c 
  cdef float x, y, z
  for i from 0 < i < h:
    for j from 0 < j < w:
      a_ = cov[0,0,i,j]; b_ = cov[0,1,i,j]; c_ = cov[0,2,i,j]
      d_ = cov[1,0,i,j]; e_ = cov[1,1,i,j]; f_ = cov[1,2,i,j]
      g_ = cov[2,0,i,j]; h_ = cov[2,1,i,j]; i_ = cov[2,2,i,j]
      
      a = -1
      b = a_ + e_ + i_
      c = d_*b_ + g_*c_ + f_*h_ - a_*e_ - a_*i_ - e_*i_
      d = a_*e_*i_ - a_*f_*h_ - d_*b_*i_ + d_*c_*h_ + g_*b_*f_ - g_*c_*e_
      
      x = ((3*c/a) - (b*b/(a*a)))/3
      y = ((2*b*b*b/(a*a*a)) - (9*b*c/(a*a)) + (27*d/a))/27
      z = y*y/4 + x*x*x/27
      
  return 0
      
