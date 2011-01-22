import calibkinect
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import normals
from pylab import *

"""
initializing:
  - boundpts, boundsptsM: corners of the table in image and metric space
  - tableplane: plane equation (normal vector) of the table in metric camera space
  - background, openglBgHi: backgrounds for use 
  
per frame:
  - mask: 
  - rect: 
"""

# This is for my setup at home!
if 1:
  RIGHT2LEFT = array([[-0.38304387,  0.56503284, -0.73076355, -0.53087166],
         [-0.5339835 ,  0.51008241,  0.67429794,  0.41361691],
         [ 0.75375007,  0.64850134,  0.10633425, -0.60915678],
         [ 0.        ,  0.        ,  0.        ,  1.        ]])

  # Use the mouse to find 4 points (x,y) on the corners of the table.
  # These will define the first ROI.
  boundptsL = np.array(((164,203),(334,146),(604,311),(310,435)))
  boundptsR = np.array((( 33,287),(274,114),(478,195),(318,435)))
else:
  # These ones are for the table at school
  RIGHT2LEFT = array([[ 0.0413361 ,  0.46774602, -0.88289436, -0.68350111],
         [-0.39881273,  0.81792914,  0.41465632,  0.29259375],
         [ 0.9160989 ,  0.3349702 ,  0.22035354, -0.6357524 ],
         [ 0.        ,  0.        ,  0.        ,  1.        ]])
  
  

  
  
  # Use the mouse to find 4 points (x,y) on the corners of the table.
  # These will define the first ROI.
  boundptsL = np.array(((132,260),(320,205),(528,287),(382,414)))
  boundptsR = np.array(((141,269),(371,186),(520,264),(311,420)))
  

#RIGHT2LEFT = np.eye(4)




import ctypes
from ctypes import POINTER as PTR, c_byte, c_ushort, c_size_t
speedup_ctypes = ctypes.cdll.LoadLibrary('speedup_ctypes.so')
speedup_ctypes.inrange.argtypes = [PTR(c_ushort), PTR(c_byte), PTR(c_ushort), PTR(c_ushort), c_size_t]

def threshold_and_mask(depth,bg):
  import scipy
  from scipy.ndimage import binary_erosion, find_objects
  global mask
  def m_():
    # Optimize this in C?
    return (depth>bg['bgLo'])&(depth<bg['bgHi']) #background
  def m2_():
    mm = np.empty((480,640),'bool')
    speedup_ctypes.inrange(depth.ctypes.data_as(PTR(c_ushort)), 
                mm.ctypes.data_as(PTR(c_byte)), 
                bg['bgHi'].ctypes.data_as(PTR(c_ushort)), 
                bg['bgLo'].ctypes.data_as(PTR(c_ushort)), 480*640)
    return mm
  mask = m2_()
  dec = 3
  dil = binary_erosion(mask[::dec,::dec],iterations=2)
  slices = scipy.ndimage.find_objects(dil)
  a,b = slices[0]
  (l,t),(r,b) = (b.start*dec-10,a.start*dec-10),(b.stop*dec+7,a.stop*dec+7)
  b += -(b-t)%16
  r += -(r-l)%16
  if t<0: t+= 16
  if l<0: l+= 16
  if r>=640: r-= 16
  if b>=480: b-= 16
  return mask, ((l,t),(r,b))


def save(filename):
  import cPickle as pickle
  with open('data/saves/%s' % filename,'w') as f:
    pickle.dump(dict(bgL=bgL, bgR=bgR), f)
      
def load(filename):
  import cPickle as pickle
  with open('data/saves/%s' % filename,'r') as f:
    globals().update(pickle.load(f)) 



def _find_plane(depth, boundpts):
  
  # Build a mask of the image inside the convex points clicked
  u,v = uv = np.mgrid[:480,:640][::-1]
  mask = np.ones((480,640),bool)
  for (x,y),(dx,dy) in zip(boundpts, boundpts - np.roll(boundpts,1,0)):
    mask &= ((uv[0]-x)*dy - (uv[1]-y)*dx)<0
  
  # Find the average plane going through here
  global n,w
  n,w = normals.normals_c(depth.astype(np.float32))
  maskw = mask & (w>0)
  abc = n[maskw].mean(0)
  abc /= np.sqrt(np.dot(abc,abc))
  a,b,c = abc
  x,y,z = [_[maskw].mean() for _ in normals.project(depth)]
  d = -(a*x+b*y+c*z)
  tableplane = np.array([a,b,c,d])
  tablemean = np.array([x,y,z])
  
  # Backproject the table plane into the image using inverse transpose 
  global tb0
  tb0 = np.dot(calibkinect.xyz_matrix().transpose(), tableplane)
  
  global boundptsM
  boundptsM = []
  for (up,vp) in boundpts:
    # First project the image points (u,v) onto to the (imaged) tableplane
    dp  = -(tb0[0]*up  + tb0[1]*vp  + tb0[3])/tb0[2]
    
    # Then project them into metric space
    xp, yp, zp, wp  = np.dot(calibkinect.xyz_matrix(), [up,vp,dp,1])
    xp  /= wp ; yp  /= wp ; zp  /= wp;
    boundptsM += [[xp,yp,zp]]
  
  # Use OpenGL and framebuffers to draw the table and the walls
  fbo = glGenFramebuffers(1)
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  
  rb,rbc = glGenRenderbuffers(2)
  glBindRenderbuffer(GL_RENDERBUFFER, rb);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 640, 480)
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
  glDrawBuffer(0)
  glReadBuffer(0)
  glClear(GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, 640, 480);    
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity()
  glOrtho(0,640,0,480,0,-3000)
  glMultMatrixf(np.linalg.inv(calibkinect.xyz_matrix()).transpose())
  def draw():
    glEnable(GL_CULL_FACE)
    glBegin(GL_QUADS)
    for x,y,z in boundptsM:
       glVertex(x,y,z,1)
    for (x,y,z),(x_,y_,z_) in zip(boundptsM,np.roll(boundptsM,1,0)):
      glVertex(x ,y       ,z ,1)
      glVertex(x_,y_      ,z_,1)
      glVertex(a,b,c,0)
      glVertex(a,b,c,0)
    glEnd()
    glDisable(GL_CULL_FACE)   
    glFinish()

  global openglbgHi, openglbgLo
  
  gf = glGetIntegerv(GL_FRONT_FACE)
  glFrontFace(GL_CCW)
  draw()
  openglbgHi = glReadPixels(0, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT).reshape(480,640);
  glFrontFace(GL_CW)
  draw()
  openglbgLo = glReadPixels(0, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT).reshape(480,640);
  glFrontFace(gf)

  openglbgHi *= 3000
  openglbgLo *= 3000
  #openglbgLo[openglbgLo>=2047] = 0
  openglbgHi[openglbgHi>=2047] = 0
  openglbgLo[openglbgHi==openglbgLo] = 0

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteRenderbuffers(1, [rb]);
  glDeleteFramebuffers(1, [fbo]);
  glReadBuffer(GL_BACK)
  glDrawBuffer(GL_BACK)
  
  background = np.array(depth)
  background[~mask] = 2047
  background = np.minimum(background,openglbgHi)
  backgroundM = normals.project(background)
  
  openglbgLo = openglbgLo.astype(np.uint16)
  background = background.astype(np.uint16)
  background[background>=3] -= 3
  openglbgLo += 3
    
  if 1:
    figure(0);
    imshow(openglbgLo)
    
    figure(1);
    imshow(background)
    
  return dict(
    bgLo = openglbgLo,
    bgHi = background,
    boundpts = boundpts,
    boundptsM = boundptsM,
    tablemean = tablemean,
    tableplane = tableplane
  )
  
def find_plane(depthL, depthR):
  global bgL, bgR
  bgL = _find_plane(depthL, boundptsL)
  bgR = _find_plane(depthR, boundptsR)