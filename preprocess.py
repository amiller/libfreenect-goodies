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

# Use the mouse to find 4 points (x,y) on the corners of the table.
# These will define the first ROI.
boundpts = (257,220),(482,227),(533,408),(97,371)

#boundpts = (286,213),(494,264),(451,412),(200,345)
#boundpts = (354,262),(502,265),(501,363),(313,337)

boundpts = np.array(boundpts)
#array([-0.01685934,  0.94029319,  0.33994767,  0.27120995])
#tableplane = np.array([-0.1340615 ,  0.66140062,  0.73795438,  0.47417785])

def threshold_and_mask(depth):
  import scipy
  from scipy.ndimage import binary_erosion, find_objects
  global mask
  def m_():
    return (depth>openglbgLo)&(depth<background) #background
  mask = m_()
  dec = 3
  dil = binary_erosion(mask[::dec,::dec],iterations=2)
  slices = scipy.ndimage.find_objects(dil)
  a,b = slices[0]
  (l,t),(r,b) = (b.start*dec-7,a.start*dec-7),(b.stop*dec+7,a.stop*dec+7)
  b += -(b-t)%16
  r += -(r-l)%16
  return mask, ((l,t),(r,b))


def save(filename):
  np.savez('data/saves/%s'%filename, 
    tableplane=tableplane, tablemean=tablemean,
    mask=mask, 
    background=background, backgroundM=backgroundM,
    openglbgHi=openglbgHi, openglbgLo=openglbgLo,
    boundpts=boundpts, boundptsM=boundptsM)
    
  
def load(filename):
  d = np.load('data/saves/%s.npz'%filename)
  globals().update(d) 
  global openglbgLo, background
  openglbgLo = openglbgLo.astype(np.uint16)
  background = openglbgHi.astype(np.uint16)
  background[background>=3] -= 3
  openglbgLo += 3
  
def find_plane(depth):
  global tableplane,tablemean,mask,maskw,background,backgroundM
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

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteRenderbuffers(1, [rb]);
  glDeleteFramebuffers(1, [fbo]);
  glReadBuffer(GL_BACK)
  glDrawBuffer(GL_BACK)
  
  
  background = np.array(depth)
  background[~mask] = 2047
  background = np.minimum(background,openglbgHi)
  backgroundM = normals.project(background)
  
  
  
  if 1:
    figure(0);
    imshow(openglbgLo)
    
    figure(1);
    imshow(background);
  
