import sys
sys.path += ['..']

from pclwindow import PCLWindow as Window
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import wx
import freenect
import calibkinect as ck
import normals

if not 'window' in globals(): window = Window(size=(640,480))
# Update the point cloud from the shell or from a background thread!

class Ball(object):
  def __init__(self):
    self.reset()
    self.quad = gluNewQuadric()
    self.mat = np.linalg.inv(ck.xyz_matrix())
    
  def draw(self):
    glPushMatrix()
    glTranslate(*self.pos)
    gluSphere(self.quad, 0.01, 10, 10)
    glPopMatrix()
   
  def update(self, dt):
    self.pos = self.pos + self.vel*dt
    
    # Find the location of the ball in the range image
    x,y,z = self.pos
    global u,v,d
    uv = np.dot(self.mat, np.array((x,y,z,1)))
    u,v,d = (uv[:3]/uv[3])
    u,v = np.floor(u),np.floor(v)
    
    if np.sqrt(np.sum(np.dot(self.pos,self.pos))) > 2: 
      self.reset()
      return

    
    #print self.depth[u,v] - d
    # Does the ball intersect here?
    try:
      if np.abs(depth[v,u] - d) < 30:
        #self.vel[2] = -self.vel[2]
    
        # Get a region of interest and run the normals on it
        global rect
        rect = (u-6,v-6),(u+6,v+6)
        (l,t),(r,b) = rect
        self.d,(self.u,self.v) = depth[t:b,l:r], np.mgrid[t:b,l:r]
        self.n,self.w = normals.normals_c(self.d.astype('f'),self.u.astype('f'),self.v.astype('f'))
        n = self.n[self.n.shape[0]/2,self.n.shape[1]/2]
        
        if np.dot(n,self.vel) < 0:
          self.vel = self.vel - 2*n*np.dot(self.vel,n)
    except:
      pass
    
    
  def reset(self):
    self.pos = np.zeros(3)
    self.vel = np.array([0,0,-0.8])

balls = []
def update_balls(dt):
  global balls
  if len(balls) < 40:
    balls += [Ball()]
  for ball in balls:
    ball.update(dt)


@window.event
def on_draw_axes():
  # Draw some axes
  #glDisable(GL_DEPTH_TEST)
  glBegin(GL_LINES)
  glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
  glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
  glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
  glEnd()
  
  update_balls(0.03)
  for ball in balls:
    ball.draw()
  
  
def update(dt=0):
  global rgb, depth
  depth_,_ = freenect.sync_get_depth()
  rgb_,_ = freenect.sync_get_video()
  rgb,depth = np.array(rgb_), np.array(depth_)
  
  rgb = rgb.clip(0,(255-70)/2)*2+70
  wx.CallAfter(window.update_kinect, depth, rgb)
  #ball.update_depth(depth)

  
def update_on(sleep = 0):
  global _updating
  if not '_updating' in globals(): _updating = False
  if _updating:
    update_off()
  
  _updating = True
  from threading import Thread
  global _thread
  def _run():
    while _updating:
      update()
      import time
      time.sleep(sleep)
  _thread = Thread(target=_run)
  _thread.start()
  
def update_off():
  global _updating
  _updating = False
  _thread.join()
  
@window.eventx
def EVT_IDLE(evt):
  return
  evt.RequestMore()
  window.Refresh()

update()

update_on()
wx.__myapp.MainLoop()