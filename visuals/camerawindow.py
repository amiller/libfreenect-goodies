from pykinectwindow import Window
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

# Window for drawing point clouds
class CameraWindow(Window):
      
  def __init__(self, *args, **kwargs):
    super(CameraWindow,self).__init__(*args, **kwargs)
    self.rotangles = [0,0]
    self.zoomdist = 1
    self.lookat = np.array([0,0,0])
    self.upvec = np.array([0,1,0])
    self._mpos = None
    self.clearcolor = [0,0,0,0]
    
    @self.eventx
    def EVT_LEFT_DOWN(event):
      self._mpos = event.Position

    @self.eventx
    def EVT_LEFT_UP(event):
      self._mpos = None

    @self.eventx
    def EVT_MOTION(event):
      if event.LeftIsDown():
        if self._mpos:
          (x,y),(mx,my) = event.Position,self._mpos
          self.rotangles[0] += y-my
          self.rotangles[1] += x-mx
          self.Refresh()    
        self._mpos = event.Position


    @self.eventx
    def EVT_MOUSEWHEEL(event):
      dy = event.WheelRotation
      self.zoomdist *= np.power(0.95, -dy)
      self.Refresh()
    self._initOK = True

  def on_draw(self):  
    if not '_initOK' in dir(self): return
    
    glClearColor(*self.clearcolor)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, 4/3., 0.3, 200)

    glMatrixMode(GL_MODELVIEW)
    # flush that stack in case it's broken from earlier
    try:
      while 1: glPopMatrix()
    except:
      pass

    glPushMatrix()
    glLoadIdentity()

    R = np.cross(self.upvec, [0,0,1])
    R /= np.sqrt(np.dot(R,R))

    glScale(self.zoomdist,self.zoomdist,1)
    glTranslate(0, 0,-2.5)
    glRotatef(self.rotangles[0], *R)
    glRotatef(self.rotangles[1], *self.upvec)

    glTranslate(*-self.lookat)

    self._wrap('on_draw_axes')

    