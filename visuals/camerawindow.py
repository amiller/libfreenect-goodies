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
    self.lookat = [0,0,0]
    self._mpos = None

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
    
    clearcolor = [0,0,0,0]
    glClearColor(*clearcolor)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    # flush that stack in case it's broken from earlier
    try:
      while glPopMatrix(): pass
    except:
      pass
      
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, 4/3., 0.3, 200)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    def mouse_rotate(xAngle, yAngle, zAngle):
      glRotatef(xAngle, 1.0, 0.0, 0.0);
      glRotatef(yAngle, 0.0, 1.0, 0.0);
      glRotatef(zAngle, 0.0, 0.0, 1.0);
    glScale(self.zoomdist,self.zoomdist,1)
    glTranslate(0, 0,-1.5)
    glTranslate(*self.lookat)
    mouse_rotate(self.rotangles[0], self.rotangles[1], 0);

    self._wrap('on_draw_axes')

    