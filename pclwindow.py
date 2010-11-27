from pykinectwindow import Window
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

# I probably need more help with these!
try: 
  TEXTURE_TARGET = GL_TEXTURE_RECTANGLE
except:
  TEXTURE_TARGET = GL_TEXTURE_RECTANGLE_ARB

def create_texture():
  global rgbtex
  rgbtex = glGenTextures(1)
  glBindTexture(TEXTURE_TARGET, rgbtex)
  glTexImage2D(TEXTURE_TARGET,0,GL_RGB,640,480,0,GL_RGB,GL_UNSIGNED_BYTE,None)

# Window for drawing point clouds
class PCLWindow(Window):
  
  def __init__(self, *args, **kwargs):
    super(PCLWindow,self).__init__(*args, **kwargs)
    self.rotangles = [0,0]
    self.zoomdist = 1
    self.XYZ = None
    self.UV = None
    self.RGB = None
    self.COLOR = None
    self._mpos = None
    #import cv
    #cv.NamedWindow('__init')
    #cv.DestroyWindow('__init')
    
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

  def on_draw(self):  
    if not 'rgbtex' in globals():
      create_texture()

    clearcolor = [0,0,0,0]
    glClearColor(*clearcolor)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    
    xyz, uv, rgb, color = self.XYZ, self.UV, self.RGB, self.COLOR
    if xyz is None: return

    if not rgb is None:
      rgb_ = (rgb.astype(np.float32) * 4 + 70).clip(0,255).astype(np.uint8)
      glBindTexture(TEXTURE_TARGET, rgbtex)
      glTexSubImage2D(TEXTURE_TARGET, 0, 0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, rgb_);

    # flush that stack in case it's broken from earlier
    glPushMatrix()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, 4/3., 0.3, 200)
    #gluOrtho2D(-10,10,-10,10)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    def mouse_rotate(xAngle, yAngle, zAngle):
      glRotatef(xAngle, 1.0, 0.0, 0.0);
      glRotatef(yAngle, 0.0, 1.0, 0.0);
      glRotatef(zAngle, 0.0, 0.0, 1.0);
    glScale(self.zoomdist,self.zoomdist,1)
    glTranslate(0, 0,-10.5)
    mouse_rotate(self.rotangles[0], self.rotangles[1], 0);
    #glTranslate(0,0,1.5)
    #glTranslate(0, 0,-1)

    # Draw some axes
    if 0:
      glBegin(GL_LINES)
      glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
      glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
      glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
      glEnd()

    # Draw the points
    glPointSize(2)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointerf(xyz)
    if not uv is None:
      glEnableClientState(GL_TEXTURE_COORD_ARRAY)
      glTexCoordPointerf(uv)
      glEnable(TEXTURE_TARGET)

    if not color is None:
      glEnableClientState(GL_COLOR_ARRAY)
      glColorPointerf(color)
      glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
      glEnable(GL_BLEND)
    glColor3f(1,1,1)

    glDrawElementsui(GL_POINTS, np.array(range(len(xyz))))
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisable(TEXTURE_TARGET)
    glDisable(GL_BLEND)

    # Draw some axes
    if 1:
      glLineWidth(3)
      glPushMatrix()
      glScalef(1.5,1.5,1.5)
      glBegin(GL_LINES)
      glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
      glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
      glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
      glEnd()
      glPopMatrix()
    #
    if 0:
        inds = np.nonzero(xyz[:,2]>-0.55)
        glPointSize(10)
        glColor3f(0,1,1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glDrawElementsui(GL_POINTS, np.array(inds))
        glDisableClientState(GL_VERTEX_ARRAY)

    if 0:
        # Draw only the points in the near plane
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glColor(0.9,0.9,1.0,0.8)
        glPushMatrix()
        glTranslate(0,0,-0.55)
        glScale(0.6,0.6,1)
        glBegin(GL_QUADS)
        glVertex3f(-1,-1,0); glVertex3f( 1,-1,0);
        glVertex3f( 1, 1,0); glVertex3f(-1, 1,0);
        glEnd()
        glPopMatrix()
        glDisable(GL_BLEND)

    glPopMatrix()

    