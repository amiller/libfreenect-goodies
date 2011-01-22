from camerawindow import CameraWindow
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *

TEXTURE_TARGET = GL_TEXTURE_RECTANGLE

# Window for drawing point clouds
class PointWindow(CameraWindow):
  
  def __init__(self, *args, **kwargs):
    super(PointWindow,self).__init__(*args, **kwargs)
    self.XYZ = np.zeros((0,3))
    self.RGBA = None
    self.clearcolor = [1,1,1,0]
    self.create_buffers()

  def create_buffers(self):
    self.rgbtex = glGenTextures(1)
    glBindTexture(TEXTURE_TARGET, self.rgbtex)
    glTexImage2D(TEXTURE_TARGET,0,GL_RGB,640,480,0,GL_RGB,GL_UNSIGNED_BYTE,None)
    
    
  
    self._depth = np.empty((480,640,3),np.int16)
    self._depth[:,:,1], self._depth[:,:,0] = np.mgrid[:480,:640]
    self.xyzbuf = glGenBuffersARB(1)
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, 640*480*3*4, None,GL_DYNAMIC_DRAW)
    self.rgbabuf = glGenBuffersARB(1)
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.rgbabuf)
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, 640*480*4*4, None,GL_DYNAMIC_DRAW)
    
  def update_points(self, XYZ=None, RGBA=None):
    if XYZ is None: XYZ = np.zeros((0,3),'f')
    assert XYZ.dtype == np.float32
    assert RGBA is None or RGBA.dtype == np.float32
    assert XYZ.shape[1] == 3
    assert RGBA is None or RGBA.shape[1] == 4
    assert RGBA is None or XYZ.shape[0] == RGBA.shape[0]
    self.XYZ = XYZ
    self.RGBA = RGBA
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
    glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, XYZ.shape[0]*3*4, XYZ)
    if not RGBA is None:
      glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.rgbabuf)
      glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, XYZ.shape[0]*4*4, RGBA)

  def on_draw(self):  
    super(PointWindow,self).set_camera()

    glClearColor(*self.clearcolor)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    
    self._wrap('pre_draw')

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
    glVertexPointerf(None)
    glEnableClientState(GL_VERTEX_ARRAY)
    if not self.RGBA is None:
      glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.rgbabuf)
      glColorPointer(4, GL_FLOAT, 0, None)
      glEnableClientState(GL_COLOR_ARRAY)
      
    # Draw the points
    glPointSize(2)
    glColor(0,0,0,1.0)
    glDrawElementsui(GL_POINTS, np.arange(len(self.XYZ)))
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)

    glDisable(GL_BLEND)
    
    self._wrap('post_draw')

    