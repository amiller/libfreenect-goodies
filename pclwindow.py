from pykinectwindow import Window
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import calibkinect

# I probably need more help with these!
try: 
  TEXTURE_TARGET = GL_TEXTURE_RECTANGLE
except:
  TEXTURE_TARGET = GL_TEXTURE_RECTANGLE_ARB



# Window for drawing point clouds
class PCLWindow(Window):
  
  def update_kinect(self, depth, rgb):
    """
    Update all the textures and vertex buffers with new data. Threadsafe.
    """
    self._depth[:,:,2] = depth
    self.canvas.SetCurrent()
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
    glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, 640*480*3*2, self._depth)
    
    glBindTexture(TEXTURE_TARGET, self.rgbtex)
    glTexSubImage2D(TEXTURE_TARGET, 0, 0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, rgb);
    
    self.Refresh()

  def create_buffers(self):
    self.rgbtex = glGenTextures(1)
    glBindTexture(TEXTURE_TARGET, self.rgbtex)
    glTexImage2D(TEXTURE_TARGET,0,GL_RGB,640,480,0,GL_RGB,GL_UNSIGNED_BYTE,None)
  
    self._depth = np.empty((480,640,3),np.int16)
    self._depth[:,:,1], self._depth[:,:,0] = np.mgrid[:480,:640]
    self.xyzbuf = glGenBuffersARB(1)
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
    #glBufferData(GL_ARRAY_BUFFER_ARB, 640*480*3*2,None,GL_DYNAMIC_DRAW)
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, 640*480*3*2, None,GL_DYNAMIC_DRAW)

      
  def __init__(self, *args, **kwargs):
    super(PCLWindow,self).__init__(*args, **kwargs)
    self.rotangles = [0,0]
    self.zoomdist = 1
    self._mpos = None
    
    self.create_buffers()
    
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
      
    glPushMatrix()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, 4/3., 0.3, 200)
    #glScale(-1,1,1)
    #gluOrtho2D(-10,10,-10,10)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    def mouse_rotate(xAngle, yAngle, zAngle):
      glRotatef(xAngle, 1.0, 0.0, 0.0);
      glRotatef(yAngle, 0.0, 1.0, 0.0);
      glRotatef(zAngle, 0.0, 0.0, 1.0);
    glScale(self.zoomdist,self.zoomdist,1)
    glTranslate(0, 0,-3.5)
    mouse_rotate(self.rotangles[0], self.rotangles[1], 0);
    glTranslate(0,0,1.5)
    #glTranslate(0, 0,-1)



    glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
    glVertexPointers(None)
    glTexCoordPointer(3, GL_SHORT, 0, None)

    glMatrixMode(GL_TEXTURE)
    glLoadIdentity()
    glMultMatrixf(calibkinect.uv_matrix().transpose())
    glMultMatrixf(calibkinect.xyz_matrix().transpose())

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glMultMatrixf(calibkinect.xyz_matrix().transpose())

    # Draw the points
    glPointSize(1)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
  
    #if not rgb is None:
    glEnable(TEXTURE_TARGET)

    #if not color is None:
      # glEnableClientState(GL_COLOR_ARRAY)
      # glColorPointerf(color)
      # glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
      # glEnable(GL_BLEND)
    
    glColor3f(1,1,1)
    glDrawElementsui(GL_POINTS, np.mgrid[:640*480])
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisable(TEXTURE_TARGET)
    glDisable(GL_BLEND)
    glPopMatrix()
    
    self._wrap('on_draw_axes')
    
    glPopMatrix()

    