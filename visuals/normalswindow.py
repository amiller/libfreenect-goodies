import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import calibkinect
from camerawindow import CameraWindow

# Window for drawing point clouds
class NormalsWindow(CameraWindow):
  
  def __init__(self, *args, **kwargs):
    super(NormalsWindow,self).__init__(*args, **kwargs)
    self.XYZ = None
    self.UV = None
    self.COLOR = None
    self.lookat = np.array([0,0,0])
    self.upvec = np.array([0,1,0])
    
  def on_draw(self):
    
    clearcolor = [1,1,1,0]
    glClearColor(*clearcolor)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    
    xyz, uv, color = self.XYZ, self.UV, self.COLOR
    if xyz is None: return
    

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    #gluPerspective(60, 4/3., 0.3, 200)
    glOrtho(-1.33,1.33,-1,1,0.3,200)

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

    # Draw the points
    glPointSize(2)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointerf(xyz)

    if not color is None:
      glEnableClientState(GL_COLOR_ARRAY)
      glColorPointerf(color)
      glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
      glEnable(GL_BLEND)
    glColor3f(1,1,1,1)

    glDrawElementsui(GL_POINTS, np.array(range(len(xyz))))
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisable(GL_BLEND)

    self._wrap('on_draw_axes')
  
    glPopMatrix()

