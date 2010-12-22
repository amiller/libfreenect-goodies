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
    
  def on_draw(self):  
    
    clearcolor = [0,0,0,0]
    glClearColor(*clearcolor)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    
    xyz, uv, color = self.XYZ, self.UV, self.COLOR
    if xyz is None: return
    

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, 4/3., 0.3, 200)
    #gluOrtho2D(-10,10,-10,10)

    glMatrixMode(GL_MODELVIEW)
    # flush that stack in case it's broken from earlier
    glPushMatrix()
    glLoadIdentity()

    def mouse_rotate(xAngle, yAngle, zAngle):
      glRotatef(xAngle, 1.0, 0.0, 0.0);
      glRotatef(yAngle, 0.0, 1.0, 0.0);
      glRotatef(zAngle, 0.0, 0.0, 1.0);
      
    glScale(self.zoomdist,self.zoomdist,1)
    glTranslate(0, 0,-1.5)
    mouse_rotate(self.rotangles[0], self.rotangles[1], 0);
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
    glColor3f(1,1,1)

    glDrawElementsui(GL_POINTS, np.array(range(len(xyz))))
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisable(GL_BLEND)

    self._wrap('on_draw_axes')
  
    glPopMatrix()

