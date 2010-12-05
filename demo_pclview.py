import pykinectwindow as wxwindow
import numpy as np
import pylab
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import freenect
import calibkinect


# I probably need more help with these!
try: 
  TEXTURE_TARGET = GL_TEXTURE_RECTANGLE
except:
  TEXTURE_TARGET = GL_TEXTURE_RECTANGLE_ARB


if not 'win' in globals(): win = wxwindow.Window(size=(640,480))

def refresh(): win.Refresh()

if not 'rotangles' in globals(): rotangles = [0,0]
if not 'zoomdist' in globals(): zoomdist = 1
if not 'projpts' in globals(): projpts = (None, None)
if not 'rgb' in globals(): rgb = None

def create_texture():
  global rgbtex
  rgbtex = glGenTextures(1)
  glBindTexture(TEXTURE_TARGET, rgbtex)
  glTexImage2D(TEXTURE_TARGET,0,GL_RGB,640,480,0,GL_RGB,GL_UNSIGNED_BYTE,None)


if not '_mpos' in globals(): _mpos = None
@win.eventx
def EVT_LEFT_DOWN(event):
  global _mpos
  _mpos = event.Position
  
@win.eventx
def EVT_LEFT_UP(event):
  global _mpos
  _mpos = None
  
@win.eventx
def EVT_MOTION(event):
  global _mpos
  if event.LeftIsDown():
    if _mpos:
      (x,y),(mx,my) = event.Position,_mpos
      rotangles[0] += y-my
      rotangles[1] += x-mx
      refresh()    
    _mpos = event.Position


@win.eventx
def EVT_MOUSEWHEEL(event):
  global zoomdist
  dy = event.WheelRotation
  zoomdist *= np.power(0.95, -dy)
  refresh()
  

clearcolor = [0,0,0,0]
@win.event
def on_draw():  
  if not 'rgbtex' in globals():
    create_texture()

  xyz, uv = projpts
  if xyz is None: return

  if not rgb is None:
    rgb_ = (rgb.astype(np.float32) * 4 + 70).clip(0,255).astype(np.uint8)
    glBindTexture(TEXTURE_TARGET, rgbtex)
    glTexSubImage2D(TEXTURE_TARGET, 0, 0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, rgb_);

  glClearColor(*clearcolor)
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glEnable(GL_DEPTH_TEST)

  # flush that stack in case it's broken from earlier
  glPushMatrix()

  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  gluPerspective(60, 4/3., 0.3, 200)

  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

  def mouse_rotate(xAngle, yAngle, zAngle):
    glRotatef(xAngle, 1.0, 0.0, 0.0);
    glRotatef(yAngle, 0.0, 1.0, 0.0);
    glRotatef(zAngle, 0.0, 0.0, 1.0);
  glScale(zoomdist,zoomdist,1)
  glTranslate(0, 0,-3.5)
  mouse_rotate(rotangles[0], rotangles[1], 0);
  glTranslate(0,0,1.5)
  #glTranslate(0, 0,-1)

  # Draw some axes
  if 0:
    glBegin(GL_LINES)
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
    glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
    glEnd()

  # We can either project the points ourselves, or embed it in the opengl matrix
  if 0:
    dec = 4
    v,u = mgrid[:480,:640].astype(np.uint16)
    points = np.vstack((u[::dec,::dec].flatten(),
                        v[::dec,::dec].flatten(),
                        depth[::dec,::dec].flatten())).transpose()
    points = points[points[:,2]<2047,:]
    
    glMatrixMode(GL_TEXTURE)
    glLoadIdentity()
    glMultMatrixf(calibkinect.uv_matrix().transpose())
    glMultMatrixf(calibkinect.xyz_matrix().transpose())
    glTexCoordPointers(np.array(points))
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glMultMatrixf(calibkinect.xyz_matrix().transpose())
    glVertexPointers(np.array(points))
  else:
    glMatrixMode(GL_TEXTURE)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glVertexPointerf(xyz)
    glTexCoordPointerf(uv)

  # Draw the points
  glPointSize(2)
  glEnableClientState(GL_VERTEX_ARRAY)
  glEnableClientState(GL_TEXTURE_COORD_ARRAY)
  glEnable(TEXTURE_TARGET)
  glColor3f(1,1,1)
  glDrawElementsui(GL_POINTS, np.array(range(xyz.shape[0])))
  glDisableClientState(GL_VERTEX_ARRAY)
  glDisableClientState(GL_TEXTURE_COORD_ARRAY)
  glDisable(TEXTURE_TARGET)
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


# A silly loop that shows you can busy the ipython thread while opengl runs
def playcolors():
  while 1:
    global clearcolor
    clearcolor = [np.random.random(),0,0,0]
    time.sleep(0.1)
    refresh()

# Update the point cloud from the shell or from a background thread!

def update(dt=0):
  global projpts, rgb, depth
  depth,_ = freenect.sync_get_depth()
  rgb,_ = freenect.sync_get_video()
  q = depth
  X,Y = np.meshgrid(range(640),range(480))
  # YOU CAN CHANGE THIS AND RERUN THE PROGRAM!
  # Point cloud downsampling
  d = 4
  projpts = calibkinect.depth2xyzuv(q[::d,::d],X[::d,::d],Y[::d,::d])
  refresh()
  
def update_join():
  update_on()
  try:
    _thread.join()
  except:
    update_off()
  
def update_on():
  global _updating
  if not '_updating' in globals(): _updating = False
  if _updating: return
  
  _updating = True
  from threading import Thread
  global _thread
  def _run():
    while _updating:
      update()
  _thread = Thread(target=_run)
  _thread.start()
  
def update_off():
  global _updating
  _updating = False
  
  
# Get frames in a loop and display with opencv
def loopcv():
  import cv
  while 1:
    cv.ShowImage('hi',get_depth().astype(np.uint8))
    cv.WaitKey(10)

update() 
#update_on()




