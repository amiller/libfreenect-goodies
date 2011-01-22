import numpy as np
import expmap
import scipy
import pylab
from OpenGL.GL import *
import normals
import calibkinect
import preprocess
from pylab import *
import colors
import opencl

def circular_mean(data, modulo):
  """
  Given data sampled from a periodic function (with known period: modulo), 
  find the phase by converting to cartesian coordinates. 
  """
  angle = data / modulo * np.pi * 2
  y = np.sin(angle)
  x = np.cos(angle)
  a2 = np.arctan2(y.mean(),x.mean()) / (2*np.pi)
  if np.isnan(a2): a2 = 0
  return a2*modulo
  
def color_axis(X,Y,Z,w,d=0.1):
  X2,Y2,Z2 = X*X,Y*Y,Z*Z
  d = 1/d
  cc = Y2+Z2, Z2+X2, X2+Y2
  cx = [w*np.maximum(1.0-(c*d)**2,0*c) for c in cc]
  return [c for c in cx]
  

def project(X, Y, Z, mat):
  x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
  y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
  z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
  w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
  w = 1/w
  return x*w, y*w, z*w
  

  
def lattice2_opencl(mat,tableplane,init_t=None):
  from main import LH,LW
  assert mat.shape == (4,4)
  global modelmat
  modelmat = np.array(mat)
  modelmat[1,3] = tableplane[3]

  # Build a matrix that can project the depth image into model space
  matxyz = np.dot(modelmat, calibkinect.xyz_matrix().astype('f'))
  
  # Returns warped coordinates, and sincos values for the lattice
  opencl.compute_lattice2(modelmat[:3,:4], LW)
  
  global cx,cz,face
  # If we don't have a good initialization for the model space translation,
  # use the centroid of the surface points.  
  if init_t:  
    global face
    X,Y,Z,face = np.rollaxis(opencl.get_modelxyz(),1)
    cx,cz = np.rollaxis(np.frombuffer(np.array(face).data, dtype='i2').reshape(-1,2),1)
    
    modelmat[:,3] -= np.round([X[cz!=0].mean()/LW, 0, Z[cx!=0].mean()/LW, 0])*LW
    opencl.compute_lattice2(modelmat[:3,:4], LW)
    

  # Find the circular mean, using weights
  def cmean(mxy,c):
    x,y = mxy / c
    a2 = np.arctan2(y,x) / (2*np.pi) * LW
    if np.isnan(a2): a2 = 0
    return a2

  global meanx,meanz
  cxcz,qx2qz2 = opencl.reduce_lattice2()
  meanx = cmean(qx2qz2[:2],cxcz[0])
  meanz = cmean(qx2qz2[2:],cxcz[1])
  modelmat[:,3] -= np.array([meanx, 0, meanz, 0])


  import main
  if main.SHOW_LATTICEPOINTS:
    _,_,_,face = np.rollaxis(opencl.get_modelxyz(),1)
    Xo,Yo,Zo,_ = np.rollaxis(opencl.get_xyz(),1)
    
    cx,cz = np.rollaxis(np.frombuffer(np.array(face).data, dtype='i2').reshape(-1,2),1)
    R,G,B = np.abs(cx),cx*0,np.abs(cz)
    update(Xo,Yo,Zo,COLOR=(R,G,B,R*0+1))
    window.Refresh()

  return modelmat
  
  
def lattice2(n,w,depth,rgb,mat,tableplane,rect,init_t=None):
  """
  Assuming we know the tableplane, find the rotation and translation in
  the XZ plane.
  - init_angle, init_t: 
      if None, the rotation (4-way 90 degree ambiguity) is defined arbitrarily
      and the translation is set to the centroid of the detected points.
  """
  from main import LH,LW
  
  (l,t),(r,b) = rect
  assert mat.shape == (4,4)
  global modelmat
  modelmat = np.eye(4,dtype='f')
  modelmat[:3,:4] = mat[:3,:4]
  modelmat[1,3] = tableplane[3]
  
  # Build a matrix that can project the depth image into model space
  matxyz = np.dot(modelmat, calibkinect.xyz_matrix().astype('f'))
  v,u = np.mgrid[t:b,l:r]
  X,Y,Z = project(u.astype('f'),v.astype('f'),depth[t:b,l:r].astype('f'), matxyz)

  # Project normals from camera space to model space (axis aligned)
  global dx,dy,dz
  global cx,cy,cz
  dx = np.dot(n,modelmat[0,:3])
  dy = np.dot(n,modelmat[1,:3])
  dz = np.dot(n,modelmat[2,:3])
  cx,cy,cz = color_axis(dx,dy,dz,w)
    
  # If we don't have a good initialization for the model space translation,
  # use the centroid of the surface points.  
  if init_t:  
    modelmat[:,3] -= [X[cx>0].mean(), 0, Z[cz>0].mean(), 0]
    matxyz = np.dot(modelmat, calibkinect.xyz_matrix())
    v,u = np.mgrid[t:b,l:r]
    X,Y,Z = project(u.astype('f'),v.astype('f'),depth[t:b,l:r].astype('f'), matxyz)
    
  
  global meanx, meany, meanz
  meanx = circular_mean(X[cx>0],LW)
  meanz = circular_mean(Z[cz>0],LW)

  ax,az = np.sum(cx>0),np.sum(cz>0)
  ax,az = [np.minimum(_/30.0,1.0) for _ in ax,az]
  modelmat[:,3] -= np.array([ax*meanx, 0, az*meanz, 0])

  X -= (ax)*meanx
  Z -= (az)*meanz
  
  # Stacked data in model space
  global XYZ, dXYZ, cXYZ
  XYZ  = ((X,Y,Z))
  dXYZ = ((dx, dy, dz))
  cXYZ = ((cx, cy, cz))


  if 1:
    Xo,Yo,Zo = project(u,v,depth[t:b,l:r], calibkinect.xyz_matrix())

    cany = (cx>0)|(cz>0)
  
    R,G,B = cx[cany],cy[cany],cz[cany]; 
    #Xv = X - LW*np.sign(dx)*.5 * (cx>0)
    #Zv = Z - LW*np.sign(dz)*.5 * (cz>0)
  
    #R = np.floor(Xv/LW)%2
    #G = np.floor(Y/LH)%2
    #B = np.floor(Zv/LW)%2
    #R,G,B = R[cany], G[cany], B[cany]
  
    #update(X[w>0],
    #       Y[w>0],
    #       Z[w>0],COLOR=(R,G,B,R+G+B))
    #update(*3*[np.array([[0]])])
    update(Xo[cany],Yo[cany],Zo[cany],COLOR=(R,G,B,R+G+B))

    window.Refresh()
    
  return modelmat


def update(X,Y,Z,UV=None,rgb=None,COLOR=None,AXES=None):
  global window
  #window.lookat = np.array([0,0,0])
  window.upvec = preprocess.bgL['tableplane'][:3]
  window.lookat = preprocess.bgL['tablemean']+window.upvec*0.05

  
  xyz = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).transpose()
  mask = Z.flatten()<10
  xyz = xyz[mask,:]
  window.XYZ = xyz

  global axes_rotation
  axes_rotation = np.eye(4)
  if not AXES is None:
    # Rotate the axes
    axes_rotation[:3,:3] = expmap.axis2rot(-AXES)

  if not UV is None:
    U,V = UV
    uv = np.vstack((U.flatten(),V.flatten())).transpose()
    uv = uv[mask,:]

  if not COLOR is None:
    R,G,B,A = COLOR
    color = np.vstack((R.flatten(), G.flatten(), B.flatten(), A.flatten())).transpose()
    color = color[mask,:]

  window.UV = uv if UV else None
  window.COLOR = color if COLOR else None
  window.RGB = rgb
    
from visuals.normalswindow import NormalsWindow
if not 'window' in globals(): 
  window = NormalsWindow(title='Lattice Tracking', size=(640,480))

window.Refresh()

  

@window.event
def on_draw_axes():
  from main import LW,LH
  #return
  glPolygonOffset(1.0,0.2)
  glEnable(GL_POLYGON_OFFSET_FILL)
  
  # Draw the gray table
  if 0:
    glBegin(GL_QUADS)
    glColor(0.6,0.7,0.7,1)
    for x,y,z in preprocess.bgL['boundptsM']:
      glVertex(x,y,z)
    glEnd()
  
  glDisable(GL_POLYGON_OFFSET_FILL)
  
  glPushMatrix() 
  glEnable(GL_BLEND)
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

  
  if 1:
    # Draw the axes for the model coordinate space
    glLineWidth(3)
    glMultMatrixf(np.linalg.inv(modelmat).transpose())
    glScalef(LW,LH,LW)
    glBegin(GL_LINES)
    glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
    glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
    glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
    glEnd()

    import grid  
    # Draw a grid for the model coordinate space
  
    glLineWidth(1)
    glBegin(GL_LINES)

    GR = grid.GRIDRAD
    glColor3f(0.2,0.2,0.4)
    for j in range(0,1):
      for i in range(-GR,GR+1):
        glVertex(i,j,-GR); glVertex(i,j,GR)
        glVertex(-GR,j,i); glVertex(GR,j,i)
    glEnd()
    glPopMatrix()

