import main
from pclwindow import PCLWindow
if not 'window' in main.__dict__: main.window = PCLWindow(size=(640,480))
window = main.window
import numpy as np
import expmap
import scipy
import scipy.optimize
import pylab
from OpenGL.GL import *


# Return a point cloud, an Nx3 array, made by projecting the kinet depth map 
# through calibration / registration
# u, v are image coordinates, depth comes from the kinect
def project(depth, u, v):
  Z = -1.0 / (-0.0030711*depth + 3.3309495)
  X = -(u.astype(np.float32) - 340.0) * Z / 590.0
  Y = (v.astype(np.float32) - 240.0) * Z / 590.0
  return X,Y,Z
  
def other_project(depth, u, v):
  f = 590.0
  a = -0.0030711
  b = 3.3309495
  cx = 340.0
  cy = 240.0
  n = np.dstack((u,v,depth))
  global mat
  mat = np.array([[1/f, 0, 0, -cx/f],
                  [0,-1/f, 0,  cy/f],
                  [0,   0, 0,    -1],
                  [0,   0, a,     b]])
  x = n[:,:,0]*mat[0,0] + n[:,:,1]*mat[0,1] + n[:,:,2]*mat[0,2] + mat[0,3]
  y = n[:,:,0]*mat[1,0] + n[:,:,1]*mat[1,1] + n[:,:,2]*mat[1,2] + mat[1,3]
  z = n[:,:,0]*mat[2,0] + n[:,:,1]*mat[2,1] + n[:,:,2]*mat[2,2] + mat[2,3]
  w = n[:,:,0]*mat[3,0] + n[:,:,1]*mat[3,1] + n[:,:,2]*mat[3,2] + mat[3,3]
  return x/w, y/w, z/w
  
def update(X,Y,Z,UV=None,rgb=None,COLOR=None,AXES=None):
  global window
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

def fast_normals(depth, u, v):
  f = 590.0
  a = -0.0030711
  b = 3.3309495
  cx = 320.0
  cy = 240.0
  
  from scipy.ndimage.filters import uniform_filter, convolve
  depth = np.array(depth)
  depth[depth==2047] = -1e8
  depth = uniform_filter(depth, 7)
  
  dx = convolve(depth, np.array([[1,0,-1]],np.int16))/np.float32(2.0)
  dy = convolve(depth, np.array([[1],[0],[-1]],np.int16))/np.float32(2.0)
  n = np.dstack((-dx, -dy, 0*dy+1))
  zw = -(n[:,:,0]*u + n[:,:,1]*v + n[:,:,2]*depth)
  global mat
  mat = np.linalg.inv(
        np.array([[1/f, 0, 0,-cx/f],
                  [0,-1/f, 0, cy/f],
                  [0,   0, 0,   -1],
                  [0,   0, a,    b]],np.float32)).transpose()
  x = n[:,:,0]*mat[0,0] + n[:,:,1]*mat[0,1] + n[:,:,2]*mat[0,2] + zw*mat[0,3]
  y = n[:,:,0]*mat[1,0] + n[:,:,1]*mat[1,1] + n[:,:,2]*mat[1,2] + zw*mat[1,3]
  z = n[:,:,0]*mat[2,0] + n[:,:,1]*mat[2,1] + n[:,:,2]*mat[2,2] + zw*mat[2,3]
  w = np.sqrt(x*x + y*y + z*z)
  w[z<0] *= -1
  weights = z*0+1
  weights[depth<-1000] = 0
  weights[(z/w)<.1] = 0
  
  return np.dstack((x/w,y/w,z/w)), weights
    
def normal_compute(x,y,z):
  from scipy.ndimage.filters import uniform_filter
  
  # Compute the xx,xy,yx,yz moments
  moments = np.zeros((3,3,x.shape[0],x.shape[1]))
  sums = np.zeros((3,x.shape[0],x.shape[1]))
  covs = np.zeros((3,3,x.shape[0],x.shape[1]))
  xyz = [x,y,z]
  filt = 8
  
  for i in range(3):
    sums[i] = uniform_filter(xyz[i], filt)
  for i in range(3):
    for j in range(i,3):
      m = uniform_filter(xyz[i] * xyz[j], filt)
      moments[i,j,:,:] = moments[j,i,:,:] = m
      covs[i,j,:,:] = covs[j,i,:,:] = m - sums[i] * sums[j]

  normals = np.zeros((x.shape[0],x.shape[1],3))
  weights = np.zeros((x.shape[0],x.shape[1]))
  for m in range(x.shape[0]):
    for n in range(x.shape[1]):
      # Find the normal vector
      w,v = np.linalg.eig(covs[:,:,m,n])
      ids = np.argsort(np.real(w)) # Find the index of the minimum eigenvalue
      #normals[m,n,:] = np.cross(v[:,ids[2]], v[:,ids[1]])
      normals[m,n,:] = v[:,ids[0]]
      if normals[m,n,:][2] < 0: normals[m,n,:] *= -1
      ww = w*w
      weights[m,n] = 1.0 - np.max(ww[ids[0]]/ww[ids[1]], ww[ids[0]]/ww[ids[2]])
    
  return normals, np.power(weights,40)

axes_rotation = np.eye(4)
@window.event
def on_draw_axes():
  # Draw some axes
  glLineWidth(3)
  glPushMatrix()
  glMultMatrixf(axes_rotation.transpose())
  glScalef(1.5,1.5,1.5)
  glBegin(GL_LINES)
  glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
  glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
  glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
  glEnd()
  glPopMatrix()

# Color score
def color_axis(normals,d=0.1):
  #n = np.log(np.power(normals,40))
  X,Y,Z = [normals[:,:,i] for i in range(3)]
  cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
  cx = [np.max((1.0-(c/d*c/d),0*c),0) for c in cc]
  return [c * 0.8 + 0.2 for c in cx]
  
  
# Returns the optimal rotation, and the per-axis error weights
def mean_shift_optimize(normals, weights, r0=np.array([0,0,0])):
  # Don't worry about convergence for now!
  mat = expmap.axis2rot(r0)
  d = 0.2 # E(p) = (x/d)^2 + (y/d)^2
  perr = 0; derr = 10
  iters = 0
  while iters < 100 and (derr > 0.0001):
    iters += 1
    n = apply_rot(mat, normals)
    X,Y,Z = [n[:,:,i] for i in range(3)]
    R,G,B = color_axis(n,d)
    #update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R+0.5,G+0.5,B+0.5,weights))
    #update(normals[:,:,0],normals[:,:,1],normals[:,:,2], 
    #      COLOR=(R+0.5,G+0.5,B+0.5,weights), AXES=expmap.rot2axis(mat))
    #window.Refresh()
    #pylab.waitforbuttonpress(0.001)
    cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
    err = np.sum(weights*[np.max((1.0-(c/d*c/d),0*c),0) for c in cc])
    derr = np.abs(perr-err)
    perr = err
    xm,ym,zm = [(c <= d*d)*weights for c in cc] # threshold mask
    dX = (Z*ym - Y*zm).sum() / (ym.sum() + zm.sum()); 
    if np.isnan(dX): dX = 0
    dY = (X*zm - Z*xm).sum() / (zm.sum() + xm.sum()); 
    if np.isnan(dY): dY = 0
    dZ = (Y*xm - X*ym).sum() / (xm.sum() + ym.sum()); 
    if np.isnan(dZ): dZ = 0
    m = expmap.euler2rot([dX, dY, dZ])
    mat = np.dot(m.transpose(), mat)
  return expmap.rot2axis(mat), weights*[np.max((1.0-(c/d*c/d),0*c),0) for c in cc]
    
  
def score(normals, weights):
  d = 0.1
  X,Y,Z = [normals[:,:,i] for i in range(3)]
  cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
  return np.sum([np.max((1.0-(c/d*c/d),0*c),0) for c in cc])
  
def apply_rot(rot, xyz):
  flat = np.rollaxis(xyz,2).reshape((3,-1))
  xp = np.dot(rot, flat)
  return xp.transpose().reshape(xyz.shape)
  
import mpl_toolkits.mplot3d.axes3d as mp3
def surface(normals, weights, r0=np.array([-0.7,-0.2,0])):
  rangex = np.arange(-0.6,0.6,0.06)
  rangey = np.arange(-0.6,0.6,0.06)
  def err(x):
    rot = expmap.axis2rot(x)
    n = apply_rot(rot, normals)
    R,G,B = color_axis(n)
    update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R+0.4,G+0.4,B+0.4,(R+G+B)))
    #window.Refresh()
    #pylab.waitforbuttonpress(0.001)
    window.processPaintEvent()
    return score(n, weights)
  x,y = np.meshgrid(rangex, rangey)
  z = [err(r0+np.array([xp,yp,0])) for xp in rangex for yp in rangey]
  z = np.reshape(z, (len(rangey),len(rangex)))
  fig = pylab.figure(1)
  fig.clf()
  global ax
  ax = mp3.Axes3D(fig)

  ax.plot_surface(x,y,-z, rstride=1,cstride=1, cmap=pylab.cm.jet,
  	linewidth=0,antialiased=False)
  fig.show()

def go():
  while 1:
    x0 = (np.random.rand(3)*2-1)*0.5 + np.array(r0)
    print mean_shift_optimize(n, weights, x0)
    #print optimize_normals(n, weights, x0)
  
  
def optimize_normals(normals, weights,x0):
  # Optimize a cost function to find the rotation
  def err(x):
    rot = expmap.axis2rot(x)
    n = apply_rot(rot, normals)
    R,G,B = color_axis(n)
    # This version keeps the axes fixed and rotates the cloud
    #update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R+0.5,G+0.5,B+0.5,weights))
    
    # This version keep the point cloud stationary and rotates the axes
    update(normals[:,:,0],normals[:,:,1],normals[:,:,2], COLOR=(R+0.5,G+0.5,B+0.5,weights),AXES=x)
    window.Refresh()
    pylab.waitforbuttonpress(0.001)
    #window.processPaintEvent(Skip())
    return -score(n, weights)
  xs = scipy.optimize.fmin(err, x0)
  return xs
  
def play_movie():
  from pylab import gca, imshow, figure
  import pylab
  foldername = 'data/movies/block2'
  frame = 0
  r0 = np.array([-0.7626858,   0.28330218,  0.17082515])
  while 1:
    
    depth = np.load('%s/depth_%05d.npz'%(foldername,frame)).items()[0][1]
    #v,u = np.mgrid[135:332,335:485]
    v,u = np.mgrid[231:371,264:434]
    d = np.load('%s/normals_%05d.npz'%(foldername,frame))
    n,weights,u,v = d['n'],d['weights'],d['u'],d['v']
    figure(1)
    gca().clear()
    imshow(depth)
    x,y,z = project(depth[v,u], u, v)
    #n, weights = normal_compute(x,y,z)
    #np.savez('%s/normals_%05d'%(foldername,frame),v=v,u=u,n=n,weights=weights)
    r0 = mean_shift_optimize(n, weights, r0)
    print r0 
    frame += 1



if __name__ == "__main__":
  rgb, depth = [x[1].astype(np.float32) for x in np.load('data/block2.npz').items()]
  v,u = np.mgrid[231:371,264:434]
  r0 = [-0.63, 0.68, 0.17]
  
  x,y,z = project(depth[v,u], u.astype(np.float32), v.astype(np.float32))
  # sub sample
  if not 'weights' in globals(): n,weights = normal_compute(x,y,z)
  #update(x,y,z,u,v,rgb)
  #update(n[:,:,0],n[:,:,1],n[:,:,2], (u,v), rgb, (weights,weights,weights*0+1,weights*0+1))

  update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(weights,weights,weights*0+1,weights*0.3))
  R,G,B = color_axis(n)
  update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R+.5,G+.5,B+.5,R+G+B))
  import cv

  window.Refresh()
