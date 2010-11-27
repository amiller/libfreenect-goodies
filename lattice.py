import numpy as np
from pclwindow import PCLWindow
import expmap
import scipy
import scipy.optimize

# Return a point cloud, an Nx3 array, made by projecting the kinet depth map 
# through calibration / registration
# u, v are image coordinates, depth comes from the kinect
def project(depth, u, v):
  Z = -1.0 / (-0.0030711*depth + 3.3309495)
  X = -(u - 320.0) * Z / 590.0
  Y = (v - 240.0) * Z / 590.0
  return X,Y,Z
  
def update(X,Y,Z,UV=None,rgb=None,COLOR=None):
  global window
  xyz = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).transpose()
  mask = Z.flatten()<10
  xyz = xyz[mask,:]
  window.XYZ = xyz
  
  if UV:
    U,V = UV
    uv = np.vstack((U.flatten(),V.flatten())).transpose()
    uv = uv[mask,:]
    
  if COLOR:
    R,G,B,A = COLOR
    color = np.vstack((R.flatten(), G.flatten(), B.flatten(), A.flatten())).transpose()
    color = color[mask,:]
    
  window.UV = uv if UV else None
  window.COLOR = color if COLOR else None
  window.RGB = rgb
  

  


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
    
  return normals, weights

	
# Color score
def color_axis(normals):
  #n = np.log(np.power(normals,40))
  X,Y,Z = [normals[:,:,i] for i in range(3)]
  cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
  cx = [np.power(1.0 - c, 20) for c in cc]
  return [c * 0.8 + 0.2 for c in cx]
  
def score(normals, weights):
  X,Y,Z = [normals[:,:,i] for i in range(3)]
  cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
  return np.sum([np.power(1.0 - c, 20)*weights for c in cc])
  
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
    update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R,G,B,(R+G+B)))
    #window.Refresh()
    #pylab.waitforbuttonpress(0.001)
    window.processPaintEvent(Skip())
    return score(n)
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

  
  
def optimize_normals(normals, weights):
  # Optimize a cost function to find the rotation
  def err(x):
    rot = expmap.axis2rot(x)
    n = apply_rot(rot, normals)
    R,G,B = color_axis(n)
    update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R,G,B,(R+G+B)))
    window.Refresh()
    pylab.waitforbuttonpress(0.001)
    #window.processPaintEvent(Skip())
    return -score(n, weights)
  xs = scipy.optimize.fmin(err, [0.0,0.0,0])
  return xs
  
      


#depth, rgb = [x[1] for x in np.load('data/ceiling.npz').items()]
#rgb, depth = [x[1] for x in np.load('data/block1.npz').items()]
#v,u = np.mgrid[160:282,330:510]
rgb, depth = [x[1] for x in np.load('data/block2.npz').items()]
v,u = np.mgrid[231:371,264:434]
#v,u = np.mgrid[:480,:640]
x,y,z = project(depth[v,u], u, v)

# sub sample
if not 'window' in globals(): window = PCLWindow(size=(640,480))
if not 'weights' in globals(): n,weights = normal_compute(x,y,z)
#update(x,y,z,u,v,rgb)
#update(n[:,:,0],n[:,:,1],n[:,:,2], (u,v), rgb, (weights,weights,weights*0+1,weights*0+1))

update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(weights,weights,weights*0+1,weights*0.3))
R,G,B = color_axis(n)
update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R,G,B,(R+G+B)))
import cv
class Skip():
  def Skip(self):
    pass


window.Refresh()
