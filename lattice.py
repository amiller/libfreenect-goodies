import main
from pclwindow import PCLWindow
if not 'window' in main.__dict__: main.window = PCLWindow(size=(640,480))
window = main.window
import numpy as np
import expmap
import scipy
import pylab
from OpenGL.GL import *
import normals

def circular_mean(data, modulo):
  angle = data / modulo * np.pi * 2
  y = np.sin(angle)
  x = np.cos(angle)
  return np.arctan2(y.mean(),x.mean()) / (2*np.pi) * modulo
  
def color_axis(normals,d=0.1):
  #n = np.log(np.power(normals,40))
  X,Y,Z = [normals[:,:,i] for i in range(3)]
  cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
  cx = [np.max((1.0-(c/d*c/d),0*c),0) for c in cc]
  return [c * 0.8 + 0.2 for c in cx]

def show_projections(rotpts,cc,weights,rotn):
  from pylab import figure, title, clf, gca, scatter, grid
  figure(1)
  global meanx, meany
  meanx = circular_mean(rotpts[:,:,0][cc[0,:,:]>0],0.0158)
  meany = circular_mean(rotpts[:,:,1][cc[1,:,:]>0],0.0192)
  meanz = circular_mean(rotpts[:,:,2][cc[2,:,:]>0],0.0158)
  mx = np.floor(rotpts[:,:,0][weights>0].mean()/.0158)*.0158
  mz = np.floor(rotpts[:,:,2][weights>0].mean()/.0158)*.0158
  my = np.floor(rotpts[:,:,1][weights>0].mean()/.0196)*.0196

  R,G,B = color_axis(rotn); R = R[weights>0]; G = G[weights>0]; B = B[weights>0]
  update(rotpts[:,:,0][weights>0]-mx,rotpts[:,:,1][weights>0]-my,rotpts[:,:,2][weights>0]-mz,COLOR=(R,G,B,R+G+B))
  window.Refresh()
  pylab.draw()
  pylab.waitforbuttonpress(0.1)

  clf(); 
  title('height vs Z')
  xlabel('Z (meters)')
  ylabel('Height/Y (meters)')
  xticks((np.arange(-10,10) + np.floor(rotpts[:,:,2][cc[2,:,:]>0].mean()/.0158))*0.0158)
  yticks((np.arange(-10,10) + np.floor(rotpts[:,:,1][cc[1,:,:]>0].mean()/.0196))*0.0196)
  scatter(rotpts[:,:,2][weights>0]-meanz, rotpts[:,:,1][weights>0]-meany)
  scatter(rotpts[:,:,2][cc[2,:,:]>0]-meanz, rotpts[:,:,1][cc[2,:,:]>0]-meany,c='r')
  gca().set_xlim(gca().get_xlim()[::-1]) 
  grid('on')
  figure(3)
  clf();
  yticks(np.arange(-10,10)*0.0158)
  xticks(np.arange(-30,-10)*0.0192)
  scatter(rotpts[:,:,1][cc[2,:,:]>0].flatten()-meany, np.mod(rotpts[:,:,2][cc[2,:,:]>0].flatten(),0.0158)-meanx)
  grid ('on')
  figure(2)
  clf()
  title('Height vs X')
  xlabel('X (meters)')
  ylabel('Height/Y (meters)')
  xticks((np.arange(-10,10) + np.floor(rotpts[:,:,0][cc[0,:,:]>0].mean()/.0158))*0.0158)
  yticks((np.arange(-10,10) + np.floor(rotpts[:,:,1][cc[1,:,:]>0].mean()/.0196))*0.0196)
  grid('on')
  scatter(rotpts[:,:,0][cc[2,:,:]>0]-meanx,rotpts[:,:,1][cc[2,:,:]>0]-meany)
  scatter(rotpts[:,:,0][cc[0,:,:]>0]-meanx,rotpts[:,:,1][cc[0,:,:]>0]-meany, c='r')
  




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
  
def play_movie():
  from pylab import gca, imshow, figure
  import pylab
  foldername = 'data/movies/test'
  frame = 0
  r0 = np.array([-0.7626858,   0.28330218,  0.17082515])
  while 1:

    depth = np.load('%s/depth_%05d.npz'%(foldername,frame)).items()[0][1].astype(np.float32)
    v,u = np.mgrid[135:332,335:485] #test
    #v,u = np.mgrid[231:371,264:434] # single
    x,y,z = normals.project(depth[v,u], u, v)
    n, weights = normals.fast_normals(depth[v,u],u.astype(np.float32),v.astype(np.float32))
    #np.savez('%s/normals_%05d'%(foldername,frame),v=v,u=u,n=n,weights=weights)
    r0, cc = normals.mean_shift_optimize(n,weights, np.array(r0))
    rot = expmap.axis2rot(r0)
    rotpts = normals.apply_rot(rot, np.dstack((x,y,z)))
    rotn = normals.apply_rot(rot, n)
    show_projections(rotpts,cc,weights,rotn)
    print r0 
    frame += 1

#depth, rgb = [x[1] for x in np.load('data/ceiling.npz').items()]
# rgb, depth = [x[1].astype(np.float32) for x in np.load('data/block1.npz').items()]
# v,u = np.mgrid[160:282,330:510]
# r0 = np.array([-0.7,-0.2,0])
rgb, depth = [x[1].astype(np.float32) for x in np.load('data/block2.npz').items()]
v,u = np.mgrid[231:371,264:434]
r0 = [-0.63, 0.68, 0.17]
#depth = np.load('data/movies/test/depth_00000.npz').items()[0][1]
#v,u = np.mgrid[175:332,365:485]
#r0 = [-0.7626858, 0.28330218, 0.17082515]
# depth = np.load('data/movies/single/depth_00000.npz').items()[0][1]
# v,u = np.mgrid[146:202,344:422]
# r0 = np.array([-0.7,-0.2,0])

x,y,z = normals.project(depth[v,u], u.astype(np.float32), v.astype(np.float32))

# sub sample
if not 'weights' in globals(): 
  n,weights = normals.fast_normals(depth[v,u],u.astype(np.float32),v.astype(np.float32))
#update(x,y,z,u,v,rgb)
#update(n[:,:,0],n[:,:,1],n[:,:,2], (u,v), rgb, (weights,weights,weights*0+1,weights*0+1))

#update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(weights,weights,weights*0+1,weights*0.3))
R,G,B = normals.color_axis(n)
normals.update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R+.5,G+.5,B+.5,R+G+B))
import cv

rotaxis, cc = normals.mean_shift_optimize(n,weights, np.array(r0))
rot = expmap.axis2rot(rotaxis)
rotpts = normals.apply_rot(rot, np.dstack((x,y,z)))
rotn = normals.apply_rot(rot, n)
show_projections(rotpts,cc,weights,rotn)


window.Refresh()
