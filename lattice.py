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
  a2 = np.arctan2(y.mean(),x.mean()) / (2*np.pi)
  return a2 * modulo if a2 >= 0 else (a2+1)*modulo
  
def color_axis(normals,d=0.1):
  #n = np.log(np.power(normals,40))
  X,Y,Z = [normals[:,:,i] for i in range(3)]
  cc = Y*Y+Z*Z, Z*Z+X*X, X*X+Y*Y
  cx = [np.max((1.0-(c/d*c/d),0*c),0) for c in cc]
  return [c * 0.8 + 0.2 for c in cx]
  
  
def voxels():
  global votes
  xv = (rotpts[cc[0]>0]-[meanx+mx,meany+my,meanz+mz])/[0.0158,0.0196,0.0158]-[.5,0,0]
  yv = (rotpts[cc[1]>0]-[meanx+mx,meany+my,meanz+mz])/[0.0158,0.0196,0.0158]-[0,.5,0]
  zv = (rotpts[cc[2]>0]-[meanx+mx,meany+my,meanz+mz])/[0.0158,0.0196,0.0158]-[0,0,.5]
  
  figure(4)
  clf();
  scatter(xv[:,0]+.5,xv[:,2])
  scatter(zv[:,0],zv[:,2]+.5,c='r')
  votes = np.floor(np.vstack((xv,zv)))
  scatter(votes[:,0],votes[:,2],c='g')
  pylab.draw()
  
  mins = votes.min(0)
  maxs = votes.max(0)
  global bins
  bins = [np.arange(mins[i],maxs[i]+2)-.5 for i in range(3)]
  global H
  H,_ = np.histogramdd(votes, bins)
  global legos
  legos = np.array(np.nonzero(H > 30)).transpose() + mins
  

def show_projections(rotpts,cc,weights,rotn):
  from pylab import figure, title, clf, gca, scatter, grid
  
  global meanx, meany, meanz, mx, my, mz
  meanx = circular_mean(rotpts[:,:,0][cc[0,:,:]>0],0.0158)
  meany = circular_mean(rotpts[:,:,1][cc[1,:,:]>0],0.0196)
  meanz = circular_mean(rotpts[:,:,2][cc[2,:,:]>0],0.0158)
  mx = np.floor(rotpts[:,:,0][weights>0].mean()/.0158)*.0158
  mz = np.floor(rotpts[:,:,2][weights>0].mean()/.0158)*.0158
  my = np.floor(rotpts[:,:,1][weights>0].mean()/.0196)*.0196

  R,G,B = color_axis(rotn); R = R[weights>0]; G = G[weights>0]; B = B[weights>0]
  update(rotpts[:,:,0][weights>0]-mx-meanx,
         rotpts[:,:,1][weights>0]-my-meany,
         rotpts[:,:,2][weights>0]-mz-meanz,COLOR=(R,G,B,R+G+B))
  window.Refresh()

  figure(1)
  clf();
  title('height vs Z')
  xlabel('Z (meters)')
  ylabel('Height/Y (meters)')
  xticks(np.arange(-10,10)*.0158 + mz)
  yticks(np.arange(-10,10)*.0196 + my)
  scatter(rotpts[:,:,2][weights>0]-meanz, rotpts[:,:,1][weights>0]-meany)
  scatter(rotpts[:,:,2][cc[2,:,:]>0]-meanz, rotpts[:,:,1][cc[2,:,:]>0]-meany,c='r')
  gca().set_xlim(gca().get_xlim()[::-1]) 
  grid('on')
  pylab.draw()
  
  figure(2)
  clf()
  title('Height vs X')
  xlabel('X (meters)')
  ylabel('Height/Y (meters)')
  xticks(np.arange(-10,10)*.0158 + mx)
  yticks(np.arange(-10,10)*.0196 + my)
  grid('on')
  scatter(rotpts[:,:,0][cc[2,:,:]>0]-meanx,rotpts[:,:,1][cc[2,:,:]>0]-meany)
  scatter(rotpts[:,:,0][cc[0,:,:]>0]-meanx,rotpts[:,:,1][cc[0,:,:]>0]-meany, c='r')
  pylab.draw()
  
  pylab.waitforbuttonpress(0.1)



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
    
if __name__ == "__main__":
  from visuals.normalswindow import PCLWindow
  if not 'window' in globals(): window = PCLWindow(size=(640,480))


  #depth, rgb = [x[1] for x in np.load('data/ceiling.npz').items()]
  # rgb, depth = [x[1].astype(np.float32) for x in np.load('data/block1.npz').items()]
  # rect = ((330,160),(510,282))
  # r0 = np.array([-0.7,-0.2,0])
  rgb, depth = [x[1].astype(np.float32) for x in np.load('data/block2.npz').items()]
  rect = ((264,231),(434,371))
  r0 = [-0.63, 0.68, 0.17]
  # depth = np.load('data/movies/test/depth_00000.npz').items()[0][1].astype(np.float32)
  # v,u = np.mgrid[175:332,365:485]
  # r0 = [-0.7626858, 0.28330218, 0.17082515]
  # depth = np.load('data/movies/single/depth_00000.npz').items()[0][1]
  # v,u = np.mgrid[146:202,344:422]
  # r0 = np.array([-0.7,-0.2,0])
  
  (l,t),(r,b) = rect
  v,u = np.mgrid[t:b,l:r]

  if 1:
    x,y,z = normals.project(depth[v,u], u.astype(np.float32), v.astype(np.float32))

    # sub sample
    n,weights = normals.normals_c(depth,rect)
    #update(x,y,z,u,v,rgb)
    #update(n[:,:,0],n[:,:,1],n[:,:,2], (u,v), rgb, (weights,weights,weights*0+1,weights*0+1))

    #update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(weights,weights,weights*0+1,weights*0.3))
    #R,G,B = normals.color_axis(n)
    #normals.update(n[:,:,0],n[:,:,1],n[:,:,2], COLOR=(R+.5,G+.5,B+.5,R+G+B))
    import cv

    rotaxis, cc = normals.mean_shift_optimize(n,weights, np.array(r0))
    rot = expmap.axis2rot(rotaxis)
    rotpts = normals.apply_rot(rot, np.dstack((x,y,z)))
    rotn = normals.apply_rot(rot, n)
    from pylab import *
    show_projections(rotpts,cc,weights,rotn)

  voxels()
  window.Refresh()
  
  from visuals.legowindow import PCLWindow as LW
  if not 'legowindow' in globals(): legowindow = LW()
  
  @legowindow.event
  def on_draw_axes():

    glPushMatrix()
    glColor(0,1,0,0.5)
    glTranslate(0,0,-1.5)
    glScale(0.0158,0.0196,0.0158)
    glBegin(GL_QUADS)
    for x,y,z in legos:
      for q in     [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
                    [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
                    [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
                    [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
                    [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
                    [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]:
        normal = np.cross(np.subtract(q[0],q[1]),np.subtract(q[0],q[2]))
        glColor(*np.abs(normal))
        for i,j,k in q:
          glVertex(x+i,y+j,z+k)
        
    glEnd()
    glPopMatrix()
    glDisable(GL_LIGHTING)
    glDisable(GL_COLOR_MATERIAL)
  legowindow.Refresh()
      

