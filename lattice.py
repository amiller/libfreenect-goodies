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

LH = 0.0180
LW = 0.0160

def circular_mean(data, modulo):
  angle = data / modulo * np.pi * 2
  y = np.sin(angle)
  x = np.cos(angle)
  a2 = np.arctan2(y.mean(),x.mean()) / (2*np.pi)
  if np.isnan(a2): a2 = 0
  return a2 * modulo if a2 >= 0 else (a2+1)*modulo
  
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
  
def lattice2(n,w,depth,rgb,mat_,tableplane,rect):
  (l,t),(r,b) = rect
  mat = np.eye(4,dtype='f')
  mat[:3,:3] = mat_
  global rotmat
  rotmat = mat
  matxyz = np.dot(mat, calibkinect.xyz_matrix())
  v,u = np.mgrid[t:b,l:r]
  X,Y,Z = project(u,v,depth[t:b,l:r], matxyz)
  XYZ = np.dstack((X,Y,Z))
  dx = np.dot(n,mat[0,:3])
  dy = np.dot(n,mat[1,:3])
  dz = np.dot(n,mat[2,:3])
  global cx,cy,cz
  cx,cy,cz = color_axis(dx,dy,dz,w)
  
  global meanx, meany, meanz
  global mx, my, mz  
  mx = np.floor(X[w>0].mean()/LW)*LW
  my = 0#my = np.floor(Y[w>0].mean()/LH)*LH
  mz = np.floor(Z[w>0].mean()/LW)*LW
  
  meanx = circular_mean(X[cx>0],LW)
  meany = -tableplane[3]
  meanz = circular_mean(Z[cz>0],LW)
  

  # Mark which blocks correspond to each pixel. We need to divide by the
  # blockheight to be in block coordinates. Since surface pixels are on 
  # the boundary between two blocks, we need to use the normal direction
  # to tell which block on the boundary it is.
  global votes
  xv = (XYZ[cx>0]-[meanx+mx,meany+my,meanz+mz])/[LW,LH,LW]
  xv[:,0] -= np.sign(dx[cx>0])*.5
  #yv = (XYZ[cy>0]-[meanx+mx,meany+my,meanz+mz])/[LW,LH,LW]-[0,.5,0]
  zv = (XYZ[cz>0]-[meanx+mx,meany+my,meanz+mz])/[LW,LH,LW]
  zv[:,2] -= np.sign(dz[cz>0])*.5
  
  votes = np.floor(np.vstack((xv,zv)))

  
  # We also want to mark which pixels have been occluded. This can be 
  # about volumes rather than boundaries, so we don't need to use the
  # surface normals. It's ok if the detected blocks are marked as occluded.
  # 
  global bgM
  bgM = project(u,v,preprocess.background[t:b,l:r], matxyz)
  occ_sample_alpha = np.random.rand(b-t,r-l)*0.9
  occ_mask = (cx>0)|(cz>0)
  global Xs,Ys,Zs
  Xs = (X*occ_sample_alpha + bgM[0]*(1-occ_sample_alpha)-(meanx+mx))/LW
  Ys = (Y*occ_sample_alpha + bgM[1]*(1-occ_sample_alpha)-(meany+my))/LH
  Zs = (Z*occ_sample_alpha + bgM[2]*(1-occ_sample_alpha)-(meanz+mz))/LW
  
  global occ_votes
  occ_votes = np.floor(np.dstack((Xs[occ_mask],Ys[occ_mask],Zs[occ_mask]))).reshape(-1,3)
  
  Xo,Yo,Zo = project(u,v,depth[t:b,l:r], calibkinect.xyz_matrix())
  R,G,B = cx,cy,cz; R = R[w>0]; G = G[w>0]; B = B[w>0]
  update(*3*[np.array([[0]])])
  #update(Xo[w>0],Yo[w>0],Zo[w>0],COLOR=(R,G,B,R+G+B))

  # update(X[w>0]-meanx-mx,
  #       Y[w>0]-meany-my,
  #       Z[w>0]-meanz-mz,COLOR=(R,G,B,R+G+B))
  window.Refresh()
  
  
  if 0:
    figure(1)
    clf();
    title('height vs Z')
    xlabel('Z (meters)')
    ylabel('Height/Y (meters)')
    xticks(np.arange(-100,100)*LW)
    yticks(np.arange(-100,100)*LH)
    scatter(Z[ w>0]-meanz,  Y[w>0]-meany)
    scatter(Z[cz>0]-meanz, Y[cz>0]-meany,c='r')
    gca().set_xlim(gca().get_xlim()[::-1]) 
    grid('on')
    pylab.draw()

    figure(2)
    clf()
    title('Height vs X')
    xlabel('X (meters)')
    ylabel('Height/Y (meters)')
    xticks(np.arange(-100,100)*LW)
    yticks(np.arange(-100,100)*LH)
    grid('on')
    scatter(X[ w>0]-meanx, Y[w>0]-meany)
    scatter(X[cx>0]-meanx, Y[cx>0]-meany, c='r')
    pylab.draw()
    
    figure(4)
    clf();
    scatter(xv[:,0],xv[:,2])
    scatter(zv[:,0],zv[:,2],c='r')
    scatter(votes[:,0],votes[:,2],c='g')
    pylab.draw()
  
  mins = votes.min(0)
  maxs = votes.max(0)
  global bins
  bins = [np.arange(mins[i],maxs[i]+2)-.5 for i in range(3)]
  global H
  
  import colors
  R,G,B = colors.project_colors(depth,rgb,rect)
  col = colors.choose_colors(R,G,B)
  colv = np.hstack((col[cx>0], col[cz>0]))
  
  H,_ = np.histogramdd(votes, bins)
  global cH
  cH_ = [histogramdd(votes,bins,weights=colv==i)[0] for i in range(4)]
  cH = np.argmax(cH_,0)
  
    
  occmins = occ_votes.min(0)
  occmaxs = occ_votes.max(0)
  occbins = [np.arange(occmins[i],occmaxs[i]+2)-.5 for i in range(3)]
  global occH
  occH,_ = np.histogramdd(occ_votes, occbins)

  
  global legos, legocolors, occblocks
  legos = np.array(np.nonzero(H > 30)).transpose() + mins  
  occblocks = np.array(np.nonzero(occH > 0)).transpose() + occmins
  legocolors = cH[H>30].flatten()
  
def voxels():
  global votes
  xv = (rotpts[cc[0]>0]-[meanx+mx,meany+my,meanz+mz])/[LW,LH,LW]-[.5,0,0]
  yv = (rotpts[cc[1]>0]-[meanx+mx,meany+my,meanz+mz])/[LW,LH,LW]-[0,.5,0]
  zv = (rotpts[cc[2]>0]-[meanx+mx,meany+my,meanz+mz])/[LW,LH,LW]-[0,0,.5]
  
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
  global legos, legocolors
  legos = np.array(np.nonzero(H > 30)).transpose() + mins
  

def show_projections(rotpts,cc,weights,rotn):
  global meanx, meany, meanz, mx, my, mz
  meanx = circular_mean(rotpts[:,:,0][cc[0,:,:]>0],LW)
  meany = circular_mean(rotpts[:,:,1][cc[1,:,:]>0],LH)
  meanz = circular_mean(rotpts[:,:,2][cc[2,:,:]>0],LW)
  mx = np.floor(rotpts[:,:,0][weights>0].mean()/LW)*LW
  mz = np.floor(rotpts[:,:,2][weights>0].mean()/LW)*LW
  my = np.floor(rotpts[:,:,1][weights>0].mean()/LH)*LH

  R,G,B = color_axis(*np.rollaxis(rotn,2),w=weights); R = R[weights>0]; G = G[weights>0]; B = B[weights>0]
  update(rotpts[:,:,0][weights>0]-mx-meanx,
         rotpts[:,:,1][weights>0]-my-meany,
         rotpts[:,:,2][weights>0]-mz-meanz,COLOR=(R,G,B,R+G+B))
  window.Refresh()

  figure(1)
  clf();
  title('height vs Z')
  xlabel('Z (meters)')
  ylabel('Height/Y (meters)')
  xticks(np.arange(-10,10)*LW + mz)
  yticks(np.arange(-10,10)*LH + my)
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
  xticks(np.arange(-10,10)*LW + mx)
  yticks(np.arange(-10,10)*LH + my)
  grid('on')
  scatter(rotpts[:,:,0][cc[2,:,:]>0]-meanx,rotpts[:,:,1][cc[2,:,:]>0]-meany)
  scatter(rotpts[:,:,0][cc[0,:,:]>0]-meanx,rotpts[:,:,1][cc[0,:,:]>0]-meany, c='r')
  pylab.draw()
  
  pylab.waitforbuttonpress(0.1)



def update(X,Y,Z,UV=None,rgb=None,COLOR=None,AXES=None):
  global window
  window.lookat = preprocess.tablemean
  
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
    
    
from visuals.normalswindow import NormalsWindow
if not 'window' in globals(): 
  window = NormalsWindow(size=(640,480))
  legolist = glGenLists(1)
  rotmat = np.eye(4)
  legos = []
  legocolors = []
  occblocks = []
  meanx = meany = meanz = mx = my = mz = 0
  
def build_list():
  glNewList(legolist, GL_COMPILE) # Build a list for the frustum
  for q in     [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
                [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
                [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
                [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
                [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
                [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]:
      glBegin(GL_LINE_STRIP)
      glVertex(*q[0]); glVertex(*q[1]); glVertex(*q[2]); glVertex(*q[3]); glVertex(*q[0])
      glEnd()
  glEndList()

build_list()

@window.event
def on_draw_axes():
  
  glBegin(GL_QUADS)
  glColor(0.6,0.7,0.7,1)
  for x,y,z in preprocess.boundptsM:
    glVertex(x,y,z)
  glEnd()
  

  glPushMatrix()
  #glTranslate(0,0,-1.5)
  glMultMatrixf(rotmat)
  glTranslate((meanx+mx),(meany+my),(meanz+mz))
  glScale(LW,LH,LW)
  glPolygonOffset(1.0,0.2)
  #glEnable(GL_LINE_SMOOTH)
  #glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
  glEnable(GL_BLEND)
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
  glEnable(GL_POLYGON_OFFSET_FILL)
  glBegin(GL_QUADS)
  for (x,y,z),cind in zip(legos,legocolors):
    for q in     [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
                  [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
                  [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
                  [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
                  [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
                  [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]:
      normal = np.cross(np.subtract(q[0],q[1]),np.subtract(q[0],q[2]))
      #glColor(*np.abs(normal).tolist()+[0.3])
      glColor(*colors.colormap[cind])
      for i,j,k in q:
        glVertex(x+i,y+j,z+k)
  glColor(0.2,0.2,0.3,0.3)
  for x,y,z in occblocks:
    for q in     [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
                  [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
                  [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
                  [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
                  [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
                  [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]:
      for i,j,k in q:
        glVertex(x+i,y+j,z+k)
  glEnd()
  glDisable(GL_POLYGON_OFFSET_FILL)
  
  glColor(1,1,1,0.8)
  for x,y,z in legos:
    glPushMatrix()
    glTranslate(x,y,z)
    glCallList(legolist)
    glPopMatrix()
  glPopMatrix()
  glDisable(GL_LIGHTING)
  glDisable(GL_COLOR_MATERIAL)
  

window.Refresh()


    
if __name__ == "__main__":
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

  if 0 or not 'n' in globals():
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
    
    





