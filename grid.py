from pylab import *
import pylab
import numpy as np
import main
import preprocess
from OpenGL.GL import *
import lattice

def initialize():
  GRIDRAD = 12
  
  global bounds
  bounds = (-GRIDRAD,0,-GRIDRAD),(GRIDRAD,8,GRIDRAD)
  b_width = [bounds[1][i]-bounds[0][i] for i in range(3)]
  
  global vote_grid, carve_grid
  vote_grid  = np.zeros(b_width)
  carve_grid = np.zeros(b_width)
  
if not 'vote_grid' in globals(): initialize()


def drift_correction(new_votes):
  """
  Using the current values for vote_grid and carve_grid, and the new histograms
  generated from the newest frame, find the translation between old and new 
  (in a 3x3 neighborhood, only considering jumps of 1 block) that minimizes 
  an error function between them.
  """
  def error(t):
    """ 
    t: x,y
    The error function is the number of error blocks that fall in a carved region.
    """
    nv = np.roll(new_votes, t[0], 0)
    nv = np.roll(nv, t[2], 2)
    return np.sum((nv>10) & (carve_grid>=1))
    
  t = [(x,y,z) for x in [0,-1,1] for y in [0] for z in [0,-1,1]]
  vals = [error(_) for _ in t]
  return t[np.argmin(vals)]

  
def image_bounds():
  """
  Find the ROI corresponding to the imaged grid.
  """
  # Project each of the 8 points in the bounds
  from main import LH,LW
  import itertools
  import calibkinect
  p = np.array(list(itertools.product(*zip(*bounds)))) * [LW,LH,LW]
  pz = np.hstack((p, 8*[[1]]))
  impts = np.dot(np.linalg.inv(np.dot(lattice.modelmat, calibkinect.xyz_matrix())), pz.transpose())
  mx = impts[0] / impts[3]
  my = impts[1] / impts[3]
  return (mx.min(), my.min()),(mx.max(),my.max())
  
def carve_background(depth, XYZ):
  """
  We can carve out points in the grid as long as the points match the background
  and they fit in the grid. We have to randomly sample distances. 
  #TODO We can probably also randomly sample points themselves.
  #TODO Use OpenGL to draw a mask of the grid.
  """
  
  from preprocess import background
  from lattice import modelmat
  from main import LW,LH
  import calibkinect
  global mask
  mask = (depth<2047)&(np.abs(depth+4>background))
  
  N = 10000
  samp = np.ones((N,4),np.int32)
  samp[:,:2] = np.floor(np.random.rand(N,2) * [640,480])
  samp[:,2] = depth[samp[:,1],samp[:,0]]
  bgm = np.dot(np.dot(modelmat,calibkinect.xyz_matrix()), samp.transpose())
  bgm = bgm[:3] / bgm[3]
  alpha = np.random.rand(N)
  orig = lattice.modelmat[:3,3].reshape(3,1)
  xyz = ((alpha) * orig + (1-alpha) * bgm).transpose() / [LW,LH,LW]
  
  m = np.all((xyz >= bounds[0]) & (xyz <= bounds[1]),1) & mask[samp[:,1],samp[:,0]]
  m = m & np.all((xyz%1 > 0.1) & (xyz%1 < 0.9),1)

  bins = [np.arange(bounds[0][i],bounds[1][i]+1) for i in range(3)]
  H,_ = np.histogramdd(xyz[m], bins)

  
  # We also want to mark which pixels have been occluded. This can be 
  # about volumes rather than boundaries, so we don't need to use the
  # surface normals. It's ok if the detected blocks are marked as occluded.
  mask = (cx>0)|(cz>0)
  XYZo = np.array([x[mask] for x in XYZ])
  occ_alpha = np.random.rand(XYZo.shape[1])*0.98
  XYZs = (orig*(1-occ_alpha) + XYZo*(occ_alpha)) / np.reshape([LW,LH,LW],(3,1))
  H2,_ = np.histogramdd(XYZs.transpose(), bins)
  
  global carve_grid
  carve_grid = carve_grid * 1 + H + H2

  
def threshold_votes():
  """
  Look through the voting spaces and threshold the elements. The result is a
  list of lego blocks we can draw.
  """

  global legos
  legos = np.array(np.nonzero(vote_grid > 30)).transpose() + bounds[0]

  window.lookat = preprocess.tablemean
  window.Refresh()
  
def add_votes(XYZ, dXYZ, cXYZ):
  """
  Mark which blocks correspond to each pixel. We need to divide by the
  blockheight to be in block coordinates. Since surface pixels are on 
  the boundary between two blocks, we need to use the normal direction
  to tell which block on the boundary it is.
  """
  from main import LH,LW
  global X,Y,Z
  global cx,cz,dx,dz
  X,Y,Z = XYZ
  dx,_,dz = dXYZ
  cx,_,cz = cXYZ
  xv = (X[cx>0],Y[cx>0],Z[cx>0]) / np.reshape([LW,LH,LW],(3,1))
  zv = (X[cz>0],Y[cz>0],Z[cz>0]) / np.reshape([LW,LH,LW],(3,1))
  
  xv[0,:] -= np.sign(dx[:,cx>0])*.5
  zv[2,:] -= np.sign(dz[:,cz>0])*.5
  
  votes = np.hstack((xv,zv))
  bins = [np.arange(bounds[0][i],bounds[1][i]+1) for i in range(3)]
  
  window.lookat = preprocess.tablemean
  window.Refresh()

  global vote_grid
  global carve_grid  
  vg,_ = np.histogramdd(votes.transpose(), bins)
  #vote_grid *= 0.8
  
  bx,_,bz = drift_correction(vg)
  if 0 and (bx,bz) != (0,0): 
    vote_grid  = np.roll(np.roll( vote_grid, -bx, 0), -bz, 2)
    carve_grid = np.roll(np.roll(carve_grid, -bx, 0), -bz, 2)
    #vg = np.roll(np.roll( vg, bx, 0), bz, 2)
    #lattice.modelmat[0,3] -= bx*LW
    #lattice.modelmat[2,3] -= bz*LW
    print "drift detected:", bx,bz
  
  
  vote_grid += vg
  if 1:
    # Only show the current frame
    vote_grid = vg
    carve_grid = carve_grid * 0


  if 0:
    figure(1)
    clf();
    title('height vs Z')
    xlabel('Z (meters)')
    ylabel('Height/Y (meters)')
    xticks(np.arange(-100,100)*LW)
    yticks(np.arange(-100,100)*LH)
    scatter(Z[(cz>0)|(cx>0)],  Y[(cz>0)|(cx>0)])
    scatter(Z[cz>0], Y[cz>0],c='r')
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
    scatter(X[(cz>0)|(cx>0)], Y[(cz>0)|(cx>0)])
    scatter(X[cx>0], Y[cx>0], c='r')
    pylab.draw()
  window.Refresh()
  return lattice.modelmat[:3,:4]


  import colors
  R,G,B = colors.project_colors(depth,rgb,rect)
  col = colors.choose_colors(R,G,B)
  colv = np.hstack((col[cx>0], col[cz>0]))

  global cH
  cH_ = [histogramdd(votes,bins,weights=colv==i)[0] for i in range(4)]
  cH = np.argmax(cH_,0)

  global legocolors
  legocolors = cH[H>30].flatten()
  
 


from visuals.camerawindow import CameraWindow
if not 'window' in globals(): 
  window = CameraWindow(title='Occupancy Grid', size=(640,480))
  window.clearcolor = [1,1,1,0]
  legos = []
  legocolors = []
  occblocks = []
  legolist = glGenLists(1)


window.Refresh()


def grid_vertices(grid):
  """
  Given a boolean voxel grid, produce a list of vertices and indices 
  for drawing quads or line strips in opengl
  """
  q = [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
       [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
       [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
       [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
       [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
       [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]
  
  normal = [np.cross(np.subtract(qz[0],qz[1]),np.subtract(qz[0],qz[2])) for qz in q]
  
  blocks = np.array(grid.nonzero()).transpose().reshape(-1,1,3)
  q = np.array(q).reshape(1,-1,3)
  vertices = (q + blocks).reshape(-1,3)
  normals = np.tile(normal, (len(blocks),4)).reshape(-1,3)
  line_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,1,2,2,3,3,0]
  quad_inds = np.arange(0,len(blocks)*6).reshape(-1,1)*4 + [0,1,2,3]
  
  return vertices, normals, line_inds, quad_inds


@window.event
def on_draw_axes():
  import lattice
  from main import LH,LW
  
  if not 'modelmat' in lattice.__dict__: return
  
  glPolygonOffset(1.0,0.2)
  glEnable(GL_POLYGON_OFFSET_FILL)

  # Draw the gray table
  glBegin(GL_QUADS)
  glColor(0.6,0.7,0.7,1)
  for x,y,z in preprocess.boundptsM:
    glVertex(x,y,z)
  glEnd()
  
  
  glPushMatrix()
  glMultMatrixf(np.linalg.inv(lattice.modelmat).transpose())
  glScale(LW,LH,LW)
  
  

    
  #glEnable(GL_LINE_SMOOTH)
  #glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
  glEnable(GL_BLEND)
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
  #for (x,y,z),cind in zip(legos,legocolors):
  
  #carve_verts, _, line_inds, quad_inds = grid_vertices((carve_grid<1))
  carve_verts, _, line_inds, quad_inds = grid_vertices((carve_grid<1)&(vote_grid>30))
  
  glPushMatrix()
  glTranslate(*bounds[0])
  
  # Draw the carved out pixels
  glColor(0.1,0.1,0.4,0.5)
  glEnableClientState(GL_VERTEX_ARRAY)
  glVertexPointeri(carve_verts)
  glDrawElementsui(GL_LINES, line_inds)
  #glDrawElementsui(GL_QUADS, quad_inds)
  glDisableClientState(GL_VERTEX_ARRAY)
  
  
  
  
  #  Draw the filled in surface faces of the legos 
  verts, norms, line_inds, quad_inds = grid_vertices((vote_grid>30)&(carve_grid<1))
  glEnableClientState(GL_VERTEX_ARRAY)
  glVertexPointeri(verts)  
  glEnableClientState(GL_COLOR_ARRAY)
  glColorPointerf(np.abs(norms))
  glDrawElementsui(GL_QUADS, quad_inds)
  glDisableClientState(GL_COLOR_ARRAY)
  glColor(1,1,1)
  glDrawElementsui(GL_LINES, line_inds)
  glDisableClientState(GL_VERTEX_ARRAY)
  glPopMatrix()
  
  # Draw the shadow blocks (occlusions)
  glDisable(GL_POLYGON_OFFSET_FILL)


  # Draw the outlines for the lego blocks
  glColor(1,1,1,0.8)

  glDisable(GL_LIGHTING)
  glDisable(GL_COLOR_MATERIAL)

  # Draw the axes for the model coordinate space
  glLineWidth(3)
  glBegin(GL_LINES)
  glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
  glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
  glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
  glEnd()

  # Draw a grid for the model coordinate space
  glLineWidth(1)
  glBegin(GL_LINES)
  GR = 8
  glColor3f(0.2,0.2,0.4)
  for j in range(0,1):
    for i in range(-GR,GR+1):
      glVertex(i,j,-GR); glVertex(i,j,GR)
      glVertex(-GR,j,i); glVertex(GR,j,i)
  glEnd()
  glPopMatrix()

  