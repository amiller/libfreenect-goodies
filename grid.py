from pylab import *
import pylab
import numpy as np
import main
import preprocess
import opencl
from OpenGL.GL import *
import lattice
import carve

GRIDRAD = 8
bounds = (-GRIDRAD,0,-GRIDRAD),(GRIDRAD,8,GRIDRAD)

import ctypes
from ctypes import POINTER as PTR, c_byte, c_size_t, c_float
speedup_ctypes = ctypes.cdll.LoadLibrary('speedup_ctypes.so')
speedup_ctypes.histogram.argtypes = [PTR(c_byte), PTR(c_float), PTR(c_float), 
                                     c_size_t, c_size_t, c_size_t, c_size_t]
                                     
guidecarve_grid,guidevote_grid = np.load('data/2001model.npy')
def fix_guide():
  # Using the guides, and the current keycarve, prepare blocks for display
  global solid_blocks, shadow_blocks, wire_blocks
  notdone = (keyvote_grid<=30)&(guidevote_grid>30)
  matching = (keyvote_grid>30)&(guidevote_grid>30)
  next = 0*notdone
  if np.any(notdone):
    first = np.nonzero(notdone)[1].min()
    next[:,first,:] = notdone[:,first,:]

    #solid_blocks = carve.grid_vertices((guidevote_grid>30))
    solid_blocks = carve.grid_vertices((keyvote_grid>30)&~next, 0.4)
    shadow_blocks = carve.grid_vertices(next)
    wire_blocks = carve.grid_vertices((carve_grid>10))
  else:
    #solid_blocks = carve.grid_vertices((guidevote_grid>30))
    solid_blocks = carve.grid_vertices((keyvote_grid>30)&~next, 1.0)
    shadow_blocks = carve.grid_vertices(next)
    wire_blocks = carve.grid_vertices((carve_grid>10))


def initialize():    
  b_width = [bounds[1][i]-bounds[0][i] for i in range(3)]
  global vote_grid, carve_grid, keyvote_grid, keycarve_grid
  keyvote_grid = np.zeros(b_width)
  keycarve_grid = np.zeros(b_width)
  vote_grid  = np.zeros(b_width)
  carve_grid = np.zeros(b_width)
  
  global shadow_blocks, solid_blocks, wire_blocks
  shadow_blocks = None
  solid_blocks = None
  wire_blocks = None
  
if not 'vote_grid' in globals(): initialize()


def refresh():
  global solid_blocks, shadow_blocks, wire_blocks
  
  solid_blocks = carve.grid_vertices((vote_grid>30))
  shadow_blocks = carve.grid_vertices((carve_grid>10)&(vote_grid>30))
  wire_blocks = carve.grid_vertices((carve_grid>10))
  
  window.Refresh()

def drift_correction(new_votes, new_carve):
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
    nc = np.roll(new_carve, t[0], 0)
    nc = np.roll(nc, t[2], 2)
    return np.sum(np.minimum(nc,vote_grid) + np.minimum(nv,carve_grid))
    
  t = [(x,y,z) for x in [0,-1,1] for y in [0] for z in [0,-1,1]]
  vals = [error(_) for _ in t]
  #print vals
  return t[np.argmin(vals)]

  
def add_votes_opencl(xfix,zfix):
  gridmin = np.zeros((4,),'f')
  gridmax = np.zeros((4,),'f')
  gridmin[:3] = bounds[0]
  gridmax[:3] = bounds[1]

  global occH, vacH
  global carve_grid,vote_grid
  
  opencl.compute_gridinds(xfix,zfix, main.LW, main.LH, gridmin, gridmax)
  global gridinds
  gridinds = opencl.get_gridinds()
  
  inds = gridinds[gridinds[:,0,3]!=0,:,:3]
  
  if len(inds) == 0: return
  
  bins = [np.arange(0,bounds[1][i]-bounds[0][i]+1)-0.5 for i in range(3)]
  occH,_ = np.histogramdd(inds[:,0,:], bins)
  vacH,_ = np.histogramdd(inds[:,1,:], bins)
  bx,_,bz = drift_correction(occH, vacH)
  if (bx,bz) != (0,0):
    lattice.modelmat[0,3] += bx*main.LW
    lattice.modelmat[2,3] += bz*main.LW
    print "drift detected:", bx,bz
  return lattice.modelmat[:3,:4]
  wx,wy,wz = [bounds[1][i]-bounds[0][i] for i in range(3)]
  

  

    # occH = np.zeros((wx,wy,wz),'f')
    # vacH = np.zeros((wx,wy,wz),'f')
    # speedup_ctypes.histogram(gridinds.ctypes.data_as(PTR(c_byte)), 
    #                          occH.ctypes.data_as(PTR(c_float)),
    #                          vacH.ctypes.data_as(PTR(c_float)), 
    #                          np.int32(gridinds.shape[0]), 
    #                          np.int32(wx), np.int32(wy), np.int32(wz))
    
  sums = np.zeros((3,3),'f')
  window.lookat = preprocess.bgL['tablemean']
  window.upvec = preprocess.bgL['tableplane'][:3]
  window.Refresh()
  
  speedup_ctypes.histogram_error(vote_grid.ctypes.data, carve_grid.ctypes.data, 
                                  occH.ctypes.data, vacH.ctypes.data, 
                                 sums.ctypes.data_as(PTR(c_float)), wx, wy, wz);
                                 
  t = np.argmin(sums)
  ts = [(z,x) for x in (-1,0,1) for z in (-1,0,1)]
  bx, bz = ts[t]
  print sums
  
  if sums.flatten()[t] < sums.flatten()[4] - 2: 
    #vote_grid  = np.roll(np.roll( vote_grid, -bx, 0), -bz, 2)
    #carve_grid = np.roll(np.roll(carve_grid, -bx, 0), -bz, 2)
    #occH = np.roll(np.roll(occH, bx, 0), bz, 2)
    #vacH = np.roll(np.roll(vacH, bx, 0), bz, 2)
    #vg = np.roll(np.roll( vg, bx, 0), bz, 2)
    lattice.modelmat[0,3] += bx*main.LW
    lattice.modelmat[2,3] += bz*main.LW
    print "drift detected:", bx,bz
  
  return lattice.modelmat[:3,:4]

  

  if 0:
    carve_grid = np.maximum(vacH,carve_grid)
    vote_grid = np.maximum(occH,vote_grid)
    vote_grid *= (carve_grid<30)
  
  refresh()

  if 0:
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
  #window.clearcolor = [0.0,0.0,0.0,0]

window.Refresh()


@window.event
def on_draw_axes():
  import lattice
  from main import LH,LW
  
  if not 'modelmat' in lattice.__dict__: return
  
  glPolygonOffset(1.0,0.2)
  glEnable(GL_POLYGON_OFFSET_FILL)

  # Draw the gray table
  if 1:
    glBegin(GL_QUADS)
    glColor(0.6,0.7,0.7,1)
    for x,y,z in preprocess.bgL['boundptsM']:
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
  
  glPushMatrix()
  glTranslate(*bounds[0])
  
  # Draw the carved out pixels
  glColor(1.0,0.1,0.1,0.9)
  glEnableClientState(GL_VERTEX_ARRAY)
  
  if 0 and wire_blocks:
    carve_verts, _, line_inds, _ = wire_blocks
    glVertexPointeri(carve_verts)
    glDrawElementsui(GL_LINES, line_inds)
  
  if 1 and shadow_blocks:
    carve_verts, _, _, quad_inds = shadow_blocks
    glVertexPointeri(carve_verts)      
    glDrawElementsui(GL_QUADS, quad_inds)
  glDisableClientState(GL_VERTEX_ARRAY)
  
  
  
  #  Draw the filled in surface faces of the legos 
  #verts, norms, line_inds, quad_inds = grid_vertices((vote_grid>30)&(carve_grid<30))
  if solid_blocks:
    verts, norms, line_inds, quad_inds = solid_blocks
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointeri(verts)  
    #glColor(0.3,0.3,0.3)
    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointerf(np.abs(norms))
    glDrawElementsui(GL_QUADS, quad_inds)
    glDisableClientState(GL_COLOR_ARRAY)
    glColor(1,1,1,1)
    #glColor(0.1,0.1,0.1)
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
  if 1:
    glLineWidth(1)
    glBegin(GL_LINES)
    GR = GRIDRAD
    glColor3f(0.2,0.2,0.4)
    for j in range(0,1):
      for i in range(-GR,GR+1):
        glVertex(i,j,-GR); glVertex(i,j,GR)
        glVertex(-GR,j,i); glVertex(GR,j,i)
    glEnd()
  glPopMatrix()

  