import numpy as np
import main
import preprocess

def initialize_grid():
  GRIDRAD = 12
  
  global bounds
  bounds = (-GRIDRAD,0,-GRIDRAD),(GRIDRAD,8,GRIDRAD)
  b_width = [bounds[1][i]-bounds[0][i] for i in range(3)]
  
  global vote_grid, carve_grid
  vote_grid  = np.zeros(b_width)
  carve_grid = np.zeros(b_width)
  
def image_bounds():
  """
  Find the ROI corresponding to the imaged grid.
  """
  pass

def carve_background():
  # We can carve out points in the grid as long as the points match the background
  # and they fit in the grid. We have to randomly sample distances. 
  #TODO We can probably also randomly sample points themselves.
  #TODO Use OpenGL to draw a mask of the grid.
  
  # Points we know aren't on a lego surface
  mask = (main.depth<2047)&(np.abs(main.depth+4>preprocess.background))
  