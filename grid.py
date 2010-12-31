

def initialize_grid():
  GRIDRAD = 12
  
  global bounds
  bounds = (-GRIDRAD,0,-GRIDRAD),(GRIDRAD,8,GRIDRAD)
  b_width = [bounds[1][i]-bounds[0][i] for i in range(3)]
  
  global vote_grid, carve_grid
  vote_grid  = np.zeros(b_width)
  carve_grid = np.zeros(b_width)
  

def carve_background():
  # We can carve out points from 