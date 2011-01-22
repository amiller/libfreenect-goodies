import multiprocessing

def reset():
  global pool, track_f, keyframe_f, found_keyframe
  if 'pool' in globals(): 
    pool.terminate()
  track_f = None
  keyframe_f = None
  found_keyframe = False
  pool = multiprocessing.Pool(processes=2)  
reset()

import numpy as np
import carve

prevstate = None

def send_test(depth):
  return [np.sqrt(depth.astype('f')).mean() for i in range(10000)]
  
def get_blocks(vote_grid, carve_grid):
  solid_blocks = carve.grid_vertices((vote_grid>30))
  shadow_blocks = carve.grid_vertices((carve_grid>10)&(vote_grid>30))
  wire_blocks = carve.grid_vertices((carve_grid>10))
  
  return dict(solid_blocks=solid_blocks, 
              shadow_blocks=shadow_blocks,
              wire_blocks=wire_blocks)

def get_update():
  result = {}
  global track_f, keyframe_f
  
  if track_f and track_f.ready(): 
    print 'got track'
    result.update(track_f.get())
    track_f = None
    
  if keyframe_f and keyframe_f.ready():
    #print 'got keyframe'
    result.update(keyframe_f.get())
    global found_keyframe
    if result.has_key('keyvote_grid'):
      found_keyframe = True
    if result.has_key('score'):
      pass
      #print result['score']
    keyframe_f = None
  
  return result


def _update_track(xyzf, LW, LH, bounds, meanx, meanz, 
                  vote_grid, carve_grid,
                  keyvote_grid, keycarve_grid):
                  
  # Copy the item
  if 0:
    occH,vacH = carve.add_votes(xyzf, LW, LH, bounds, meanx, meanz)
  
    vote_grid = np.maximum(np.maximum(occH, vote_grid), keyvote_grid)
    carve_grid = np.maximum(np.maximum(vacH, carve_grid), keycarve_grid)
  else:
    vote_grid = keyvote_grid
    carve_grid = keycarve_grid

  vote_grid *= (keycarve_grid<=10)
  carve_grid *= (keyvote_grid<=10)
  
  result = dict(vote_grid=vote_grid,
                carve_grid=carve_grid)
  result.update(get_blocks(vote_grid, carve_grid))
  return result
  
  
def _update_keyframe(xyzf, LW, LH, bounds, meanx, meanz,
                     keyvote_grid, keycarve_grid, 
                     depthL, depthR, matL, matR, KK, 
                     maskL, maskR, rectL, rectR,
                     rgbL, rgbR):
  def from_rect(m,rect):
    (l,t),(r,b) = rect
    return m[t:b,l:r]
                     
  # Check if we should process a key frame this turn.
  global prevstate
  if prevstate:
    cleancount, rect, depth = prevstate
    # Compare the frames
    newdepth = from_rect(depthL, rect)
    mask = (depth<2047) & (newdepth<2047)
    diff = np.abs(depth.astype('f')-newdepth.astype('f'))
    score = (diff*mask).mean()
    
    if score < 1.5: cleancount += 1
    else: cleancount = 0
    
    # Store the state
    prevstate =  cleancount, rectL, from_rect(depthL, rectL)
    
    if cleancount == 6:
      prevstate = 0, rectL, from_rect(depthL, rectL)
      
      occH,vacH = carve.add_votes(xyzf, LW, LH, bounds, meanx, meanz)
      
      #ccL = colors.project_colors(depthL, rgbL, rectL)
      # build the histogram
      #bins_
      # threshold the histogram
      #cH = colors.choose_colors(*ccL)
      
      #ccR = colors.project_colors(depthR, rgbR, rectR)
      
      
      keycarve_grid = np.maximum(keycarve_grid, vacH)        
      keyvote_grid = np.maximum(occH,keyvote_grid)
      
      if 0:    
        HL = carve.carve_background(depthL, LW, LH, bounds, matL, KK)
        HR = carve.carve_background(depthR, LW, LH, bounds, matR, KK)
        
        keycarve_grid = np.maximum(keycarve_grid, (HR+HL)*30)
    
      # Merge the grids
      keycarve_grid *= (occH<30)
      
      import scipy.ndimage
      labels,nlabels = scipy.ndimage.label(keyvote_grid>=30,np.ones((3,3,3)))
      for i in range(1,nlabels+1):
        if not np.any(labels[:,0,:]==i):
          keyvote_grid[labels==i] = 0

        
      
      result = dict(keyvote_grid=keyvote_grid, keycarve_grid=keycarve_grid)
      result.update(_update_track(xyzf, LW, LH, bounds, meanx, meanz, 
                        keyvote_grid, keycarve_grid,
                        keyvote_grid, keycarve_grid))
      return result
        
    else:
      return dict(score=score)

  else: 
    prevstate = 0, rectL, from_rect(depthL, rectL)
    
  return {}
  
  

  
  # Return the results, and the vertices
  #return _, _, keyvote_grid, keycarve_grid,
  
def update_track(*args):
  # Check if the pool is ready for a new items
  global track_f
  if not track_f:
    track_f = pool.apply_async(_update_track, args=args)
  
def update_keyframe(*args):
  global keyframe_f
  if not keyframe_f:
    keyframe_f = pool.apply_async(_update_keyframe, args)

  