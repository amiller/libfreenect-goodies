import freenect
import numpy as np
import os
import cv

def record_movie(filename):
  foldername = 'data/movies/%s/' % filename
  try:
    os.mkdir(foldername)
  except:
    pass
  frame = 0
  while 1:
    (depth,_),(rgb,_) = freenect.sync_get_depth(), freenect.sync_get_video()
    
    np.savez('%s/depth_%05d.npz' % (foldername,frame), depth)
    
    cv.SaveImage('%s/depthim_%05d.jpg' % (foldername,frame), depth.astype(np.uint8))
    
    cv.SaveImage('%s/rgb_%05d.jpg' % (foldername,frame), rgb[:,:,::-1])
    
    print 'frame: %d' % frame
    frame = frame + 1