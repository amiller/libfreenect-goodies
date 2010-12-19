import freenect
import normals
import scipy
import cv

# Use the mouse to find 4 points (x,y) on the corners of the table.
# These will define the first ROI.
boundpts = (260,218),(484,222),(570,409),(150,350)
boundpts = np.array(boundpts)
tableplane = np.array([-0.1340615 ,  0.66140062,  0.73795438,  0.47417785])

def find_plane(depth):
  global tableplane,mask,background
  # Build a mask of the image inside the convex points clicked
  uv = np.mgrid[:480,:640][::-1]
  mask = np.ones((480,640),bool)
  for (x,y),(dx,dy) in zip(boundpts, boundpts - np.roll(boundpts,1,0)):
    mask &= ((uv[0]-x)*dy - (uv[1]-y)*dx)<0
  
  # Find the average plane going through here
  global n,w
  n,w = normals.normals_c(depth.astype(np.float32))
  mask &= (w>0)
  abc = n[mask].mean(0)
  abc /= np.sqrt(np.dot(abc,abc))
  a,b,c = abc
  x,y,z = [_[mask].mean() for _ in normals.project(depth)]
  d = -(a*x+b*y+c*z)
  tableplane = np.array([a,b,c,d])
  background = np.array(depth)
  background[~mask] = 0
  figure(1);
  imshow(background);

def threshold_and_mask(depth):
  from scipy.ndimage import binary_erosion, find_objects
  mask = depth+3 < background
  dec = 3
  dil = binary_erosion(mask[::dec,::dec],iterations=2)
  slices = scipy.ndimage.find_objects(dil)
  a,b = slices[0]
  rect = (b.start*dec,a.start*dec),(b.stop*dec,a.stop*dec)
  return mask, rect

def grab():
  global depth, rgb
  (depth,_),(rgb,_) = freenect.sync_get_depth(), freenect.sync_get_video()

# Grab a blank frame!
def init_stage0():
  grab()
  find_plane(depth)

# Grab a frame, assume we already have the table found
def init_stage1():
  import pylab
  grab()
  global mask, rect, r0
  mask,rect = threshold_and_mask(depth)
  (l,t),(r,b) = rect
  print (b-t),'x', (r-l)
  n,w = normals.normals_opencl(depth.astype('f'), rect, 6)
  r0,_ = normals.mean_shift_optimize(n, w, r0, rect)
  cv.ShowImage('rgb',rgb[::2,::2,::-1].clip(0,255/2)*2)
  #normals.show_normals(depth.astype('f'),rect, 5)
  pylab.waitforbuttonpress(0.001)
  #figure(1);
  #clf()
  #imshow(depth[t:b,l:r])
  
def go():
  global r0
  r0 = np.array([0,0,0])
  while 1:
    init_stage1()
  
if __name__ == "__main__":
  pass
