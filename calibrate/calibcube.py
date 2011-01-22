import freenect
import calibkinect
import pclwindow

"""
This is an experimental attempt at using the calibration cube that I hope I won't need to deal with
anymore. :(

However what worked best was using the calibration to 
"""

def xyz_matrix():
  fx = 595.0
  fy = 595.0
  a = -0.0030711016
  b = 3.3309495161
  cx = 320
  cy = 240
  mat = np.array([[1/fx, 0, 0, -cx/fx],
                  [0, -1/fy, 0, cy/fy],
                  [0,   0, 0,    -1],
                  [0,   0, a,     b]])
  return mat

xyz_matrix = xyz_matrix().astype('f')

def project(depth, u=None, v=None, mat=xyz_matrix):
  if u is None or v is None: v,u = np.mgrid[:480,:640].astype('f')
  X,Y,Z = u,v,depth
  x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
  y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
  z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
  w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
  w = 1/w
  return x*w, y*w, z*w

def grab():
  global depth, rgb
  (depth,_),(rgb,_) = freenect.sync_get_depth(0), (None,None)#freenect.sync_get_video(0)

background = np.load('data/background.npy')

def show_points():
  xyz = np.dstack(project(depth.astype('f'))).reshape(-1,3)
  xyz *= (xyz[:,2] < 0).reshape(-1,1)
  window.update_points(xyz)
  window.Refresh()

def get_background():
  global background
  grab()
  background = np.array(depth)
  np.save('data/background',background)
  
def ransac_fit_planes(XYZ):

  def ransac_fit_plane(points, iters=2000, thresh=0.002, ):
    bestinliers = None
    for i in range(iters):
      # Select a number of points
      inds = np.random.randint(0,points.shape[0],6)
    
      # Fit a plane to the points
      plane = fit_plane(points[inds,:])
    
      # Check for inliers
      dist = np.dot(points, plane[:3]) + plane[3]
      (inliers,) = np.nonzero(np.abs(dist) < thresh)
      
      if bestinliers is None or len(inliers) > len(bestinliers):
        bestinliers = inliers

    return fit_plane(points[bestinliers]), bestinliers
  
  def fit_plane(points):
    # Points is Nx3
    assert points.shape[1] == 3
    assert points.shape[0] >= 3
    pmean = points.mean(0)
    u,s,vh = np.linalg.svd((points-pmean),full_matrices=False)
    plane = np.zeros(4)
    plane[:3] = vh[-1,:]
    plane[3] = -np.dot(pmean,plane[:3])
    return plane
    
  remaining = np.arange(len(XYZ))
  RGBA = np.zeros((XYZ.shape[0],4),'f')
  RGBA[:,3] = 1
  planes = []
  for i in range(6):
    plane, inliers = ransac_fit_plane(XYZ[remaining,:])
    planes += [plane]
    RGBA[remaining[inliers],:3] = [i%3,(i+1)%3,(i+2)%3]
    rm = remaining*0+1
    rm[inliers] = 0
    remaining = remaining[rm>0]
  window.update_points(XYZ,RGBA)
  window.Refresh()
  print 'angle (degrees): ', np.rad2deg(np.arccos(np.abs(np.dot(planes[0][:3],planes[2][:3]))))
  print 'angle (degrees): ', np.rad2deg(np.arccos(np.abs(np.dot(planes[0][:3],planes[1][:3]))))
  print 'distance (m): ', np.abs(np.abs(planes[0][3])-np.abs(planes[1][3]))
  return planes
  #return plane, inliers
  
def get_foreground():
  global foreground,inds,XYZ
  grab()
  mask = (background != 2047) & (depth != 2047)
  foreground = (depth+5 < background)*mask
  inds = np.nonzero(foreground)
  X,Y,Z = project(depth[inds].astype('f'), inds[1].astype('f'), inds[0].astype('f'))
  XYZ = np.vstack((X,Y,Z)).transpose()
  window.update_points(XYZ)
  window.lookat = XYZ.mean(0)
  window.Refresh()
  
  
if not 'window' in globals():
  from visuals.pointwindow import PointWindow
  window = PointWindow(size=(640,480))