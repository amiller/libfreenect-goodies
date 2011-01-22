import freenect 
import numpy as np
import cv
import sys
sys.path += ['..']
import expmap 

"""
  Instructions are incomplete here, but do your best!
  
  This is for computing depth calibration, it assumes that intrinsic calibration is already 
  done. (See intrinsic.py)
  
  First use mkdir to create a directory to store your calibration images. Then use record()
  to start capturing images. Basically, it will save only those images that pass a 'found corners' 
  check. Skip this step if you don't have a kinect. You can use the included recording if you like 
  as a starting point.


  Use find_all_corners() to find all the corners and store them for later, yes this is repeated
  You can use load() to load all the corners and everything else for an individual frame
  backproject_depth() takes a chunk of the depth image and shows it in 3D along with the backprojected
    image points. 
    
  plot_planes() is the main function. It finds the corners in each image, and 
    saves the statistics, i.e. the mean depth and the mean Z from the intrinsics.
    After it's done, you need to fit a least squares line to the m_depth and -1.0/z_intrinsic, which are the 
    mean depths for each frame according to the depth camera and the intrinsics.

    It's possible to turn off the visualizations, but you should leave them on (it should take less than 3 
    minutes). They'll be how you'll check everything is working right.
    The intrinsic corner points are displayed as black dots.
    The pixels in the depth image are displayed in green.
    If the depth mapping is correct, then the black dots will align with the green dots. Otherwise,
    they should at least only be off by translation in Z. The least square solution mentioned above 
    shows how to do this.
    
    You can modify the xyz_matrix here until you get it right and are satisfied. Then you need to copy
    into ../calbkinect.py because that's where everything else will look for it.
"""

# Corner size: this isn't used yet. 
#   I'm sorry, you'll just have to look everywhere 6 and 8 are found near eachother). 
#   Keep in mind this is the number of 'internal corners', 6 and 8 is for the 8.5"x11" paper. Try to figure
#   out how to print a perfectly sized checkerboard
corner_count = (6,8)

# This is the directory to store depth map frames, ir, and corners
dirname = 'calibdata/pair2'

if not 'window' in globals():
  from visuals.pointwindow import PointWindow
  window = PointWindow(size=(640,480))

def xyz_matrix():
  fx = 583.0
  fy = 583.0
  cx = 321
  cy = 249
  a = -0.0028300396
  b = 3.1006268
  if 0: # Old parameters from nic burrus
    fx = 583.0
    fy = 583.0
    a = -0.0030711
    b = 3.3309495
    cx = 321.5
    cy = 246.5
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
  
def load(i=20):
  global im
  im = cv.LoadImage('%s/ir_%05d.jpg' % (dirname,i))
  im = np.fromstring(im.tostring(),'u1').reshape(480,640,3)
  global depth
  depth = np.load('%s/depth_%05d.npy' % (dirname,i))

def find_corners(im, w=8,h=6):
  found,corners = cv.FindChessboardCorners(im, (w,h))
  corners = np.array(corners)
  if found:
    figure(1)
    clf()
    imshow(im)
    scatter(corners[:,0],corners[:,1],c='r')
  return found,corners
  
def find_all_corners():
  for i in range(100):
    im = cv.LoadImage('%s/ir_%05d.jpg' % (dirname,i))
    im = np.fromstring(im.tostring(),'u1').reshape(480,640,3)
    found,corners = find_corners(im)
    if found:
      print 'found: ', i
      np.save('%s/corners_%05d.npy' % (dirname,i), corners)
      pylab.waitforbuttonpress(0.01)
      
def plot_planes():
  
  global z_intrinsic,z_depth,m_depth
  z_intrinsic = []
  z_depth = []
  m_depth = []
  
  for i in range(100):
    try:
      load(i)
      corners = np.load('%s/corners_%05d.npy' % (dirname,i))
    except:
      continue
      
    backproject_depth(im,depth,corners)

    pylab.waitforbuttonpress(0.1)
  
    # Fit a plane to the points inside the corners
    z_intrinsic += [pts1.mean(1)[2]]
    z_depth += [pts2.mean(1)[2]]
    m_depth += [depth[mask].astype('f').mean()]
    
  
def backproject_depth(im, depth, corners):
  # Use 'extrinsic rectification' to find plane equation
  global obj, cam, distcc, rvec, tvec,bp
  obj = np.mgrid[:6,:8,:1].astype('f').reshape(3,-1)*0.0254
  f = 583.0
  cx = 321
  cy = 249
  distcc = np.zeros((4,1),'f')
  rvec = np.zeros((3,1),'f')
  tvec = np.zeros((3,1),'f')
  cam = np.array([[f,0,cx],[0,f,cy],[0,0,1]])
  cv.FindExtrinsicCameraParams2(obj, corners, cam, distcc, rvec,tvec)

  # Back project to see how it went
  bp = np.array(corners)
  cv.ProjectPoints2(obj, rvec, tvec, cam, distcc, bp)

  global pts1
  # Show the points in a point cloud, using rvec and tvec
  RT = np.eye(4).astype('f')
  RT[:3,:3] = expmap.axis2rot(rvec.squeeze())
  RT[:3,3] = tvec.squeeze()
  #RT = np.linalg.inv(RT)
  pts1 = np.dot(RT[:3,:3], obj) + RT[:3,3].reshape(3,1)
  pts1[1,:] *= -1
  pts1[2,:] *= -1
  rgb1 = np.zeros((pts1.shape[1],4),'f')
  rgb1[:,:] = [0,0,0,1]

  # Also show the points in the region, using the calib image 
  global mask,pts2
  bounds = corners[[0,7,5*8+7,5*8],:]
  polys = (tuple([tuple(x) for x in tuple(bounds)]),)
  mask = np.zeros((480,640),'f')
  cv.FillPoly(mask, polys, [1,0,0])
  mask = (mask>0)&(depth<2047)
  v,u = np.mgrid[:480,:640].astype('f')
  pts2 = np.vstack(project(depth[mask].astype('f'),u[mask],v[mask]))
  rgb2 = np.zeros((pts2.shape[1],4),'f')
  rgb2[:,:] = [0,1,0,1]

  if np.any(np.isnan(pts2.mean(1))): return

  window.update_points(np.hstack((pts1,pts2)).transpose(),np.vstack((rgb1,rgb2)))
  window.lookat = pts1.mean(1)
  window.Refresh()
  
  return
  
  
  
def showimagegray(name, data):
  image = cv.CreateImageHeader((data.shape[1], data.shape[0]),
                             cv.IPL_DEPTH_8U,
                             1)
                        
  cv.SetData(image, data.tostring(), data.dtype.itemsize * data.shape[1])
  cv.ShowImage(name, image)


def grab():
  global depth, ir
  depth,_ = freenect.sync_get_depth()
  ir,_ = freenect.sync_get_video(0, freenect.VIDEO_IR_8BIT)
  
def record():
  for i in range(0,100):
    while 1:
      grab()
      found,_ = find_corners(ir)
      if not found: continue
      showimagegray('depth', (depth/8).astype('u1'))
      showimagegray('ir', ir)  
      filename = '%s/ir_%05d.jpg' % (dirname, i)
      cv.SaveImage(filename, ir) 
      filename = '%s/depth_%05d.jpg' % (dirname, i)
      cv.SaveImage(filename, (depth/8).astype('u1')) 
      np.save('%s/depth_%05d.npy' % (dirname, i), depth)
      cv.WaitKey(10)
      print 'saved ', i
      break