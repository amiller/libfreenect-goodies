"""
These are some functions to help work with kinect camera calibration and projective
geometry. 

Tasks:
- Convert the kinect depth image to a metric 3D point cloud
- Convert the 3D point cloud to texture coordinates in the RGB image

Notes about the coordinate systems:
 There are three coordinate systems to worry about. 
 1. Kinect depth image:
    u,v,depth
    u and v are image coordinates, (0,0) is the top left corner of the image
                               (640,480) is the bottom right corner of the image
    depth is the raw 11-bit image from the kinect, where 0 is infinitely far away
      and larger numbers are closer to the camera
      (2047 indicates an error pixel)
      
 2. Kinect rgb image:
    u,v
    u and v are image coordinates (0,0) is the top left corner
                              (640,480) is the bottom right corner
                              
 3. XYZ world coordinates:
    x,y,z
    The 3D world coordinates, in meters, relative to the depth camera. 
    (0,0,0) is the camera center. 
    Negative Z values are in front of the camera, and the positive Z direction points
       towards the camera. 
    The X axis points to the right, and the Y axis points up. This is the standard 
       right-handed coordinate system used by OpenGL.
    

"""
import numpy as np


def depth2xyzuv(depth, u=None, v=None):
  """
  Return a point cloud, an Nx3 array, made by projecting the kinect depth map 
    through intrinsic / extrinsic calibration matrices
  Parameters:
    depth - comes directly from the kinect 
    u,v - are image coordinates, same size as depth (default is the original image)
  Returns:
    xyz - 3D world coordinates in meters (Nx3)
    uv - image coordinates for the RGB image (Nx3)
  
  You can provide only a portion of the depth image, or a downsampled version of
    the depth image if you want; just make sure to provide the correct coordinates
    in the u,v arguments. 
    
  Example:
    # This downsamples the depth image by 2 and then projects to metric point cloud
    u,v = mgrid[:480:2,:640:2]
    xyz,uv = depth2xyzuv(freenect.sync_get_depth()[::2,::2], u, v)
    
    # This projects only a small region of interest in the upper corner of the depth image
    u,v = mgrid[10:120,50:80]
    xyz,uv = depth2xyzuv(freenect.sync_get_depth()[v,u], u, v)
  """
  if u is None or v is None:
    u,v = np.mgrid[:480,:640]
  
  # Build a 3xN matrix of the d,u,v data
  C = np.vstack((u.flatten(), v.flatten(), depth.flatten(), 0*u.flatten()+1))

  # Project the duv matrix into xyz using xyz_matrix()
  X,Y,Z,W = np.dot(xyz_matrix(),C)
  X,Y,Z = X/W, Y/W, Z/W
  xyz = np.vstack((X,Y,Z)).transpose()
  xyz = xyz[Z<0,:]

  # Project the duv matrix into U,V rgb coordinates using rgb_matrix() and xyz_matrix()
  U,V,_,W = np.dot(np.dot(uv_matrix(), xyz_matrix()),C)
  U,V = U/W, V/W
  uv = np.vstack((U,V)).transpose()    
  uv = uv[Z<0,:]       

  # Return both the XYZ coordinates and the UV coordinates
  return xyz, uv



def uv_matrix():
  """
  Returns a matrix you can use to project XYZ coordinates (in meters) into
      U,V coordinates in the kinect RGB image
  """
  rot = np.array([[ 9.99846e-01,   -1.26353e-03,   1.74872e-02], 
                  [-1.4779096e-03, -9.999238e-01,  1.225138e-02],
                  [1.747042e-02,   -1.227534e-02,  -9.99772e-01]])
  trans = np.array([[1.9985e-02, -7.44237e-04,-1.0916736e-02]])
  m = np.hstack((rot, -trans.transpose()))
  m = np.vstack((m, np.array([[0,0,0,1]])))
  KK = np.array([[529.2, 0, 329, 0],
                 [0, 525.6, 267.5, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
  m = np.dot(KK, (m))
  return m

def xyz_matrix():
  fx = 594.21
  fy = 591.04
  a = -0.0030711
  b = 3.3309495
  cx = 339.5
  cy = 242.7
  mat = np.array([[1/fx, 0, 0, -cx/fx],
                  [0, -1/fy, 0, cy/fy],
                  [0,   0, 0,    -1],
                  [0,   0, a,     b]])
  return mat