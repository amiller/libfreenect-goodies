

if __name__ == "__main__":
  
  import main
  from pclwindow import PCLWindow
  if not 'window' in main.__dict__: main.window = PCLWindow(size=(640,480))
  
  import normals
  
  movie = "block1"
  if movie == 'block1':
    #depth, rgb = [x[1] for x in np.load('data/ceiling.npz').items()]
    rgb, depth = [x[1] for x in np.load('data/block1.npz').items()]
    v,u = np.mgrid[160:282,330:510]
    r0 = [-0.7,-0.2,0]
  #rgb, depth = [x[1].astype(np.float32) for x in np.load('data/block2.npz').items()]
  #v,u = np.mgrid[231:371,264:434]
  #r0 = [-0.63, 0.68, 0.17]
  #depth = np.load('data/movies/test/depth_00000.npz').items()[0][1]
  #v,u = np.mgrid[175:332,365:485]
  #r0 = [-0.7626858, 0.28330218, 0.17082515]
  #depth = np.load('data/movies/single/depth_00000.npz').items()[0][1]
  #v,u = np.mgrid[:480,:640]
  x,y,z = project(depth[v,u], u.astype(np.float32), v.astype(np.float32))
  
  