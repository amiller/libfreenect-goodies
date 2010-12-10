from IPython.Shell import IPShellEmbed
ip = IPShellEmbed(user_ns=globals()).IP.ipmagic

import normals
from timeit import timeit

rgb, depth = [x[1] for x in np.load('data/block2.npz').items()]
depth = depth.astype('f')
rect = ((0,0),(200,200))



def tests():
  global times
  px = []
  p_np = []
  p_c = []
  p_cl = []
  iters=15
  for pixels in range(10,480,10):
    px += [pixels*pixels]
    print pixels
    rect = ((0,0),(pixels,pixels))
    p_np += [timeit(lambda:normals.normals_numpy(depth,rect), number=iters,)]
    p_c += [timeit(lambda:normals.normals_c(depth,rect), number=iters,)]
    p_cl += [timeit(lambda:normals.normals_opencl(depth,rect), number=iters,)]
  times = np.vstack((px,p_np,p_c,p_cl))
  times[1:,:] /= iters

figure(1)
clf()
plot(times[0,:].transpose()/1000,times[1:,:].transpose())
xlabel('Number of Pixels (x1000)')
ylabel('Compute Time (s)')
title('Surface Normal compute time')
legend(('NumPy','C','OpenCL'),loc=2)
# print "normals numpy"
# 
# ip('timeit normals.normals_numpy(depth,rect)')
# ip('timeit normals.normals_c(depth,rect)')
# ip('timeit normals.normals_opencl(depth,rect)')