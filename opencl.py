import pyglet.gl
from OpenGL.GL import *
from OpenGL.GLU import *
import calibkinect
import pyopencl as cl
import numpy as np

def print_info(obj, info_cls):
    for info_name in sorted(dir(info_cls)):
        if not info_name.startswith("_") and info_name != "to_string":
            info = getattr(info_cls, info_name)
            try:
                info_value = obj.get_info(info)
            except:
                info_value = "<error>"

            print "%s: %s" % (info_name, info_value)

context = cl.Context(dev_type = cl.device_type.GPU)
print_info(context.devices[0], cl.device_info)
queue = cl.CommandQueue(context)
mf = cl.mem_flags
sampler = cl.Sampler(context, True, cl.addressing_mode.CLAMP, cl.filter_mode.LINEAR)

def matmul(name, mat):
  def tup(i):
    return '(float4)' + repr(tuple(mat[i].tolist()))
  return """
    float4 %s(const float4 r1) {
    return (float4)(dot(%s,r1),dot(%s,r1),dot(%s,r1),dot(%s,r1));
    }  
    """ % (name, tup(0), tup(1), tup(2), tup(3))

kernel_normals = """
%s

const float EPSILON = 1e-5;
const float PI = 3.1415;
const int FILT = 3;

kernel void filter_compute(
  global float *output,
  global short *input
)
{
 	unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
	unsigned int width = get_global_size(0);
	unsigned int height = get_global_size(1);
	unsigned int index = (y * width) + x;
	
  float o = 0;
  if (x-FILT<0 || x+FILT>=width || y-FILT<0 || y+FILT>=height) return;
	for (int i = -FILT; i <= FILT; i++) {
    int w = index+width*i;
	  for (int j = -FILT; j <= FILT; j++) {
	    o += input[w+j];
	  }
  }
  output[index] = o / ((FILT*2+1)*(FILT*2+1));
}

kernel void filter_compute_im(
  global float *output,
  read_only image2d_t depth,
  sampler_t sampler
)
{
 	unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
	unsigned int width = get_global_size(0);
	unsigned int height = get_global_size(1);
	unsigned int index = (y * width) + x;
	
  float o = 0;
  
	for (int i = -FILT; i <= FILT; i++) {
	  for (int j = -FILT; j <= FILT; j++) {
	    o += read_imagef(depth, sampler, (int2)(x+i,y+j)).x;
	  }
  }
  output[index] = o / ((FILT*2+1)*(FILT*2+1));
}
kernel void filter_compute_im2(
  write_only image2d_t output,
  read_only image2d_t depth,
  sampler_t sampler
)
{
 	unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
	unsigned int width = get_global_size(0);
	unsigned int height = get_global_size(1);
	unsigned int index = (y * width) + x;
	
  float o = 0;
  
	for (int i = -FILT; i <= FILT; i++) {
	  for (int j = -FILT; j <= FILT; j++) {
	    o += read_imagef(depth, sampler, (int2)(x+i,y+j)).x;
	  }
  }
  write_imagef(output, (int2)(x,y),  (float4)(o / ((FILT*2+1)*(FILT*2+1))));
}

// Main Kernel code
kernel void normal_compute(
	global float4 *output,
	global const short *depth,
	sampler_t sampler
)
{	
	unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
	unsigned int width = get_global_size(0);
	unsigned int height = get_global_size(1);
	unsigned int index = (y * width) + x;

	float dx = ((x+0.5) / (float) width)*2.0f-1.0f; // 0.5 increment is to reach the center of the pixel.
	float dy = ((y+0.5) / (float) height)*2.0f-1.0f;
	
	output[index] = (float4)(0,1,2,3);
}
""" % matmul('mult_xyz', calibkinect.xyz_matrix())

program = cl.Program(context, kernel_normals).build("-cl-mad-enable")
print program.get_build_info(context.devices[0], cl.program_build_info.LOG)
def print_all():
  print_info(program, cl.program_info)
  print_info(program.normal_compute, cl.kernel_info)
  print_info(queue, cl.command_queue_info)
  
print_all()
normals = np.empty((480,640,4),'f')
normals_buf = cl.Buffer(context, mf.WRITE_ONLY, normals.nbytes)

depth_fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT16)
depth_img = cl.Image(context, mf.READ_ONLY, depth_fmt, shape=(480,640))
depth_buf = cl.Buffer(context, mf.READ_ONLY, 480*640*2)

filt_fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
filt_img = cl.Image(context, mf.READ_WRITE, filt_fmt, shape=(480,640))
filt_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4)

norm_fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
norm_img = cl.Image(context, mf.READ_WRITE, filt_fmt, shape=(480,640))


def load_depth(depth):
  assert depth.dtype == np.int16
  assert depth.shape == (480,640)
  cl.enqueue_write_buffer(queue, depth_buf, depth).wait()
  df = np.empty((480,640,4),np.int16)
  df[:,:,0] = depth
  cl.enqueue_write_image(queue, depth_img, (0,0), (480,640), df).wait()
  
def compute_filter():
  off = 3
  #program.filter_compute(queue, (480-off*2,640-off*2), None, filt_buf, depth_buf, global_offset=(off,off)).wait()
  #program.filter_compute(queue, (480,640), None, filt_buf, depth_buf, global_offset=(off,off)).wait()
  #program.filter_compute_im(queue, (480,640), None, filt_buf, depth_img, sampler).wait()
  #program.filter_compute_im2(queue, (480,640), None, filt_img, depth_img, sampler).wait()
  
def get_filter():
  filt = np.empty((480,640),'f')
  cl.enqueue_read_buffer(queue, filt_buf, filt).wait()
  return filt
  
def compute_normals(rect=((0,480),(0,640))):
  program.normal_compute(queue, normals.shape[:2], None, normals_buf, depth_buf, sampler)
  cl.enqueue_read_buffer(queue, normals_buf, normals).wait()
  return normals
	
