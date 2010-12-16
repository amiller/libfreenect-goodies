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
    
def matmul3(name, mat):
  def tup(i):
    return '(float4)' + repr(tuple(mat[i].tolist()))
  return """
    float4 %s(const float4 r1) {
    return (float4)(dot(%s,r1),dot(%s,r1),dot(%s,r1),0);
    }  
    """ % (name, tup(0), tup(1), tup(2))

kernel_normals = """
%s

const float EPSILON = 1e-5;
const float PI = 3.1415;
const int FILT = 3;

kernel void filter_compute(
  global float *output,
  global const short *input,
  float4 bounds
)
{
 	unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
	unsigned int width = get_global_size(0);
	unsigned int height = get_global_size(1);
	unsigned int index = (y * width) + x;
	
  float o = 0;
  if (x<bounds[0] || x>=bounds[2] || y<bounds[1] || y>=bounds[3]) return;

	for (int i = -FILT; i <= FILT; i++) {
    int w = index+i*width;
	  for (int j = -FILT; j <= FILT; j++) {
	    o += input[w+j];
	  }
  }
  output[index] = o / ((FILT*2+1)*(FILT*2+1));
}

// Main Kernel code
kernel void normal_compute(
	global float4 *output,
	global const float *filt,
	float4 bounds
)
{	
	unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
	unsigned int width = get_global_size(0);
	unsigned int height = get_global_size(1);
	unsigned int index = (y * width) + x;
	
	if (x<bounds[0] || x>=bounds[2] || y<bounds[1] || y>=bounds[3]) return;

  float dx = (filt[index+1] - filt[index-1])/2;
  float dy = (filt[index+width] - filt[index-width])/2;

  float4 XYZW = (float4)(-dx, -dy, 1, -(-dx*x + -dy*y + filt[index]));
  float4 xyz = mult_xyz(XYZW);
  xyz = normalize(xyz);
  if (xyz.z < 0) xyz = -xyz;
  xyz.w = 1*(xyz.z>.1)*(filt[index]<1000)*(fabs(dx)+fabs(dy)<10);
  	
	output[index] = xyz;
}
""" % matmul3('mult_xyz', np.linalg.inv(calibkinect.xyz_matrix()).transpose())

program = cl.Program(context, kernel_normals).build("-cl-mad-enable")
print program.get_build_info(context.devices[0], cl.program_build_info.LOG)
def print_all():
  print_info(context.devices[0], cl.device_info)
  print_info(program, cl.program_info)
  print_info(program.normal_compute, cl.kernel_info)
  print_info(queue, cl.command_queue_info)
  
#print_all()
normals = np.empty((480,640,4),'f')
normals_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
depth_buf = cl.Buffer(context, mf.READ_ONLY, 480*640*2)
filt_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4)

def load_depth(depth):
  assert depth.dtype == np.int16
  assert depth.shape == (480,640)
  cl.enqueue_write_buffer(queue, depth_buf, depth).wait()

def load_filt(filt, rect=((0,0),(640,480))):
  assert filt.dtype == np.float32
  assert filt.shape == (480,640)
  (l,t),(r,b) = rect
  return cl.enqueue_write_buffer(queue, filt_buf, filt).wait()
  
def compute_filter(rect=((0,0),(640,480))):
  (L,T),(R,B) = rect; F = 7; bounds = np.array((L+F,T+F,R-F,B-F),'f')
  return program.filter_compute(queue, (640,480), None, filt_buf, depth_buf, bounds).wait()
  
def get_filter():
  filt = np.empty((480,640),'f')
  cl.enqueue_read_buffer(queue, filt_buf, filt).wait()
  return filt
  
def get_normals():
  cl.enqueue_read_buffer(queue, normals_buf, normals).wait()
  return normals
  
def compute_normals(rect=((0,0),(640,480))):
  (L,T),(R,B) = rect; F = 1; bounds = np.array((L+F,T+F,R-F,B-F),'f')
  return program.normal_compute(queue, (640,480), None, normals_buf, filt_buf, bounds).wait()
	
