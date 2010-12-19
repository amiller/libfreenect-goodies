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
const float DIST = 0.2; // Maximum distance away from the axis

float4 matmul3(const float4 mat[3], const float4 r1) {
  return (float4)(dot(mat[0],r1),dot(mat[1],r1),dot(mat[2],r1), 1);
}

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
	
	if (x<1 || x>=width-1 || y<1 || y>=height-1) return;
	x += bounds[0];
	y += bounds[1];
	
	if (filt[index]<-1000) {
	  output[index] = (float4)(0);
	  return;
	}

  float dx = (filt[index+1] - filt[index-1])/2;
  float dy = (filt[index+width] - filt[index-width])/2;
  
  if (fabs(dx)+fabs(dy)>10) {
    output[index] = (float4)(0);
    return;
  }

  float4 XYZW = (float4)(-dx, -dy, 1, -(-dx*x + -dy*y + filt[index]));
  float4 xyz = mult_xyz(XYZW);
  xyz = normalize(xyz);
  if (xyz.z < 0) xyz = -xyz;
  xyz.w = (xyz.z>.15);
  	
	output[index] = xyz;
}

kernel void meanshift_compute(
  global float4 *output,
  global float4 *cmout,
	global const float4 *norm,
	float4 m0, float4 m1, float4 m2
)
{
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
  unsigned int width = get_global_size(0);
  unsigned int height = get_global_size(1);
  unsigned int index = (y * width) + x;
  if (x<0 || x>=width || y<0 || y>=height) return;
  if (norm[index].w == 0) { // Quit early if the weight is too low!
    output[index] = (float4)(0);
    return;  
  }
  
  float4 n = norm[index]; n = (float4)(dot(m0,n), dot(m1,n), dot(m2,n), 1);
  float4 n2 = n*n;
  float4 c2 = n2.yzxw + n2.zxyw;
  float4 cm = (float4) step(c2, (float4)(DIST*DIST)); // c2<DIST^2, peraxis weight
  cmout[index] = cm;
  //cmout[index] = norm[index];
  
  float4 dd = n.zxyw*cm.yzxw - n.yzxw*cm.zxyw;
  output[index] = dd;
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
normals_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
filt_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4)

axisweight_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
axismean_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)


def load_filt(filt, rect=((0,0),(640,480))):
  (L,T),(R,B) = rect; 
  assert filt.dtype == np.float32
  assert filt.shape == (B-T,R-L)
  return cl.enqueue_write_buffer(queue, filt_buf, filt).wait()
  
def get_filter(rect=((0,0),(640,480))):
  (L,T),(R,B) = rect;
  filt = np.empty((B-T,R-L),'f')
  cl.enqueue_read_buffer(queue, filt_buf, filt).wait()
  return filt
  
def get_normals(normals=None, rect=((0,0),(640,480))):
  (L,T),(R,B) = rect;
  if normals is None: normals = np.empty((B-T,R-L,4),'f')
  assert normals.dtype == np.float32
  assert normals.shape == (B-T,R-L,4)
  cl.enqueue_read_buffer(queue, normals_buf, normals).wait()
  return normals
  
def get_meanshift(axismean=None, axisweight=None, rect=((0,0),(640,480))):
  (L,T),(R,B) = rect
  if   axismean is None:   axismean = np.empty((B-T,R-L,4),'f')
  if axisweight is None: axisweight = np.empty((B-T,R-L,4),'f')
  assert   axismean.dtype == np.float32
  assert axisweight.dtype == np.float32
  assert   axismean.shape == (B-T,R-L,4)
  assert axisweight.shape == (B-T,R-L,4)
  cl.enqueue_read_buffer(queue,   axismean_buf,   axismean)
  cl.enqueue_read_buffer(queue, axisweight_buf, axisweight).wait()
  return axismean, axisweight

def compute_normals(rect=((0,0),(640,480))):
  (L,T),(R,B) = rect; bounds = np.array((L,T,R,B),'f')
  return program.normal_compute(queue, (R-L,B-T), None, normals_buf, filt_buf, bounds)
  
def compute_meanshift(mat=np.eye(3), rect=((0,0),(640,480)),):
  mat_ = np.eye(4,dtype='f')
  mat_[:3,:3] = mat
  (L,T),(R,B) = rect;
  return program.meanshift_compute(queue, (R-L,B-T), None, 
    axismean_buf, axisweight_buf, normals_buf, mat_[0,:], mat_[1,:], mat_[2,:])
    


	
