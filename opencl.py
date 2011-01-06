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
const float TAU = 6.2831853071;
const int FILT = 3;
const float DIST = 0.2; // Maximum distance away from the axis

inline float4 matmul3(const float4 mat[3], const float4 r1) {
  return (float4)(dot(mat[0],r1),dot(mat[1],r1),dot(mat[2],r1), 1);
}
inline float4 matmul4h(const float4 mat[4], const float4 r1) {
  float W = 1.0 / dot(mat[3],r1);
  return (float4)(W*dot(mat[0],r1),W*dot(mat[1],r1),W*dot(mat[2],r1), 1);
}

inline float4 color_axis(float4 n)
{
  float4 n2 = n*n;
  float4 c2 = n2.yzxw + n2.zxyw;
  float4 cm = (float4) step(c2, (float4)(0.1)); // c2<DIST^2, peraxis weight
  return cm;
}

// Main Kernel code
kernel void normal_compute(
	global float4 *output,
	global const float *filt,
	global const char *mask,
	float4 bounds
)
{	
	unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
	unsigned int width = get_global_size(0);
	unsigned int height = get_global_size(1);
	unsigned int index = (y * width) + x;
	
	if (x<1 || x>=width-1 || y<1 || y>=height-1) {
	  output[index] = (float4)(0);
	  return;
	}
	x += bounds[0];
	y += bounds[1];
	
	if (!mask[index] || filt[index]<-1000) {
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
  xyz.w = (xyz.z>0.1);
  	
	output[index] = xyz;
}

kernel void flatrot_compute(
	global float4 *output,
	global const float4 *norm,
	float4 v0, float4 v1,	float4 v2,
	float4 bounds
)
{
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
  unsigned int width = get_global_size(0);
  unsigned int height = get_global_size(1);
  unsigned int index = (y * width) + x;
  if (norm[index].w == 0) { // Quit early if the weight is too low!
    output[index] = (float4)(0);
    return;  
  }
  float4 n = norm[index];
  float dx = dot(n, v0);
  float dy = dot(n, v1);
  float dz = dot(n, v2);
  
  float qz = 4*dz*dx*dx*dx - 4*dz*dz*dz * dx;
  float qx = dx*dx*dx*dx - 6*dx*dx*dz*dz + dz*dz*dz*dz;
  
  if (dy<0.3) output[index] = (float4)(qx, dy,qz, 1);  
  else        output[index] = (float4)(0,0,0,0);
}


kernel void lattice2_compute(
	global float4 *output,
	global const float4 *norm,
	global const float *filt,
	float modulo,
	float4 mx0, float4 mx1, float4 mx2, float4 mx3,
	float4 mm0, float4 mm1, float4 mm2,
	float4 bounds
)
{
 unsigned int x = get_global_id(0);
 unsigned int y = get_global_id(1);
 unsigned int width = get_global_size(0);
 unsigned int height = get_global_size(1);
 unsigned int index = (y * width) + x;
 //output[index*3+0] = (float4)(width,height,0,0);
 //output[index*3+1] = (float4)(get_local_size(0),get_local_size(1),0,0);
 //return;
 if (norm[index].w == 0) { // Quit early if the weight is too low!
   output[index*3+0] = (float4)(0,0,0,0);
   output[index*3+1] = (float4)(0,0,0,0);
   output[index*3+2] = (float4)(0,0,0,0);
   return;  
 }

 float4 mxyz[4] = {mx0, mx1, mx2, mx3};
 float4 mmat[3] = {mm0, mm1, mm2};
 
 // Project the depth image
 float4 XYZ = (float4)(x+bounds[0],y+bounds[1],filt[index],1);
 XYZ = matmul4h(mxyz, XYZ);

 // Project the normals
 float4 dx_z_ = norm[index]; dx_z_.w = 0;
 dx_z_ = matmul3(mmat, dx_z_);

 // Find the color axis 
 float4 cx_z_ = color_axis(dx_z_);
 
 // Finally do the trig functions
 float2 qsin, qcos;
 qsin = sincos(XYZ.xz * modulo * TAU, &qcos);
 float2 qx = (float2)(qcos.x,qsin.x);
 float2 qz = (float2)(qcos.y,qsin.y);
 if (cx_z_.x == 0) qx = (float2)(0);
 if (cx_z_.z == 0) qz = (float2)(0);
  
 // output structure: 
 // XYZ_, dxz,cxz, qx2z2
 output[index*3+0] = XYZ;
 output[index*3+1] = (float4)(dx_z_.xz, cx_z_.xz); 
 output[index*3+2] = (float4)(qx,qz);
}


""" % matmul3('mult_xyz', np.linalg.inv(calibkinect.xyz_matrix()).transpose())

program = cl.Program(context, kernel_normals).build("-cl-mad-enable")
print program.get_build_info(context.devices[0], cl.program_build_info.LOG)
def print_all():
  print_info(context.devices[0], cl.device_info)
  print_info(program, cl.program_info)
  print_info(program.normal_compute, cl.kernel_info)
  print_info(queue, cl.command_queue_info)
  
  
# I have no explanation for this workaround. Presumably it's fixed in 
# another version of pyopencl. Wtf. Getting the kernel like this
# makes it go much faster when we __call__ it.
def bullshit(self):
  return self
cl.Kernel.bullshit = bullshit
program.flatrot_compute = program.flatrot_compute.bullshit()
program.normal_compute = program.normal_compute.bullshit()
program.lattice2_compute = program.lattice2_compute.bullshit()
  
#print_all()
mask_buf = cl.Buffer(context, mf.READ_WRITE, 480*640)

normals_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
filt_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4)

qxdyqz_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)

lattice_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4*3)

def load_mask(mask, rect=((0,0),(640,480))):
  (L,T),(R,B) = rect; 
  assert mask.dtype == np.bool
  assert mask.shape == (B-T,R-L)
  return cl.enqueue_write_buffer(queue, mask_buf, mask).wait()

def load_filt(filt, rect=((0,0),(640,480))):
  (L,T),(R,B) = rect; 
  assert filt.dtype == np.float32
  assert filt.shape == (B-T,R-L)
  return cl.enqueue_write_buffer(queue, filt_buf, filt).wait()
  
def get_normals(normals=None, rect=((0,0),(640,480))):
  (L,T),(R,B) = rect;
  if normals is None: normals = np.empty((B-T,R-L,4),'f')
  assert normals.dtype == np.float32
  assert normals.shape == (B-T,R-L,4)
  cl.enqueue_read_buffer(queue, normals_buf, normals).wait()
  return normals

def get_flatrot(rect=((0,0),(640,480))):
  (L,T),(R,B) = rect  
  qxdyqz = np.empty((B-T,R-L,4),'f')
  cl.enqueue_read_buffer(queue, qxdyqz_buf, qxdyqz).wait()
  return qxdyqz
  
def get_lattice2(rect=((0,0),(640,480))):
  (L,T),(R,B) = rect  
  lattice = np.empty((B-T,R-L,12),'f')
  cl.enqueue_read_buffer(queue, lattice_buf, lattice).wait()
  output = np.rollaxis(lattice,2)
  X   = output[ 0   ]
  Y   = output[ 1   ]
  Z   = output[ 2   ]
  dx  = output[ 4   ]
  dz  = output[ 5   ]
  cx  = output[ 6   ]
  cz  = output[ 7   ]
  qx2 = output[ 8:10]
  qz2 = output[10:12]
  return X,Y,Z,dx,dz,cx,cz,qx2,qz2
  

def compute_normals(rect=((0,0),(640,480))):
  (L,T),(R,B) = rect; bounds = np.array((L,T,R,B),'f')
  evt = program.normal_compute(queue, (R-L,B-T), None, normals_buf, filt_buf, mask_buf, bounds)
  import main
  if main.WAIT_COMPUTE: evt.wait()
  return evt

def compute_flatrot(mat, rect=((0,0),(640,480))):
  assert mat.dtype == np.float32
  assert mat.shape == (3,4)
  def f(m): return m.astype('f')
  (L,T),(R,B) = rect; bounds = np.array((L,T,R,B),'f')

  evt = program.flatrot_compute(queue, (R-L,B-T), None,
    qxdyqz_buf, normals_buf, f(mat[0,:]), f(mat[1,:]), f(mat[2,:]), bounds)
  
  import main
  if main.WAIT_COMPUTE: evt.wait()
  return evt
  
def compute_lattice2(modelmat, xyzmat, modulo, rect=((0,0),(640,480))):
  assert modelmat.dtype == np.float32
  assert   xyzmat.dtype == np.float32
  assert modelmat.shape == (3,4)
  assert   xyzmat.shape == (4,4)
  def f(m): return m.astype('f')
  (L,T),(R,B) = rect; bounds = np.array((L,T,R,B),'f')
  evt = program.lattice2_compute(queue, (R-L,B-T), None, 
    lattice_buf, normals_buf, filt_buf, np.float32(1.0/modulo),
    f(  xyzmat[0,:]), f(  xyzmat[1,:]), f(  xyzmat[2,:]), f(xyzmat[3,:]),
    f(modelmat[0,:]), f(modelmat[1,:]), f(modelmat[2,:]),
    bounds)

  import main
  if main.WAIT_COMPUTE: evt.wait()
  return evt

	
