import pyglet.gl
from OpenGL.GL import *
from OpenGL.GLU import *
import calibkinect
import pyopencl as cl
import numpy as np
import preprocess

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
    
def normal_maker(name, mat, matw, matr):
  def tup(i):
    return '(float4)' + repr(tuple(mat[i].tolist()))
    
  def tupw(i):
    return '(float4)' + repr(tuple(matw[i].tolist()))
    
  def tupr(i):
    return '(float4)' + repr(tuple(matr[i].tolist()))


  return """
  inline float4 matmul3_%s(const float4 r1) {
    return (float4)(dot(%s,r1),dot(%s,r1),dot(%s,r1),0);
  }
  
  inline float4 matmul4_%s(const float4 r1) {
    return (float4)(dot(%s,r1),dot(%s,r1),dot(%s,r1),dot(%s,r1));
  }
  
  inline float4 matmul3z_%s(const float4 r1) {
    return (float4)(dot(%s,r1),dot(%s,r1),dot(%s,r1),0);
  }
  
  
  // Main Kernel code
  kernel void normal_compute_%s(
  	global float4 *output,
  	global float4 *xoutput,
  	global const float *filt,
  	global const char *mask,
  	const float4 bounds, const int offset
  )
  {	
  	unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
  	unsigned int width = get_global_size(0);
  	unsigned int height = get_global_size(1);
  	unsigned int index = (y * width) + x + offset;

  	if (x<1 || x>=width-1 || y<1 || y>=height-1 || !mask[index] || filt[index]<-1000) {
  	   output[index] = (float4)(0);
  	  xoutput[index] = (float4)(0);
  	  return;
  	}
  	x += bounds[0];
  	y += bounds[1];

    float dx = (filt[index+1] - filt[index-1])/2;
    float dy = (filt[index+width] - filt[index-width])/2;

    if (fabs(dx)+fabs(dy)>10) {
       output[index] = (float4)(0);
      xoutput[index] = (float4)(0);
      return;
    }

    //float nw = -(-dx*x+ -dy*y + (filt[index] + dx*x + dy*y)*filt[index]);
    //float4  XYZW = (float4)(-dx, -dy, filt[index] + dx*x + dy*y, nw);
    float4  XYZW = (float4)(-dx, -dy, 1, -(-dx*x + -dy*y + filt[index]));
    float4 xXYZW = (float4)(  x,   y, filt[index], 1);
    float4  xyz = matmul3_%s ( XYZW);
    float4 _xyz = matmul4_%s (xXYZW);
    _xyz /= _xyz.w;
    xyz = normalize(xyz);
    if (xyz.z < 0) xyz = -xyz;
    float w = (xyz.z>0.1);
    xyz.w = 1;
    xyz = matmul3z_%s(xyz);
    xyz.w = w;

  	 output[index] =  xyz;
  	xoutput[index] = _xyz;
  }
  """ % (name, tup (0), tup (1), tup (2), 
         name, tupw(0), tupw(1), tupw(2), tupw(3),
         name, tupr(0), tupr(1), tupr(2),
         name, name, name, name)

kernel_normals = """

const float EPSILON = 1e-5;
const float TAU = 6.2831853071;
const int FILT = 3;
const float DIST = 0.2; // Maximum distance away from the axis

inline float4 matmul3(const float4 mat[3], const float4 r1) {
  return (float4)(dot(mat[0],r1),dot(mat[1],r1),dot(mat[2],r1), 0);
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

%s

%s

kernel void flatrot_compute(
	global float4 *output,
	global const float4 *norm,
	float4 v0, float4 v1,	float4 v2
)
{
  unsigned int index = get_global_id(0);
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

kernel void float4_sum(
	global float4 *result,
	local float4 *scratch,
	global const float4 *input,
	const int length
)
{
  int global_index = get_global_id(0);
  float4 accum = (float4)(0);
  while (global_index < length) {
    accum += input[global_index];
    global_index += get_global_size(0);
  }
  int local_index = get_local_id(0);
  scratch[local_index] = accum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = get_local_size(0) / 2;
           offset > 0;
           offset = offset / 2) {
    if (local_index < offset) {
      float4 other = scratch[local_index + offset];
      float4 mine  = scratch[local_index];
      scratch[local_index] = mine + other;
    } 
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}

kernel void lattice2_compute(
	global float4 *face_label,
	global float4 *qx2z2,
	global float4 *modelxyz,
	global const float4 *norm,
	global const float4 *xyz,
	float modulo,
	float4 mm0, float4 mm1, float4 mm2
)
{
  unsigned int index = get_global_id(0);
    
  if (norm[index].w == 0) { // Quit early if the weight is too low!
   face_label[index] = (float4)(0,0,0,0);
   qx2z2[index] = (float4)(0,0,0,0);
   modelxyz[index] = (float4)(0,0,0,0);
   return;
  }

  float4 mmat[3] = {mm0, mm1, mm2};

  // Project the depth image
  float4 XYZ = matmul3(mmat, xyz[index]);

  // Project the normals
  float4 dxyz_ = norm[index]; dxyz_.w = 0;
  dxyz_ = matmul3(mmat, dxyz_);

  // Threshold the normals and pack it into one number as a label
  const float CLIM = 0.9486;
  float2 dxz = dxyz_.xz;
  //float2 cxz = color_axis(dxyz_).xz;
  float2 cxz = (float2)-1 + step(dxz,(float2)(-CLIM)) + step(dxz,(float2)(CLIM));
  float label = as_float((short2)(cxz.s0,cxz.s1));
  XYZ.w = label;

  // Finally do the trig functions
  float2 qsin, qcos;
  qsin = sincos(XYZ.xz * modulo * TAU, &qcos);
  float2 qx = (float2)(qcos.x,qsin.x);
  float2 qz = (float2)(qcos.y,qsin.y);
  if (cxz.s0 == 0) qx = (float2)(0);
  if (cxz.s1 == 0) qz = (float2)(0);

  // output structure: 
  modelxyz[index] = XYZ;
  face_label[index] = (float4)(cxz.s0!=0,cxz.s1!=0,0,0);
  qx2z2[index] = (float4)(qx,qz);
}


kernel void gridinds_compute(
  global uchar4 *gridinds,
  global const float4 *modelxyz,
  const float xfix, const float zfix, 
  const float LW, const float LH,
  const float4 gridmin, const float4 gridmax
)
{
  unsigned int index = get_global_id(0);
  float4 xyzf = modelxyz[index];
  short2 cxz = as_short2(xyzf.w);
  float4 f1 = (float4)(cxz.s0*.5,0,cxz.s1*.5,0);
  float4 fix = (float4)(xfix,0,zfix,0);
  
  float4 mod = (float4)(LW,LH,LW,1);
  
  uchar4 occ = convert_uchar4(floor(-gridmin + (xyzf-fix)/mod + f1));
  uchar4 vac = occ - (uchar4)(cxz.s0,0,cxz.s1,0);
  occ.w = cxz.s0 + cxz.s1;
  vac.w = occ.w;
  gridinds[2*index+0] = occ;
  gridinds[2*index+1] = vac;
}

""" % (normal_maker('LEFT',  np.linalg.inv(calibkinect.xyz_matrix()).transpose(),
                            calibkinect.xyz_matrix(), np.eye(4)),
       normal_maker('RIGHT', np.linalg.inv(calibkinect.xyz_matrix()).transpose(),
                          np.dot(preprocess.RIGHT2LEFT, calibkinect.xyz_matrix()),
                             np.linalg.inv(preprocess.RIGHT2LEFT).transpose()))

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
program.normal_compute_LEFT = program.normal_compute_LEFT.bullshit()
program.normal_compute_RIGHT = program.normal_compute_RIGHT.bullshit()
program.lattice2_compute = program.lattice2_compute.bullshit()
program.float4_sum = program.float4_sum.bullshit()
program.gridinds_compute = program.gridinds_compute.bullshit()

#print_all()
mask_buf    = cl.Buffer(context, mf.READ_WRITE, 480*640)

normals_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
xyz_buf     = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
filt_buf    = cl.Buffer(context, mf.READ_WRITE, 480*640*4)

qxdyqz_buf  = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)

face_buf    = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
qxqz_buf    = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
model_buf   = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)

gridinds_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*2)

#debug_buf     = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
reduce_buf    = cl.Buffer(context, mf.READ_WRITE, 8*4*100)
reduce_scratch = cl.LocalMemory(64*8*4)


def set_rect(_rectL, _rectR):
  global rectL, rectR
  rectL = _rectL
  rectR = _rectR
  (L1,T1),(R1,B1) = rectL;
  (L2,T2),(R2,B2) = rectR;
  global lengthL, lengthR
  lengthL = (B1-T1)*(R1-L1)
  lengthR = (B2-T2)*(R2-L2)
  assert lengthL + lengthR < 480*640
  
def load_mask(mask, spot):
  assert spot=='LEFT' or spot=='RIGHT'
  rect = rectL if spot=='LEFT' else rectR
  offset = 0 if spot=='LEFT' else lengthL
  (L,T),(R,B) = rect
  assert mask.dtype == np.bool
  assert mask.shape[0] == B-T 
  assert mask.shape[1] == R-L
  return cl.enqueue_write_buffer(queue, mask_buf, mask, offset*1, is_blocking=False)

def load_filt(filt, spot):
  assert spot=='LEFT' or spot=='RIGHT'
  rect = rectL if spot=='LEFT' else rectR
  offset = 0 if spot=='LEFT' else lengthL
  (L,T),(R,B) = rect
  assert filt.dtype == np.float32
  assert filt.shape[0] == B-T 
  assert filt.shape[1] == R-L
  return cl.enqueue_write_buffer(queue, filt_buf, filt, offset*4, is_blocking=False)
  
def get_xyz():
  xyz = np.empty((lengthL+lengthR,4),'f')
  cl.enqueue_read_buffer(queue, xyz_buf, xyz).wait()
  return xyz
  
def get_normals():
  (L1,T1),(R1,B1) = rectL
  (L2,T2),(R2,B2) = rectR
  
  normals = np.empty((lengthL+lengthR,4), 'f')
  cl.enqueue_read_buffer(queue, normals_buf, normals).wait()
  
  return (normals[:lengthL,:].reshape(T1-B1,R1-L1,4),
          normals[lengthL:,:].reshape(T2-B2,R2-L2,4))

def get_flatrot():
  qxdyqz = np.empty((lengthL+lengthR,4),'f')
  cl.enqueue_read_buffer(queue, qxdyqz_buf, qxdyqz).wait()
  return qxdyqz
  
def get_modelxyz():
  model   = np.empty((lengthL+lengthR,4),'f')
  cl.enqueue_read_buffer(queue, model_buf, model).wait()
  return model
  
def get_gridinds():
  gridinds = np.empty((lengthL+lengthR,2,4), 'u1')
  cl.enqueue_read_buffer(queue, gridinds_buf, gridinds).wait()
  return gridinds

def compute_normals(spot):
  assert spot=='LEFT' or spot=='RIGHT'
  rect = rectL if spot=='LEFT' else rectR
  offset = 0 if spot=='LEFT' else lengthL
  (L,T),(R,B) = rect; bounds = np.array((L,T,R,B),'f')

  kernel = program.normal_compute_LEFT if spot =='LEFT' else program.normal_compute_RIGHT
  evt = kernel(queue, (R-L,B-T), None, normals_buf, xyz_buf,
        filt_buf, mask_buf, bounds, np.int32(offset))
  import main
  #if main.WAIT_COMPUTE: evt.wait()
  return evt

def compute_flatrot(mat):
  assert mat.dtype == np.float32
  assert mat.shape == (3,4)
  def f(m): return m.astype('f')
  
  evt = program.flatrot_compute(queue, (lengthL+lengthR,), None,
    qxdyqz_buf, normals_buf, f(mat[0,:]), f(mat[1,:]), f(mat[2,:]))
  
  import main
  if main.WAIT_COMPUTE: evt.wait()
  return evt
  
def compute_lattice2(modelmat, modulo):
  assert modelmat.dtype == np.float32
  assert modelmat.shape == (3,4)
  def f(m): return m.astype('f')
  
  evt = program.lattice2_compute(queue, (lengthL+lengthR,), None, 
    face_buf, qxqz_buf, model_buf,
    normals_buf, xyz_buf, np.float32(1.0/modulo),
    f(modelmat[0,:]), f(modelmat[1,:]), f(modelmat[2,:]))

  import main
  if main.WAIT_COMPUTE: evt.wait()
  return evt
  
def compute_gridinds(xfix, zfix, LW, LH, gridmin, gridmax):
  # The model x and z coordinates from the lattice2 step are off by the
  # translation component found in that stage. Pass them along here.
  assert gridmin.shape == (4,)
  assert gridmin.dtype == np.float32
  assert gridmin[3] == 0
  evt = program.gridinds_compute(queue, (lengthL+lengthR,), None,
    gridinds_buf, model_buf,
    np.float32(xfix), np.float32(zfix), 
    np.float32(LW),   np.float32(LH), 
    gridmin, gridmax)
    
  import main
  if main.WAIT_COMPUTE: evt.wait()
  return evt  

def reduce_flatrot():
  sums = np.empty((8,4),'f')  
  evt = program.float4_sum(queue, (64*8,), (64,), 
    reduce_buf, reduce_scratch, 
    qxdyqz_buf, np.int32(lengthL+lengthR))
  cl.enqueue_read_buffer(queue, reduce_buf, sums).wait()
  return sums.sum(0)
    
def reduce_lattice2():
  sums = np.empty((8,4),'f') 
  
  evt = program.float4_sum(queue, (64*8,), (64,), 
    reduce_buf, reduce_scratch, 
    qxqz_buf, np.int32(lengthL+lengthR))
  cl.enqueue_read_buffer(queue, reduce_buf, sums).wait()
  qxqz = sums.sum(0)  
  
  evt = program.float4_sum(queue, (64*8,), (64,), 
    reduce_buf, reduce_scratch, 
    face_buf, np.int32(lengthL+lengthR))
  cl.enqueue_read_buffer(queue, reduce_buf, sums).wait()
  cxcz = sums[:2,:].sum(0)

  return cxcz,qxqz
  
  
