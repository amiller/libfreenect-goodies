#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void inrange(unsigned short *a, char *out,
			 unsigned short *hi, unsigned short *lo, size_t length)
{
	int i;
	for (i = 0; i < length; i++) 
		out[i] = a[i] > lo[i] && a[i] < hi[i];
}

void gradient(float *depth, float *dx, float *dy, int h, int w)
{
  int i, j;
  for (i = 1; i < h-1; i++) {
    for (j = 1; j < w-1; j++) {
      dx[i*w+j] = (depth[i*w+j+1] - depth[i*w+j-1])/2;
      dy[i*w+j] = (depth[(i+1)*w+j] - depth[(i-1)*w+j])/2;
    }
  }
}

void normals(float *depth, 
             float *u, float *v, 
             float *nx, float *ny, float *nz, 
             float mat[4][4], int h, int w)
{
  int i, j;
  for (i = 1; i < h-1; i++) {
    for (j = 1; j < w-1; j++) {
      float dx = (depth[i*w+j+1] - depth[i*w+j-1])/2;
      float dy = (depth[(i+1)*w+j] - depth[(i-1)*w+j])/2;
      float X=-dx, Y=-dy, Z=1, W=-(-dx*u[i*w+j] + -dy*v[i*w+j] + depth[i*w+j]);
      
      float x = X*mat[0][0] + Y*mat[0][1] + Z*mat[0][2] + W*mat[0][3];
      float y = X*mat[1][0] + Y*mat[1][1] + Z*mat[1][2] + W*mat[1][3];
      float z = X*mat[2][0] + Y*mat[2][1] + Z*mat[2][2] + W*mat[2][3];
      float w_ = sqrt(x*x + y*y + z*z);
      if (z<0) w_ *= -1;

      nx[i*w+j] = x/w_;
      ny[i*w+j] = y/w_;
      nz[i*w+j] = z/w_;
    }
  }  
}


void histogram(char *inds, float *grid0, float *grid1,
 				int len, int wx, int wy, int wz) 
{
	int i;
	for (i = 0; i < len; i++) {
		{
			int  x = inds[i*4*2+0+0];
			int  y = inds[i*4*2+0+1];
			int  z = inds[i*4*2+0+2];
			char w = inds[i*4*2+0+3];
			if (w==0) return;
			grid0[x*wy*wz + y*wz + z]++;
		}
		{
			int  x = inds[i*4*2+4+0];
			int  y = inds[i*4*2+4+1];
			int  z = inds[i*4*2+4+2];
			char w = inds[i*4*2+4+3];	
			if (w==0) return;
			grid1[x*wy*wz + y*wz + z]++;
		}
	}
}


void histogram_error(float *vote_grid, float *carve_grid, float *new_vote, float *new_carve,
				     float *sums,
					 int wx, int wy, int wz)
{
	int x,y,z;
	int i,j;
	
	for (x = 1; x < wx-1; x++) {
		for (y = 1; y < wy-1; y++) {
			for (z = 1; z < wz-1; z++) {
				float vg =  vote_grid[x*wy*wz + y*wz + z];
				float cg = carve_grid[x*wy*wz + y*wz + z];
				
				for (i = -1; i <= 1; i++) { // tx
					for (j = -1; j <= 1; j++) { // tz
						sums[(i+1)*3+(j+1)] += fmin( new_vote[(x-i)*wy*wz + (y-j)*wz + z], cg);
						sums[(i+1)*3+(j+1)] += fmin(new_carve[(x-i)*wy*wz + (y-j)*wz + z], vg);
						sums[(i+1)*3+(j+1)] -= 0.6 * fmax(new_vote[(x-i)*wy*wz + (y-j)*wz + z],vg);
					}
				}
			}
		}
	} 
}

