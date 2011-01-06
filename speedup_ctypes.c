#include <stdlib.h>
#include <math.h>
#include <stdio.h>


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


void lattice(float *depth, 
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

