#include <stdlib.h>



void gradient(float *depth, float *dx, float *dy, int h, int w)
{
  int i, j;
  for (i = 1; i < h-1; i++) {
    for (j = 1; j < w-1; j++) {
      dx[i*w+j] = depth[i*w+j+1] - depth[i*w+j-1];
      dy[i*w+j] = depth[(i+1)*w+j] - depth[(i-1)*w+j];
    }
  }
}
