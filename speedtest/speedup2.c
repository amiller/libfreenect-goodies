#include <stdlib.h>

float project(float *depth, float *u, float *v, int h, int w) {
  return 0;
}
void scramble(float v) {
  
}

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
