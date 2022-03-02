
#include <stdio.h>
#include <stdlib.h>

#define DIM 1000

struct complex {
  float x, i;
  complex(float a, float b) : r(a), i(b) {}
  __device__ float mag2(void) {
    return r * r + i * i;
  }
  __device__ complex operator *(const complex& a) {
    return complex(r*a.r - i*a.i, i*a.r + r*a.i);
  }
  __device__ complex operator +(const complex& a) {
    return complex(r+a.r, i+a.i);
  }
};

__device__ int julia(int x, int y) {
  const float scale = 1.5;
  float jx = scale * (float)(DIM/2 - x) / (DIM/2);
  float jy = scale * (float)(DIM/2 - y) / (DIM/2);
  
  complex c(-0.8, 0.156);
  complex a(jx, jy);
  
  for (int i=0; i<200; i++) {
    a = a * a + c;
    if (a.mag2() > 1000)
      return 0;
  }
  
  return 1;
}

__global__ void kernel(unsigned char *p) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int z = x + y * gridDim.x;
  
  p[z*4 + 0] = 255 * julia(x,y);
//  p[z*4 + 1] = 0;
//  p[z*4 + 2] = 0;
//  p[z*4 + 3] = 255;
}

int main(void) {
  unsigned char *bm = (void*) malloc(DIM*DIM);
  unsigned char *bm_dev;
  
  cudaMalloc((void**)&bm_dev, DIM*DIM);
  kernel<<<DIM, DIM, 1>>>(bm_dev);
  cudaMemcpy(bm, bm_dev, DIM*DIM, cudaMemcpyDeviceToHost);
  
  img = fopen("julia.pgm", "wb");
  if (img == NULL) {
    perror("ERROR: Cannot open output file");
    exit(EXIT_FAILURE);
  }

  fprintf(img, "P5\n");
  fprintf(img, "%d %d\n", DIM, DIM);
  fprintf(img, "255\n");

  for (int i=0 ;i<DIM*DIM; i++)
    fputc(bm[i]);

  cudaFree(bm_dev);
}
