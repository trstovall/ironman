
#include <stdio.h>
#include <stdlib.h>

#define DIM 1000

__device__ int julia(int x, int y) {
  float r0 = 0, r = 1.5 * (float)(DIM/2 - x) / (DIM/2);
  float i0 = 0, i = 1.5 * (float)(DIM/2 - y) / (DIM/2);
  float m = 0;
  
  complex c(-0.8, 0.156);
  complex a(jx, jy);
  
  for (int i=0; i<200; i++) {
    r0 = r*r - i*i - 0.8;
    i0 = i*r + i*r + 0.156;
    m = r*r + i*i;
    if (m > 1000)
      return 0;
    r = r0;
    i = i0;
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
  unsigned char *bm = (unsigned char*) malloc(DIM*DIM);
  unsigned char *bm_dev;
  
  cudaMalloc((void**)&bm_dev, DIM*DIM);
  kernel<<<DIM, DIM, 1>>>(bm_dev);
  cudaDeviceSynchronize();
  cudaMemcpy(bm, bm_dev, DIM*DIM, cudaMemcpyDeviceToHost);
  
  FILE *img = fopen("julia.pgm", "wb");
  if (img == NULL) {
    perror("ERROR: Cannot open output file");
    exit(EXIT_FAILURE);
  }

  fprintf(img, "P5\n");
  fprintf(img, "%d %d\n", DIM, DIM);
  fprintf(img, "255\n");

  for (int i=0 ;i<DIM*DIM; i++)
    fputc(bm[i], img);

  fclose(img);

  cudaFree(bm_dev);
}
