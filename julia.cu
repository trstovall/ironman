
#include <stdio.h>
#include <stdlib.h>

#define DIM 1000

__device__ int julia(int x, int y) {
  float r0 = 0, r = 1.5 * (float)(DIM/2 - x) / (DIM/2);
  float i0 = 0, i = 1.5 * (float)(DIM/2 - y) / (DIM/2);
  float m = 0;
  
  for (int n=0; n<200; n++) {
    // printf("n: %d r0: %.2f i0: %.2f\n", n, r0, i0);
    r0 = r*r - i*i - 0.8;
    i0 = i*r + i*r + 0.156;
    m = r0*r0 + i0*i0;
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
  
  if (x < DIM && y < DIM) {
    p[z] = 255 * julia(x, y);
    // printf("p: %d z: %d\n", p[z], z);
  }
}

int main(void) {
  unsigned char *bm = (unsigned char*) malloc(DIM*DIM);
  unsigned char *bm_dev;
  
  cudaError_t err;
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    exit(EXIT_FAILURE);
  }

  err = cudaMalloc((void**)&bm_dev, DIM*DIM);
  if (err != cudaSuccess) {
    perror("ERROR: cudaMalloc");
    exit(EXIT_FAILURE);
  }

  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    perror("ERROR: cudaStreamCreate");
    exit(EXIT_FAILURE);
  }

  dim3 grid(DIM, DIM);
  kernel<<<grid, 1, 0, stream>>>(bm_dev);
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    perror("ERROR: cudaStreamSynchronize");
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(bm, bm_dev, DIM*DIM, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    perror("ERROR: cudaMemcpy D2H");
    exit(EXIT_FAILURE);
  }

  err = cudaStreamDestroy(stream);
  if (err != cudaSuccess) {
    perror("ERROR: cudaStreamDestroy");
    exit(EXIT_FAILURE);
  }

  FILE *img = fopen("julia.pgm", "wb");
  if (img == NULL) {
    perror("ERROR: Cannot open output file");
    exit(EXIT_FAILURE);
  }

  fprintf(img, "P5\n");
  fprintf(img, "%d %d\n", DIM, DIM);
  fprintf(img, "255\n");

  for (int i=0; i<DIM*DIM; i++)
    fputc(bm[i], img);
    // fprintf(img, "%d\n", bm[i]);

  fclose(img);

  cudaFree(bm_dev);
}
