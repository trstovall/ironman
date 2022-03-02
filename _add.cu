
#include "_add.cuh"

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
  p[z*4 + 1] = 0;
  p[z*4 + 2] = 0;
  p[z*4 + 3] = 255;
}

int main(void) {
  CPUBitmap bm(DIM, DIM);
  unsigned char *bm_dev;
  
  cudaMalloc((void**)&bm_dev, bm.image_size());
  kernel<<<DIM, DIM, 1>>>(bm_dev);
  cudaMemcpy(bm.get_ptr(), bm_dev, bm.image_size(), cudaMemcpyDeviceToHost);
  
  bm.display_and_exit();
  cudaFree(bm_dev);
}
