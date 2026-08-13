#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello() { printf("Hello from GPU!\n"); }

int main() {
  hello<<<1, 1>>>();

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  err = cudaDeviceSynchronize();

  if (err != cudaSuccess) {
    printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  return 0;
}
