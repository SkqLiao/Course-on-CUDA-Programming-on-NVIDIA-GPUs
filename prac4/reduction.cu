#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_cuda.h"

#define WARP_SIZE 32

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

float reduction_gold(float *idata, int len) {
  float sum = 0.0f;
  for (int i = 0; i < len; i++) sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float warpReduceSum(float sum, int blockSize) {
  if (blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
  if (blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
  if (blockSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
  if (blockSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
  if (blockSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

__global__ void reduction(float *g_odata, float *g_idata, int n) {
  int tid = threadIdx.x;

  int i = blockIdx.x * blockDim.x + tid;
  int imax = min(n, (blockIdx.x + 1) * blockDim.x);

  float sum = i < imax ? g_idata[i] : 0.0f;

  extern __shared__ float warpLevelSums[];
  int laneId = threadIdx.x % WARP_SIZE;
  int warpId = threadIdx.x / WARP_SIZE;

  sum = warpReduceSum(sum, blockDim.x);

  if (laneId == 0) warpLevelSums[warpId] = sum;
  __syncthreads();
  sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) atomicAdd(&g_odata[blockIdx.x], sum);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  int num_elements, num_blocks, num_threads, mem_size, shared_mem_size;

  float *h_data, *h_odata, sum;
  float *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_elements = 260817;
  num_threads = 1024;
  num_blocks = (num_elements + num_threads - 1) / num_threads;
  mem_size = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float *)malloc(mem_size);
  h_odata = (float *)malloc(num_blocks * sizeof(float));

  for (int i = 0; i < num_elements; i++)
    h_data[i] = floorf(10.0f * (rand() / (float)RAND_MAX));

  // compute reference solution

  sum = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors(cudaMalloc((void **)&d_idata, mem_size));
  checkCudaErrors(cudaMalloc((void **)&d_odata, num_blocks * sizeof(float)));

  // copy host memory to device input array

  checkCudaErrors(
      cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

  // execute the kernel

  shared_mem_size = sizeof(float) * WARP_SIZE;
  reduction<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata,
                                                          num_elements);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors(cudaMemcpy(h_odata, d_odata, num_blocks * sizeof(float),
                             cudaMemcpyDeviceToHost));

  // check results

  float h_sum = 0.0f;
  for (int i = 0; i < num_blocks; i++) h_sum += h_odata[i];

  printf("reduction error = %f\n", h_sum - sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
