#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <csignal>

#include "helper_cuda.h"

///////////////////////////////////////////////////////////////////////////////
// CPU routine
///////////////////////////////////////////////////////////////////////////////

void scan_gold(double *odata, double *isdata, double *iudata,
               const unsigned int len) {
  odata[0] = iudata[0];
  for (int i = 1; i < len; i++) odata[i] = isdata[i] * odata[i - 1] + iudata[i];
}

///////////////////////////////////////////////////////////////////////////////
// GPU routine
///////////////////////////////////////////////////////////////////////////////

__device__ volatile int current_block = 0;
__device__ volatile double current_sum = 0.0f;

__global__ void scan(int N, double *g_odata, double *g_isdata,
                     double *g_iudata) {
  // Dynamically allocated shared memory for scan kernels

  extern __shared__ double tmp[];

  int tid = threadIdx.x;
  int rid = tid + blockDim.x * blockIdx.x;
  int laneId = tid % warpSize;
  int warpId = tid / warpSize;

  if (rid >= N) return;
  // read input into shared memory

  double temp = g_idata[rid];

  // scan up the tree

  for (int d = 1; d < 32; d = 2 * d) {
    __syncthreads();

    double t = __shfl_up_sync(0xffffffff, temp, d);

    if (laneId - d >= 0) {
      temp += t;
    }
  }

  __syncthreads();

  if (laneId == 31) {
    tmp[warpId] = temp;
  }

  __syncthreads();

  if (warpId == 0) {
    double tt = tmp[tid], t;
    for (int d = 1; d < 32; d = 2 * d) {
      t = __shfl_up_sync(0xffffffff, tt, d);

      if (laneId >= d) tt += t;
    }
    tmp[tid] = tt;
  }
  __syncthreads();

  if (warpId > 0) temp += tmp[warpId - 1];
  __syncthreads();
  do {
  } while (current_block < blockIdx.x);
  temp += current_sum;
  __syncthreads();
  if (tid == blockDim.x - 1) {
    current_sum = temp;
    current_block++;
  }
  if (rid < N) g_odata[rid + 1] = temp;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  int num_elements, num_threads, num_blocks, mem_size, shared_mem_size;

  double *h_sdata, *h_udata, *reference;
  double *d_isdata, *d_iudata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_elements = 50000;
  num_threads = 1024;
  num_blocks = (num_elements + num_threads - 1) / num_threads;
  mem_size = sizeof(double) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 1000

  h_sdata = (double *)malloc(mem_size);
  h_udata = (double *)malloc(mem_size);

  for (int i = 0; i < num_elements; i++) {
    h_sdata[i] = 1.5 * (rand() / (double)RAND_MAX);
    h_udata[i] = 2 * (rand() / (double)RAND_MAX);
  }

  // compute reference solution

  reference = (double *)malloc(mem_size);
  scan_gold(reference, h_sdata, h_udata, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors(cudaMalloc((void **)&d_isdata, mem_size));
  checkCudaErrors(cudaMalloc((void **)&d_iudata, mem_size));
  checkCudaErrors(cudaMalloc((void **)&d_odata, mem_size));

  // copy host memory to device input array

  checkCudaErrors(
      cudaMemcpy(d_isdata, h_sdata, mem_size, cudaMemcpyHostToDevice));

  checkCudaErrors(
      cudaMemcpy(d_iudata, h_udata, mem_size, cudaMemcpyHostToDevice));

  // execute the kernel

  shared_mem_size = sizeof(double) * 32;

  scan<<<num_blocks, num_threads, shared_mem_size>>>(num_elements, d_odata,
                                                     d_isdata, d_iudata);

  getLastCudaError("scan kernel execution failed");

  // copy result from device to host

  checkCudaErrors(
      cudaMemcpy(h_data, d_odata, mem_size, cudaMemcpyDeviceToHost));

  // check results

  double err = 0.0;
  for (int i = 0; i < num_elements; i++) {
    err += (h_data[i] - reference[i]) * (h_data[i] - reference[i]);
    // printf("%d %f %f \n", i, h_data[i], reference[i]);
  }
  printf("rms scan error  = %f\n", sqrt(err / num_elements));

  // cleanup memory

  free(h_data);
  free(reference);
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
