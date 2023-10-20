//
// Program to perform Backward Euler time-marching on a 1D grid
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_cuda.h"

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

// #include "trid_kernel.h"
// #include "trid_kernel2.h"
#include "trid_kernel3.h"

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void gold_trid(int, int, float *, float *);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  int NX = 32, niter = 2;

  float *h_u, *h_v, *h_c, *d_u;

  // initialise card

  findCudaDevice(argc, argv);

  // allocate memory on host and device

  h_u = (float *)malloc(sizeof(float) * NX);
  h_v = (float *)malloc(sizeof(float) * NX);
  h_c = (float *)malloc(sizeof(float) * NX);

  checkCudaErrors(cudaMalloc((void **)&d_u, sizeof(float) * NX));

  // GPU execution

  for (int i = 0; i < NX; i++) h_u[i] = 1.0f;

  checkCudaErrors(
      cudaMemcpy(d_u, h_u, sizeof(float) * NX, cudaMemcpyHostToDevice));

  // int shared_mem_size = sizeof(float) * NX * 3;

  // GPU_trid<<<1, NX, shared_mem_size>>>(NX, niter, d_u);

  GPU_trid<<<1, NX>>>(NX, niter, d_u);

  checkCudaErrors(
      cudaMemcpy(h_u, d_u, sizeof(float) * NX, cudaMemcpyDeviceToHost));

  // CPU execution

  for (int i = 0; i < NX; i++) h_v[i] = 1.0f;

  gold_trid(NX, niter, h_v, h_c);

  // print out array

  for (int i = 0; i < NX; i++) {
    printf(" %d  %f  %f  %f \n", i, h_u[i], h_v[i], h_u[i] - h_v[i]);
  }

  // Release GPU and CPU memory

  checkCudaErrors(cudaFree(d_u));

  free(h_u);
  free(h_v);
  free(h_c);
}
