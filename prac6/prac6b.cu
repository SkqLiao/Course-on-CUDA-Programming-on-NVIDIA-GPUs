//
// include files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//
// template kernel routine
//

template <class T>
__global__ void my_first_kernel(T *x) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  x[tid] = threadIdx.x;
}

//
// CUDA routine to be called by main code
//

extern int prac6(int nblocks, int nthreads) {
  float *h_x, *d_x;
  int *h_i, *d_i;
  double *h_y, *d_y;
  int nsize, n;

  // allocate memory for arrays

  nsize = nblocks * nthreads;

  h_x = (float *)malloc(nsize * sizeof(float));
  cudaMalloc((void **)&d_x, nsize * sizeof(float));

  h_i = (int *)malloc(nsize * sizeof(int));
  cudaMalloc((void **)&d_i, nsize * sizeof(int));

  h_y = (double *)malloc(nsize * sizeof(double));
  cudaMalloc((void **)&d_y, nsize * sizeof(double));

  // execute kernel for float

  my_first_kernel<<<nblocks, nthreads>>>(d_x);
  cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost);
  for (n = 0; n < nsize; n++) printf(" n,  x  =  %d  %f \n", n, h_x[n]);

  // execute kernel for ints

  my_first_kernel<<<nblocks, nthreads>>>(d_i);
  cudaMemcpy(h_i, d_i, nsize * sizeof(int), cudaMemcpyDeviceToHost);
  for (n = 0; n < nsize; n++) printf(" n,  i  =  %d  %d \n", n, h_i[n]);

  my_first_kernel<<<nblocks, nthreads>>>(d_y);
  cudaMemcpy(h_y, d_y, nsize * sizeof(double), cudaMemcpyDeviceToHost);
  for (n = 0; n < nsize; n++) printf(" n,  i  =  %d  %lf \n", n, h_y[n]);

  // free memory

  cudaFree(d_x);
  free(h_x);
  cudaFree(d_i);
  free(h_i);

  return 0;
}
