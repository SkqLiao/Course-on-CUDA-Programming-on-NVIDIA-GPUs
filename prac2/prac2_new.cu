#include <curand.h>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "helper_cuda.h"

__constant__ int n;
__constant__ float a, b, c;
__device__ double sum;

__global__ void calc1(float *d_z, float *d_v) {
  int ind = threadIdx.x + n * blockIdx.x * blockDim.x;
  for (int i = 0; i < n; i++) {
    float tmp = a * d_z[ind] * d_z[ind] + b * d_z[ind] + c;
    atomicAdd(&d_v[i + blockIdx.x], tmp);
    ind += blockDim.x;
  }
}

__global__ void calc2(float *d_z) {
  int ind = threadIdx.x + n * blockIdx.x * blockDim.x;
  for (int i = 0; i < n; i++) {
    float tmp = a * d_z[ind] * d_z[ind] + b * d_z[ind] + c;
    atomicAdd(&sum, tmp);
    ind += blockDim.x;
  }
}

struct MyFunction {
  float a, b, c;
  MyFunction(float _a, float _b, float _c) : a(_a), b(_b), c(_c) {}

  __host__ __device__ float operator()(const float &x) const {
    return a * x * x + b * x + c;
  }
};

void getTime(cudaEvent_t &start, cudaEvent_t &stop, const char *msg = NULL) {
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milli;
  cudaEventElapsedTime(&milli, start, stop);
  if (msg != NULL) {
    cudaDeviceSynchronize();
    printf("%s: %f ms", msg, milli);
  }
}

int main(int argc, const char **argv) {
  // initialise card
  findCudaDevice(argc, argv);

  int h_N = 1280000, h_n = 1000;
  float *d_z, *d_v;
  float h_a = 1.0f, h_b = 2.0f, h_c = 3.0f;
  checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(float) * h_N * h_n));
  checkCudaErrors(cudaMalloc((void **)&d_v, sizeof(float) * h_N));

  // random number generation

  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  checkCudaErrors(curandGenerateNormal(gen, d_z, h_N * h_n, 0.0f, 1.0f));
  checkCudaErrors(cudaMemcpyToSymbol(n, &h_n, sizeof(h_n)));
  checkCudaErrors(cudaMemcpyToSymbol(a, &h_a, sizeof(h_a)));
  checkCudaErrors(cudaMemcpyToSymbol(b, &h_b, sizeof(h_b)));
  checkCudaErrors(cudaMemcpyToSymbol(c, &h_c, sizeof(h_c)));
  int type = 4;
  printf("a = %f, b = %f, c = %f\n", h_a, h_b, h_c);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (type == 1) {
    cudaEventRecord(start);
    calc1<<<h_N / 128, 128>>>(d_z, d_v);
    float *h_v = (float *)malloc(sizeof(double) * h_N);
    double h_sum = 0;
    checkCudaErrors(
        cudaMemcpy(h_v, d_v, sizeof(float) * h_N, cudaMemcpyDeviceToHost));
    for (int i = 0; i < h_N; i++) {
      h_sum += h_v[i];
    }

    printf("sum = %lf\n", h_sum / (h_N * h_n));
    free(h_v);
    getTime(start, stop, "Method 1");
  }
  if (type == 2) {
    cudaEventRecord(start);
    calc2<<<h_N / 128, 128>>>(d_z);
    double h_sum;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_sum, sum, sizeof(sum)));
    printf("sum = %lf\n", h_sum / (h_N * h_n));
    getTime(start, stop, "Method 2");
  }
  if (type == 3) {
    cudaEventRecord(start);
    MyFunction f(h_a, h_b, h_c);
    double h_sum = thrust::transform_reduce(
        thrust::device, d_z, d_z + h_N * h_n, f, 0.0, thrust::plus<float>());
    printf("sum = %lf\n", h_sum / (h_N * h_n));
    getTime(start, stop, "Method 3");
  }
  if (type == 4) {
    cudaEventRecord(start);
    double h_sum = 0;
    float *h_z = (float *)malloc(sizeof(float) * h_N * h_n);
    checkCudaErrors(cudaMemcpy(h_z, d_z, sizeof(float) * h_N * h_n,
                               cudaMemcpyDeviceToHost));
    for (int i = 0; i < h_N * h_n; i++) {
      h_sum += h_a * h_z[i] * h_z[i] + h_b * h_z[i] + h_c;
    }
    printf("sum = %lf\n", h_sum / (h_N * h_n));
    getTime(start, stop, "Method 4");
  }
  checkCudaErrors(curandDestroyGenerator(gen));
  checkCudaErrors(cudaFree(d_z));
  checkCudaErrors(cudaFree(d_v));
  return 0;
}