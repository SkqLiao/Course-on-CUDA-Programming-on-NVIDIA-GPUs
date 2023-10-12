#include <cuda_runtime.h>

#include <iostream>

int main() {
  int deviceId = 0;  // 选择要查询的CUDA设备的ID，这里假设使用设备0

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceId);

  int maxBlocksPerSM;
  int maxThreadsPerBlock;
  int maxThreadsPerSM;

  // 查询最大可分配线程块数量
  cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor,
                         deviceId);

  // 查询线程块中的线程数量
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock,
                         deviceId);

  // 计算nsize（nblocks * nthreads）
  maxThreadsPerSM = maxBlocksPerSM * maxThreadsPerBlock;

  std::cout << "Max Blocks Per SM: " << maxBlocksPerSM << std::endl;
  std::cout << "Max Threads Per Block: " << maxThreadsPerBlock << std::endl;
  std::cout << "Max Threads Per SM (nsize): " << maxThreadsPerSM << std::endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device name: " << prop.name << std::endl;
  std::cout << "Compute capability: " << prop.major << "." << prop.minor
            << std::endl;

  return 0;
}
