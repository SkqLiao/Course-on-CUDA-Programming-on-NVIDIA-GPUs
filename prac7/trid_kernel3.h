__global__ void GPU_trid(int NX, int niter, float *u) {
  float aa, bb, cc, dd, bbi, lambda = 1.0;
  int tid;

  for (int iter = 0; iter < niter; iter++) {
    // set tridiagonal coefficients and r.h.s.

    tid = threadIdx.x;
    bbi = 1.0f / (2.0f + lambda);

    if (tid > 0)
      aa = -bbi;
    else
      aa = 0.0f;

    if (tid < blockDim.x - 1)
      cc = -bbi;
    else
      cc = 0.0f;

    if (iter == 0)
      dd = lambda * u[tid] * bbi;
    else
      dd = lambda * dd * bbi;

    // forward pass
    for (int s = 1; s < 32; s *= 2) {
      bb = 1.0f / (1.0f - aa * __shfl_up_sync(0xffffffff, cc, s) -
                   cc * __shfl_down_sync(0xffffffff, aa, s));
      dd = (dd - aa * __shfl_up_sync(0xffffffff, dd, s) -
            cc * __shfl_down_sync(0xffffffff, dd, s)) *
           bb;
      aa = -aa * __shfl_up_sync(0xffffffff, aa, s) * bb;
      cc = -cc * __shfl_down_sync(0xffffffff, cc, s) * bb;
    }
  }

  u[tid] = dd;
}
