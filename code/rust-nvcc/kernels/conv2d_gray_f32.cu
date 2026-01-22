
#define MAX_K 31

extern "C" __global__ void conv2d_gray_f32(const float *input, float *output,
                                           const float *weights,
                                           const int width, const int height,
                                           const int kSize) {

  const int x = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int y = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int lx = (int)threadIdx.x;
  const int ly = (int)threadIdx.y;
  const int bx = (int)blockDim.x;

  if (x >= width || y >= height)
    return;

  __shared__ float kLocal[MAX_K * MAX_K];

  const int lid = lx * bx + ly;
  if (lid < kSize * kSize) {
    kLocal[lid] = weights[lid];
  }

  __syncthreads();

  const int r = (kSize - 1) / 2;
  float acc = 0.0f;

  // Convolution sum
  for (int ky = 0; ky < kSize; ++ky) {
    int iy = y + ky - r;
    iy = max(0, min(iy, height - 1));

    const int rowBase = iy * width;

    for (int kx = 0; kx < kSize; ++kx) {
      int ix = x + kx - r;
      ix = max(0, min(ix, width - 1));

      const float p = input[rowBase + ix];
      const float w = kLocal[ky * kSize + kx];
      acc += p * w;
    }
  }
  output[y * width + x] = acc;
}
