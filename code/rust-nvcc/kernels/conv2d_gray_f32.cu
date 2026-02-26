// Maximum supported kernel width/height for shared-memory staging.
// Keep this >= runtime `kSize`.
#define MAX_K 31

extern "C" __global__ void conv2d_gray_f32(const float *input, float *output,
                                           const float *weights,
                                           const int width, const int height,
                                           const int kSize) {

  // Global pixel coordinates for this thread.
  const int x = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int y = (int)blockIdx.y * blockDim.y + threadIdx.y;
  // Thread coordinates within the current block.
  const int lx = (int)threadIdx.x;
  const int ly = (int)threadIdx.y;
  // Block width (used for linear local indexing).
  const int bx = (int)blockDim.x;

  // Skip threads outside image bounds.
  if (x >= width || y >= height)
    return;

  // Shared-memory copy of convolution weights.
  __shared__ float kLocal[MAX_K * MAX_K];

  // Linear thread index in the block.
  const int lid = ly * bx + lx;
  // First kSize*kSize threads cooperatively load weights.
  if (lid < kSize * kSize) {
    kLocal[lid] = weights[lid];
  }

  // Ensure all weights are visible before convolution.
  __syncthreads();

  // Kernel radius for odd-sized kernels.
  const int r = (kSize - 1) / 2;
  // Accumulator for one output pixel.
  float acc = 0.0f;

  // Convolution with border clamping.
  for (int ky = 0; ky < kSize; ++ky) {
    int iy = y + ky - r;
    iy = max(0, min(iy, height - 1));

    const int rowBase = iy * width;

    for (int kx = 0; kx < kSize; ++kx) {
      int ix = x + kx - r;
      ix = max(0, min(ix, width - 1));

      const float p = input[rowBase + ix];
      // Read weight from shared memory (faster than repeated global loads).
      const float w = kLocal[ky * kSize + kx];
      acc += p * w;
    }
  }
  output[y * width + x] = acc;
}
