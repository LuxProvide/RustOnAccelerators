extern "C" __global__
void vec_add(const float* a, const float* b, float* out, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

