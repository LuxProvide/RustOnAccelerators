#define MAX_K 31

kernel void conv2d_gray_f32( __global const float* input, 
                             __global float* output, 
                             __global const float* weights, 
                             const int width,  
                             const int height, 
                             const int kSize) {

    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    const int lx = (int)get_local_id(0);
    const int ly = (int)get_local_id(1);
    const int by = (int)get_local_size(1);


    if (x >= width || y >= height) return;


    __local float kLocal[MAX_K * MAX_K];

    const int lid = lx * by + ly;
    if(lid < kSize * kSize){
     kLocal[lid] = weights[lid];
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    const int r = (kSize-1) / 2;
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
            acc += p*w; 
        }
    }

    output[y * width + x] = acc;
  }
