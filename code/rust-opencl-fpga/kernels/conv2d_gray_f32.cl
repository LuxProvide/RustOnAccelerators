#define MAX_K 31

kernel void conv2d_gray_f32( __global const float* restrict input, 
                             __global float* restrict output, 
                             __global const float* restrict weights, 
                             const int width,  
                             const int height, 
                             const int kSize) {

    // Cache weights in local memory to reduce repeated global reads.
    __local float kLocal[MAX_K * MAX_K];

    // Fully unroll to encourage hardware replication on FPGA.
    #pragma unroll
    for (int i = 0; i <= kSize * kSize; i++) {
        kLocal[i] = weights[i];
    }

    // Main convolution loop; FPGA can pipeline these operations.
    float acc;
    const int r = (kSize-1) / 2;
    int ii,jj;
    for( int i = r; i < height-r; i++){
       for(int j = r; j < width-r; j++){
            acc = 0;
            #pragma unroll
            for( int k = 0; k < kSize*kSize; k++){
                jj = (k%kSize) + j - r;
                ii = (k/kSize) + i - r;
                acc += input[ii*width+jj] * kLocal[k];
            }
            output[i*width+j] = acc;
        }
    }
  }
