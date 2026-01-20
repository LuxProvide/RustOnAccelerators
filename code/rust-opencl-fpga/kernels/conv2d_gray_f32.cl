#define MAX_K 31

kernel void conv2d_gray_f32( __global const float* input, 
                             __global float* output, 
                             __global const float* weights, 
                             const int width,  
                             const int height, 
                             const int kSize) {

    __local float kLocal[MAX_K * MAX_K];
    #pragma unroll
    for(int i=0; i<= kSize*kSize; i++){

            kLocal[i] = weights[i];

    }

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
