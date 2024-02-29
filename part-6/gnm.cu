#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

template <int CHUNK_SIZE>
__global__ void parallelGroupNormMishKernel(float* d_input, float* d_output, float* d_weights, float* d_bias, const int input_length) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ float sharedSum[CHUNK_SIZE]; // Shared memory for storing sum and sum of squares

    //1. Load data into shared memory
    float dataValue = d_input[bid * CHUNK_SIZE + tid]; 
    sharedSum[tid] = dataValue;
    __syncthreads();

    //2. Reduction to compute total sum (mean calculation)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        __syncthreads();
    }

    //3. calculates the square of the difference between the value for the current thread and the mean for the group
    float mean = sharedSum[0] / CHUNK_SIZE;
    float diff = dataValue - mean;
    sharedSum[tid] = diff * diff; //save the squared difference in shared memory
    __syncthreads();

    //4. Reduction to compute total sum of squares (variance calculation)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        __syncthreads();
    }

    //5. calculates the normalized value
    float variance = sharedSum[0] / CHUNK_SIZE;
    int global_td = bid * CHUNK_SIZE + tid;
    int weight_bias_idx = global_td / input_length;
    float invStdDev = rsqrtf(variance + 1e-5); // Using epsilon = 1e-5
     float normVal = (dataValue - mean) * invStdDev;
    
    //6. scale normalized value by weight and bias
    float weightedNormVal = normVal * d_weights[weight_bias_idx] + d_bias[weight_bias_idx];

    //7. apply Mish activation using CUDA special functions and store the result in global memory
    float mishVal = weightedNormVal * tanhf(log1pf(expf(weightedNormVal)));
    d_output[bid * CHUNK_SIZE + tid] = mishVal;
}