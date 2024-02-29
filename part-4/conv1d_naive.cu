#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

template <int InputChannels, int InputLength, int Padding, int KernelSize>
__global__ void conv1d_naive(float *d_input, float *d_weight, float *d_bias, float *d_output)
{
    //define constants 
    const int padded_input_length = InputLength + 2 * Padding;
    const int weight_offset = blockIdx.x * InputChannels * KernelSize;
    
    //allocate register memory
    float regInput[padded_input_length] = {0};
    float regKernel[KernelSize];

    //load input tensor from global memory into thread registers
    for(int inputIdx = 0; inputIdx < InputLength; ++inputIdx){
        regInput[Padding + inputIdx] = d_input[threadIdx.x * InputLength + inputIdx];
    }

    //load convolution kernels from global memory into thread registers
    for(int kernelIdx = 0; kernelIdx < KernelSize; ++kernelIdx){
        regKernel[kernelIdx] = d_weight[weight_offset + threadIdx.x*KernelSize + kernelIdx];
    }

    //allocate shared memory based on input channel size
    __shared__ float sumReduce[InputChannels];


    //outer loop over input length, calculates each element of the output in one iterations
    for (int tileIdx = 0; tileIdx < InputLength; ++tileIdx) {
        //inner loop performs dot product between conv kernel & input tensor
        float res = 0.0;
        for(int dotIdx = 0; dotIdx < KernelSize; ++dotIdx) {
            res += regInput[tileIdx + dotIdx] * regKernel[dotIdx];
        }
        
        //store the result of the dot product in shared memory
        sumReduce[threadIdx.x] = res;
        
        __syncthreads();
        
        //reduce the sum of the dot product in shared mem
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sumReduce[threadIdx.x] += sumReduce[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        //store the reduced sum of the dot product in shared memory, acccounting for bias
        if (threadIdx.x == 0) {
                d_output[blockIdx.x * InputLength + tileIdx] = sumReduce[0] + d_bias[blockIdx.x];
        }
        __syncthreads();
    }
}

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
    assert(result == cudaSuccess);
  }
  return result;
}

void runKernelWithMockData() {
    // Define tensor dimensions
    const int batch_size = 1024;
    const int input_channels = 1024;
    const int output_channels = 1024;
    const int kernel_size = 5;
    const int length = 4; // Assuming length is the spatial dimension

    // Calculate sizes
    size_t input_size = batch_size * input_channels * length * sizeof(float);
    size_t weight_size = output_channels * input_channels * kernel_size * sizeof(float);
    size_t bias_size = batch_size * output_channels * sizeof(float);
    size_t output_size = 1 * output_channels * length * sizeof(float);

    // Allocate memory on the host
    float* h_input = new float[input_size];
    float* h_weight = new float[weight_size];
    float* h_bias = new float[bias_size];
    float* h_output = new float[output_size];

    // Initialize host memory to zero (mock data)
    std::fill_n(h_input, input_size / sizeof(float), 0.0f);
    std::fill_n(h_weight, weight_size / sizeof(float), 0.0f);
    std::fill_n(h_bias, bias_size / sizeof(float), 0.0f);
    std::fill_n(h_output, output_size / sizeof(float), 0.0f);

    // Allocate memory on the device
    float *d_input, *d_weight, *d_bias, *d_output;
    checkCuda(cudaMalloc((void **)&d_input, input_size));
    checkCuda(cudaMalloc((void **)&d_weight, weight_size));
    checkCuda(cudaMalloc((void **)&d_bias, bias_size));
    checkCuda(cudaMalloc((void **)&d_output, output_size));

    // Copy data from host to device
    checkCuda(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_bias, h_bias, bias_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_output, h_output, output_size, cudaMemcpyHostToDevice));

    // Set up kernel dimensions
    dim3 blocks(output_channels, 1, 1);
    dim3 threads(input_channels, 1, 1);

    // Call the kernel
    conv1d_naive<1024, 4, 2, 5><<<blocks, threads>>>(d_input, d_weight, d_bias, d_output);

    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    checkCuda(cudaDeviceSynchronize());

    // Cleanup
    checkCuda(cudaFree(d_input));
    checkCuda(cudaFree(d_weight));
    checkCuda(cudaFree(d_bias));
    checkCuda(cudaFree(d_output));

    delete[] h_input;
    delete[] h_weight;
    delete[] h_bias;
    delete[] h_output;
}

int main() {
    runKernelWithMockData();
    return 0;
}
