#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

template<typename T>
constexpr T constexpr_max(T a, T b) {
    return (a > b) ? a : b;
}

template <int InputChannels, int InputLength, int Padding, int KernelSize, int ChannelsPerThread>
__global__ void conv1d_optimized(float *d_input, float *d_weight, float *d_bias, float *d_output)
{
    //define constants
    constexpr int SharedMemLength = constexpr_max(InputLength, KernelSize);
    const int blockId = blockIdx.x;
    const int tdIdx = threadIdx.x;
    const int laneIdx = threadIdx.x % warpSize;
    const int warpIdx = threadIdx.x / warpSize;

    const int input_accesses_per_thread = (InputChannels * InputLength)/(4 * blockDim.x); 
    const int weight_accesses_per_thread = (InputChannels * KernelSize)/(blockDim.x); 
    const int weight_offset = blockId * InputChannels * KernelSize;
    const int padded_input_length = InputLength + Padding * 2;
    const int shared_mem_offset_denom = (InputLength * ChannelsPerThread) < 32 ? 32 : (InputLength * ChannelsPerThread);


    //static mem allocations
    float regInput[padded_input_length*ChannelsPerThread] = {0};
    float regFilter[KernelSize*ChannelsPerThread];
    __shared__ float shared_mem[InputChannels * SharedMemLength];

    //load input from global memory into shared memory 
    for (int channelIndex = 0; channelIndex < input_accesses_per_thread; ++channelIndex){
        int td_offset = 4 * (channelIndex * blockDim.x + tdIdx); 
        int smem_offset = td_offset/shared_mem_offset_denom; 
        float4 data = *reinterpret_cast<float4*>(&d_input[td_offset]);
        shared_mem[td_offset + smem_offset + 0] = data.x; 
        shared_mem[td_offset + smem_offset + 1] = data.y; 
        shared_mem[td_offset + smem_offset + 2] = data.z; 
        shared_mem[td_offset + smem_offset + 3] = data.w; 
    }

    __syncthreads(); 

    //load input from shared memory into thread registers
    for (int channelIndex = 0; channelIndex < ChannelsPerThread; ++channelIndex){
        for (int colIndex = 0; colIndex < InputLength; ++colIndex){
            int regIndex = Padding + channelIndex * padded_input_length + colIndex;
            int sharedMemIndex = InputLength * (ChannelsPerThread * tdIdx + channelIndex) + colIndex;
            int smem_offset = sharedMemIndex/shared_mem_offset_denom; 
            regInput[regIndex] = shared_mem[sharedMemIndex + smem_offset];
        }
    }

    __syncthreads(); 

    //load weights from global memory into shared memory 
    for (int channelIndex = 0; channelIndex < weight_accesses_per_thread; ++channelIndex){
        int td_offset = (channelIndex * blockDim.x) + tdIdx;
        shared_mem[td_offset] = d_weight[td_offset + weight_offset];
    }

    __syncthreads(); 

    //load weights from shared memory to thread registers
    for (int channelIndex = 0; channelIndex < ChannelsPerThread; ++channelIndex){
        for (int colIdx = 0; colIdx < KernelSize; ++colIdx){
            int regIndex = channelIndex * KernelSize + colIdx;
            int sharedMemIndex = KernelSize * (ChannelsPerThread * tdIdx + channelIndex) + colIdx;
            regFilter[regIndex] = shared_mem[sharedMemIndex];
        }
    }

    //outer loop iterates over each element in output vector
    for (int tileIdx = 0; tileIdx < InputLength; ++tileIdx){
        float res = 0.0;
        
        //inner loop performs dot product over all kernel positions and accumulates results
        for(int dotIdx = 0; dotIdx < KernelSize; ++dotIdx){
            for(int channelIndex = 0; channelIndex < ChannelsPerThread; ++channelIndex){
                res += regInput[tileIdx + dotIdx + (channelIndex * padded_input_length)] * regFilter[dotIdx + (channelIndex * KernelSize)];
            }
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            res += __shfl_down_sync(0xffffffff, res, offset);
        }
        
        if (threadIdx.x == 0) {
            atomicAdd(&d_output[blockIdx.x * InputLength + tileIdx], d_bias[blockIdx.x]);
        }

        if (laneIdx == 0) {
            atomicAdd(&d_output[blockIdx.x * InputLength + tileIdx], res);
        }
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
    const int channels_per_thread = 4; 

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
    dim3 threads(input_channels/channels_per_thread, 1, 1);

    // Call the kernel
    conv1d_optimized<1024, 4, 2, 5, 4><<<blocks, threads>>>(d_input, d_weight, d_bias, d_output);

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
