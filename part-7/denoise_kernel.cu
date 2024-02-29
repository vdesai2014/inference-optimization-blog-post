#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

__global__ void denoise(
    float* model_output, 
    float* sample, 
    float* diffusion_constants, 
    long* timestep, 
    float* out, 
    float* diffusion_noise
) {
    __shared__ float shared_diffusion_constants[5];
    int tid = threadIdx.x;

    if (tid < 5) {
        shared_diffusion_constants[tid] = diffusion_constants[(*timestep) * 5 + tid];
    }

    __syncthreads();

    float static_pred_original_sample_value = (sample[tid] - shared_diffusion_constants[0] * model_output[tid]) / shared_diffusion_constants[1];
    static_pred_original_sample_value = fmax(-1.0, fmin(1.0, static_pred_original_sample_value));
    float static_pred_prev_sample = shared_diffusion_constants[3] * static_pred_original_sample_value + shared_diffusion_constants[2] * sample[tid];
    if (*timestep != 0){
        static_pred_prev_sample = static_pred_prev_sample + diffusion_noise[tid] * shared_diffusion_constants[4];
    }

    out[tid] = static_pred_prev_sample;
}


torch::Tensor denoise_cuda(
    torch::Tensor& model_output,
    torch::Tensor& sample,
    torch::Tensor& diffusion_constants,
    torch::Tensor& timestep,
    torch::Tensor& diffusion_noise
){
    model_output = model_output.contiguous();
    sample = sample.contiguous();
    diffusion_constants = diffusion_constants.contiguous();
    timestep = timestep.contiguous();
    diffusion_noise = diffusion_noise.contiguous();

    float* d_model_output = model_output.data_ptr<float>();
    float* d_sample = sample.data_ptr<float>();
    float* d_diffusion_constants = diffusion_constants.data_ptr<float>();
    long* d_timestep = timestep.data_ptr<long>();
    float* d_diffusion_noise = diffusion_noise.data_ptr<float>();
    
    auto options = sample.options();
    auto out = torch::empty({1, 2, 16}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    denoise<<<1, 32, 0, stream>>>(d_model_output, d_sample, d_diffusion_constants, d_timestep, out.data_ptr<float>(), d_diffusion_noise);

    return out;
}