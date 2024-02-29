
# Inference Optimization for Diffusion Policy - Blog Post Supplement

This repository contains supplemental code to accompany my [blog post](https://www.vrushankdes.ai/diffusion-inference-optimization) on optimizing inference for [Diffusion Policy](https://github.com/real-stanford/diffusion_policy). Special thanks to Cheng Chi and the team at TRI/Columbia for their clean code release, which has been instrumental for pedagogical purposes.

## Contents

- [Part 3 - Profiling a Pytorch Forward Pass](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-iii---profiling-a-pytorch-forward-pass)
  - `diffusion_inference.py`: Code to run an end-to-end evaluation of Diffusion Policy with a 2D Push-T environment, including coarse/fine profiling of program run-time.
  - `log/diffusion/unet_prof.pt.trace.json`: Pytorch profile trace for U-Net forward pass. Can be viewed using `chrome://tracing`.
  - `hta.ipynb`: A Jupyter notebook demonstrating the use of Meta's Holistic Trace Analysis tool for detailed U-Net GPU utilization and kernel-level performance metrics analysis.

- [Part 4 - 1D Convolution in CUDA (Naive)](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-iv---1d-convolution-in-cuda-naive)
  - `conv1d_naive.cu`: Standalone version of the naive 1D convolution kernel.
  - `conv1d_naive.ncu-rep`: NCU profile of the naive 1D convolution kernel's performance.

- [Part 5 - 1D Convolution in CUDA (Optimized)](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-v---1d-convolution-in-cuda-optimized)
  - `conv1d_optimized.cu`: Standalone version of the optimized 1D convolution kernel discussed in the blog post.
  - `conv1d_optimized.ncu-rep`: NCU profile of the optimized 1D convolution kernel's performance.

- [Part 6 - Kernel Fusion in CUDA](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-vi---kernel-fusion-in-cuda)
  - `gnm.cu`: Standalone version of the kernel fusion example discussed in the blog post.

- [Part 7 - A Dive Into DDPMs & a CUDA kernel for Denoising](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-vii---a-dive-into-ddpms-cuda-kernel-for-denoising)
  - `denoise_kernel.cu`: Standalone version of the denoising kernel.

- [Part 8 - Integrating a Custom CUDA Kernel & CUDA Graphs in Pytorch](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-viii---integrating-a-custom-cuda-kernel-cuda-graphs-in-pytorch)
  - `conv1d.cpp`: C++ file with Python binding & CUDA kernel wrapper for the 1D Convolution kernel.
  - `conv1d_kernel.cu`: CUDA file with the Conv1D kernel and driver function.
  - `cuda_graph_example.py`: Script demonstrating how to integrate a custom CUDA kernel into Pytorch, including the use of CUDA graphs to reduce CPU overhead.

