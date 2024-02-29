import torch.nn as nn
import torch 
from torch.utils.cpp_extension import load

#use JIT compiled custom CUDA kernel integration provided by torch.utils module
conv1d_module = load(
    name="conv1d",
    sources=["conv1d.cpp", "conv1d_kernel.cu"],
    verbose=True
)

#initialize network block
class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.out_channels = out_channels
        self.padding = kernel_size // 2
        self.kernel_size = kernel_size

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x, mode='default'):
        if mode == 'custom':
            conv_out = conv1d_module.conv1d(x, self.block[0].weight, self.block[0].bias, self.out_channels, self.padding, self.kernel_size)
            return self.block[1:](conv_out) 
        else:
            return self.block(x)

#define constants
input_channels = 1024
output_channels = 1024
length = 4
batch_size = 1
kernel_size = 5

#perform forward pass with pytorch and custom kernel
conv1d_block = Conv1dBlock(inp_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size).to('cuda')
input = torch.randn((batch_size, input_channels, length)).to('cuda')
custom_conv_gnm_out = conv1d_block(input, mode='custom')
torch_conv_gnm_out = conv1d_block(input, mode='torch')

#verify correctness
assert torch.allclose(custom_conv_gnm_out, torch_conv_gnm_out, atol=0.005), "The outputs from the custom and torch implementations are not close enough."

#define static input tensor that we will copy new inputs into later on
static_input = torch.randn((1, 1024, 4)).to('cuda')

#perform a few warm-up iterations in a side-stream
torch.cuda.synchronize()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for _ in range(3):
        _ = conv1d_block(input, mode='custom')
torch.cuda.current_stream().wait_stream(s)

#graph the network with the static input tensor
conv1d_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(conv1d_graph), torch.no_grad():
    static_output = conv1d_block(static_input, mode='custom') 

#take new input and copy into static input
new_input = torch.randn((batch_size, input_channels, length)).to('cuda')
static_input.copy_(new_input)
conv1d_graph.replay()

#now the 'static_output' tensor has been populated with the new output values. we perform a sanity check with a regular torch forward pass.
ground_truth_output = conv1d_block(new_input, mode='default')
assert torch.allclose(static_output, ground_truth_output, atol=0.005), "The static output and ground truth output are not close enough."
print('Successfully completed Pytorch CUDA graph demonstration.')