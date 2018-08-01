"""Implements layers for the spectral CNN."""
from .spectral_pool import _crop_spectrum, _treat_corner_cases, _remove_redundancy, _pad_spectrum, _recover_map
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# Import pytorch-fft if available
try:
    import pytorch_fft.fft as fft 
except ModuleNotFoundError: 
    pass
    

class spectral_pool_cuda(torch.autograd.Function):
    """function for spectral pool using CUDA tensors and pytorch-fft""" 
    
    @staticmethod
    def forward(ctx, input, filter_height, filter_width): 
        ctx.save_for_backward(input) 
        y_spectral = fft.fft2(input) 
        y_spectral = _crop_spectrum(y_spectral) 
        y_spectral = _treat_corner_cases(y_spectral) 
        cropped = fft.ifft2(y_spectral) 
        return cropped.float()
    
    @staticmethod 
    def backward(ctx, grad_output): 
        # Retrieve original tensor shape for _pad_spectrum
        orig = ctx.saved_variables 
        orig_shape = orig[0].shape[3]
        
        z = fft.fft2(z) 
        z = _remove_redundancy(z) 
        z = _pad_spectrum(z, orig_shape) 
        z = _recover_map(z) 
        return Variable(z), None, None 
    

class spectral_pool(torch.autograd.Function):
    """function for spectral pool"""
    
    @staticmethod
    def forward(ctx, input, filter_height, filter_width):
        """
        In the forward method we receive a Tensor which is converted to a numpy array. 
        This is then transformed into the frequency domain and cropped according to 
        Algorithm 1 in section 3. 
        
        Args:
            x: Tensor with shape (b, c, h, w) 
            filter_height: height of filtered image
            filter_width: width of filtered image 
        """
        ctx.save_for_backward(input)
        
        # convert to numpy to use fft 
        if not isinstance(input, np.ndarray): 
            input = input.numpy() 
        y_spectral = np.fft.fft2(input) 
        # center the frequency 
        y_spectral = np.fft.fftshift(y_spectral)     
        
        y_spectral = _crop_spectrum(y_spectral, filter_height, filter_width) 
        y_spectral,_ = _treat_corner_cases(y_spectral) 
        cropped = np.abs(np.fft.ifft2(y_spectral))
        return torch.from_numpy(cropped).float()
        
        
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve original tensor shape for _pad_spectrum
        orig = ctx.saved_variables 
        orig_shape = orig[0].shape[3]
        
        z = grad_output.data.numpy() 
        z = np.fft.fft2(z) 
        z = _remove_redundancy(z) 
        z = _pad_spectrum(z, orig_shape) 
        z = _recover_map(z) 
        return Variable(torch.from_numpy(z).float()), None, None 
         

class SpectralPool(nn.Module):
    """Spectral pooling layer."""
    
    def __init__(self, filter_height, filter_width):
        super(SpectralPool, self).__init__()
        self.filter_height = filter_height
        self.filter_width = filter_width
        
    def forward(self, input): 
        return spectral_pool.apply(input, self.filter_height, self.filter_width)

            
class SpectralConv2d(nn.Module):
    """
    Initialize filter in spectral domain and use inverse DFT to convert to spatial 
    Perform standard convolution afterwards 
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding=0):

        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding
        
        # Initialize weight
        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels,
                                             self.kernel_size, self.kernel_size), requires_grad=True)
        real_init = np.random.rand(self.out_channels, self.in_channels,self.kernel_size, self.kernel_size) 
        real_fft = np.abs(np.fft.fft2(real_init)) 
        self.weight.data = torch.from_numpy(real_fft).float()

        # Initialize bias 
        self.bias = Parameter(torch.Tensor(self.out_channels), requires_grad=True) 
        nn.init.constant(self.bias.data, val=0) 
            
    def forward(self, input): 
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding) 
    


        

