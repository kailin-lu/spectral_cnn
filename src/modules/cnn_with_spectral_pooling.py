"""Implement a CNN with spectral pooling and frequency dropout."""
from .layers import SpectralPool
import numpy as np
import time
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader 
from torch.autograd import Variable
import torch.nn.functional as F

class CNN_Spectral_Pool(nn.Module): 
    def __init__(self, 
                 kernel_size=3,
                 epochs=1, 
                 M=5, 
                 max_num_filters=288, 
                 learning_rate=0.0088,
                 gamma=.85, 
                 num_output=10, 
                 extra_conv_layer=True, 
                 use_global_avg=False,
                 batch_size=512, 
                 device_type='cpu'):
        super(CNN_Spectral_Pool, self).__init__() 
        self.epochs = epochs
        self.M = M 
        self.max_num_filters = max_num_filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_output = num_output
        self.extra_conv_layer = extra_conv_layer
        self.use_global_avg = use_global_avg 
        self.batch_size = batch_size
        self.device_type = device_type
        
        self.conv_pools = nn.ModuleList() 
        self.channels = [3] 
        self.dims = [32]
        
        for m in range(self.M): 
            # Add padding to produce convolution output the same size as input 
            self.conv_pools.append(nn.ConstantPad2d(padding=self._get_pad(self.dims[m]), value=0)) 
            
            # Get number of out_channels
            self.channels.append(self._get_cnn_num_filters(m))
            
            self.conv_pools.append(nn.Conv2d(in_channels=self.channels[m], 
                                             out_channels=self.channels[m+1], 
                                             kernel_size=self.kernel_size, 
                                             stride=1)) 
            self.conv_pools.append(nn.ReLU())
            
            # Calculate pooling dimensions 
            self.dims.append(self._get_sp_dim(self.dims[m]))
            self.conv_pools.append(SpectralPool(filter_height=self.dims[m+1], filter_width=self.dims[m+1]))
            self.conv_pools.append(nn.ReLU()) 
        
        if self.extra_conv_layer: 
            # Add padding to produce convolution output the same size as input 
            self.conv_pools.append(nn.ConstantPad2d(padding=self._get_pad(self.dims[M-1]), value=0)) 
            # Get number of filters 
            self.channels.append(self._get_cnn_num_filters(M))  
            self.conv_pools.append(nn.Conv2d(in_channels=self.channels[M], 
                                             out_channels=self.channels[M+1], 
                                             kernel_size=1, 
                                             stride=1))
        
        # Finally, if we are using global averaging,
        # the last 1x1 convolutional layer is followed by an additional
        # 1x1 convolutional layer with output_dim equal to the possible
        # number of output classes, followed by a final global averaging layer.
        if self.use_global_avg:
            # Add padding to produce convolution output the same size as input 
            self.pad = ConstantPad2d(padding=self._get_pad(self.dims[M-1]), value=0)
            self.conv = nn.Conv2d(in_channels=self.channels[M+1], 
                                  out_channels=self.num_output, 
                                  kernel_size=1, 
                                  stride=1)
            
            self.ga = nn.AvgPool2d(kernel_size=self.dims[M-1]) 
        else: 
            self.fc1 = nn.Linear(256*14*14, self.num_output)

    
    def _get_cnn_num_filters(self, m):
        """
        Get number of filters for CNN.

        Args:
            m: current layer number
        """
        return min(self.max_num_filters,
                   96 + 32 * m)
    
    
    def _get_pad(self, orig_size): 
        """
        Get padding size needed for same size output 
        
        Args:
            orig_size: Input size 
        """
        return int(self.kernel_size/2)

    
    def _get_sp_dim(self, old_size): 
        return int(old_size*self.gamma) 
                                       
    
    def _get_num_features(self, x): 
        """Return number of flat features"""
        S = x.size()[1:]  # get all dimensions besides batch size 
        features = 1
        for s in S: 
            features *= s    
        return features 
    
    
    def forward(self, x):
        for i, l in enumerate(self.conv_pools): 
            x = l(x) 
        x = x.view(-1, self._get_num_features(x)) 
        if self.use_global_avg: 
            x = self.pad(x) 
            x = self.conv(x)
            x = ga(x)
        else: 
            x = self.fc1(x)
        return x
    
    
    def evaluate(self, output, input_y): 
        _,pred = torch.max(output, dim=1) 
        return torch.nonzero(pred.float() - input_y.float()).shape[0]
            
        
    def train(self, 
              traindata, 
              epochs=100,
              batch_size=512): 
        trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True) 
        alldata = DataLoader(traindata, batch_size=len(traindata)) 
                          
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.learning_rate) 
        criterion = nn.CrossEntropyLoss() 
                                        
        for epoch in range(epochs): 
            for i, data in enumerate(trainloader): 
                img, label = data
                if self.device_type == 'gpu' and torch.cuda.is_available(): 
                    img, label = img.cuda(), label.cuda() 
                img, label = Variable(img), Variable(label)

                optimizer.zero_grad()

                output = self.forward(img) 
                loss = criterion(output, label) 
                loss.backward() 
                optimizer.step() 
                
            for i, data in enumerate(alldata): 
                test_img, test_label = data 
                test_img, test_label = Variable(test_img), Variable(test_label)
                output = self.forward(test_img) 
                
            num_error = self.evaluate(output, test_label) 
            error = num_error / len(traindata) 
            print('Epoch {} Train Error: {}'.format(epoch, error)) 
        
            

