from .layers import SpectralConv2d
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torch.autograd import Variable

class CNN_Spectral_Param(nn.Module):
    """
    This class builds and trains the generic and deep CNN architectures 
    as described in section 5.2 of the paper with and without spectral pooling
    """
    def __init__(self, 
                 kernel_size=3, 
                 num_output=10, 
                 batch_size=512, 
                 learning_rate=1e-4,
                 use_spectral_params=False, 
                 random_seed=0, 
                 device_type='cpu', 
                 architecture='generic'):   
        """
        :param kernel_size 
        :param num_output 
        :param architecture 
        """
        super(CNN_Spectral_Param, self).__init__()
        self.kernel_size = kernel_size
        self.num_output = num_output
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_spectral_params = use_spectral_params
        self.architecture = architecture
        self.layers = nn.ModuleList() 
        self.device_type = device_type 
                
        if self.architecture == 'generic': 
            """
            Generic architecture: 
            conv96 > maxpool > conv192 > maxpool > fc1024 > fc512 > softmax  
            """
            self.layers.append(nn.ConstantPad2d(padding=self._get_pad(32), value=0))
            
            if self.use_spectral_params:
                self.layers.append(SpectralConv2d(in_channels=3, out_channels=96, 
                                                  kernel_size=kernel_size, stride=1))
            else: 
                self.layers.append(nn.Conv2d(in_channels=3, out_channels=96, 
                                                  kernel_size=kernel_size, stride=1))
                
            self.layers.append(nn.ReLU())
            self.layers.append(nn.ZeroPad2d((0,1,0,1)))
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            
            self.layers.append(nn.ConstantPad2d(padding=self._get_pad(16), value=0))
            if self.use_spectral_params:
                self.layers.append(SpectralConv2d(in_channels=96, out_channels=192, 
                                                  kernel_size=kernel_size, stride=1))

            else: 
                self.layers.append(nn.Conv2d(in_channels=96, out_channels=192, 
                                             kernel_size=kernel_size, stride=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.ZeroPad2d((0,1,0,1)))
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            
            self.fc1 = nn.Linear(in_features=192*8*8, out_features=1024)
            self.fc2 = nn.Linear(in_features=1024, out_features=512)
            self.fc3 = nn.Linear(in_features=512, out_features=self.num_output)
            
        elif self.architecture == 'deep':
            """
            Deep architecture: .
            conv96 > conv96 > maxpool > conv192 > conv192 > conv192 > maxpool > conv192(1x1) > ga > softmax
            """
            self.layers.append(nn.ConstantPad2d(padding=self._get_pad(32), value=0))
            if self.use_spectral_params: 
                self.layers.append(SpectralConv2d(in_channels=3, out_channels=96, 
                                                  kernel_size=kernel_size, stride=1))
            else: 
                self.layers.append(nn.Conv2d(in_channels=3, out_channels=96, 
                                                  kernel_size=kernel_size, stride=1))
            
            self.layers.append(nn.ReLU())
            self.layers.append(nn.ConstantPad2d(padding=self._get_pad(32), value=0))
            
            if self.use_spectral_params: 
                self.layers.append(SpectralConv2d(in_channels=96, out_channels=96, 
                                                  kernel_size=kernel_size, stride=1))
            else: 
                self.layers.append(nn.Conv2d(in_channels=96, out_channels=96, 
                                                  kernel_size=kernel_size, stride=1))
            self.layers.append(nn.ReLU())
            
            self.layers.append(nn.ZeroPad2d((0,1,0,1)))
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            
            self.layers.append(nn.ConstantPad2d(padding=self._get_pad(16), value=0))
            if self.use_spectral_params:
                self.layers.append(SpectralConv2d(in_channels=96, out_channels=192, 
                                                  kernel_size=kernel_size, stride=1))
            else: 
                self.layers.append(nn.Conv2d(in_channels=96, out_channels=192, 
                                                  kernel_size=kernel_size, stride=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.ConstantPad2d(padding=self._get_pad(16), value=0))
            
            if self.use_spectral_params:
                self.layers.append(SpectralConv2d(in_channels=192, out_channels=192, 
                                                  kernel_size=kernel_size, stride=1))
            else: 
                self.layers.append(nn.Conv2d(in_channels=192, out_channels=192, 
                                                  kernel_size=kernel_size, stride=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.ConstantPad2d(padding=self._get_pad(16), value=0))
            
            if self.use_spectral_params:
                self.layers.append(SpectralConv2d(in_channels=192, out_channels=192, 
                                                  kernel_size=kernel_size, stride=1))
            else: 
                self.layers.append(nn.Conv2d(in_channels=192, out_channels=192, 
                                                  kernel_size=kernel_size, stride=1))
            self.layers.append(nn.ReLU())
            
            self.layers.append(nn.ZeroPad2d((0,1,0,1)))
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            
            self.layers.append(nn.ConstantPad2d(padding=self._get_pad(16), value=0))
            if self.use_spectral_params: 
                self.layers.append(SpectralConv2d(in_channels=192, out_channels=192, 
                                                  kernel_size=1, stride=1))
            else: 
                self.layers.append(nn.Conv2d(in_channels=192, out_channels=192, 
                                                  kernel_size=1, stride=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.AvgPool2d(kernel_size=10))
            
            self.fc1 = nn.Linear(in_features=192, out_features=10)
                
        
        else: 
            raise Exception('Architecture \'' + self.architecture + '\' not defined')
            
            
    def _get_pad(self, orig_size): 
        """
        Get padding size needed for same size output 
        
        Args:
            orig_size: Input size 
        """
        return int(self.kernel_size/2)
    
    
    def forward(self, x): 
        if self.architecture == 'generic':
            for i, l in enumerate(self.layers): 
                x = l(x) 
            x = x.view([-1, self._get_num_features(x)])
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        else: 
            for i, l in enumerate(self.layers): 
                x = l(x)
            x = x.view([-1, self._get_num_features(x)])
            x = self.fc1(x)
            return x

        
    def _get_num_features(self, x): 
        """Return number of flat features"""
        S = x.size()[1:]  # get all dimensions besides batch size 
        features = 1
        for s in S: 
            features *= s    
        return features 
        
        
    def evaluate(self, output, input_y): 
        _,pred = torch.max(output, dim=1) 
        return torch.nonzero(pred.float() - input_y.float()).shape[0]
    
    
    def train(self, 
              traindata, 
              batch_size=512,
              epochs=1):
        
        trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True) 
        alldata = DataLoader(traindata, batch_size=len(traindata))  
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) 
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
                    
                    