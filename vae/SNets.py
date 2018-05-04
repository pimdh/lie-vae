
import torch
import torch.nn as nn

from s2cnn.nn.soft.so3_conv import SO3Convolution
from s2cnn.nn.soft.s2_conv import S2Convolution
from s2cnn.nn.soft.so3_integrate import so3_integrate
from s2cnn.ops.so3_localft import near_identity_grid as so3_near_identity_grid
from s2cnn.ops.s2_localft import near_identity_grid as s2_near_identity_grid

from pytorch_util import MLP


class S2ConvNet(nn.Module):

    def __init__(self, f_list=[1, 10, 10], b_list=[30, 10, 6], activation=nn.ReLU):
        super(S2ConvNet, self).__init__()
        
        #TODO make boolean for integrate
        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()
        
        self.f_list = f_list
        self.b_list = b_list
        
        #self.mlp_dim = mlp_dim.copy()
        
        #self.mlp_dim.insert(0, f_list[-1]*(b_list[-1]*2)**3)
        #print(self.mlp_dim)
        self.activation = activation

        modules = []
        conv1 = S2Convolution(
            nfeature_in= f_list[0],
            nfeature_out=f_list[1],
            b_in=b_list[0],
            b_out=b_list[1],
            grid=grid_s2)
        modules.append(conv1)
        modules.append(self.activation())
    
        for f_in, f_out, b_in, b_out in zip(f_list[1:-1], f_list[2:], b_list[1:-1], b_list[2:]):
            conv = SO3Convolution(
                                    nfeature_in=f_in,
                                    nfeature_out=f_out,
                                    b_in=b_in,
                                    b_out=b_out,
                                    grid=grid_so3)
            
            modules.append(conv)
            modules.append(self.activation())
            
        self.conv_module = nn.Sequential(*modules) 
        
        #self.mlp_module = MLP(H = self.mlp_dim, activation = self.activation)

    def forward(self, x):
       
        x = self.conv_module(x)
        #x = self.mlp_module(x.view(-1,self.mlp_dim[0]))
        #x = so3_integrate(x)

        return x
    
    
class S2DeconvNet(nn.Module):

    def __init__(self, f_list=[10,10,1], b_list=[5,10,30], mlp_dim=[10], max_pooling=True, activation=nn.ReLU):

        super(S2DeconvNet, self).__init__()

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()
        
        self.f_list = f_list
        self.b_list = b_list
        self.mlp_dim = mlp_dim.copy()
        self.mlp_dim.append( f_list[0]*(b_list[0]*2)**3)

        self.activation = activation
        self.max_pooling = max_pooling
        
        self.mlp_module = MLP(H=self.mlp_dim, activation=self.activation)

        modules = []
  
        for f_in, f_out, b_in, b_out in zip(f_list[:-1], f_list[1:], b_list[:-1], b_list[1:]):
        
            modules.append(self.activation())
            
            if b_in < b_out:
                modules.append(torch.nn.Upsample(size=b_out * 2,  mode='nearest'))
                
            conv = SO3Convolution(nfeature_in=f_in,
                                  nfeature_out=f_out,
                                  b_in=b_out,
                                  b_out=b_out,
                                  grid=grid_so3)
            
            modules.append(conv)
         
        self.conv_module = nn.Sequential(*modules) 

    def forward(self, x):
        
        x = self.mlp_module(x)
        shape = x.size()[:-1]
        x = x.view(-1, self.f_list[0], self.b_list[0]*2, self.b_list[0]*2, self.b_list[0]*2)
        
        x = self.conv_module(x)
  
        x = x.view(*shape, self.f_list[-1], self.b_list[-1]*2, self.b_list[-1]*2, self.b_list[-1]*2)

        #x = so3_integrate(x)

        # TODO better reduce gamma channel
        if self.max_pooling:
            x = x.max(dim = -1)[0]
        else:
            x = x.mean(dim = -1)[0]
            
        return x 
