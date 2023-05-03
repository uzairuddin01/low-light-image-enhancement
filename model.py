import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
torch.set_printoptions(linewidth=120)



class NNetwork(nn.Module):
    def __init__(self):
        super(NNetwork,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,padding=0)#16
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,padding=0)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=8,out_channels=4,kernel_size=4)
        self.deconv2 = nn.ConvTranspose2d(in_channels=4,out_channels=3,kernel_size=4)
        
    
    def forward(self,t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=3, stride = 1)
        
        t = self.conv2(t)
        t = F.relu(t)
        
        t = self.deconv1(t)
        t = F.relu(t)

        t = self.deconv2(t)
        t = F.relu(t)
        
        return t