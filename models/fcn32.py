import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from torchvision import models
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
from PIL import Image
from collections import Counter



class FCN32_VGG16(torch.nn.Module): # Archi FCN-32s du papier
    def __init__(self,vgg16):
        super(FCN32_VGG16,self).__init__()
        self.features_im_net = nn.Sequential(*list(vgg16.features.children()))
        # On ne considère que les features et on supprime les couches fully connected.
        # pour les remplacer par des conv 
        # on ne garde pas l'avg pool
        # [512,7,7]
        
        self.conv1 = nn.Conv2d(512, 4096, 7, 7,padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.conv2 = nn.Conv2d(4096,4096,kernel_size=(1,1),padding=2)
        self.conv3 = nn.Conv2d(4096,21,kernel_size=(1,1),padding=1)       

        ### get the weights from the fc layer
        #self.conv1.load_state_dict({"weight":fc1["weight"].view(4096, 512, 7, 7),
         #                         "bias":fc1["bias"]})

        self.conv1.weight = nn.Parameter(vgg16.classifier[0].weight.view(4096,512,7,7))
        self.conv1.bias = vgg16.classifier[0].bias
        self.conv2.weight = nn.Parameter(vgg16.classifier[3].weight.unsqueeze(2).unsqueeze(3))
        self.conv2.bias = vgg16.classifier[3].bias


        self.upsampling = nn.Sequential(
            #nn.ConvTranspose2d(21,1,kernel_size=(7,7)), # Pas sur mais pour augmenter d'un facteur 32 pour arriver à 224
            #, il nous faut des fmaps de taille 7 #Pas sur pour dim_out =1 non plus.
            nn.UpsamplingBilinear2d(scale_factor=32)#,
            #nn.Sigmoid()

        )
        


    def forward(self,x):
        x = self.features_im_net(x)
        x = self.conv1(x)
        #print("after conv1 ",x.size())
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        #print("after conv2 ",x.size())
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        #print("after conv3 ",x.size())
        x = self.upsampling(x)  
        #print("after upsampling ",x.size())
        return x # [224,224]



def get_fcn32(device):
    vgg16 = models.vgg16(pretrained=True)
    fcn32 = FCN32_VGG16(vgg16)
    fcn32.to(device)

    return fcn32