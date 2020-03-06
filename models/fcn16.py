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



class FCN16_VGG16(torch.nn.Module): # Archi FCN-16s du papier
    def __init__(self,fcn32):
        super(FCN16_VGG16,self).__init__()
        
        self.features_pool4 = nn.Sequential(*list(fcn32.features_im_net.children())[:24]) # Features coupé à la couche pool4
        self.features_fc32 = nn.Sequential(*list(fcn32.features_im_net.children())[24:]) # Features complet de fc32
        # Classifieur FCN32
        self.conv1 = nn.Conv2d(512, 4096, 7,padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.conv2 = nn.Conv2d(4096,4096,kernel_size=(1,1),padding=0)
        self.conv3 = nn.Conv2d(4096,21,kernel_size=(1,1),padding=0)
        
        # Copy Weight of FCN32
        self.conv1.weight = nn.Parameter(fcn32.conv1.weight)
        self.conv1.bias = fcn32.conv1.bias
        self.conv2.weight = nn.Parameter(fcn32.conv2.weight)
        self.conv2.bias = fcn32.conv2.bias
        self.conv3.weight = nn.Parameter(fcn32.conv3.weight)
        self.conv3.bias = fcn32.conv3.bias

        # Classifieur branche FCN16

        self.conv_fc16 = nn.Conv2d(512,21,kernel_size=(1,1)) # [14,14]
        

        # Upsampling
        self.final_upsampling = nn.UpsamplingBilinear2d(scale_factor=16)
        self.combine_classif_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        


    def forward(self,x):
        x = self.features_pool4(x)
        # branche fc32
        x1 = self.features_fc32(x)
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout(x1)
        x1 = self.conv2(x1)
        x1 = self.relu2(x1)
        x1 = self.dropout(x1)
        x1 = self.conv3(x1) # [7,7]
        x1 = self.combine_classif_upsampling(x1) # [14,14]
        
        # branche fc16
        x2 = self.conv_fc16(x)

        # Somme des classifieurs
        x = x1+x2 
        mask = self.final_upsampling(x)  

        return mask # [224,224]



SAVE_DIR_CNAM = '/home/yannis/Documents/stage_segmentation/SoTa_models_segmentation/model_saved'
SAVE_DIR_HOME = '/Users/ykarmim/Documents/Cours/Master/stage_segmentation/SoTa_models/saved_model'


def get_fcn16(device):
    try:
        fcn32=torch.load(os.path.join(SAVE_DIR_CNAM,'fc32.pt'))
    except:
        print("Le modèle fc32.pt n'a pas encore été enregistré ou le chemin d'accès est erroné.")
    fcn16 = FCN16_VGG16(fcn32)
    fcn16.to(device)

    return fcn16