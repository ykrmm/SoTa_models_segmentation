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



class FCN8_VGG16(torch.nn.Module): # Archi FCN-8s du papier
    def __init__(self,fcn16):
        super(FCN8_VGG16,self).__init__()
        self.features_pool3 = nn.Sequential(*list(fcn16.features_pool4.children())[:17]) # Features coupé à la couche pool3
        self.features_pool4 = nn.Sequential(*list(fcn16.features_pool4.children())[17:24]) # Features coupé à la couche pool4
        self.features_fc32 = nn.Sequential(*list(fcn16.features_fc32.children())) # Features complet de fc32
        # Classifieur FCN32
        self.conv1 = nn.Conv2d(512, 4096, 7, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.conv2 = nn.Conv2d(4096,4096,kernel_size=(1,1),padding=0)
        self.conv3 = nn.Conv2d(4096,21,kernel_size=(1,1),padding=0)

        # Copy weight from classifier
        self.conv1.weight = nn.Parameter(fcn16.conv1.weight)
        self.conv1.bias = fcn16.conv1.bias
        self.conv2.weight = nn.Parameter(fcn16.conv2.weight)
        self.conv2.bias = fcn16.conv2.bias
        self.conv3.weight = nn.Parameter(fcn16.conv3.weight)
        self.conv3.bias = fcn16.conv3.bias

        # Classifieur branche FCN16

        self.conv_fc16 = nn.Conv2d(512,21,kernel_size=(1,1)) # [14,14]

        # Classifieur branche FCN8

        self.conv_fc8 = nn.Conv2d(256,21,kernel_size=(1,1)) # [28,28]
        

        # Upsampling
        self.final_upsampling = nn.UpsamplingBilinear2d(scale_factor=8)
        self.combine_classif_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        


    def forward(self,x):
        x0 = self.features_pool3(x)
        # branche fc32 et fc16
        x1 = self.features_pool4(x0)

        # branche fc32
        x2 = self.features_fc32(x1)
        x2 = self.conv1(x2)
        x2 = self.relu1(x2)
        x2 = self.dropout(x2)
        x2 = self.conv2(x2)
        x2 = self.relu2(x2)
        x2 = self.dropout(x2)
        x2 = self.conv3(x2) # [7,7]
        x2 = self.combine_classif_upsampling(x2) # [14,14]
        
        # branche fc16
        x3 = self.conv_fc16(x1) # [14,14]

        # branche fc8
        x4 = self.conv_fc8(x0) # [28,28]

        # Somme des classifieurs fc16 et fc32
        x5 = x2+x3 # [14,14]
        x5 = self.combine_classif_upsampling(x5) # [28,28]

        # somme des classifieurs fc8 et fc16+32
        x6 = x5 + x4

        mask = self.final_upsampling(x6)  

        return mask # [224,224]


SAVE_DIR_CNAM = '/home/yannis/Documents/stage_segmentation/SoTa_models_segmentation/model_saved'
SAVE_DIR_HOME = '/Users/ykarmim/Documents/Cours/Master/stage_segmentation/SoTa_models/saved_model'


def get_fcn16(device):
    try:
        fcn16=torch.load(os.path.join(SAVE_DIR_CNAM,'fc16.pt'))
    except:
        print("Le modèle fc16.pt n'a pas encore été enregistré ou le chemin d'accès est erroné.")
    fcn8 = FCN8_VGG16(fcn16)
    fcn8.to(device)

    return fcn8







