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





#Lambda transforms
def change_type_input(img):
  return img.float()

def to_tensor_target(img):
  img = np.array(img)
  # border
  img[img==255] = 0 # border = background 
  return torch.LongTensor(img)



class VOC_Dataset:
    def __init__(self,batch_size=20,year='2012',dataroot='/home/yannis/Documents/stage_segmentation/dataset'):

      dataroot = os.path.join(dataroot,year)
      self.train_dataset = dset.VOCSegmentation(dataroot,year=year, image_set='train', download=True,transform=transforms.Compose([
                                                    transforms.Resize((384,512)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ]),target_transform=transforms.Compose([
                                                    transforms.Resize((384,512)),
                                                    transforms.Lambda(to_tensor_target)
                                                ]))

      self.val_dataset = dset.VOCSegmentation(dataroot,year=year, image_set='val', download=True,transform=transforms.Compose([
                                                    transforms.Resize((384,512)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ]),target_transform=transforms.Compose([
                                                    transforms.Resize((384,512)),
                                                    transforms.Lambda(to_tensor_target)
                                                ]))
      self.VOC_CLASSES = ('background',  # always index 0
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor')
      self.NUM_CLASSES = len(self.VOC_CLASSES) + 1
      self.dataloader_train = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,shuffle=True)
      self.dataloader_val = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size)

    def get_dataset(self):
      """
        return train dataset and val dataset
      """
      return self.train_dataset,self.val_dataset

    def get_dataloader(self):
      """
        return train dataloader and val dataloader
      """
      return self.dataloader_train,self.dataloader_val

    def get_voc_classe(self):

      return self.VOC_CLASSES,self.NUM_CLASSES
