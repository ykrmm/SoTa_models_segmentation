
   
# # Fully convolutional Networks for Semantic Segmentation
# https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
from PIL import Image
from collections import Counter
from torch.utils.tensorboard import SummaryWriter


# mon code 
from utils import dataset
from metrics import metric
from models import fcn32,fcn16,fcn8



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataroot = '/home/yannis/Documents/stage_segmentation/dataset'
#dataroot = '/Users/ykarmim/Documents/Cours/Master/stage_segmentation/dataset'
voc = dataset.VOC_Dataset(batch_size=1,dataroot=dataroot)
dataloader_train,dataloader_val = voc.get_dataloader()
model = fcn32.get_fcn32(device)

learning_rate = 10e-4
moment = 0.9
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=moment, weight_decay=2e-4)
n_epochs = 175
writer = SummaryWriter()
criterion = nn.CrossEntropyLoss(ignore_index=21)

print_image = False
SAVE_DIR_CNAM = '/home/yannis/Documents/stage_segmentation/SoTa_models_segmentation/model_saved'
SAVE_DIR_HOME = '/Users/ykarmim/Documents/Cours/Master/stage_segmentation/SoTa_models/saved_model'
iou_train = []
loss_train = []
iou_test = []
loss_test = []
all_loss_train = []
all_loss_test = []
all_iou = []
for ep in range(n_epochs):
    print("EPOCH",ep)

    for i,(x,mask) in enumerate(dataloader_train):
          optimizer.zero_grad()
          x = x.to(device)
          mask = mask.to(device)

          model.train()
          pred = model(x)
          #pred = pred.squeeze()
          #mask = mask.squeeze()
          loss = criterion(pred,mask)
          iou,treshold = metric.iou(pred.argmax(dim=1),mask)
          p_acc = metric.scores(pred.argmax(dim=1),mask)["Pixel Accuracy"]
          writer.add_scalar('Loss/train',loss.item(),ep)
          writer.add_scalar('MIOU/train',iou,ep)
          writer.add_scalar('Accuracy/train',p_acc,ep)
          writer.add_scalar('Treshold/train',treshold,ep)
          loss.backward()
          all_loss_train.append(loss.item())
          all_iou.append(iou)

          optimizer.step()
          
    loss_train.append(np.array(all_loss_train).mean()) #.item() pour eviter fuite memoire
    iou_train.append(np.array(all_iou).mean())
    all_loss_train = []
    all_iou = []

    for i,(x,mask) in enumerate(dataloader_val):
          x = x.to(device)
          mask = mask.to(device)

          model.eval()
          with torch.no_grad():
            pred = model(x)
          #pred = pred.squeeze()
          #mask = mask.squeeze()
          loss = criterion(pred,mask)
          iou,treshold = metric.iou(pred.argmax(dim=1),mask)
          p_acc = metric.scores(pred.argmax(dim=1),mask)["Pixel Accuracy"]
          writer.add_scalar('Loss/test',loss.item(),ep)
          writer.add_scalar('MIOU/test',iou,ep)
          writer.add_scalar('Accuracy/test',p_acc,ep)
          writer.add_scalar('Treshold/test',treshold,ep)
          all_loss_test.append(loss.item())
          all_iou.append(iou)
      
    loss_test.append(np.array(all_loss_test).mean())
    iou_test.append(np.array(all_iou).mean())
    all_loss_test = []
    all_iou = []

    if ep%20==0 and print_image:
      try:
        i = 0
        fig = plt.figure()

        a = fig.add_subplot(1,2,1)
        plt.imshow(np.transpose(pred.argmax(dim=1)[i].cpu()))

        a = fig.add_subplot(1,2,2)
        plt.imshow(mask.cpu().detach().numpy()[i])

        plt.show()
        class_pred = []
        class_mask = []

        for p in pred.argmax(dim=1)[i].unique():
          class_pred.append(VOC_CLASSES[int(p)])
        for m in mask[i].unique():
          class_mask.append(VOC_CLASSES[int(m)])

        print("Classe prédite : ",class_pred)
        print("Classe réelle : ",class_mask)
      except:
        print('something wrong with the plot image function')
      try:
        writer.add_image('Mask/pred',pred.argmax(dim=1)[i].cpu())
        writer.add_image('Mask/ground_truth',mask.cpu())
        writer.add_image('Mask/input',x.cpu())
      except:
        print('something wrong with writer.add_image')

try:
  torch.save(model,SAVE_DIR_CNAM)
except:
  try:
    torch.save(model,SAVE_DIR_HOME)
  except:
    print('le modèle n\'a pas pu etre enregistré')

metric.evaluate_model(model,dataloader_val,device=device)