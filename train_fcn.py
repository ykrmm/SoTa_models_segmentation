
   
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






learning_rate = 10e-4
moment = 0.9
optimizer = torch.optim.SGD(fcn32.parameters(),lr=learning_rate,momentum=moment, weight_decay=2e-4)
n_epochs = 175
writer = SummaryWriter()
criterion = nn.CrossEntropyLoss(ignore_index=21)


SAVE_DIR = '/tmp/model'
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
          x = x.to(device)
          mask = mask.to(device)

          fcn32.train()
          pred = fcn32(x)
          #pred = pred.squeeze()
          #mask = mask.squeeze()

          #print('i =',i)

          loss = criterion(pred,mask)
          all_loss_train.append(loss.item())
          loss.backward()

          all_iou.append(float(iou(pred.argmax(dim=1),mask)))

          optimizer.step()
          optimizer.zero_grad()

    loss_train.append(np.array(all_loss_train).mean()) #.item() pour eviter fuite memoire
    iou_train.append(np.array(all_iou).mean())
    all_loss_train = []
    all_iou = []
    for i,(x,mask) in enumerate(dataloader_val):
          x = x.to(device)
          mask = mask.to(device)

          fcn32.eval()
          with torch.no_grad():
            pred = fcn32(x)
          #pred = pred.squeeze()
          #mask = mask.squeeze()

          loss = criterion(pred,mask)
          all_loss_test.append(loss.item())
          all_iou.append(float(iou(pred.argmax(dim=1),mask)))
      
    loss_test.append(np.array(all_loss_test).mean())
    iou_test.append(np.array(all_iou).mean())
    all_loss_test = []
    all_iou = []

    if ep%20==0:
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
        print('something with the plot image function')