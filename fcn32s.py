# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %%  '
# %%
from IPython import get_ipython

   
# # Fully convolutional Networks for Semantic Segmentation
# https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

 
"""googlecolab = True

if googlecolab:
    from os.path import exists
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
    cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
    accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

    !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
    #!pip install Pillow==4.1.1"""


 
#get_ipython().run_line_magic('matplotlib', 'inline')
import argparse
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


dataroot = '/home/yannis/Documents/stage_segmentation/dataset/voc2007'
#dset.VOCSegmentation(dataroot,year='2007', image_set='train', download=True)



"""Pascal VOC Dataset Segmentation Dataloader"""


VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = len(VOC_CLASSES) + 1


class PascalVOCDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

        self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)
        """
        data = {
                    'image': torch.FloatTensor(image),
                    'mask' : torch.LongTensor(gt_mask)
                    }"""
        data = (torch.FloatTensor(image),torch.LongTensor(gt_mask))

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

            raw_image = Image.open(mask_path).resize((224, 224))
            imx_t = np.array(raw_image).reshape(224*224)
            imx_t[imx_t==255] = len(VOC_CLASSES)

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = np.transpose(raw_image.resize((224, 224)), (2,1,0))
        imx_t = np.array(raw_image, dtype=np.float32)/255.0

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))
        imx_t = np.array(raw_image)
        # border
        imx_t[imx_t==255] = 0 # On délimite la bordure comme étant la classe background. 
        # Sinon on peut mettre = 21 mais on doit spécifier dans la loss que c une valeure à ignorer.
        return imx_t


data_root = os.path.join(dataroot,'VOCdevkit','VOC2007')
list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
img_dir = os.path.join(data_root, "JPEGImages")
mask_dir = os.path.join(data_root, "SegmentationObject")

## Train
train_dataset = PascalVOCDataset(list_file=list_file_path,
                                    img_dir=img_dir,
                                    mask_dir=mask_dir)

## train_val 
list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "trainval.txt")

trainval_dataset = PascalVOCDataset(list_file=list_file_path,
                                    img_dir=img_dir,
                                    mask_dir=mask_dir)
## val
list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "val.txt")

val_dataset = PascalVOCDataset(list_file=list_file_path,
                                    img_dir=img_dir,
                                    mask_dir=mask_dir)



# Set batch_size
batch_size = 20

dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
dataloader_train_val = torch.utils.data.DataLoader(trainval_dataset, batch_size=batch_size)
dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device :",device)

# Plot some training images
"""
real_batch = next(iter(dataloader_train))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].float()[:16], padding=2, normalize=True,nrow=4).cpu()))
plt.show()

rb=torch.stack([real_batch[1],real_batch[1],real_batch[1]],dim=1)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Mask")
plt.imshow(np.transpose(vutils.make_grid(rb.float()[:16], padding=2, normalize=True,nrow=4).cpu(),(1,2,0)))
plt.show()

# Plot some val images
real_batch = next(iter(dataloader_val))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Val Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:16], padding=2, normalize=True,nrow=4).cpu()))
plt.show()
rb=torch.stack([real_batch[1],real_batch[1],real_batch[1]],dim=1)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Val Mask")
plt.imshow(np.transpose(vutils.make_grid(rb.float().to(device)[:16], padding=2, normalize=True,nrow=4).cpu(),(1,2,0)))
plt.show()"""


# ### info datasets

"""
print("Taille dataset train :",len(train_dataset))
print("Taille dataset trainval :",len(trainval_dataset))
print("Taille dataset val :",len(val_dataset))
"""

"""
print(val_dataset.get_class_probability())

sample = val_dataset[2]
image, mask = sample[0], sample[1]

image.transpose_(0, 2)

fig = plt.figure()

a = fig.add_subplot(1,2,1)
plt.imshow(image)

a = fig.add_subplot(1,2,2)
plt.imshow(mask)

plt.show()
"""


#image.size()


#mask.max()


## Metric code
### pixel accuracy 



### mean accuracy

### IOU


SMOOTH = 1e-6
def iou(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch
    


 
iou(next(iter(dataloader_train))[1],next(iter(dataloader_train_val))[1])

   
# ### frequency weighted IU

 


   
# ## FCN for Semantic Segmentation models
  
# ### Premièrement il faut charger un modèle bgg16 préentrainé sur imageNet.


print('Instanciation de VGG16')
vgg16 = models.vgg16(pretrained=True)
vgg16

  
# ### On implémente le modèle FCN -32s du papier en gardant les couches convolutives de vgg16


class FCN32_VGG16(torch.nn.Module): # Archi FCN-32s du papier
    def __init__(self,vgg16):
        super(FCN32_VGG16,self).__init__()
        self.features_im_net = nn.Sequential(*list(vgg16.features.children()))
        # On ne considère que les features et on supprime les couches fully connected.
        # pour les remplacer par des conv 
        # on ne garde pas l'avg pool
        # [512,7,7]
        self.conv_classif = nn.Sequential(
            
            nn.Conv2d(512,4096,kernel_size=(1,1)), #mettre poids w des lineaires vgg
            nn.ReLU(inplace=True),
            nn.Conv2d(4096,4096,kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096,21,kernel_size=(1,1))
        )
        # [21,1,1]
        self.upsampling = nn.Sequential(
            #nn.ConvTranspose2d(21,1,kernel_size=(7,7)), # Pas sur mais pour augmenter d'un facteur 32 pour arriver à 224
            #, il nous faut des fmaps de taille 7 #Pas sur pour dim_out =1 non plus.
            nn.UpsamplingBilinear2d(scale_factor=32)#,
            #nn.Sigmoid()

        )
        


    def forward(self,x):
        x = self.features_im_net(x)
        x = self.conv_classif(x)
        x = self.upsampling(x)        
        return x # [224,224]

  
# ### Rapide test si les dimensions et le modèle sont ok


model = FCN32_VGG16(vgg16)
model.to(device)


"""
batch_test = next(iter(dataloader_train))[0]
output_f = model(batch_test.to(device))




fig = plt.figure()

a = fig.add_subplot(1,2,1)
plt.imshow(np.transpose(batch_test[0].cpu()))

a = fig.add_subplot(1,2,2)
plt.imshow(output_f[0][0].cpu().detach().numpy())

plt.show() # test si image meme dimension et si un batch passe dans le réseau 

"""
# ## Training


learning_rate = 10e-4
moment = 0.9
optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=moment)
n_epochs = 175
writer = SummaryWriter()
criterion = nn.CrossEntropyLoss()



SAVE_DIR = '/home/yannis/Documents/stage_segmentation/SoTa_models_segmentation/model_saved'

for ep in range(n_epochs):
    print("EPOCH",ep)

    for i,(x,mask) in enumerate(dataloader_train):
        if i!=5:
            x = x.to(device)
            mask = mask.to(device)

            model.train()
            pred = model(x)
            pred = pred.squeeze()

            print('i =',i)

            loss = criterion(pred,mask)
            writer.add_scalar('Loss/train',loss.item(),ep) #.item() pour eviter fuite memoire
            writer.add_scalar('IOU/train',iou(pred.max(dim=1)[1],mask),ep)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    torch.cuda.empty_cache()
    model.to(device)
    for i,(x,mask) in enumerate(dataloader_val):
        if i!=5:
            x = x.to(device)
            mask = mask.to(device)

            model.eval()
            pred = model(x)
            pred = pred.squeeze()

            loss = criterion(pred,mask)
            writer.add_scalar('Loss/test',loss.item(),ep)
            writer.add_scalar('IOU/test',iou(pred.max(dim=1)[1],mask),ep)

    if ep%20==0:
        writer.add_image('Mask/seg',pred[0],ep/20)
    
    if ep%10==0:
        try:
            path = os.path.join(SAVE_DIR,'model'+str(ep)+'.pt')
            torch.save(model.cpu().state_dict())
            torch.cuda.empty_cache()
            model.to(device)
        except:
            print("Erreur sauvegarde du modèle epochs:",ep)
        
































pred.size()



VOC_CLASSES[20]



mask



mask[0][100]



mask[0][100]-1
m = mask[0][100]-1



m+=2
m.size()



m=m[m.nonzero()]-1



m.size()



bool(m[m>0])





