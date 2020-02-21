import torch
import numpy as np

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
    
    return iou  # Or thresholded.mean() if you are interested in average across the batch
    


 
iou(next(iter(dataloader_train))[1],next(iter(dataloader_train_val))[1])

   
# ### frequency weighted IU

 """

def mean_iou(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])+np.sum(hist_matrix[:,i])-hist_matrix[i,i]))
        print('class',class_names[i],'miou',class_scores[i])
    print('Mean IOU:',np.mean(class_scores))
    return class_scores

def mean_pixel_accuracy(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])))
        print('class',class_names[i],'mean_pixel_accuracy',class_scores[i])
    return class_scores

def pixel_accuracy(hist_matrix):
    num = np.trace(hist_matrix)
    p_a =  num/max(1,np.sum(hist_matrix).astype('float'))
    print('Pixel accuracy:',p_a)
    return p_a

def freq_weighted_miou(hist_matrix, class_names):
    classes = len(class_names)
    class_scores = np.zeros((classes))
    for i in range(classes):
        class_scores[i] = np.sum(hist_matrix[i,:])*hist_matrix[i,i]/(max(1,np.sum(hist_matrix[i,:])))
    fmiou = np.sum(class_scores)/np.sum(hist_matrix).astype('float')
    print('Frequency Weighted mean accuracy:',fmiou)
    return fmiou
    
 """