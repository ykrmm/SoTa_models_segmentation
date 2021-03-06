# State of the arts segmentations models

Implementation and experiments are detailed in the notebooks/

## Fully Convolutional Neural Network for Semantic Segmentation
https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

<img src="figures/fcn.png" width="350">

<img src="figures/3fcn.png">





### Results on VOC 2012

#### FCN 32

<img src="figures/meanIoufcn32.png" width="300"> <img src="figures/lossfcn32.png" width="300">  




<img src="figures/inputfcn32.png" width="300">


<img src="figures/masque_fcn32.png" width="300">


#### FCN 16

<img src="figures/meanIOUfcn16.png" width="300"> <img src="figures/loss_fcn16.png" width="300">


<img src="figures/inputfcn32.png" width="300">


<img src="figures/maskFcn16.png" width="300">


#### FCN 8 

<img src="figures/meanIOUfcn8.png" width="300"> <img src="figures/lossFCN8.png" width="300">


<img src="figures/inputfcn32.png" width="300">


<img src="figures/maskTrainFCN8.png" width="300"> <img src="figures/maskPersonFcn8.png" width="300">

Test on a picture of my cat :

<img src="figures/my_cat.jpg" width="300">

#### Table results
I followed the original hyper parameters that were in the paper. Except that I resized the images in 224x224.
That could explain the difference between the score on the paper and mine. 
|    224x224       | FCN 32 | FCN 16 | FCN 8 |
| ----------| ------ | ------ | ----- |
| Mean IOU  | 0.497   |  0.523  |  0.561 |

|    512x384       | FCN 32 | FCN 16 | FCN 8 |
| ----------| ------ | ------ | ----- |
| Mean IOU  | 0.541   |  0.572  |  0.591 |

Original score in the paper 

|           | FCN 32 | FCN 16 | FCN 8 |
| ----------| ------ | ------ | ----- |
| Mean IOU  | 0.594   |  0.624  |  0.627 |



