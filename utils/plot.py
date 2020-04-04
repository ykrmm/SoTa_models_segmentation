
import matplotlib.pyplot as plt



def plot_figures(loss_train,loss_test,iou_train,iou_test,model_name='FCN32'):
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.title(model_name,"loss train")
    plt.plot(loss_train)
    plt.xlabel("iterations")
    plt.ylabel("Loss")


    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.title(model_name,"loss test")
    plt.plot(loss_test)
    plt.xlabel("iterations")
    plt.ylabel("Loss")


    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.title(model_name,"mean iou train")
    plt.plot(iou_train)
    plt.xlabel("iterations")
    plt.ylabel("Mean IOU")


    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.title(model_name,"mean iou test")
    plt.plot(iou_test)
    plt.xlabel("iterations")
    plt.ylabel("Mean IOU")