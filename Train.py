# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""
import torchvision.utils as vutils
from torchvision import transforms
import torchvision
from torch.autograd import Variable
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import kornia
from Fusion import MMutli_Fusion
import pytorch_msssim
###############MultiDetect 数据加载
from DetectNet.dataset.vocdataset_GRAY import *
from DetectNet.dataset.dataloading import *
from DetectNet.dataset.data_augment_GRAY import TrainTransform

    

# 
ssim_loss = pytorch_msssim.msssim  #mssim损失被重写在文件夹中    
fusionloss =pytorch_msssim.Fusionloss()
ssim = kornia.losses.SSIMLoss(11,reduction='mean')
# =============================================================================
# Hyperparameters Setting 
# =============================================================================

train_data_path = '/root/autodl-tmp/M3FD_Detection/'
train_path = '/root/Fusion/FUSIONLOSSONLY/'

device = "cuda"
input_size=(240,240)#########M3FD数据集比例为4：3（1024：768 或者 800：480）
batch_size=8
epochs = 100
lr = 1e-5

Train_Image_Number=len(os.listdir(train_data_path+'ir'))

Iter_per_epoch=(Train_Image_Number % batch_size!=0)+Train_Image_Number//batch_size

# =============================================================================
# Models
# =============================================================================
models = MMutli_Fusion()
is_cuda = True
if is_cuda:
    models=models.cuda()

optimizer1 = optim.Adam(models.parameters(), lr = lr)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [epochs//2,epochs], gamma=0.1)

# =============================================================================
# Training
# =============================================================================
print('============ Training Begins ===============')
loss_train=[]
loss_1_train=[]
loss_2_train=[]
loss_3_train=[]

lr_list1=[]
alpha_list=[]

for iteration in range(epochs):
    
    models.train()
    train_sets = 'train'
    dataset = VOCDetection(root=train_data_path,
                     image_sets = train_sets,
                     input_dim = input_size,
                     preproc=TrainTransform(rgb_means=None,std=None,max_labels=30))
    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size,drop_last=False,input_dimension=input_size)
    dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)

    for iter_i,  (imgir,imgvi, targets,img_info,idx) in enumerate(dataloader):
        imgir = Variable(imgir.to(device))
        imgvi = Variable(imgvi.to(device))
        targets = Variable(targets.to(device), requires_grad=False) 

        if is_cuda:
            data_VIS=imgvi.cuda()
            data_IR=imgir.cuda()
        
        optimizer1.zero_grad()
       
       
        
        outputs,loss_detect =models(data_IR,data_VIS,targets,train=True)

        outputs=(outputs+1)/2
        # outputs_save= (outputs+1)/2
        # outputs_save =(outputs_save*255).cpu().detach().numpy().astype('uint8')
        # # print(outputs_save.shape)
        # imsave('Fusionprocess'+'/' +'1.jpg',outputs_save[0:1,:, :, :].squeeze())
                # resolution loss: between fusion image and visible image
        x_ir = Variable(data_IR.data.clone(), requires_grad=False)
        x_vi = Variable(data_VIS.data.clone(), requires_grad=False)

        # print(outputs.shape)
        loss1_value = ssim(outputs, x_ir)+ssim(outputs, x_vi)#可见光细节损失
        loss2_value = 2*fusionloss(x_vi,x_ir,outputs)   
        
        loss = loss1_value+ loss2_value+loss_detect


        loss.backward()
        optimizer1.step()
################检测损失显示#################
#    print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
#                     '[Loss: anchor %.2f, iou %.2f, l1 %.2f, conf %.2f, cls %.2f, imgsize %d, time: %.2f]'
#                 % (epoch, epochs, iter_i, epoch_size, tmp_lr,
#                  sum(anchor_loss for anchor_loss in loss_dict_reduced['anchor_losses']).item(),
#                  sum(iou_loss for iou_loss in loss_dict_reduced['iou_losses']).item(),
#                  sum(l1_loss for l1_loss in loss_dict_reduced['l1_losses']).item(),
#                  sum(conf_loss for conf_loss in loss_dict_reduced['conf_losses']).item(),
#                  sum(cls_loss for cls_loss in loss_dict_reduced['cls_losses']).item(),
#                  input_size[0], end-start),
#                 flush=True)
###############记录损失并打印损失###############    
        los = loss.item()
        los_1= loss1_value.item()
        los_2 = loss2_value.item()
        los_3= loss_detect.item()
        print('Epoch/step: %d/%d, loss: %.7f, lr: %f' %(iteration+1, iter_i+1, los, optimizer1.state_dict()['param_groups'][0]['lr']))
       

        #Save Loss
        loss_train.append(loss.item())
        loss_1_train.append(loss1_value.item())
        loss_2_train.append(loss2_value.item())
        loss_3_train.append(loss_detect.item())
        # mse_loss_IF_train.append(mse_loss_IF.item())
        # Gradient_loss_train.append(Gradient_loss.item())

       



    scheduler1.step()

    lr_list1.append(optimizer1.state_dict()['param_groups'][0]['lr'])

   
    loss_detectc=loss_detect.cpu().detach().numpy() 
    if (iteration+1)%1==0:
        torch.save( {'weight': models.state_dict(), 'epoch':epochs}, 
        os.path.join(train_path,'weight'+str(iteration)+'.pkl'))

        scio.savemat(os.path.join(train_path, 'TrainData.mat'), 
                                 {'Loss': np.array(loss_train),
                                  'loss1' : np.array(loss_1_train),
                                  'loss2': np.array(loss_2_train),
                                   'loss3': np.array(loss_detectc),

                                  })
        #'Base_layer_loss'  : np.array(mse_loss_B_train),
        scio.savemat(os.path.join(train_path, 'TrainData_plot_loss.mat'), 
                                 {'loss_train': np.array(loss_train),
                                  'loss1'  : np.array(loss_1_train),
                                  'loss2': np.array(loss_2_train),
                                   'loss3': np.array(loss_detectc),
                                  })
        #'mse_loss_B_train'  : np.array(mse_loss_B_train),
        # plot
        def Average_loss(loss):
            return [sum(loss[i*Iter_per_epoch:(i+1)*Iter_per_epoch])/Iter_per_epoch for i in range(int(len(loss)/Iter_per_epoch))]
        plt.figure(figsize=[6,4])
        plt.subplot(2,3,1), plt.plot(Average_loss(loss_train)), plt.title('Loss')
        plt.subplot(2,3,2), plt.plot(Average_loss(loss_1_train)), plt.title('loss1')
        # plt.subplot(2,3,3), plt.plot(Average_loss(mse_loss_D_train)), plt.title('Detail_layer_loss')
        plt.subplot(2,3,3), plt.plot(Average_loss(loss_2_train)), plt.title('loss2')
        plt.subplot(2,3,4), plt.plot(Average_loss(loss_detectc)), plt.title('loss3')
        # plt.subplot(2,3,4), plt.plot(Average_loss(mse_loss_IF_train)), plt.title('I_recon_loss')
        # plt.subplot(2,3,5), plt.plot(Average_loss(Gradient_loss_train)), plt.title('Gradient_loss')
        plt.tight_layout()
        plt.savefig(os.path.join(train_path,'curve_per_'+str(iteration)+'.png'))    

