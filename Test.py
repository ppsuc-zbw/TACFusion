# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""

from Fusion import MMutli_Fusion


from torch.autograd import Variable
import numpy as np
import torch
import os
from PIL import Image
from skimage.io import imsave
import cv2

import numpy as np
import torch

import torch.nn.functional as F

device='cuda'
# =============================================================================
# Test Details 
# =============================================================================
device='cuda'
test_data_path = '/root/autodl-tmp/map/'
modelpath=  "/root/Fusion/Code/Epoch60.pkl"
imagelist=os.listdir(test_data_path+'/ir/')
# Determine the number of files
Test_Image_Number=len(os.listdir(test_data_path+'/ir/'))

# =============================================================================
# Test
# =============================================================================

def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]


def Test_fusion(img_test1,img_test2,addition_mode='net'):
    MMutli_Fusion1 = MMutli_Fusion().to(device)
    MMutli_Fusion1.load_state_dict(torch.load(
          modelpath
            )['weight'])    

    MMutli_Fusion1.eval()

    img_test1 = np.array(img_test1, dtype='float32')/255# 将其转换为一个矩阵
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))

    img_test2 = np.array(img_test2, dtype='float32')/255 # 将其转换为一个矩阵
    img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))
    
    img_test1=img_test1.cuda()
    img_test2=img_test2.cuda()
    
    with torch.no_grad():
        Out=MMutli_Fusion1(img_test1,img_test2,train=False)

   
        Out=127.5*(Out+1)
    return output_img(Out)
# =============================================================================

for i in range(Test_Image_Number):
   
    Test_IR = Image.open(test_data_path+'/ir/'+str(imagelist[i])) # infrared image
    Test_Vis = Image.open(test_data_path+'/vi/'+str(imagelist[i])) # visible image
 
    Fusion_image = Test_fusion(Test_IR,Test_Vis)
    #print((Fusion_image.shape))
    Fusion_image=Fusion_image.astype(np.uint8)
    print(imagelist[i])

    imsave('/root/autodl-tmp/FMB(S)/Epoch50/'+imagelist[i],Fusion_image)

