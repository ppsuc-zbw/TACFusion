import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from PCD import *
from DetectNet.MultiDetect import *
from skimage.io import imsave

class MMutli_Fusion(nn.Module):
    def __init__(self,):
        super(MMutli_Fusion, self).__init__()
        self.pyramidir = pyramid()
        self.pyramidvi = pyramid()
        self.encoderir=AE_Encoder()
        self.encodervi=AE_Encoder()
        self.fuse1 =Fusion_network(64)
        self.fuse2 =Fusion_network(64)
        self.fuse3 = Fusion_network(64)
        
        self.Detectfeature=Detectfeature()
        self.Featureject=Feature_enject()

        self.YOLOv3=YOLOv3()
        self.asff = ASFF(level=2)
        self.decoder = AE_Decoder()
    def forward(self, data_IR,data_VI,targets=None,train=False):
        
        nbr_feat_l=self.pyramidir(data_IR)
        ref_feat_l=self.pyramidvi(data_VI)
        aligned_featIR3,aligned_featIR2,aligned_featIR1= self.encoderir( ref_feat_l,nbr_feat_l)
        aligned_featVI3,aligned_featVI2,aligned_featVI1= self.encodervi( nbr_feat_l,ref_feat_l)
        fuse3=self.fuse3(aligned_featIR3,aligned_featVI3)
        fuse2=self.fuse2(aligned_featIR2,aligned_featVI2)
        fuse1=self.fuse1(aligned_featIR1,aligned_featVI1)
        fuse_level3,fuse_level2,fuse_level1=self.Detectfeature(fuse3,fuse2,fuse1)
        enjectfuse3=self.Featureject(fuse3,fuse_level3)
        ASFFfeature=self.asff(enjectfuse3, fuse2, fuse1)
        Fusedimage = self.decoder(ASFFfeature)
        
        if train:
   
            loss_dict=self.YOLOv3(fuse_level3,fuse_level2,fuse_level1,targets)

            loss_detect = sum(loss for loss in loss_dict['losses']) 
            # output=self.YOLOv3(fuse_level3,fuse_level2,fuse_level1,targets,train=False)
            # return 检测结果
            return Fusedimage,loss_detect
        else:
            # output=self.YOLOv3(fuse_level3,fuse_level2,fuse_level1,targets,train=False)
            return Fusedimage
       

class pyramid(nn.Module):
    def __init__(self, 
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_extract_block=1,):
        super(pyramid, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
       
        #创建金字塔结构
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        
    def forward(self,data_ma):
        
        b, c, h, w = data_ma.size()

        #main image 特征金字塔
        feat_l1 = self.lrelu(self.conv_first(data_ma))

        featma_l1 = self.feature_extraction(feat_l1)
        # L2
        featma_l2 = self.lrelu(self.conv_l2_1(featma_l1))
        featma_l2 = self.lrelu(self.conv_l2_2(featma_l2))
        # L3
        featma_l3 = self.lrelu(self.conv_l3_1(featma_l2))
        featma_l3 = self.lrelu(self.conv_l3_2(featma_l3))

        featma_l1 = featma_l1.view(b,  -1, h, w)
        featma_l2 = featma_l2.view(b,  -1, h // 2, w // 2)
        featma_l3 = featma_l3.view(b,  -1, h // 4, w // 4)

#         #ref image 特征金字塔
#         featref_l1 = self.lrelu(self.conv_first(data_ref))

#         featref_l1 = self.feature_extraction(featref_l1)
#         # L2
#         featref_l2 = self.lrelu(self.conv_l2_1(featref_l1))
#         featref_l2 = self.lrelu(self.conv_l2_2(featref_l2))
#         # L3
#         featref_l3 = self.lrelu(self.conv_l3_1(featref_l2))
#         featref_l3 = self.lrelu(self.conv_l3_2(featref_l3))

#         featref_l1 = featref_l1.view(b,  -1, h, w)
#         featref_l2 = featref_l2.view(b,  -1, h // 2, w // 2)
#         featref_l3 = featref_l3.view(b,  -1, h // 4, w // 4)
        # PCD alignment
        ref_feat_l = [  # reference feature list
            featma_l1[:,  :, :, :].clone(),
            featma_l2[:,  :, :, :].clone(),
            featma_l3[:,  :, :, :].clone()
        ]
        
      
        return ref_feat_l
   
class AE_Encoder(nn.Module):
    def __init__(self,
                 num_feat=64,
                 deformable_groups=8,):    
        super(AE_Encoder, self).__init__()

        self.pcd_align = PCDAlignment(
            num_feat=num_feat, deformable_groups=deformable_groups)
        
    def forward(self, data_ma,data_ref):
        aligned_feat3,aligned_feat2,aligned_feat1 = self.pcd_align(data_ma, data_ref)   
        return  aligned_feat3,aligned_feat2,aligned_feat1


class Fusion_network(nn.Module):
    def __init__(self, featdim):
        super(Fusion_network, self).__init__()
        

        self.fusion_block1 = FusionBlock_res(featdim)
 

    def forward(self, en_ir, en_vi):
        fuse = self.fusion_block1(en_ir, en_vi)
 
        return fuse


class ASFF(nn.Module):
    def __init__(self, level):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [64, 64, 64]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = add_conv(128, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)  ####expand表示最终网络输出的该尺度特征的维度
        elif level==1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        elif level==2:
            self.compress_level_0 = add_conv(64, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 64, 3, 1)

        compress_c = 8 

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)


    def forward(self, x_level_0, x_level_1, x_level_2):

        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2
        # print(level_0_resized.shape,level_1_resized.shape,level_2_resized.shape)   #12,128,32,32   12,128,64,64    12,128,128,128
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)



        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        # print(levels_weight.shape)
        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)
        return out



class AE_Decoder(nn.Module):
    def __init__(self,
                 ):
        
        super(AE_Decoder, self).__init__()
        self.cov5=nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), 
            # nn.BatchNorm2d(64),
            nn.PReLU(),
            )
        self.cov6=nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            nn.PReLU(),
            )
        self.cov7=nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.PReLU(),
            )
        self.cov8=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, 3, padding=0),
            nn.Tanh(),
            )
    def forward(self,feature):
        # Output1 = self.cov5(feature)   
        Output2 = self.cov6(feature) 
        Output3 = self.cov7(Output2)
        Output4 = self.cov8(Output3)

        return Output4
    
class Feature_enject(nn.Module):
    def __init__(self,
                 ):
        
        super(Feature_enject, self).__init__()
        self.CBAMFUSE=CBAM(64)
        self.CBAMDetect=CBAM(64)
        self.fuse =Fusion_network(64)
        self.conv1 =nn.Conv2d(256, 128, 3, padding=1)
        self.conv2 =nn.Conv2d(128, 64, 3, padding=1)

    def forward(self,fuse,fuse_level):
        fuse_level=self.conv2(self.conv1(fuse_level))
        CBAMfuse=self.CBAMFUSE(fuse)
        CBAMDetect=self.CBAMFUSE(fuse_level)
        enject=self.fuse(CBAMfuse,CBAMDetect)
        return enject
    

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

    