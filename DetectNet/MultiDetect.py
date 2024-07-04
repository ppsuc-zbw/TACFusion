from torch import nn
import torch
from .network_blocks import ASFFmobile,add_conv,RFBblock,resblock
from .yolov3_head import YOLOv3Head

class YOLOv3(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """
    def __init__(self, num_classes = 6, ignore_thre=0.6, label_smooth = False, rfb=False, vis=False, asff=False):
        """
        Initialization of YOLOv3 class.
        Args:
            ignore_thre (float): used in YOLOLayer.
        """
        super(YOLOv3, self).__init__()
        # self.module_list = create_yolov3_mobilenet_v2(num_classes)

        if asff:
            self.level_0_conv =ASFFmobile(level=0,rfb=rfb,vis=vis)
        else:
            self.level_0_conv =add_conv(in_ch=256, out_ch=512, ksize=3, stride=1,leaky=False)  

        self.level_0_header = YOLOv3Head(anch_mask=[6, 7, 8], n_classes=num_classes, stride=4, in_ch=512,
                              ignore_thre=ignore_thre,label_smooth = label_smooth, rfb=rfb, sep=False)

        if asff:
            self.level_1_conv =ASFFmobile(level=1,rfb=rfb,vis=vis)
        else:
            self.level_1_conv =add_conv(in_ch=128, out_ch=256, ksize=3, stride=1,leaky=False)  

        self.level_1_header = YOLOv3Head(anch_mask=[3, 4, 5], n_classes=num_classes, stride=2, in_ch=256,
                              ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=rfb, sep=False)

        if asff:
            self.level_2_conv =ASFFmobile(level=2,rfb=rfb,vis=vis)
        else:
            self.level_2_conv =add_conv(in_ch=64, out_ch=128, ksize=3, stride=1,leaky=False)  

        self.level_2_header = YOLOv3Head(anch_mask=[0, 1, 2], n_classes=num_classes, stride=1, in_ch=128,
                              ignore_thre=ignore_thre, label_smooth = label_smooth, rfb=rfb, sep=False)
        self.asff = asff
       

    def forward(self, fuse3,fuse2,fuse1, targets=None,train=True, epoch=0):

        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """

        output = []
        anchor_losses= []
        iou_losses = []
        l1_losses = []
        conf_losses = []
        cls_losses = []
        route_layers = []

   

        

        Detect_feature = [  # neighboring feature list
                fuse3[:,  :, :, :].clone(), fuse2[:, :, :, :].clone(),
                fuse1[:,  :, :, :].clone()
            ]


        for l in range(3):
            conver = getattr(self, 'level_{}_conv'.format(l))
            header = getattr(self, 'level_{}_header'.format(l))
            if self.asff:
                f_conv= conver(route_layers[2],route_layers[3],route_layers[4])
            else:
                # f_conv = conver(route_layers[l+2])
                f_conv = conver(Detect_feature[l])
                
            if train:
                x, anchor_loss, iou_loss, l1_loss, conf_loss, cls_loss = header(f_conv, targets)
                anchor_losses.append(anchor_loss)
                iou_losses.append(iou_loss)
                l1_losses.append(l1_loss)
                conf_losses.append(conf_loss)
                cls_losses.append(cls_loss)
            else:
                x = header(f_conv)

            output.append(x)

        if train:
            losses = torch.stack(output, 0).unsqueeze(0).sum(1,keepdim=True)
            anchor_losses = torch.stack(anchor_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            iou_losses = torch.stack(iou_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            l1_losses = torch.stack(l1_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            conf_losses = torch.stack(conf_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            cls_losses = torch.stack(cls_losses, 0).unsqueeze(0).sum(1,keepdim=True)
            loss_dict = dict(
                    losses = losses,
                    anchor_losses = anchor_losses,
                    iou_losses = iou_losses,
                    l1_losses = l1_losses,
                    conf_losses = conf_losses,
                    cls_losses = cls_losses,
            )
            return loss_dict
        else:
            return torch.cat(output, 1)
        
# def reduce_loss_dict(loss_dict):
#     """
#     Reduce the loss dictionary from all processes so that process with rank
#     0 has the averaged results. Returns a dict with the same fields as
#     loss_dict, after reduction.
#     """
#     world_size = get_world_size()
#     if world_size < 2:
#         return loss_dict
#     with torch.no_grad():
#         loss_names = []
#         all_losses = []
#         for k in sorted(loss_dict.keys()):
#             loss_names.append(k)
#             all_losses.append(loss_dict[k])
#         all_losses = torch.stack(all_losses, dim=0)
#         torch.distributed.reduce(all_losses, dst=0)
#         if torch.distributed.get_rank() == 0:
#             # only main process gets accumulated, so only divide by
#             # world_size in this case
#             all_losses /= world_size
#         reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
#     return reduced_losses

class Detectfeature(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ):
       
        super(Detectfeature, self).__init__()


        self.rfb_fuse1  =RFBblock(64)
        self.rfb_fuse2  =RFBblock(128)
        self.rfb_fuse3  =RFBblock(256)
        self.conv1=add_conv(in_ch=64, out_ch=64, ksize=3, stride=1,leaky=False)
        self.conv2=add_conv(in_ch=64, out_ch=128, ksize=3, stride=1,leaky=False)
        self.conv3=add_conv(in_ch=64, out_ch=256, ksize=3, stride=1,leaky=False)

    def forward(self, fuse3,fuse2,fuse1):
        fuse_level1= self.rfb_fuse1(self.conv1(fuse1))
        fuse_level2= self.rfb_fuse2(self.conv2(fuse2))
        fuse_level3= self.rfb_fuse3(self.conv3(fuse3))

        return fuse_level3,fuse_level2,fuse_level1

