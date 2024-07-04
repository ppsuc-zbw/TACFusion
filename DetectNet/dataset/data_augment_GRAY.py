"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
import random
import math
from ..utils.utils import matrix_iou, visual

#DEBUG = True
DEBUG = False

def _crop(imageir, imagevi,boxes, labels, ratios = None):
    height, width= imageir.shape

    if len(boxes)== 0:
        return imageir, imagevi,boxes, labels, ratios

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return  imageir, imagevi, boxes, labels, ratios

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3,1.)
            min_ratio = max(0.5, scale*scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)


            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_irt = imageir[roi[1]:roi[3], roi[0]:roi[2]]
            image_vit = imagevi[roi[1]:roi[3], roi[0]:roi[2]]
            

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                     .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if ratios is not None:
                ratios_t = ratios[mask].copy()
            else:
                ratios_t=None

            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_irt, image_vit,boxes_t,labels_t, ratios_t


def _distort(imageir,imagevi):
    def _convert(imageir,imagevi, alpha=1, beta=0):
        
        tmp = imageir.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        imageir[:] = tmp
        
        tmp1 = imagevi.astype(float) * alpha + beta
        tmp1[tmp1 < 0] = 0
        tmp1[tmp1 > 255] = 255
        imagevi[:] = tmp1

    imageir = imageir.copy()
    imagevi = imagevi.copy()
    

    if random.randrange(2):
        _convert(imageir, imagevi,beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(imageir, imagevi, alpha=random.uniform(0.5, 1.5))
############该部分代码进行HSV色彩空间增强，因此灰度图像应该注释掉###########
    # imageir = cv2.cvtColor(imageir, cv2.COLOR_BGR2HSV)
    # imagevi = cv2.cvtColor(imagevi, cv2.COLOR_BGR2HSV)

    # if random.randrange(2):
    #     rand=random.randint(-18, 18)
    #     tmp = imageir[:, :, 0].astype(int) + rand
    #     tmp %= 180
    #     imageir[:, :, 0] = tmp
        
    #     tmp1 = imagevi[:, :, 0].astype(int) + rand
    #     tmp1 %= 180
    #     imagevi[:, :, 0] = tmp1

    # if random.randrange(2):
    #     _convert(imageir[:, :, 1], imagevi[:, :, 1],alpha=random.uniform(0.5, 1.5))

    # # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    # imageir = cv2.cvtColor(imageir, cv2.COLOR_HSV2BGR)
    # imagevi = cv2.cvtColor(imagevi, cv2.COLOR_HSV2BGR)
############该部分代码进行HSV色彩空间增强，因此灰度图像应该注释掉###########
        # print(imageir.shape)
    return imageir,imagevi


def _expand(imageir, imagevi,boxes,fill, p):
    if random.random() > p:
        return imageir, imagevi,boxes

    height, width = imageir.shape
    for _ in range(50):
        scale = random.uniform(1,4)

        min_ratio = max(0.5, 1./scale/scale)
        max_ratio = min(2, scale*scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale*ratio
        hs = scale/ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)
        # print(h,w)
        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)


        expand_image1 = np.empty(
            (h, w),
            dtype=imageir.dtype)
        expand_image1[:, :] = fill
        expand_image1[top:top + height, left:left + width] = imageir
        
        expand_image2 = np.empty(
            (h, w),
            dtype=imagevi.dtype)
        expand_image2[:, :] = fill
        expand_image2[top:top + height, left:left + width] = imagevi
        
        imageir = expand_image1
        imagevi = expand_image2
        

        return imageir,imagevi, boxes_t


def _mirror(imageir,imagevi, boxes):
    _, width= imageir.shape
    if random.randrange(2):
        imageir = imageir[:, ::-1]
        imagevi = imagevi[:, ::-1]
        
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return imageir, imagevi,boxes


def _random_affine(imgir,imgvi, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    #height = max(img.shape[0], img.shape[1]) + border * 2
    height, width= imgir.shape 

    # Rotation and Scale
    R_ir = np.eye(3)
    R_vi = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R_ir[:2] = cv2.getRotationMatrix2D(angle=a, center=(imgir.shape[1] / 2, imgir.shape[0] / 2), scale=s)
    R_vi[:2] = cv2.getRotationMatrix2D(angle=a, center=(imgvi.shape[1] / 2, imgvi.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * imgir.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * imgir.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R_ir  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imir = cv2.warpPerspective(imgir, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    imvi = cv2.warpPerspective(imgvi, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 0:4].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            x1 = np.clip(xy[:,0], 0, width)
            y1 = np.clip(xy[:,1], 0, height)
            x2 = np.clip(xy[:,2], 0, width)
            y2 = np.clip(xy[:,3], 0, height)
            boxes = np.concatenate((x1, y1, x2, y2)).reshape(4, n).T

        return imir,imvi, boxes, M
    else:
        return imir,imvi

def preproc_for_test(imageir,imagevi, input_size, mean, std):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    imageir = cv2.resize(imageir, input_size,interpolation=interp_method)
    imagevi  = cv2.resize(imagevi , input_size,interpolation=interp_method)
    
    imageir = imageir.astype(np.float32)
    # imageir = imageir[:,:]
    imageir /= 255.
    # print(imageir.shape,'################')
    imageir=np.expand_dims(imageir,axis=0)
    
    # print(imagevi.shape)
    imagevi = imagevi.astype(np.float32)
    
    # imagevi = imagevi[:,:,::]
    imagevi /= 255.
    imagevi=np.expand_dims(imagevi,axis=0)
    # print(imagevi.shape)
    # mean=np.expand_dims(mean,axis=0)
    # std=np.expand_dims(std,axis=0)
    if mean is not None:
        imagevi -= mean
        imageir -= mean
    if std is not None:
        imagevi /= std
        imageir /= std
    # print(imageir.transpose(2, 0, 1).shape)
    return imageir,imagevi


class TrainTransform(object):

    def __init__(self, p=0.5, rgb_means=None, std = None,max_labels=50):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels=max_labels

    def __call__(self, imageir,imagevi, targets, input_dim):
        boxes = targets[:,:4].copy()
        labels = targets[:,4].copy()
        if targets.shape[1] > 5:
            mixup=True
            ratios = targets[:,-1].copy()
            ratios_o = targets[:,-1].copy()
        else:
            mixup=False
            ratios = None
            ratios_o = None
        lshape = 6 if mixup else 5
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels,lshape),dtype=np.float32)
            imageir,imagevi = preproc_for_test(imageir, imagevi,input_dim, self.means, self.std)
            imageir = np.ascontiguousarray(imageir, dtype=np.float32)
            imagevi = np.ascontiguousarray(imagevi, dtype=np.float32)
            return torch.from_numpy(imageir), torch.from_numpy(imagevi),torch.from_numpy(targets)

        image_o = imageir.copy()
        targets_o = targets.copy()
        height_o, width_o = image_o.shape
        boxes_o = targets_o[:,:4]
        labels_o = targets_o[:,4]
        b_x_o = (boxes_o[:, 2] + boxes_o[:, 0])*.5
        b_y_o = (boxes_o[:, 3] + boxes_o[:, 1])*.5
        b_w_o = (boxes_o[:, 2] - boxes_o[:, 0])*1.
        b_h_o = (boxes_o[:, 3] - boxes_o[:, 1])*1.
        boxes_o[:,0] = b_x_o
        boxes_o[:,1] = b_y_o
        boxes_o[:,2] = b_w_o
        boxes_o[:,3] = b_h_o
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        boxes_o[:, 0::2] *= input_dim[0]
        boxes_o[:, 1::2] *= input_dim[1]
        #labels_o = np.expand_dims(labels_o,1)
        #targets_o = np.hstack((boxes_o,labels_o))
        #targets_o = np.hstack((labels_o,boxes_o))

        # image_irt,imagevit = _distort(imageir,imagevi)
        image_irt=imageir
        imagevit =imagevi
        if self.means is not None:
            fill = [m * 255 for m in self.means]
            fill = fill[::-1]
            # print(fill)
        else:
            fill = 127.5
        # image_irt, imagevit,boxes = _expand(image_irt,imagevit,boxes, fill, self.p)
        # image_irt, imagevit,boxes, labels, ratios = _crop(image_irt, imagevit,boxes, labels, ratios)
        # image_irt,imagevit, boxes = _mirror(image_irt, imagevit,boxes)

        # if random.randrange(2):
        #     image_irt,imagevit, boxes, _ = _random_affine(image_irt,imagevit, boxes, borderValue=fill)

        height, width= image_irt.shape

        if DEBUG:
            image_t = np.ascontiguousarray(image_t, dtype=np.uint8)
            img = visual(image_t, boxes,labels) 
            cv2.imshow('DEBUG', img)
            cv2.waitKey(0)

        image_irt,imagevit = preproc_for_test(image_irt, imagevit,input_dim, self.means, self.std)
        boxes  = boxes.copy()
        b_x = (boxes[:, 2] + boxes[:, 0])*.5
        b_y = (boxes[:, 3] + boxes[:, 1])*.5
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        boxes[:,0] = b_x
        boxes[:,1] = b_y
        boxes[:,2] = b_w
        boxes[:,3] = b_h
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        boxes[:, 0::2] *= input_dim[0]
        boxes[:, 1::2] *= input_dim[1]
        mask_b= np.minimum(boxes[:,2], boxes[:,3]) > 6
        #mask_b= (boxes[:,2]*boxes[:,3]) > 32**2
        #mask_b= (boxes[:,2]*boxes[:,3]) > 48**2
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()
        if mixup:
            ratios_t = ratios[mask_b].copy()

        '''
        if len(boxes_t)==0:
            targets = np.zeros((self.max_labels,lshape),dtype=np.float32)
            image = preproc_for_test(image_o, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return torch.from_numpy(image), torch.from_numpy(targets)
        '''
        #if len(boxes_t)==0 or random.random() > 0.97:
        if len(boxes_t)==0:
            image_irt,imagevit = preproc_for_test(imageir,imagevi, input_dim, self.means, self.std)
            boxes_t = boxes_o
            labels_t = labels_o
            ratios_t = ratios_o

        labels_t = np.expand_dims(labels_t,1)
        if mixup:
            ratios_t = np.expand_dims(ratios_t,1)
            targets_t = np.hstack((labels_t,boxes_t,ratios_t))
        else:
            targets_t = np.hstack((labels_t,boxes_t))
        padded_labels = np.zeros((self.max_labels,lshape))
        padded_labels[range(len(targets_t))[:self.max_labels]] = targets_t[:self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_irt = np.ascontiguousarray(image_irt, dtype=np.float32)
        imagevit = np.ascontiguousarray(imagevit, dtype=np.float32)
        
        return torch.from_numpy(image_irt), torch.from_numpy(imagevit),torch.from_numpy(padded_labels)
        # return torch.from_numpy(image_t), torch.from_numpy(padded_labels)



class ValTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """
    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std=std

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), input_size,
                        interpolation = interp_method).astype(np.float32)
        # img = img[:,:,::-1]
        img /= 255.
        if self.means is not None:
            img -= self.means
        if self.std is not None:
            img /= self.std
        img=np.expand_dims(img,axis=0)
        print(img.shape)
        # img = img.transpose(self.swap)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return torch.from_numpy(img), torch.zeros(1,5)
