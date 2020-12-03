import glob
from vgg import vgg16
from googlenet import googlenet
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

color_palette = np.loadtxt('/home/duy/phd/tmp_half/libs/palette.txt').astype(np.uint8)

def color_image_w_masks(image, masks):
    image = image.astype(np.uint8)

    for index in range(23):
        mask = (masks == index).astype(np.uint8)
        if mask.sum() == 0:
            continue
        color = color_palette[index]
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)
        mask = mask * np.array(color).reshape((-1, 3)) + (1 - mask) * image
        mask = mask.astype(np.uint8)
        image = cv2.addWeighted(image, .5, mask, .5, 0)
    return image

def inference_on_whole_image(img, model):
    h,w,c = img.shape
    if h % 256 != 0:
        h_ = (h // 256 + 1) * 256
    else:
        h_ = h
    if w %  256 != 0:
        w_ = (w // 256 + 1) * 256
    else:
        w_ = w

    img = cv2.resize(img, (w_, h_))
    img = img.astype(np.float32).transpose(2,0,1)
    img[0,:,:] -= 104
    img[1,:,:] -= 117
    img[2,:,:] -= 124
    img = torch.FloatTensor(img)
    img = img.cuda()
    img = img.unsqueeze(0)

    softmax = nn.Softmax(dim=1)

    nh = h_ // 256
    nw = w_ // 256
    prob = np.zeros((h_, w_, 23))

    for i in range(nh):
        for j in range(nw):
            img_patch = img[:,:,i*256:(i+1)*256, j*256:(j+1)*256]
            pred = model(img_patch)
            pred = softmax(pred).squeeze().cpu().numpy().transpose(1,2,0)
            pred = cv2.resize(pred, (256, 256))
            prob[i*256:(i+1)*256, j*256:(j+1)*256, :] = pred

    return prob

def multi_scale_inference(img, model):
    h,w,c = img.shape
    scales = [.5, 1, 1.5]
    prob = np.zeros((h,w,23))

    for scale in scales:
        img_ = cv2.resize(img, (int(w*scale), int(h*scale)))
        prob_ = inference_on_whole_image(img_, model)
        prob += cv2.resize(prob_, (w,h))

    prob /= 3
    return prob

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


if __name__ == '__main__':
    model0 = googlenet
    #model1 = vgg16

    m0 = model0()
    #m1 = model1()

    #m1.load_state_dict(torch.load('minc-vgg16.pth'), strict=False)
    #m1.cuda().eval()
    m0.load_state_dict(torch.load('minc-googlenet.pth'), strict=False)
    m0.cuda().eval()

    torch.set_grad_enabled(False)
    #img_paths = glob.glob('/home/duy/phd/lucasdu/duy/material/photos/*')
    img_paths = glob.glob('/home/duy/phd/lucasdu/duy/material/ycb/test/000055/rgb/*')
    labels = open('categories.txt', 'r').readlines()
    labels = [i.strip() for i in labels]

    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    for i in range(10):
        img_path = np.random.choice(img_paths)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1024, 1024))

        prob0 = multi_scale_inference(img, m0)
        #prob1 = multi_scale_inference(img, m1)
        prob = prob0 #(prob0 + prob1) / 2

        prob = cv2.resize(prob, (480, 320))
        img = cv2.resize(img, (480, 320))
        prob = prob.transpose(2,0,1)
        #img_ = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        prob = postprocessor(img, prob)
        labelmap = np.argmax(prob, axis=0)

        #mask = color_image_w_masks(img, labelmap)
        #img = np.concatenate([img, mask], axis=1)
        #cv2.imshow('img', img)
        #cv2.waitKey()

        #'''
        plt.figure(figsize=(10, 10))
        plt.imshow(img[:, :, ::-1])

        plt.figure(figsize=(15, 15))

        for i in range(23):
            mask = labelmap == i
            ax = plt.subplot(4, 6, i + 1)
            ax.set_title(labels[i])
            ax.imshow(img[:, :, ::-1])
            ax.imshow(mask.astype(np.float32), alpha=0.5)
            ax.axis("off")

        plt.tight_layout()
        plt.show()
        #'''
