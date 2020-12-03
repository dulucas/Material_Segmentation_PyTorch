import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

def shift_pool(x, pool2d):
    n,c,h,w = x.size()
    p1d = (1, 1, 1, 1)
    x = F.pad(x, p1d, "constant", 0)

    x0 = pool2d(x[:, :, :-1, :-1]).unsqueeze(2)
    x1 = pool2d(x[:, :, :-1, 1: ]).unsqueeze(2)
    x2 = pool2d(x[:, :, 1: , :-1]).unsqueeze(2)
    x3 = pool2d(x[:, :, 1: , 1: ]).unsqueeze(2)

    x = torch.cat((x0, x1, x2, x3), 2)
    x = x.view(n, c, 4, 64)
    x = x.permute(0,1,3,2)
    x = x.view(n,c,8,8,2,2)
    x = x.permute(0,1,2,4,3,5)
    x = x.reshape(n,c,16,16)
    return x

class LRN(nn.Module):
    def __init__(self, local_size=5, alpha=1e-4, beta=0.75):
        super(LRN, self).__init__()
        self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                stride=1,
                padding=(int((local_size-1.0)/2), 0, 0))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        div = x.pow(2).unsqueeze(1)
        div = self.average(div).squeeze(1) #* 5
        div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class googlenet(nn.Module):
    def __init__(self):
        super(googlenet, self).__init__()

        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.lrn = LRN()
        self.relu = nn.ReLU(inplace=True)

        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_3a_3x3 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_3a_5x5_reduce = nn.Conv2d(192, 16, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_3a_5x5 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0, bias=True)

        self.inception_3b_1x1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_3b_3x3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_3b_5x5_reduce = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_3b_5x5 = nn.Conv2d(32, 96, kernel_size=5, stride=1, padding=2, bias=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception_4a_1x1 = nn.Conv2d(480, 192, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4a_3x3_reduce = nn.Conv2d(480, 96, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4a_3x3 = nn.Conv2d(96, 208, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_4a_5x5_reduce = nn.Conv2d(480, 16, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4a_5x5 = nn.Conv2d(16, 48, kernel_size=5, stride=1, padding=2, bias=True)
        self.inception_4a_pool_proj = nn.Conv2d(480, 64, kernel_size=1, stride=1, padding=0, bias=True)

        self.inception_4b_1x1 = nn.Conv2d(512, 160, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4b_3x3_reduce = nn.Conv2d(512, 112, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4b_3x3 = nn.Conv2d(112, 224, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_4b_5x5_reduce = nn.Conv2d(512, 24, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4b_5x5 = nn.Conv2d(24, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.inception_4b_pool_proj = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=True)

        self.inception_4c_1x1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4c_3x3_reduce = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4c_3x3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_4c_5x5_reduce = nn.Conv2d(512, 24, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4c_5x5 = nn.Conv2d(24, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.inception_4c_pool_proj = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=True)

        self.inception_4d_1x1 = nn.Conv2d(512, 112, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4d_3x3_reduce = nn.Conv2d(512, 144, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4d_3x3 = nn.Conv2d(144, 288, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_4d_5x5_reduce = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4d_5x5 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.inception_4d_pool_proj = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=True)

        self.inception_4e_1x1 = nn.Conv2d(528, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4e_3x3_reduce = nn.Conv2d(528, 160, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4e_3x3 = nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_4e_5x5_reduce = nn.Conv2d(528, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_4e_5x5 = nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2, bias=True)
        self.inception_4e_pool_proj = nn.Conv2d(528, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception_5a_1x1 = nn.Conv2d(832, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_5a_3x3_reduce = nn.Conv2d(832, 160, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_5a_3x3 = nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_5a_5x5_reduce = nn.Conv2d(832, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_5a_5x5 = nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2, bias=True)
        self.inception_5a_pool_proj = nn.Conv2d(832, 128, kernel_size=1, stride=1, padding=0, bias=True)

        self.inception_5b_1x1 = nn.Conv2d(832, 384, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_5b_3x3_reduce = nn.Conv2d(832, 192, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_5b_3x3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True)
        self.inception_5b_5x5_reduce = nn.Conv2d(832, 48, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_5b_5x5 = nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2, bias=True)
        self.inception_5b_pool_proj = nn.Conv2d(832, 128, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv_fc8 = nn.Conv2d(1024, 23, kernel_size=1, stride=1, padding=0, bias=True)
        self.inception_pool2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.adapool = nn.AdaptiveAvgPool2d(output_size=(16, 16))

    def forward(self, rgb):
        x = self.conv1_7x7_s2(rgb)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.lrn(x)

        x = self.conv2_3x3_reduce(x)
        x = self.relu(x)
        x = self.conv2_3x3(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.maxpool2(x)

        x0 = self.inception_3a_1x1(x)
        x0 = self.relu(x0)
        x1 = self.inception_3a_3x3_reduce(x)
        x1 = self.relu(x1)
        x1 = self.inception_3a_3x3(x1)
        x1 = self.relu(x1)
        x2 = self.inception_3a_5x5_reduce(x)
        x2 = self.relu(x2)
        x2 = self.inception_3a_5x5(x2)
        x2 = self.relu(x2)
        x3 = self.inception_pool2d(x)
        x3 = self.inception_3a_pool_proj(x3)
        x3 = self.relu(x3)
        x = torch.cat([x0, x1, x2, x3], 1)

        x0 = self.inception_3b_1x1(x)
        x0 = self.relu(x0)
        x1 = self.inception_3b_3x3_reduce(x)
        x1 = self.relu(x1)
        x1 = self.inception_3b_3x3(x1)
        x1 = self.relu(x1)
        x2 = self.inception_3b_5x5_reduce(x)
        x2 = self.relu(x2)
        x2 = self.inception_3b_5x5(x2)
        x2 = self.relu(x2)
        x3 = self.inception_pool2d(x)
        x3 = self.inception_3b_pool_proj(x3)
        x3 = self.relu(x3)
        x = torch.cat([x0, x1, x2, x3], 1)
        x = self.maxpool3(x)

        x0 = self.inception_4a_1x1(x)
        x0 = self.relu(x0)
        x1 = self.inception_4a_3x3_reduce(x)
        x1 = self.relu(x1)
        x1 = self.inception_4a_3x3(x1)
        x1 = self.relu(x1)
        x2 = self.inception_4a_5x5_reduce(x)
        x2 = self.relu(x2)
        x2 = self.inception_4a_5x5(x2)
        x2 = self.relu(x2)
        x3 = self.inception_pool2d(x)
        x3 = self.inception_4a_pool_proj(x3)
        x3 = self.relu(x3)
        x = torch.cat([x0, x1, x2, x3], 1)

        x0 = self.inception_4b_1x1(x)
        x0 = self.relu(x0)
        x1 = self.inception_4b_3x3_reduce(x)
        x1 = self.relu(x1)
        x1 = self.inception_4b_3x3(x1)
        x1 = self.relu(x1)
        x2 = self.inception_4b_5x5_reduce(x)
        x2 = self.relu(x2)
        x2 = self.inception_4b_5x5(x2)
        x2 = self.relu(x2)
        x3 = self.inception_pool2d(x)
        x3 = self.inception_4b_pool_proj(x3)
        x3 = self.relu(x3)
        x = torch.cat([x0, x1, x2, x3], 1)

        x0 = self.inception_4c_1x1(x)
        x0 = self.relu(x0)
        x1 = self.inception_4c_3x3_reduce(x)
        x1 = self.relu(x1)
        x1 = self.inception_4c_3x3(x1)
        x1 = self.relu(x1)
        x2 = self.inception_4c_5x5_reduce(x)
        x2 = self.relu(x2)
        x2 = self.inception_4c_5x5(x2)
        x2 = self.relu(x2)
        x3 = self.inception_pool2d(x)
        x3 = self.inception_4c_pool_proj(x3)
        x3 = self.relu(x3)
        x = torch.cat([x0, x1, x2, x3], 1)

        x0 = self.inception_4d_1x1(x)
        x0 = self.relu(x0)
        x1 = self.inception_4d_3x3_reduce(x)
        x1 = self.relu(x1)
        x1 = self.inception_4d_3x3(x1)
        x1 = self.relu(x1)
        x2 = self.inception_4d_5x5_reduce(x)
        x2 = self.relu(x2)
        x2 = self.inception_4d_5x5(x2)
        x2 = self.relu(x2)
        x3 = self.inception_pool2d(x)
        x3 = self.inception_4d_pool_proj(x3)
        x3 = self.relu(x3)
        x = torch.cat([x0, x1, x2, x3], 1)

        x0 = self.inception_4e_1x1(x)
        x0 = self.relu(x0)
        x1 = self.inception_4e_3x3_reduce(x)
        x1 = self.relu(x1)
        x1 = self.inception_4e_3x3(x1)
        x1 = self.relu(x1)
        x2 = self.inception_4e_5x5_reduce(x)
        x2 = self.relu(x2)
        x2 = self.inception_4e_5x5(x2)
        x2 = self.relu(x2)
        x3 = self.inception_pool2d(x)
        x3 = self.inception_4e_pool_proj(x3)
        x3 = self.relu(x3)
        x = torch.cat([x0, x1, x2, x3], 1)
        x = shift_pool(x, self.maxpool4)

        x0 = self.inception_5a_1x1(x)
        x0 = self.relu(x0)
        x1 = self.inception_5a_3x3_reduce(x)
        x1 = self.relu(x1)
        x1 = self.inception_5a_3x3(x1)
        x1 = self.relu(x1)
        x2 = self.inception_5a_5x5_reduce(x)
        x2 = self.relu(x2)
        x2 = self.inception_5a_5x5(x2)
        x2 = self.relu(x2)
        x3 = self.inception_pool2d(x)
        x3 = self.inception_5a_pool_proj(x3)
        x3 = self.relu(x3)
        x = torch.cat([x0, x1, x2, x3], 1)

        x0 = self.inception_5b_1x1(x)
        x0 = self.relu(x0)
        x1 = self.inception_5b_3x3_reduce(x)
        x1 = self.relu(x1)
        x1 = self.inception_5b_3x3(x1)
        x1 = self.relu(x1)
        x2 = self.inception_5b_5x5_reduce(x)
        x2 = self.relu(x2)
        x2 = self.inception_5b_5x5(x2)
        x2 = self.relu(x2)
        x3 = self.inception_pool2d(x)
        x3 = self.inception_5b_pool_proj(x3)
        x3 = self.relu(x3)
        x = torch.cat([x0, x1, x2, x3], 1)

        #x = self.adapool(x)
        x = self.conv_fc8(x)

        return x

if __name__ == "__main__":
    f = googlenet()
    f.load_state_dict(torch.load('minc-googlenet.pth'), strict=False)
    f.cuda()
    f.eval()
    torch.set_grad_enabled(False)

    import cv2
    import numpy as np

    #img = cv2.imread('example.jpg')
    img = cv2.imread('/home/duy/phd/lucasdu/duy/material/photos/10.jpg')
    #img_ori = cv2.resize(img, (224, 224))
    img_ori = img[900:1156, 600:856, :]
    img = img_ori.astype(np.float32)
    img = img.transpose(2,0,1)
    img[0,:,:] -= 104
    img[1,:,:] -= 117
    img[2,:,:] -= 124
    img = torch.FloatTensor(img)
    img = img.cuda()
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = f(img)
    softmax = nn.Softmax(dim=1)
    pred = softmax(pred)
    pred = torch.argmax(pred, dim=1)
    pred = pred.squeeze().cpu().numpy()

    plt.figure()
    plt.imshow(img_ori[:,:,::-1])
    plt.figure()
    plt.imshow(pred)
    plt.show()
