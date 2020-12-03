import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def shift_pool(x, pool2d):
    n,c,h,w = x.size()
    p1d = (1,1,1,1)
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


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.conv1_1 = conv3x3(3, 64)
        self.conv1_2 = conv3x3(64, 64)
        self.conv2_1 = conv3x3(64, 128)
        self.conv2_2 = conv3x3(128, 128)
        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(256, 256)
        self.conv3_3 = conv3x3(256, 256)
        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(512, 512)
        self.conv4_3 = conv3x3(512, 512)
        self.conv5_1 = conv3x3(512, 512)
        self.conv5_2 = conv3x3(512, 512)
        self.conv5_3 = conv3x3(512, 512)
        self.pool2d = nn.MaxPool2d(2,2,0,1)
        self.adapool = nn.AdaptiveAvgPool2d(output_size=(8, 8))

        self.relu = nn.ReLU(inplace=True)

        self.conv_fc6 = nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv_fc7 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fc8 = nn.Conv2d(4096, 23, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, rgb):
        x = self.conv1_1(rgb)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.pool2d(x)

        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.pool2d(x)

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.pool2d(x)

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.pool2d(x)

        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        x = shift_pool(x, self.adapool)

        x = self.conv_fc6(x)
        x = self.conv_fc7(x)
        x = self.relu(x)
        x = self.conv_fc8(x)
        return x

if __name__ == '__main__':
    f = vgg16()
    weights = torch.load('../minc-vgg16.pth')
    f.load_state_dict(weights, strict=False)
    f.cuda()
    f.eval()
    torch.set_grad_enabled(False)

    import cv2
    import numpy as np

    img = cv2.imread('xxx')
    print(img.shape)
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
