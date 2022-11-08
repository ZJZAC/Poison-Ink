import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import torchvision
from PIL import Image
import skimage
import torchvision.utils as vutils

import numpy as np
import	numpy	



def main():
   
    cudnn.benchmark = True


    coverdir = "/data-x/g12/zhangjie/deepfake_watermark/dataset/celebA_test/256x256"
    stegodir1 = "/data-x/g12/zhangjie/deepfake_watermark/dataset/celebA_test/128x128/train"
    stegodir2 = "/data-x/g12/zhangjie/deepfake_watermark/dataset/celebA_test/128x128/val"



    cover_imgs = os.listdir(coverdir)
    imgNum = len(cover_imgs)
    # print(cover_imgs)
    print(imgNum)


    for i in range (imgNum-500):
        cover_img_name=cover_imgs[i].split('.')[0]
        cover_img = Image.open(coverdir + "/"+ cover_imgs[i])
        loader = transforms.Compose([  transforms.Resize(256),
            transforms.ToTensor(),
        ])

        cover_img = loader(cover_img)
        cover_img.cuda()
        cover_img = cover_img.unsqueeze(0)
        cover_img = cover_img[:,:,88:216, 64:192]
        resultImgName = '%s/%s.png' % (stegodir1,cover_img_name)
        vutils.save_image(cover_img, resultImgName, padding=0, normalize=False)


    # for i in range (imgNum-500, imgNum):
    #     cover_img_name=cover_imgs[i].split('.')[0]
    #     cover_img = Image.open(coverdir + "/"+ cover_imgs[i])
    #     loader = transforms.Compose([  transforms.Resize(256),
    #         transforms.ToTensor(),
    #     ])
    #
    #     cover_img = loader(cover_img)
    #     cover_img.cuda()
    #     cover_img = cover_img.unsqueeze(0)
    #     cover_img = cover_img[:,:,88:216, 64:192]
    #     resultImgName = '%s/%s.png' % (stegodir2,cover_img_name)
    #     vutils.save_image(cover_img, resultImgName, padding=0, normalize=False)




if __name__ == '__main__':
    main()