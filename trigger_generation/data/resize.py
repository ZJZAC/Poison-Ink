import argparse
import os


import utils.transformed as transforms
import torchvision
from PIL import Image
import skimage
import torchvision.utils as vutils
import numpy
from skimage import filters
import math
import torch.nn.functional as F
import numpy as np


def main():

    cudnn.benchmark = True

    coverdir = "/data-x/g11/zhangjie/ECCV/datasets/backdoor/original/ImageNet100/train"
    datasetdir = "/data-x/g11/zhangjie/ECCV/datasets/backdoor/original_256/ImageNet100/train_copy"

    if not os.path.exists(datasetdir):
        creatdir(datasetdir)


    for target in sorted(os.listdir(coverdir)):
        coverdir_child = os.path.join(coverdir, target)
        datasetdir_child = os.path.join(datasetdir, target)
        if not os.path.exists(datasetdir_child):
            creatdir(datasetdir_child)

        cover_imgs = os.listdir(coverdir_child)
        imgNum = len(cover_imgs)
        if imgNum < 1300:
            print(coverdir_child)
            print(imgNum)

        # print(coverdir_child)
        # print(datasetdir_child)
        # print(imgNum)

        # with torch.no_grad():
        #     for i in range (imgNum):
        #         cover_img_name=cover_imgs[i].split('.')[0]
        #         cover_img = Image.open(coverdir_child + "/"+ cover_imgs[i]).convert("RGB")
        #         img_resize = cover_img.resize((256,256), Image.ANTIALIAS)
        #         resultImgName = '%s/%s.png' % (datasetdir_child,cover_img_name)
        #         img_resize.save(resultImgName)

def creatdir(path):
    folders = []
    while not os.path.isdir(path):
        path, suffix = os.path.split(path)
        folders.append(suffix)
    for folder in folders[::-1]:
        path = os.path.join(path, folder)
        os.mkdir(path)


if __name__ == '__main__':
    main()
