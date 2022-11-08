import argparse
import os
import sys
import torch
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import skimage
import torchvision.utils as vutils  
import numpy
from skimage import filters
sys.path.append("..") 
from models.HidingRes import HidingRes
from models.ReflectionUNet import UnetGenerator2
from models.up_UNet import UNet
from models.up_UNet_rp import UNet_rp
import utils.transformed as transforms

import math
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', type=int, default=1,  
                    help='number of GPUs to use')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')

parser.add_argument('--test', action='store_true', help=' make data for test set ')
parser.add_argument('--adv', action='store_true', help=' make adv_data for train set ')
parser.add_argument('--sig', action='store_true', help='signal pattern ')



def main():

    mask =  np.zeros((32,32))

    for i in range(32):
        for j in range(32):
            mask[i,j] =  255*math.sin(2*math.pi*j*6/32)

    mask_img = mask[:,:,np.newaxis]
    mask_img = mask_img.repeat(3, axis=2)
    print(mask)
    print(mask_img)
    im = Image.fromarray(np.uint8(mask))
    im.save('SIG_255.png')

    
    # mask =  np.zeros((32,32))

    # for i in range(32):
    #     for j in range(32):
    #         if (i%2 == 0) and (j%2 == 0):
    #             mask[i,j] = 255
    #         else:
    #             mask[i,j] = 0
    #         # print(i,j)

    # print(mask.shape)
    # mask_img = mask[:,:,np.newaxis]
    # mask_img = mask_img.repeat(3, axis=2)
    # print(mask_img.shape)
    # print(mask_img[:,:,0])
    # print(mask_img[:,:,1])
    # print(mask_img[:,:,0] == mask_img[:,:,1])
    # im = Image.fromarray(np.uint8(mask))
    # im.save('check_board_255.png')

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

#checkboard
    # mask =  np.zeros((32,32))

    # for i in range(32):
    #     for j in range(32):
    #         if (i%2 == 0) and (j%2 == 0):
    #             mask[i,j] = 10
    #         else:
    #             mask[i,j] = 0
    #         # print(i,j)

    # print(mask.shape)
    # mask_img = mask[:,:,np.newaxis]
    # mask_img = mask_img.repeat(3, axis=2)
    # print(mask_img.shape)
    # print(mask_img[:,:,0])
    # print(mask_img[:,:,1])
    # print(mask_img[:,:,0] == mask_img[:,:,1])
    # im = Image.fromarray(np.uint8(mask))
    # im.save('check_board_10.png')



# ['n01883070', 'n02190166', 'n02361337', 'n02445715', 'n02526121', 'n02804414', 'n03208938', 'n03476684', 'n03935335', 'n04442312']
# ['n01558993', 'n01692333', 'n01729322', 'n01735189', 'n01749939', 'n01773797', 'n01820546', 'n01855672',
#  'n01978455', 'n01980166', 'n01983481', 'n02009229', 'n02018207', 'n02085620', 'n02086240', 'n02086910',
#  'n02087046', 'n02089867', 'n02089973', 'n02090622', 'n02091831', 'n02093428', 'n02099849', 'n02100583',
#  'n02104029', 'n02105505', 'n02106550', 'n02107142', 'n02108089', 'n02109047', 'n02113799', 'n02113978',
#  'n02114855', 'n02116738', 'n02119022', 'n02123045', 'n02138441', 'n02172182', 'n02231487', 'n02259212',
#  'n02326432', 'n02396427', 'n02483362', 'n02488291', 'n02701002', 'n02788148', 'n02804414', 'n02859443',
#  'n02869837', 'n02877765', 'n02974003', 'n03017168', 'n03032252', 'n03062245', 'n03085013', 'n03259280',
#  'n03379051', 'n03424325', 'n03492542', 'n03494278', 'n03530642', 'n03584829', 'n03594734', 'n03637318',
#  'n03642806', 'n03764736', 'n03775546', 'n03777754', 'n03785016', 'n03787032', 'n03794056', 'n03837869',
#  'n03891251', 'n03903868', 'n03930630', 'n03947888', 'n04026417', 'n04067472', 'n04099969', 'n04111531',
#  'n04127249', 'n04136333', 'n04229816', 'n04238763', 'n04336792', 'n04418357', 'n04429376', 'n04435653',
#  'n04485082', 'n04493381', 'n04517823', 'n04589890', 'n04592741', 'n07714571', 'n07715103', 'n07753275',
#  'n07831146', 'n07836838', 'n13037406', 'n13040303']