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
import torchvision.transforms as trans

import torchvision
from PIL import Image
import skimage
import torchvision.utils as vutils  
import numpy
from skimage import filters
import sys 
sys.path.append("..") 
from models.HidingRes import HidingRes
from models.ReflectionUNet import UnetGenerator2
from models.up_UNet import UNet
from models.up_UNet_rp import UNet_rp
from metrics import PSNR, SSIM
import utils.transformed as transforms

import random
import math
import torch.nn.functional as F
import  shutil
import lpips

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--classes', type=str, default='airplane', help='number of GPUs to use')

parser.add_argument('--sig', action='store_true', help='signal pattern ')
parser.add_argument('--blend', action='store_true', help='blend pattern ')

def main():
    
    opt = parser.parse_args()
    cudnn.benchmark = True
 
    class_name = opt.classes
    coverdir = "../dataset/cifar10/test/" + class_name

    if opt.sig:
        metricdir = "./test_result/SIG/"+class_name
        watermark = Image.open('./png/SIG_20.png').convert('RGB')
    elif opt.blend:
        metricdir = "./test_result/Blend/"+class_name
        watermark = Image.open('./png/kitty.jpg').convert('RGB')
    else:
        metricdir = "./test_result/Static/"+class_name
        watermark = Image.open('./png/check_board_255.png').convert('RGB')

    stegodir = metricdir+"/em"
    datasetdir = metricdir+"/em/data"
    paperdir = metricdir+"/paper"

    if not os.path.exists(stegodir):
        creatdir(stegodir)
    em_psnrdir = os.path.join(stegodir,'psnr')
    if not os.path.exists(em_psnrdir):
        os.mkdir(em_psnrdir)
    embeddir = os.path.join(stegodir,'embed')
    if not os.path.exists(embeddir):
        os.mkdir(embeddir)   
    if not os.path.exists(datasetdir):
        creatdir(datasetdir)
    if not os.path.exists(paperdir):
        creatdir(paperdir)


    shutil.copy("./test.py", metricdir)


    ############################################### embedding step   ##################################################################
    cover_imgs = os.listdir(coverdir)
    imgNum = len(cover_imgs)
    print(cover_imgs)
    print(imgNum)
    em_total_psnr = 0
    em_total_ssim = 0

    em_total_lpips= 0
    loss_fn_alex = lpips.LPIPS(net = 'alex').cuda()

    metric_name = os.path.join(metricdir,'metric.txt')
    f = open(metric_name, 'w+')

    loader = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
    watermark = loader(watermark)
    watermark  = watermark.cuda()
    watermark = watermark.unsqueeze(0)
    if opt.sig:
        watermark = watermark * 20 / 255
    elif opt.blend:
        watermark = watermark
    else:
        watermark = watermark * 6 / 255

    with torch.no_grad():
        for i in range (100):
        # for i in range (imgNum):
            cover_img_name = cover_imgs[i].split('.')[0]
            cover_img = Image.open(coverdir + "/"+ cover_imgs[i])

            cover_img = loader(cover_img)
            cover_img  = cover_img.cuda()
            cover_img = cover_img.unsqueeze(0)
            cover_B = cover_img
            cover_B_np = cover_B.squeeze(0).mul(255).cpu().numpy().transpose((1,2,0))  #记得乘255

            if opt.blend:
                stego_img =0.8 * cover_B + 0.2 * watermark
            else:
                stego_img = cover_B + watermark
            
            stego_img = torch.clamp(stego_img, 0, 1)


            stego_img_np = stego_img.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))
            em_psnr = PSNR(stego_img_np, cover_B_np)
            em_total_psnr += em_psnr
            em_ssim = SSIM(stego_img.cpu(), cover_B.cpu()) 
            em_total_ssim += em_ssim 

            stego_norm = stego_img
            cover_norm = cover_B
            # stego_norm = (stego_img -0.5)/0.5
            # cover_norm = (cover_B -0.5)/0.5
            em_lpips = loss_fn_alex(stego_norm, cover_norm)
            # sys.exit(0)
            em_total_lpips += em_lpips

            diff = stego_img - cover_B
            resultImg1 = torch.cat([ cover_B, diff, stego_img], 0)
            resultImgName1 = '%s/%s_psnr_%04f_ssim_%04f_lpips_%04f.png' % (em_psnrdir,cover_img_name,em_psnr,em_ssim,em_lpips)
            vutils.save_image(resultImg1, resultImgName1, nrow=3, padding=1, normalize=False)


            resultImg3 =  stego_img.clone()
            resultImgName3 = '%s/%s.png' % (embeddir,cover_img_name)
            vutils.save_image(resultImg3, resultImgName3, nrow=3, padding=0, normalize=False)

            stego_img0 = stego_img
            watermark0 = watermark
            resultImg_paper = torch.cat([ cover_B,  stego_img0, (stego_img0-cover_B), 10*(stego_img0-cover_B), watermark0], 0)
            resultImgName_paper = '%s/%s.png' % (paperdir,cover_img_name)
            vutils.save_image(resultImg_paper, resultImgName_paper, nrow=10, padding=0, normalize=False)


        print(' ####################AVERAGE METRIC ###################', file=f)
        print('em_average_psnr:%s'%(em_total_psnr/(i+1)), file=f)
        print('em_average_ssim:%s'%(em_total_ssim/(i+1)), file=f)
        print('em_average_lpips:%s'%(em_total_lpips/(i+1)), file=f)


        print(' ####################AVERAGE METRIC ###################')
        print('em_average_psnr:%s'%(em_total_psnr/(i+1)))
        print('em_average_ssim:%s'%(em_total_ssim/(i+1)))
        print('em_average_lpips:%s'%(em_total_lpips/(i+1)))



class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()

        # if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
        # if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            # kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
            #           torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.conv = F.conv2d


    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)




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