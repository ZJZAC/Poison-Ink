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
from models.Huang_UNet import UnetGenerator_H

from metrics import PSNR, SSIM
import utils.transformed as transforms

import random
import math
import torch.nn.functional as F
import  shutil
import lpips
import cv2
import numpy as np
import scipy.stats as st

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--classes', type=str, default='airplane', help='number of GPUs to use')
parser.add_argument('--color', type=str, default='green', help='number of GPUs to use')
parser.add_argument('--aug', type=str, default='base', help='data augmentation to process stego')


def main():
    
    opt = parser.parse_args()
    cudnn.benchmark = True
    
    class_name = opt.classes
    coverdir = "../dataset/cifar10/test/" + class_name

    data_aug = opt.aug
    metricdir = "./test_result/"+"refool/"+class_name
    stegodir = metricdir+"/em"


    if not os.path.exists(stegodir):
        creatdir(stegodir)
    em_psnrdir = os.path.join(stegodir,'psnr')
    if not os.path.exists(em_psnrdir):
        os.mkdir(em_psnrdir)
    embeddir = os.path.join(stegodir,'embed')
    if not os.path.exists(embeddir):
        os.mkdir(embeddir)   




    shutil.copy("./refool.py", metricdir)

 ##################################################################
    cover_imgs = os.listdir(coverdir)
    imgNum = len(cover_imgs)
    print(cover_imgs)
    print(imgNum)
    em_total_psnr = 0
    ex_total_psnr = 0
    cl_ex_total_psnr = 0
    em_total_ssim = 0
    ex_total_ssim = 0

    em_total_lpips= 0
    loss_fn_alex = lpips.LPIPS(net = 'alex').cuda()


    metric_name = os.path.join(metricdir,'metric.txt')
    f = open(metric_name, 'w+')



    img_rf = cv2.imread('./png/kitty.jpg')
    img_rf = cv2.resize(img_rf, (32,32))

    with torch.no_grad():
        # for i in range (100):
        for i in range (imgNum):

            cover_img_name=cover_imgs[i].split('.')[0]
            img_bg = cv2.imread(coverdir + "/" + cover_imgs[i])
            cover_B_np = img_bg
            stego_img_cv, _, _ = blend_images(img_bg, img_rf, ghost_rate=0.39)

            cover_img_np = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
            stego_img_np = cv2.cvtColor(stego_img_cv, cv2.COLOR_BGR2RGB)

            cover_img_t =  torch.from_numpy(cover_img_np.transpose((2, 0, 1))).float().div(255)
            stego_img_t =  torch.from_numpy(stego_img_np.transpose((2, 0, 1))).float().div(255)
            cover_B = cover_img_t.unsqueeze(0).cuda()
            stego_img = stego_img_t.unsqueeze(0).cuda()
            cover_B_np = cover_B.squeeze(0).mul(255).cpu().numpy().transpose((1,2,0))  #记得乘255
            stego_img_np = stego_img.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))
            

            em_psnr = PSNR(stego_img_np, cover_B_np)
            em_total_psnr += em_psnr
            em_ssim = SSIM(stego_img.cpu(), cover_B.cpu()) 
            em_total_ssim += em_ssim 

            # stego_norm = (stego_img -0.5)/0.5
            # cover_norm = (cover_B -0.5)/0.5
            stego_norm = stego_img
            cover_norm = cover_B
            em_lpips = loss_fn_alex(stego_norm, cover_norm)
            # sys.exit(0)
            em_total_lpips += em_lpips

            diff = stego_img - cover_B
            resultImg1 = torch.cat([ cover_B, diff, stego_img], 0)
            resultImgName1 = '%s/%s_psnr_%04f_ssim_%04f_lpips_%04f.png' % (em_psnrdir,cover_img_name,em_psnr,em_ssim,em_lpips)
            vutils.save_image(resultImg1, resultImgName1, nrow=3, padding=1, normalize=False)




            print('%s.png  #######################################'%cover_img_name, file=f)
            print('em_psnr:%s'%em_psnr, file=f)
            print('em_ssim:%s'%em_ssim, file=f)
            print('em_lpips:%s'%em_lpips, file=f)
 



            resultImg3 =  stego_img.clone()
            resultImgName3 = '%s/%s.png' % (embeddir,cover_img_name)
            vutils.save_image(resultImg3, resultImgName3, nrow=3, padding=0, normalize=False)


        print(' ####################AVERAGE METRIC ###################', file=f)
        print('em_average_psnr:%s'%(em_total_psnr/(i+1)), file=f)
        print('em_average_ssim:%s'%(em_total_ssim/(i+1)), file=f)
        print('em_average_lpips:%s'%(em_total_lpips/(i+1)), file=f)


        print(' ####################AVERAGE METRIC ###################')
        print('em_average_psnr:%s'%(em_total_psnr/(i+1)))
        print('em_average_ssim:%s'%(em_total_ssim/(i+1)))
        print('em_average_lpips:%s'%(em_total_lpips/(i+1)))

def blend_images(img_t, img_r, max_image_size=32, ghost_rate=0.49, alpha_t=-1., offset=(0, 0), sigma=-1,
                 ghost_alpha=-1.):
    """
    Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
    return the blended image and precessed reflection image
    """
    t = np.float32(img_t) / 255.
    r = np.float32(img_r) / 255.
    h, w, _ = t.shape
    # convert t.shape to max_image_size's limitation
    scale_ratio = float(max(h, w)) / float(max_image_size)
    w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
        else (int(round(w / scale_ratio)), max_image_size)
    t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
    r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

    if alpha_t < 0:
        alpha_t = 1. - random.uniform(0.05, 0.45)

    if random.randint(0, 100) < ghost_rate * 100:
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        # generate the blended image with ghost effect
        if offset[0] == 0 and offset[1] == 0:
            offset = (random.randint(3, 8), random.randint(3, 8))
        r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                         'constant', constant_values=0)
        r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                         'constant', constant_values=(0, 0))
        if ghost_alpha < 0:
            ghost_alpha_switch = 1 if random.random() > 0.5 else 0
            ghost_alpha = abs(ghost_alpha_switch - random.uniform(0.15, 0.5))

        ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
        ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
        reflection_mask = ghost_r * (1 - alpha_t)

        blended = reflection_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)

        ghost_r = np.power(reflection_mask, 1 / 2.2)
        ghost_r[ghost_r > 1.] = 1.
        ghost_r[ghost_r < 0.] = 0.

        blended = np.power(blended, 1 / 2.2)
        blended[blended > 1.] = 1.
        blended[blended < 0.] = 0.

        ghost_r = np.power(ghost_r, 1 / 2.2)
        ghost_r[blended > 1.] = 1.
        ghost_r[blended < 0.] = 0.

        reflection_layer = np.uint8(ghost_r * 255)
        blended = np.uint8(blended * 255)
        transmission_layer = np.uint8(transmission_layer * 255)
    else:
        # generate the blended image with focal blur
        if sigma < 0:
            sigma = random.uniform(1, 5)

        t = np.power(t, 2.2)
        r = np.power(r, 2.2)
        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        # get the reflection layers' proper range
        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        def gen_kernel(kern_len=100, nsig=1):
            """Returns a 2D Gaussian kernel array."""
            interval = (2 * nsig + 1.) / kern_len
            x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
            # get normal distribution
            kern1d = np.diff(st.norm.cdf(x))
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            kernel = kernel / kernel.max()
            return kernel

        h, w = r_blur.shape[0: 2]
        new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
        new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0

        g_mask = gen_kernel(max_image_size, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)

        r_blur_mask = np.multiply(r_blur, alpha_r)
        blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
        blend = r_blur_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)
        r_blur_mask = np.power(blur_r, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        blended = np.uint8(blend * 255)
        reflection_layer = np.uint8(r_blur_mask * 255)
        transmission_layer = np.uint8(transmission_layer * 255)

    return blended, transmission_layer, reflection_layer


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