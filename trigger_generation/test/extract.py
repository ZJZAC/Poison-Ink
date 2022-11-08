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
import utils.transformed as transforms
import torchvision.transforms as trans

import torchvision
from PIL import Image
import skimage
import torchvision.utils as vutils
import numpy
from skimage import filters
from models.HidingRes import HidingRes
from models.ReflectionUNet import UnetGenerator2
from metrics import PSNR, SSIM
import random
import math
import torch.nn.functional as F
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--Rnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')

parser.add_argument('--red', action='store_true', help=' red color ')
parser.add_argument('--yellow', action='store_true', help=' yellow color ')
parser.add_argument('--green', action='store_true', help=' yellow color ')
parser.add_argument('--ft', action='store_true', help=' ft_chk_Rnet ')
parser.add_argument('--cifar10', action='store_true', help=' make data for cifar10 ')
parser.add_argument('--cifar100', action='store_true', help=' make data for cifar100 ')
parser.add_argument('--imagenet10', action='store_true', help=' make data for imagenet10 ')
parser.add_argument('--imagenet100', action='store_true', help=' make data for imagenet100 ')


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


def main():
    opt = parser.parse_args()
    cudnn.benchmark = True

    loader = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomCrop(28),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    coverdir = "/data-x/g11/zhangjie/ECCV/datasets/backdoor/green_cifar10_l2_1001/_crop28_flip_resize32_from_1001_0802/double_poison_label/test_trigger/truck"

    metricdir = "/public/zhangjie/ECCV_2020/backdoor/edge_watermark_0804/extract_result/green_cifar10_l2_1001/_crop28_flip_resize32_from_1001_0802/double_poison_label/test_trigger/truck/28_32-f-epoch116"
    revealdir = "/public/zhangjie/ECCV_2020/backdoor/edge_watermark_0804/extract_result/green_cifar10_l2_1001/_crop28_flip_resize32_from_1001_0802/double_poison_label/test_trigger/truck/28_32-f-epoch116/ex"

    paperdir = "/public/zhangjie/ECCV_2020/backdoor/edge_watermark_0804/extract_result/green_cifar10_l2_1001/_crop28_flip_resize32_from_1001_0802/double_poison_label/test_trigger/truck/28_32-f-epoch116/paper"


    if not os.path.exists(metricdir):
        creatdir(metricdir)

    if not os.path.exists(paperdir):
        creatdir(paperdir)

    if not os.path.exists(revealdir):
        creatdir(revealdir)
    extractdir = os.path.join(revealdir, 'extract')
    if not os.path.exists(extractdir):
        os.mkdir(extractdir)

    shutil.copy("./extract.py", metricdir)
    ###############################################    ##################################################################

    if opt.cifar10:

        # opt.Rnet = '/data-x/g11/zhangjie/ECCV/pth/backdoor/CIFAR/10_L2/_adv_crop28_flip_resize32_from_1001-0802/R179.pth'
        # opt.Rnet = '/data-x/g11/zhangjie/ECCV/pth/backdoor/CIFAR/10_L2/1111_gan_rot90_28_32/R199.pth'
        # opt.Rnet = '/data-x/g11/zhangjie/ECCV/pth/backdoor/CIFAR/10_L2/1001/R190.pth'
        opt.Rnet = '/data-x/g11/zhangjie/ECCV/pth/backdoor/CIFAR/10_L2/_crop28_flip_resize32_from_1001-0802/R116.pth'


    if opt.ft:
        opt.Rnet = "/data-x/g11/zhangjie/cvpr/pth/R_ft/20_90_all_128_fix/FT_L2_u128_rot30_224_128/netR_epoch_6,sumloss=234.805710,Rloss=219.709488.pth"

    ###############################################    ##################################################################
    Rnet = HidingRes(in_c=3, out_c=3)
    Rnet.cuda()

    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet).cuda()

    Rnet.load_state_dict(torch.load(opt.Rnet))
    Rnet.eval()

    ############################################### embedding step   ##################################################################
    cover_imgs = os.listdir(coverdir)
    imgNum = len(cover_imgs)
    print(cover_imgs)
    print(imgNum)
    ex_total_psnr = 0
    ex_total_ssim = 0
    rev_success_rate_5 = 0
    rev_success_rate_10 = 0
    rev_success_rate_15 = 0
    rev_success_rate_20 = 0

    random_R = 80
    random_G = 160
    random_B = 80

    metric_name = os.path.join(metricdir, 'metric.txt')
    f = open(metric_name, 'w+')

    print("RRRRR%d" % random_R, file=f)
    print("GGGGG%d" % random_G, file=f)
    print("BBBBB%d" % random_B, file=f)

    with torch.no_grad():
        for i in range(imgNum):

            cover_img_name = cover_imgs[i].split('.')[0]
            cover_img = Image.open(coverdir + "/" + cover_imgs[i])


            cover_img = loader(cover_img)
            cover_img = cover_img.cuda()
            cover_img = cover_img.unsqueeze(0)
            clean_img = Image.open("/data-x/g11/zhangjie/secret/clean.png")
            clean_img = loader(clean_img)
            clean_img = clean_img.unsqueeze(0).cuda()
            if opt.cifar10:
                clean_img = clean_img[:, :, 0:32, 0:32]


            cover_B = cover_img

            cover_B_np = cover_B.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))  # 记得乘255
            cover_B_gray_np = 0.299 * cover_B_np[:, :, 0] + 0.587 * cover_B_np[:, :, 1] + 0.114 * cover_B_np[:, :, 2]
            mask = filters.sobel(cover_B_gray_np)
            mask = mask > 28
            mask = mask.astype(numpy.uint8)  # edge 作为水印和原图进行拼接   uint 8  (256,768) 值为0，1
            mask = mask[:, :, numpy.newaxis]  # (256,256,1)
            mask = mask
            mask_np = mask * 255
            mask_t = torch.from_numpy(mask_np.transpose((2, 0, 1)))
            mask_t = mask_t.float().div(255)
            mask_t = mask_t.repeat(3, 1, 1).unsqueeze(0).cuda()

            color = torch.ones((3)).cuda()
            color[0] = random_R
            color[1] = random_G
            color[2] = random_B

            # edge_np = numpy.ones((256,256,3))

            if opt.cifar10:
                edge_np = numpy.ones((32, 32, 3))
            if opt.cifar100:
                edge_np = numpy.ones((32, 32, 3))
            if opt.imagenet100:
                edge_np = numpy.ones((256, 256, 3))
            if opt.imagenet10:
                edge_np = numpy.ones((256, 256, 3))

            edge_np[:, :, 0:1] = random_R * mask
            edge_np[:, :, 1:2] = random_G * mask
            edge_np[:, :, 2:3] = random_B * mask
            edge_t = torch.from_numpy(edge_np.transpose((2, 0, 1)))
            edge_t = edge_t.float().div(255).unsqueeze(0).cuda()

            watermark = edge_t * mask_t + clean_img * (1 - mask_t)
            stego_img = cover_img




            # extratct
            watermark_np = watermark.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))

            reveal_img = Rnet(stego_img)
            reveal_img_np = reveal_img.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))
            ex_psnr = PSNR(reveal_img_np, watermark_np)
            ex_total_psnr += ex_psnr
            ex_ssim = SSIM(reveal_img.cpu(), watermark.cpu())
            ex_total_ssim += ex_ssim

            resultImg11 = torch.cat([stego_img, watermark, reveal_img], 0)
            resultImgName11 = '%s/%s_psnr_%04f_ssim_%04f.png' % (extractdir, cover_img_name, ex_psnr, ex_ssim)
            vutils.save_image(resultImg11, resultImgName11, nrow=3, padding=1, normalize=False)


            print('%s.png  #######################################' % cover_img_name, file=f)
            print('ex_psnr:%s' % ex_psnr, file=f)
            print('ex_ssim:%s' % ex_ssim, file=f)

            reveal_img_edge = reveal_img * mask_t
            reveal_img_edge = reveal_img_edge.view(1, 3, -1)
            zero_num_mask = reveal_img_edge == 0

            zero_num = torch.sum(zero_num_mask, 2).float()
            if opt.cifar10 or opt.cifar100:
                non_zero_num = 32 * 32 - zero_num
            else:
                non_zero_num = 256 * 256 - zero_num
            rev_code_round = torch.sum(reveal_img_edge, 2).div(non_zero_num).mul(255).round()  # mean 四舍五入
            code_err = (rev_code_round - color).abs()
            print("code_err:%s" % code_err, file=f)

            err_th_5 = torch.ones_like(code_err) * 5  # Error bit corrected by threshold 5
            code_err_th_5 = code_err - err_th_5
            code_err_th_max_5 = torch.max(code_err_th_5, 1)
            code_err_th_max_5 = code_err_th_max_5[0]
            rev_success_5 = (code_err_th_max_5 <= 0).float()
            print("rev_success_5%s" % rev_success_5, file=f)
            rev_success_rate_5 += rev_success_5

            err_th_10 = torch.ones_like(code_err) * 10
            code_err_th_10 = code_err - err_th_10
            code_err_th_max_10 = torch.max(code_err_th_10, 1)
            code_err_th_max_10 = code_err_th_max_10[0]  # max return 2 item
            rev_success_10 = (code_err_th_max_10 <= 0).float()
            print("rev_success_10%s" % rev_success_10, file=f)
            rev_success_rate_10 += rev_success_10

            err_th_15 = torch.ones_like(code_err) * 15
            code_err_th_15 = code_err - err_th_15
            code_err_th_max_15 = torch.max(code_err_th_15, 1)
            code_err_th_max_15 = code_err_th_max_15[0]  # max return 2 item
            rev_success_15 = (code_err_th_max_15 <= 0).float()
            print("rev_success_15%s" % rev_success_15, file=f)
            rev_success_rate_15 += rev_success_15

            err_th_20 = torch.ones_like(code_err) * 20
            code_err_th_20 = code_err - err_th_20
            code_err_th_max_20 = torch.max(code_err_th_20, 1)
            code_err_th_max_20 = code_err_th_max_20[0]  # max return 2 item
            rev_success_20 = (code_err_th_max_20 <= 0).float()
            print("rev_success_20%s" % rev_success_20, file=f)
            rev_success_rate_20 += rev_success_20



            resultImg_paper = torch.cat([cover_B, watermark, reveal_img, mask_t], 0)
            resultImgName_paper = '%s/%s.png' % (paperdir, cover_img_name)
            vutils.save_image(resultImg_paper, resultImgName_paper, nrow=10, padding=0, normalize=False)

        print(' ####################AVERAGE METRIC ###################', file=f)

        print('ex_average_psnr:%s' % (ex_total_psnr / (i + 1)), file=f)
        print('ex_average_ssim:%s' % (ex_total_ssim / (i + 1)), file=f)
        print('average_rev_success_rate_5:%s' % (rev_success_rate_5 / (i + 1)), file=f)
        print('average_rev_success_rate_10:%s' % (rev_success_rate_10 / (i + 1)), file=f)
        print('average_rev_success_rate_15:%s' % (rev_success_rate_15 / (i + 1)), file=f)
        print('average_rev_success_rate_20:%s' % (rev_success_rate_20 / (i + 1)), file=f)

        print(' ####################AVERAGE METRIC ###################')

        print('ex_average_psnr:%s' % (ex_total_psnr / (i + 1)))
        print('ex_average_ssim:%s' % (ex_total_ssim / (i + 1)))
        print('average_rev_success_rate_5:%s' % (rev_success_rate_5 / (i + 1)))
        print('average_rev_success_rate_10:%s' % (rev_success_rate_10 / (i + 1)))
        print('average_rev_success_rate_15:%s' % (rev_success_rate_15 / (i + 1)))
        print('average_rev_success_rate_20:%s' % (rev_success_rate_20 / (i + 1)))


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