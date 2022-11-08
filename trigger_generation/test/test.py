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

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--classes', type=str, default='automobile', help='number of GPUs to use')
parser.add_argument('--color', type=str, default='green', help='number of GPUs to use')
parser.add_argument('--aug', type=str, default='base', help='data augmentation to process stego')
parser.add_argument('--rp', action='store_true', help='reflection padding in Unet ')
parser.add_argument('--uh', action='store_true', help='reflection padding in Huang_Unet ')
parser.add_argument('--num_downs', type=int, default=5, help='nums of  Unet downsample')


def main():
    
    opt = parser.parse_args()
    cudnn.benchmark = True
    pth_name = 'Up_Rp_HuangUNet_down5_L1_incremental/try2/4_color_crop24_ct3_flip_resize_rot15_ct'
    opt.Hnet = '../chk/'+pth_name+'/pth/H199.pth'
    opt.Rnet = '../chk/'+pth_name+'/pth/R199.pth'
    
    color_name = opt.color
    random_R = 80
    random_G = 160
    random_B = 80
    
    class_name = opt.classes
    coverdir = "../dataset/cifar10/test/" + class_name

    data_aug = opt.aug
    metricdir = "./test_result/"+pth_name+"/"+data_aug+"/"+color_name+"/"+class_name
    stegodir = metricdir+"/em"
    datasetdir = metricdir+"/em/data"
    revealdir = metricdir+"/ex"
    cleandir = metricdir+"/clean_ex"
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

    if not os.path.exists(revealdir):
        creatdir(revealdir)  
    extractdir = os.path.join(revealdir,'extract')
    if not os.path.exists(extractdir):
        os.mkdir(extractdir)

    if not os.path.exists(cleandir):
        creatdir(cleandir)

    shutil.copy("./test.py", metricdir)
    ###############################################    ##################################################################


    if opt.rp:
        Hnet = UNet_rp(n_channels=6, n_classes=3,  requires_grad=True)
    elif opt.uh:
        Hnet = UnetGenerator_H(input_nc=6, output_nc=3, num_downs=opt.num_downs, output_function=nn.Sigmoid, requires_grad=True)
    else:
        Hnet = UNet(n_channels=6, n_classes=3,  requires_grad=True)

    Hnet.cuda()

    Rnet = HidingRes(in_c=3, out_c=3)
    Rnet.cuda()


    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
        Rnet = torch.nn.DataParallel(Rnet).cuda()

    Hnet.load_state_dict(torch.load(opt.Hnet))
    Hnet.eval()
    Rnet.load_state_dict(torch.load(opt.Rnet)) 
    Rnet.eval()

    ############################################### embedding step   ##################################################################
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

    cl_ex_total_ssim = 0
    rev_success_rate_5 = 0
    rev_success_rate_10 = 0
    rev_success_rate_15 = 0
    rev_success_rate_20 = 0

    metric_name = os.path.join(metricdir,'metric.txt')
    f = open(metric_name, 'w+')

    print("RRRRR%d"%random_R, file=f)
    print("GGGGG%d"%random_G, file=f)
    print("BBBBB%d"%random_B, file=f)


    with torch.no_grad():
        for i in range (100):
        # for i in range (imgNum):

            cover_img_name=cover_imgs[i].split('.')[0]
            cover_img = Image.open(coverdir + "/"+ cover_imgs[i])
            loader = transforms.Compose([transforms.ToTensor()])

            cover_img = loader(cover_img)
            cover_img  = cover_img.cuda()
            cover_img = cover_img.unsqueeze(0)
            clean_img = Image.open("../dataset/clean.png")
            clean_img = loader(clean_img)
            clean_img = clean_img.unsqueeze(0).cuda()
            clean_img = clean_img[:, :, 0:32, 0:32]

            clean_img_np = clean_img.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))

            cover_B = cover_img

            cover_B_np = cover_B.squeeze(0).mul(255).cpu().numpy().transpose((1,2,0))  #记得乘255
            cover_B_gray_np = 0.299 * cover_B_np[:,:,0] +  0.587 * cover_B_np[:,:,1] + 0.114 * cover_B_np[:,:,2]
            mask = filters.sobel(cover_B_gray_np)
            mask = mask > 28
            mask = mask.astype(numpy.uint8)    #edge 作为水印和原图进行拼接   uint 8  (256,768) 值为0，1
            mask = mask[:,:,numpy.newaxis]  #(256,256,1)
            mask = mask
            mask_np = mask * 255
            mask_t = torch.from_numpy(mask_np.transpose((2, 0, 1)))
            mask_t = mask_t.float().div(255)
            mask_t = mask_t.repeat(3,1,1).unsqueeze(0).cuda()



            color = torch.ones((3)).cuda()
            color[0] = random_R
            color[1] = random_G
            color[2] = random_B


            edge_np = numpy.ones((32, 32, 3))
            edge_np[:,:,0:1] = random_R * mask 
            edge_np[:,:,1:2] = random_G * mask 
            edge_np[:,:,2:3] = random_B * mask 
            edge_t = torch.from_numpy(edge_np.transpose((2, 0, 1)))
            edge_t = edge_t.float().div(255).unsqueeze(0).cuda()


            watermark = edge_t * mask_t + clean_img * (1 - mask_t)
            watermark0 = watermark
            # watermark_np = watermark.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))

            concat_img = torch.cat([cover_B, watermark], dim=1)
            stego_img = Hnet(concat_img)
            stego_img0 = stego_img
            # stego_img = torch.clamp(stego_img, 0, 1)



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
            diff_0 = diff
            resultImg1 = torch.cat([ cover_B, diff, 5*diff, 10*diff, stego_img], 0)
            resultImgName1 = '%s/%s_psnr_%04f_ssim_%04f_lpips_%04f.png' % (em_psnrdir,cover_img_name,em_psnr,em_ssim,em_lpips)
            vutils.save_image(resultImg1, resultImgName1, nrow=5, padding=5, pad_value=1, normalize=False)

            resultImgName1_1 = '%s/%s_ori.png' % (datasetdir,cover_img_name)
            vutils.save_image(cover_B, resultImgName1_1, normalize=False)


            resultImgName1_2 = '%s/%s_diff.png' % (datasetdir,cover_img_name)
            vutils.save_image(diff, resultImgName1_2, normalize=False)

            diff_5 = torch.clamp(diff_0*5,0,1)
            resultImgName1_3 = '%s/%s_5diff.png' % (datasetdir,cover_img_name)
            vutils.save_image(diff_5, resultImgName1_3, normalize=False)

            diff_10 = torch.clamp(diff_0*10,0,1)
            resultImgName1_4 = '%s/%s_10diff.png' % (datasetdir,cover_img_name)
            vutils.save_image(diff_10, resultImgName1_4, normalize=False)

            resultImgName1_5 = '%s/%s_poison.png' % (datasetdir,cover_img_name)
            vutils.save_image(stego_img, resultImgName1_5, normalize=False)
# norm
#             mean=torch.tensor([0.485, 0.456, 0.406]).cuda()
#             std=torch.tensor([0.229, 0.224, 0.225]).cuda()
#             stego_img.sub_(mean[None,:,None,None]).div_(std[None,:,None,None])
#
#             color = color.sub(mean[:,None,None]).div(std[:,None,None])


            # #baseline
#             stego_img = F.pad(stego_img,(4,4,4,4),'constant',0)
#
#             opt.cropsize = 32
#             k = random.randint(0, 40 - opt.cropsize)
#             j = random.randint(0, 40 - opt.cropsize)
#             stego_img = stego_img[:, :, k:k + opt.cropsize, j:j + opt.cropsize]

# crop and resize and flip

            if data_aug == 'crop28_resize32':
                # print(data_aug)
                opt.cropsize = 28
                k = random.randint(0, 32 - opt.cropsize)
                j = random.randint(0, 32 - opt.cropsize)
                stego_img = stego_img[:, :, k:k + opt.cropsize, j:j + opt.cropsize]
                
                # stego_img = torch.flip(stego_img,[3])
                
                opt.resizesize = 32
                stego_img = F.interpolate(stego_img, [opt.resizesize, opt.resizesize], mode='bilinear')
                stego_img = torch.flip(stego_img,[3])
            


            cover_B_np = stego_img.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))  # 记得乘255
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

# watermark re-compute

            edge_np[:, :, 0:1] = random_R * mask
            edge_np[:, :, 1:2] = random_G * mask
            edge_np[:, :, 2:3] = random_B * mask
            edge_t = torch.from_numpy(edge_np.transpose((2, 0, 1)))
            edge_t = edge_t.float().div(255).unsqueeze(0).cuda()

            watermark = edge_t * mask_t + clean_img * (1 - mask_t)

#extratct
            watermark_np = watermark.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))

            reveal_img = Rnet(stego_img)
            reveal_img_np = reveal_img.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))
            ex_psnr = PSNR(reveal_img_np, watermark_np)
            ex_total_psnr += ex_psnr
            ex_ssim = SSIM(reveal_img.cpu(), watermark.cpu())
            ex_total_ssim += ex_ssim

            resultImg11 = torch.cat([stego_img, watermark, reveal_img], 0)
            resultImgName11 = '%s/%s_psnr_%04f_ssim_%04f.png' % (extractdir,cover_img_name,ex_psnr,ex_ssim)
            vutils.save_image(resultImg11, resultImgName11, nrow=3, padding=1, normalize=False)


            clean_rev_img = Rnet(cover_B)
            clean_rev_img_np = clean_rev_img.squeeze(0).mul(255).cpu().numpy().transpose((1, 2, 0))
            cl_ex_psnr = PSNR(clean_rev_img_np, clean_img_np)
            cl_ex_total_psnr += cl_ex_psnr
            cl_ex_ssim = SSIM(clean_rev_img.cpu(), clean_img.cpu())
            cl_ex_total_ssim += cl_ex_ssim

            resultImg111 = torch.cat([cover_B, clean_img, clean_rev_img], 0)
            resultImgName111 = '%s/%s_psnr_%04f_ssim_%04f.png' % (cleandir,cover_img_name,ex_psnr,ex_ssim)
            vutils.save_image(resultImg111, resultImgName111, nrow=3, padding=1, normalize=False)


            print('%s.png  #######################################'%cover_img_name, file=f)
            print('em_psnr:%s'%em_psnr, file=f)
            print('em_ssim:%s'%em_ssim, file=f)
            print('em_lpips:%s'%em_lpips, file=f)
            print('ex_psnr:%s'%ex_psnr, file=f)
            print('ex_ssim:%s'%ex_ssim, file=f)


            reveal_img_edge = reveal_img * mask_t
            reveal_img_edge = reveal_img_edge.view(1,3,-1)
            zero_num_mask = reveal_img_edge ==0

            zero_num = torch.sum(zero_num_mask,2).float()
            non_zero_num = 32 * 32 - zero_num

            rev_code_round = torch.sum(reveal_img_edge,2).div(non_zero_num).mul(255).round() # mean 四舍五入
            code_err = (rev_code_round - color).abs()
            print("code_err:%s"%code_err, file=f)

            err_th_5 = torch.ones_like(code_err) * 5  #Error bit corrected by threshold 5
            code_err_th_5 = code_err - err_th_5  
            code_err_th_max_5 = torch.max(code_err_th_5,1)
            code_err_th_max_5  = code_err_th_max_5[0]
            rev_success_5 = (code_err_th_max_5 <= 0).float()
            print("rev_success_5%s"%rev_success_5, file=f)
            rev_success_rate_5 += rev_success_5


            err_th_10 = torch.ones_like(code_err) * 10 
            code_err_th_10 = code_err - err_th_10
            code_err_th_max_10 = torch.max(code_err_th_10,1)  
            code_err_th_max_10  = code_err_th_max_10[0]   #max return 2 item
            rev_success_10 = (code_err_th_max_10 <= 0).float()
            print("rev_success_10%s"%rev_success_10, file=f)
            rev_success_rate_10 += rev_success_10


            err_th_15 = torch.ones_like(code_err) * 15 
            code_err_th_15 = code_err - err_th_15
            code_err_th_max_15 = torch.max(code_err_th_15,1)  
            code_err_th_max_15  = code_err_th_max_15[0]   #max return 2 item
            rev_success_15 = (code_err_th_max_15 <= 0).float()
            print("rev_success_15%s"%rev_success_15, file=f)
            rev_success_rate_15 += rev_success_15

            err_th_20 = torch.ones_like(code_err) * 20 
            code_err_th_20 = code_err - err_th_20
            code_err_th_max_20 = torch.max(code_err_th_20,1)  
            code_err_th_max_20  = code_err_th_max_20[0]   #max return 2 item
            rev_success_20 = (code_err_th_max_20 <= 0).float()
            print("rev_success_20%s"%rev_success_20, file=f)
            rev_success_rate_20 += rev_success_20         


            resultImg3 =  stego_img.clone()
            resultImgName3 = '%s/%s.png' % (embeddir,cover_img_name)
            vutils.save_image(resultImg3, resultImgName3, nrow=3, padding=0, normalize=False)


            resultImg_paper = torch.cat([ cover_B,  stego_img0, (stego_img0-cover_B), 10*(stego_img0-cover_B), watermark0, stego_img, watermark, reveal_img,mask_t], 0)
            resultImgName_paper = '%s/%s.png' % (paperdir,cover_img_name)
            vutils.save_image(resultImg_paper, resultImgName_paper, nrow=10, padding=0, normalize=False)


        print(' ####################AVERAGE METRIC ###################', file=f)
        print('em_average_psnr:%s'%(em_total_psnr/(i+1)), file=f)
        print('em_average_ssim:%s'%(em_total_ssim/(i+1)), file=f)
        print('em_average_lpips:%s'%(em_total_lpips/(i+1)), file=f)

        print('ex_average_psnr:%s'%(ex_total_psnr/(i+1)), file=f)
        print('ex_average_ssim:%s'%(ex_total_ssim/(i+1)), file=f)
        print('average_rev_success_rate_5:%s'%(rev_success_rate_5/(i+1)), file=f)
        print('average_rev_success_rate_10:%s'%(rev_success_rate_10/(i+1)), file=f)
        print('average_rev_success_rate_15:%s'%(rev_success_rate_15/(i+1)), file=f)
        print('average_rev_success_rate_20:%s'%(rev_success_rate_20/(i+1)), file=f)

        print(' ####################AVERAGE METRIC ###################')
        print('em_average_psnr:%s'%(em_total_psnr/(i+1)))
        print('em_average_ssim:%s'%(em_total_ssim/(i+1)))
        print('em_average_lpips:%s'%(em_total_lpips/(i+1)))

        print('ex_average_psnr:%s'%(ex_total_psnr/(i+1)))
        print('ex_average_ssim:%s'%(ex_total_ssim/(i+1)))
        print('average_rev_success_rate_5:%s'%(rev_success_rate_5/(i+1)))
        print('average_rev_success_rate_10:%s'%(rev_success_rate_10/(i+1)))
        print('average_rev_success_rate_15:%s'%(rev_success_rate_15/(i+1)))
        print('average_rev_success_rate_20:%s'%(rev_success_rate_20/(i+1)))


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