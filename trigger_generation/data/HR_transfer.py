# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: main.py
@time: 2018/3/20

"""




import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

import utils.transformed as transforms
from data.HR_Dataset import ZJFolder

from models.HidingUNet import UnetGenerator
from models.HidingUNet import UnetGenerator_IN

from models.ReflectionUNet import UnetGenerator2


from models.networks import ResnetGenerator
from models.HidingRes import HidingRes

from models.Edgenet import ExtractNet


from models.RevealNet import RevealNet
from models.Discriminator import PixelDiscriminator
from models.Discriminator import PatchDiscriminator
from models.Discriminator_SN import Discriminator_SN
from models.Discriminator import Discriminator
from models.Discriminator_Switch import Discriminator_Switch

import numpy as np
from PIL import Image
import pytorch_ssim
from vgg import Vgg16

import random
from random import choice
import torch.nn.functional as F
import math

from skimage import filters

# from torchsample.transforms.affine_transforms import RandomRotate
# from torchsample.transforms.affine_transforms import Rotate

# import torchsnooper #检查tensor 类型

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--cropsize', type=int, default=196,
                    help='the number of frames')
parser.add_argument('--degree', type=int, default=45,
                    help='the number of frames')
parser.add_argument('--resizesize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=200,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.001') # stasblized training
parser.add_argument('--lr_R', type=float, default=0.00002,
                    help='learning rate, default=0.001') # stasblized training                    
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--Dnet', default='',
                    help="path to Discriminator (to continue training)")
parser.add_argument('--trainpics', default='/data-x/g11/zhangjie/cvpr/exp_chk/HR/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='/data-x/g11/zhangjie/cvpr/exp_chk/HR/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='/data-x/g11/zhangjie/cvpr/exp_chk/HR/',
                    help='folder to output test images')
parser.add_argument('--runfolder', default='/data-x/g11/zhangjie/cvpr/exp_chk/HR/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='/data-x/g11/zhangjie/cvpr/exp_chk/HR/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='/data-x/g11/zhangjie/cvpr/exp_chk/HR/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='/data-x/g11/zhangjie/cvpr/exp_chk/HR/',
                    help='folder to save the experiment codes')


parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')


#datasets to train
parser.add_argument('--datasets', type=str, default='M1_edge',
                    help='denoise/derain')

# #read secret image
# parser.add_argument('--secret', type=str, default='ustc',
#                     help='secret folder')

#hyperparameter of loss

parser.add_argument('--beta', type=float, default=1,
                    help='hyper parameter of beta :secret_reveal err')
parser.add_argument('--betagan', type=float, default=1,
                    help='hyper parameter of beta :gans weight')
parser.add_argument('--betagans', type=float, default=0.01,
                    help='hyper parameter of beta :gans weight')
parser.add_argument('--betapix', type=float, default=1.0,
                    help='hyper parameter of beta :pixel_loss weight')

parser.add_argument('--betamse', type=float, default=10000,
                    help='hyper parameter of beta: mse_loss')
parser.add_argument('--betacons', type=float, default=0,
                    help='hyper parameter of beta: consist_loss')
parser.add_argument('--betaedge', type=float, default=13,
                    help='hyper parameter of beta: edge_loss')                    
parser.add_argument('--betaclean', type=float, default=100,
                    help='hyper parameter of beta: clean_loss')
parser.add_argument('--betacleanA', type=float, default=0,
                    help='hyper parameter of beta: clean_loss')
parser.add_argument('--betacleanB', type=float, default=1,
                    help='hyper parameter of beta: clean_loss')
parser.add_argument('--betassim', type=float, default=0,
                    help='hyper parameter of beta: ssim_loss')
parser.add_argument('--ssimws', type=float, default=11,
                    help='hyper parameter of beta: ssim  window_size')
parser.add_argument('--betavgg', type=float, default=1,
                    help='hyper parameter of beta: vgg_loss')
parser.add_argument('--betapsnr', type=float, default=0,
                    help='hyper parameter of beta: psnr_loss')
parser.add_argument('--Dnorm', type=str, default='instance', help=' [instance | spectral | switch]')

parser.add_argument('--num_downs', type=int, default= 7 , help='nums of  Unet downsample')

parser.add_argument('--clip', action='store_true', help='clip container_img')

parser.add_argument('--G_net', type=str, default='unet', help=' [unet | resnet ]')
parser.add_argument('--G_norm', type=str, default='nn.BatchNorm2d', help=' [nn.BatchNorm2d | nn.InstanceNorm2d ]')

parser.add_argument('--R_net', type=str, default='resnet', help=' [unet | resnet | extract]')
parser.add_argument('--R_loss', type=str, default='mse', help=' [mse | ssim ]')
parser.add_argument('--padding', type=str, default='reflect', help=' [reflect | same ]')

parser.add_argument('--adv', action='store_true', help='add adversarial training ')
parser.add_argument('--vgg_train', action='store_true', help='add vgg loss into training ')

parser.add_argument('--load_chk', action='store_true', help='use per_trained model ')

parser.add_argument('--nocrop', action='store_true', help='do not crop ')
parser.add_argument('--noresize', action='store_true', help='do not resize ')
parser.add_argument('--norot', action='store_true', help='do not rot')

parser.add_argument('--rot_90', action='store_true', help='range 90 ')


parser.add_argument('--color_step', type=int, default= 1 , help='color range, 0:only two color')


parser.add_argument('--TH', type=float, default=10, help='threshold of err ')

def main():
    ############### define global parameters ###############
    global opt, optimizerH, optimizerR, optimizerD, writer, logPath, SRPath, schedulerH, schedulerR, schedulerD
    global val_loader, smallestLoss,  mse_loss, ssim_loss, gan_loss, pixel_loss, patch, criterion_GAN, criterion_pixelwise,vgg, vgg_loss

    #################  输出配置参数   ###############
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True

    ############  create the dirs to save the result #############

    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    experiment_dir = opt.hostname  + "_" + opt.remark + "_" + cur_time
    opt.outckpts += experiment_dir + "/checkPoints"
    opt.trainpics += experiment_dir + "/trainPics"
    opt.validationpics += experiment_dir + "/validationPics"
    opt.outlogs += experiment_dir + "/trainingLogs"
    opt.outcodes += experiment_dir + "/codes"
    opt.testPics += experiment_dir + "/testPics"
    opt.runfolder += experiment_dir + "/run"

    if not os.path.exists(opt.outckpts):
        os.makedirs(opt.outckpts)
    if not os.path.exists(opt.trainpics):
        os.makedirs(opt.trainpics)
    if not os.path.exists(opt.validationpics):
        os.makedirs(opt.validationpics)
    if not os.path.exists(opt.outlogs):
        os.makedirs(opt.outlogs)
    if not os.path.exists(opt.outcodes):
        os.makedirs(opt.outcodes)
    if not os.path.exists(opt.runfolder):
        os.makedirs(opt.runfolder)        
    if (not os.path.exists(opt.testPics)) and opt.test != '':
        os.makedirs(opt.testPics)



    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)
    SRPath  = opt.outlogs + '/%s_%d_SR.txt' % (opt.dataset, opt.batchSize)

    # 保存模型的参数
    print_log(str(opt), logPath)
    # 保存本次实验的代码
    save_current_codes(opt.outcodes)
    # tensorboardX writer
    writer = SummaryWriter(log_dir=opt.runfolder, comment='**' + opt.hostname + "_" + opt.remark)




    ##############   获取数据集   ############################
    DATA_DIR_root = './datasets/'
    DATA_DIR = os.path.join(DATA_DIR_root, opt.datasets)

    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')
    # secretdir = os.path.join(DATA_DIR_root, opt.secret)
    

    
    train_dataset = ZJFolder(traindir, opt)
    val_dataset = ZJFolder(valdir, opt)
		

		
    assert train_dataset
    assert val_dataset
    # assert secret_dataset


    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                              shuffle=True, num_workers=int(opt.workers))

    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                            shuffle=True, num_workers=int(opt.workers))    	

    ##############   所使用网络结构   ############################


    Hnet = UnetGenerator2(input_nc=6, output_nc=3, num_downs= 8, output_function=nn.Sigmoid)
    Hnet.cuda()
    Hnet.apply(weights_init)

    Rnet = HidingRes(in_c=3, out_c=3)
    Rnet.cuda()
    Rnet.apply(weights_init)

    if opt.adv:

        Dnet = Discriminator(in_channels=3, requires_grad= True)
    else:
        Dnet = Discriminator(in_channels=3, requires_grad= False)

    Dnet.cuda()
    # Dnet.apply(weights_init)
    
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.imageSize // 2 ** 4, opt.imageSize // 2 ** 4)


    # setup optimizer
    optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)
    schedulerH = StepLR(optimizerH, step_size=80)
    if opt.adv:
        optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr_R, betas=(opt.beta1, 0.999))
    # schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)
    else:
        optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    schedulerR = StepLR(optimizerR, step_size=80)

    optimizerD = optim.Adam(Dnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=5, verbose=True)
    schedulerD = StepLR(optimizerD, step_size=80)


    if opt.load_chk:

        if opt.color_step == 1:
            print("LOAD STEP 1")
            opt.Hnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/all_90_all_128/netH_epoch_104,sumloss=605.632568,Hloss=553.303345.pth"
            opt.Rnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/all_90_all_128/netR_epoch_104,TH5 SR=0.294000,TH10 SR=0.654000,sumloss=605.632568,Rloss=52.329250.pth"
            opt.Dnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/all_90_all_128/netD_epoch_104,sumloss=605.632568,Dloss=658.134216.pth"

        elif opt.color_step == 10:
            print("LOAD STEP 10")
            opt.Hnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/10_90_all_0/netH_epoch_175,sumloss=316.128296,Hloss=299.303650.pth"
            opt.Rnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/10_90_all_0/netR_epoch_175,TH5 SR=0.612000,TH10 SR=0.898000,sumloss=316.128296,Rloss=16.824657.pth"
            opt.Dnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/10_90_all_0/netD_epoch_175,sumloss=316.128296,Dloss=660.032410.pth"

        elif opt.color_step == 20:
            print("LOAD STEP 20")
            # opt.Hnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/20_90_all_0/netH_epoch_155,sumloss=313.462097,Hloss=295.896088.pth"
            # opt.Rnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/20_90_all_0/netR_epoch_155,TH5 SR=0.708000,TH10 SR=0.918000,sumloss=313.462097,Rloss=17.566023.pth"
            # opt.Dnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/20_90_all_0/netD_epoch_155,sumloss=313.462097,Dloss=662.467468.pth"
            opt.Hnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/20_90_all_128_fix/netH_epoch_143,sumloss=479.333344,Hloss=439.185669.pth"
            opt.Rnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/20_90_all_128_fix/netR_epoch_143,TH5 SR=0.322000,TH10 SR=0.752000,sumloss=479.333344,Rloss=40.147686.pth"
            opt.Dnet ="/data-x/g11/zhangjie/cvpr/pth/HR_model/noadv/20_90_all_128_fix/netD_epoch_143,sumloss=479.333344,Dloss=621.136475.pth"


    # 判断是否接着之前的训练
    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet))
    # 两块卡加这行
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
    print_network(Hnet)


    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet).cuda()
    print_network(Rnet)

    if opt.Dnet != '':
        Dnet.load_state_dict(torch.load(opt.Dnet))
    if opt.ngpu > 1:
        Dnet = torch.nn.DataParallel(Dnet).cuda()
    print_network(Dnet)


    # define loss
    mse_loss = nn.MSELoss().cuda()
    criterion_GAN = nn.MSELoss().cuda()
    criterion_pixelwise = nn.L1Loss().cuda()
    ssim_loss = pytorch_ssim.SSIM(window_size = opt.ssimws).cuda()
    vgg = Vgg16(requires_grad=False).cuda()

    smallestLoss = 10000
    print_log("training is beginning .......................................................", logPath)
    for epoch in range(opt.niter):
        ######################## train ##########################################
        # rev_success_5_rate, rev_success_10_rate,  val_hloss, val_rloss, val_r_mseloss, val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses,vgg_loss, val_sumloss = validation(val_loader,  epoch, Hnet=Hnet, Rnet=Rnet, Dnet=Dnet)

        train(train_loader,  epoch, Hnet=Hnet, Rnet=Rnet, Dnet=Dnet)

        ####################### validation  #####################################
        rev_success_5_rate, rev_success_10_rate,  val_hloss, val_rloss, val_r_mseloss, val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses,vgg_loss, val_sumloss = validation(val_loader,  epoch, Hnet=Hnet, Rnet=Rnet, Dnet=Dnet)

        ####################### adjust learning rate ############################
        # schedulerH.step(val_sumloss)
        # schedulerR.step(val_rloss)
        # schedulerD.step(val_dloss)

        schedulerH.step()
        schedulerR.step()
        schedulerD.step()

        torch.save(Hnet.module.state_dict(),
                   '%s/netH_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (
                       opt.outckpts, epoch, val_sumloss, val_hloss))
        torch.save(Rnet.module.state_dict(),
                   '%s/netR_epoch_%d,TH5 SR=%.6f,TH10 SR=%.6f,sumloss=%.6f,Rloss=%.6f.pth' % (
                       opt.outckpts, epoch, rev_success_5_rate, rev_success_10_rate,val_sumloss, val_rloss))
        torch.save(Dnet.module.state_dict(),
                   '%s/netD_epoch_%d,sumloss=%.6f,Dloss=%.6f.pth' % (
                       opt.outckpts, epoch, val_sumloss, val_dloss))

    writer.close()

##################################################random  TRAIN###########################################################
def train(train_loader,  epoch, Hnet, Rnet, Dnet):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  # 纪录每个epoch H网络的loss
    Rlosses = AverageMeter()  # 纪录每个epoch R网络的loss
    R_mselosses = AverageMeter()
    Dlosses = AverageMeter()
    FakeDlosses = AverageMeter()
    RealDlosses = AverageMeter()
    Ganlosses = AverageMeter()
    Pixellosses = AverageMeter()
    Vgglosses =AverageMeter()    
    SumLosses = AverageMeter()  # 纪录每个epoch Hloss + β*Rloss
    success_rate_10 = AverageMeter()
    success_rate_5 = AverageMeter()


    # switch to train mode
    Hnet.train()
    Rnet.train()
    Dnet.train()


    # Tensor type
    Tensor = torch.cuda.FloatTensor 

    start_time = time.time()



    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Hnet.zero_grad()
        Rnet.zero_grad()

        this_batch_size = int(data[0].size()[0])
        opt.batchSize = this_batch_size  # 处理每个epoch 最后一个batch可能不足opt.bachsize
        
        cover_img_A, cover_img_B, watermark, mask, color, degree , matrix,cover_img_A_rot, cover_img_B_rot, watermark_rot, mask_rot = data


        cover_img_A = cover_img_A[0:this_batch_size, :, :, :]
        cover_img_B = cover_img_B[0:this_batch_size, :, :, :]
        watermark = watermark[0:this_batch_size, :, :, :]
        mask = mask[0:this_batch_size, :, :, :]
        cover_img_A_rot = cover_img_A_rot[0:this_batch_size, :, :, :]
        cover_img_B_rot = cover_img_B_rot[0:this_batch_size, :, :, :]
        watermark_rot = watermark_rot[0:this_batch_size, :, :, :]
        mask_rot = mask_rot[0:this_batch_size, :, :, :]

        loader = transforms.Compose([  transforms.ToTensor(), ])
        clean_img = Image.open("./datasets/secret/clean.png")
        clean_img = loader(clean_img) 
        clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)         
        clean_img = clean_img[0:this_batch_size, :, :, :]  # 1,3,256,256    


        cover_img_A = cover_img_A.cuda()
        cover_img_A_rot = cover_img_A_rot.cuda()
        cover_img_B = cover_img_B.cuda()
        cover_img_B_rot = cover_img_B_rot.cuda()
        watermark = watermark.cuda() 
        watermark_rot = watermark_rot.cuda()       
        clean_img = clean_img.cuda()
        mask = mask.cuda()
        mask_rot = mask_rot.cuda()
        matrix = matrix.cuda()
        color = color.cuda()
        degree = degree.cuda()

        # secret = watermark


        watermark = watermark * mask + clean_img * (1 - mask)
        secret = watermark
        concat_img = torch.cat([cover_img_B, secret], dim=1)
        watermark_rot = watermark_rot * mask_rot + clean_img * (1 - mask_rot)


        concat_imgv = Variable(concat_img)  # concatImg 作为H网络的输入
        cover_imgv = Variable(cover_img_B)  # coverImg 作为H网络的label
        container_img = Hnet(concat_imgv)  # 得到藏有secretimg的containerImg

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((cover_imgv.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((cover_imgv.size(0), *patch))), requires_grad=False)
        # pred_fake = Dnet(container_img, cover_imgv)
        pred_fake = Dnet(container_img)
        gan_loss = criterion_GAN(pred_fake, valid)
        pixel_loss = criterion_pixelwise(container_img, cover_imgv)
        vgg_loss = mse_loss(vgg(container_img).relu2_2, vgg(cover_imgv).relu2_2)

        if opt.adv:
            # print("AAAAAAAAAAAAA")
            # errH = 10000 * (mse_loss(container_img, cover_imgv) +  0.01 * (opt.betagans * gan_loss + opt.betapix * pixel_loss) + opt.betavgg * vgg_loss  )
            # errH = 10000 * (mse_loss(container_img, cover_imgv) +  opt.betagan * (opt.betagans * gan_loss + opt.betapix * pixel_loss) + opt.betavgg * vgg_loss  )
            errH = 10000 * (mse_loss(container_img, cover_imgv) +  0.1 * pixel_loss )
            # errH = 10000 * (mse_loss(container_img, cover_imgv) + opt.betagans * gan_loss )
        elif  opt.vgg_train:
            errH = 10000 * (mse_loss(container_img, cover_imgv) +  opt.betavgg * vgg_loss)  
        else:
            errH = 10000 * (mse_loss(container_img, cover_imgv))

        with torch.no_grad(): 
            rev_secret_img = Rnet(container_img)
            clean_rev_secret_img_A = Rnet(cover_img_A)
            clean_rev_secret_img_B = Rnet(cover_imgv)

        ##############################################random rotate#########################################

        # container_img_rot = ZJ_ROT(container_img, degree)
        container_img_rot = ZJ_ROT(container_img, matrix)
        # container_img_rot = container_img
        cover_imgv_rot  = cover_img_B_rot

        ##############################################crop_resize#########################################
        # opt.cropsize = choice([128, 144, 160, 176, 192, 200, 224, 256 ])


        if opt.nocrop :
            opt.cropsize = 256

        elif random.randint(0,1):
            # print("CROPCROPCROPCROPCROPCROPCROPCROPCROP")
            opt.cropsize = random.randint(32, 128)*2
        else:
            # print("NOCROP NOCROP NOCROP")
            opt.cropsize = 256

        # opt.cropsize = random.randint(64，256)
        k = random.randint(0, 256 - opt.cropsize)
        j = random.randint(0, 256 - opt.cropsize)

        container_img_crop = container_img_rot[:, :, k:k+opt.cropsize, j:j+opt.cropsize]
        cover_img_A_crop = cover_img_A_rot[:, :, k:k+opt.cropsize, j:j+opt.cropsize]
        cover_imgv_crop = cover_imgv_rot[:, :, k:k+opt.cropsize, j:j+opt.cropsize]
        clean_img = clean_img[:, :, k:k+opt.cropsize, j:j+opt.cropsize]


        # opt.resizesize = random.randint(8,32)*8

        if opt.noresize :
            opt.resizesize = opt.cropsize
            container_img_resize =container_img_crop
            cover_img_A_resize =cover_img_A_crop
            cover_imgv_resize = cover_imgv_crop
            clean_img_resize = clean_img           

        opt.resizesize = 128

        container_img_resize = F.interpolate(container_img_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
        cover_img_A_resize = F.interpolate(cover_img_A_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
        cover_imgv_resize = F.interpolate(cover_imgv_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
        clean_img_resize = F.interpolate(clean_img, [opt.resizesize,opt.resizesize], mode='bilinear')

        # elif random.randint(0,1):
        #     # print("RESIZE RESIZE RESIZE")
        #     opt.resizesize = 128

        #     container_img_resize = F.interpolate(container_img_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
        #     cover_img_A_resize = F.interpolate(cover_img_A_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
        #     cover_imgv_resize = F.interpolate(cover_imgv_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
        #     clean_img_resize = F.interpolate(clean_img, [opt.resizesize,opt.resizesize], mode='bilinear')

        # else:
        #     opt.resizesize = opt.cropsize
        #     # print("NORESIZE NORESIZE NORESIZE")
        #     # print("FFFFFFFFFFFFF",opt.resizesize)

        #     container_img_resize =container_img_crop
        #     cover_img_A_resize =cover_img_A_crop
        #     cover_imgv_resize = cover_imgv_crop
        #     clean_img_resize = clean_img


        Watermark_resize = torch.ones_like(container_img_resize).cuda()
        Mask_resize = torch.ones_like(container_img_resize).cuda()
        for m in range(this_batch_size):
            cover_imgv_resize_m = cover_imgv_resize[m:m+1,:,:,:]
            cover_imgv_resize_m = cover_imgv_resize_m.squeeze(0)
            cover_imgv_resize_np = cover_imgv_resize_m.mul(255).cpu().numpy().transpose((1, 2, 0))
            cover_imgv_resize_np = 0.299 * cover_imgv_resize_np[:,:,0] +  0.587 * cover_imgv_resize_np[:,:,1] + 0.114 * cover_imgv_resize_np[:,:,2]
            mask_resize = filters.sobel(cover_imgv_resize_np)
            mask_resize = mask_resize > 28

            mask_resize = mask_resize.astype(np.uint8)    #edge 作为水印和原图进行拼接   uint 8  (256,256) 值为0，1
            mask_resize = mask_resize[:,:,np.newaxis]  #(256,256,1)
            mask_resize_t = torch.from_numpy(mask_resize.transpose((2, 0, 1)))
            mask_resize_t = mask_resize_t.repeat(3,1,1).unsqueeze(0)

            watermark_resize = torch.ones((1,3,opt.resizesize,opt.resizesize)).cuda()

            watermark_resize[:,0:1,:,:] = color[m,0] * mask_resize_t[:,0:1,:,:]
            watermark_resize[:,1:2,:,:] = color[m,1]  * mask_resize_t[:,0:1,:,:]
            watermark_resize[:,2:3,:,:] = color[m,2]  * mask_resize_t[:,0:1,:,:]
            watermark_resize = watermark_resize.float().div(255)

            Mask_resize[m:m+1,:,:,:] = mask_resize_t
            Watermark_resize[m:m+1,:,:,:] = watermark_resize


        ##############################################crop_resize#########################################

        container_img_transfer = container_img_resize
        cover_img_A_transfer = cover_img_A_resize
        cover_imgv_transfer = cover_imgv_resize
        clean_img_transfer = clean_img_resize
        mask_transfer = Mask_resize.cuda()
        watermark_transfer = Watermark_resize.cuda()
        watermark_transfer = watermark_transfer * mask_transfer + clean_img_transfer * (1 - mask_transfer)


        rev_secret_img_transfer = Rnet(container_img_transfer)
        watermark_transfer = Variable(watermark_transfer)


        # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',container_img_transfer.shape)
        # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',rev_secret_img_transfer.shape)
        # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',mask_transfer.shape)
        # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',(rev_secret_img_transfer*mask_transfer).shape)
        # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',watermark_transfer.shape)
        # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',(watermark_transfer*mask_transfer).shape)



        err_edge = 10000 * mse_loss(rev_secret_img_transfer * mask_transfer , watermark_transfer * mask_transfer )
        err_back = 10000 * mse_loss(rev_secret_img_transfer * (1-mask_transfer) , watermark_transfer * (1-mask_transfer) )         
        errR_mse = opt.betaedge * err_edge + err_back
        # errR_mse.backward()
        # errR_mse = 10000 * mse_loss(rev_secret_img_transfer , watermark_transfer)
        
        clean_imgv = Variable(clean_img_transfer)
        clean_rev_secret_img_transfer_A = Rnet(cover_img_A_transfer)
        errR_clean_A = 10000 * mse_loss(clean_rev_secret_img_transfer_A, clean_imgv)

        clean_rev_secret_img_transfer_B = Rnet(cover_imgv_transfer)
        errR_clean_B = 10000 * mse_loss(clean_rev_secret_img_transfer_B, clean_imgv)

    ################################################################ Extract loss#########################################


        errR_clean =opt.betacleanA * errR_clean_A + opt.betacleanB * errR_clean_B 
        errR = errR_mse + opt.betaclean * errR_clean
        # errR.backward()
        betaerrR_secret = opt.beta * errR
        err_sum = errH + betaerrR_secret 


    #     # 计算梯度
        err_sum.backward()
        # 优化两个网络的参数
        optimizerH.step()
        optimizerR.step()


        #  Train Discriminator
        Dnet.zero_grad()
        # Real loss
        pred_real = Dnet(cover_imgv)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = Dnet(container_img.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        errD = 10000 * 0.5 * (loss_real + loss_fake)
        
        if opt.adv:
            # print("DDDDDDDDDDDDDDDDD")
            errD.backward()

        optimizerD.step()


####################################################code err calcualate##############################################

        rev_secret_img_edge = rev_secret_img_transfer * mask_transfer
        rev_secret_img_edge = rev_secret_img_edge.view(this_batch_size,3,-1)
        zero_num_mask = rev_secret_img_edge ==0
        zero_num = torch.sum(zero_num_mask,2).float()
        non_zero_num = opt.resizesize * opt.resizesize - zero_num
        rev_code_round = torch.sum(rev_secret_img_edge,2).div(non_zero_num).mul(255).round() # mean 四舍五入
        code_err = (rev_code_round - color).abs()

        err_th_5 = torch.ones_like(code_err) * 5  #Error bit corrected by threshold 5
        code_err_th_5 = code_err - err_th_5  
        code_err_th_max_5 = torch.max(code_err_th_5,1)
        code_err_th_max_5  = code_err_th_max_5[0]
        rev_success_5 = (code_err_th_max_5 <= 0).float()
        rev_success_rate_5 = torch.mean(rev_success_5,0)

        err_th_10 = torch.ones_like(code_err) * 10 
        code_err_th_10 = code_err - err_th_10
        code_err_th_max_10 = torch.max(code_err_th_10,1)  
        code_err_th_max_10  = code_err_th_max_10[0]   #max return 2 item
        rev_success_10 = (code_err_th_max_10 <= 0).float()
        rev_success_rate_10 = torch.mean(rev_success_10,0)


        Hlosses.update(errH.data, this_batch_size)  # 纪录H loss值
        Rlosses.update(errR.data, this_batch_size)  # 纪录R loss值
        R_mselosses.update(errR_mse.data, this_batch_size) # 纪录R_mse loss值


        Dlosses.update(errD.data, this_batch_size)  # 纪录D loss值
        FakeDlosses.update(loss_fake.data, this_batch_size)  # 纪录fakeD loss值
        RealDlosses.update(loss_real.data, this_batch_size)  # 纪录realD loss值
        Ganlosses.update(gan_loss.data, this_batch_size) #记录gan loss
        Pixellosses.update(pixel_loss.data, this_batch_size) #记录pixel loss
        Vgglosses.update(vgg_loss.data, this_batch_size)        
        SumLosses.update(err_sum.data, this_batch_size)
        success_rate_5.update(rev_success_rate_5.data, this_batch_size)
        success_rate_10.update(rev_success_rate_10.data, this_batch_size)

        # 更新一个batch的时间
        batch_time.update(time.time() - start_time)
        start_time = time.time()


        # 日志信息
        log = '[%d/%d][%d/%d]\t TH5 SUCCESS Rate: %.4f TH10 SUCCESS Rate: %.4f Loss_H: %.4f Loss_R: %.4f Loss_R_mse: %.4f  Loss_D: %.4f Loss_FakeD: %.4f Loss_RealD: %.4f Loss_Gan: %.4f Loss_Pixel: %.4f Loss_Vgg: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f\t '% (
            epoch, opt.niter, i, len(train_loader),
            success_rate_5.val, success_rate_10.val ,Hlosses.val, Rlosses.val, R_mselosses.val,  Dlosses.val, FakeDlosses.val, RealDlosses.val, Ganlosses.val, Pixellosses.val, Vgglosses.val, SumLosses.val, data_time.val, batch_time.val)


        # 屏幕打印日志信息
        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)

        #######################################   存储记录等相关操作       #######################################3
        # 100个step就生成一张图片

        diff = 50 * (container_img - cover_imgv)
        diff_transfer = 50 * (container_img_transfer - cover_imgv_transfer)
        
        if epoch % 1 == 0 and i % opt.resultPicFrequency == 0:

            save_result_pic2(this_batch_size, (rev_secret_img_transfer * mask_transfer).data, (watermark_transfer * mask_transfer).data ,
            (rev_secret_img_transfer * (1-mask_transfer)).data ,( watermark_transfer * (1-mask_transfer)).data,
            epoch, i, opt.trainpics)

            save_result_pic(this_batch_size, cover_img_A, cover_imgv.data, container_img.data, diff.data, watermark.data,
                rev_secret_img.data, clean_rev_secret_img_A.data, clean_rev_secret_img_B.data,
                cover_img_A_transfer, cover_imgv_transfer.data, container_img_transfer.data,diff_transfer.data,
                watermark_transfer.data, rev_secret_img_transfer.data, clean_rev_secret_img_transfer_A.data, clean_rev_secret_img_transfer_B.data,epoch, i, opt.trainpics)


            rev_success_5_str = str(rev_success_5.cpu().numpy())
            rev_success_10_str = str(rev_success_10.cpu().numpy())
            degree_str = str(degree.cpu().numpy())

            SR_log = '[%d/%d][%d/%d]\t TH5 SUCCESS Matrix: %s TH5 SUCCESS Rate: %.4f TH10 SUCCESS Matrix: %s TH10 SUCCESS Rate: %.4f Degree: %s'% (epoch, opt.niter, i, len(train_loader),
            rev_success_5_str , success_rate_5.val, rev_success_10_str , success_rate_10.val, degree_str)
            print_log(SR_log, SRPath)



    # 输出一个epoch所用时间
    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f      optimizerR_lr = %.8f     optimizerD_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr'], optimizerD.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_Hloss=%.6f\tepoch_Rloss=%.6f\tepoch_R_mseloss=%.6f\tepoch_Dloss=%.6f\tepoch_FakeDloss=%.6f\tepoch_RealDloss=%.6f\tepoch_GanLoss=%.6fepoch_Pixelloss=%.6f\tepoch_Vggloss=%.6f\tepoch_sumLoss=%.6f" % (
        Hlosses.avg, Rlosses.avg, R_mselosses.avg,  Dlosses.avg, FakeDlosses.avg, RealDlosses.avg, Ganlosses.avg, Pixellosses.avg, Vgglosses.avg, SumLosses.avg)


    print_log(epoch_log, logPath)


    # 纪录learning rate
    writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/D_lr", optimizerD.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/beta", opt.beta, epoch)
    # 每个epoch纪录一次平均loss 在tensorboard展示
    writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('train/R_mse_loss', R_mselosses.avg, epoch)

    writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('train/D_loss', Dlosses.avg, epoch)
    writer.add_scalar('train/FakeD_loss', FakeDlosses.avg, epoch) 
    writer.add_scalar('train/RealD_loss', RealDlosses.avg, epoch)   
    writer.add_scalar('train/Gan_loss', Ganlosses.avg, epoch)  
    writer.add_scalar('train/Pixel_loss', Pixellosses.avg, epoch)
    writer.add_scalar('train/Vgg_loss', Vgglosses.avg, epoch)    
    writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)
    writer.add_scalar('train/success_rate_5', success_rate_5.avg, epoch)
    writer.add_scalar('train/success_rate_10', success_rate_10.avg, epoch)



def validation(val_loader,  epoch, Hnet, Rnet, Dnet):
    print(
        "#################################################### validation begin ########################################################")
    
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Dnet.eval()
    Hlosses = AverageMeter()  # 纪录每个epoch H网络的loss
    Rlosses = AverageMeter()  # 纪录每个epoch R网络的loss
    R_mselosses = AverageMeter() 

    Dlosses = AverageMeter()  # 纪录每个epoch D网络的loss
    FakeDlosses = AverageMeter()
    RealDlosses = AverageMeter()
    Ganlosses = AverageMeter()
    Pixellosses = AverageMeter()
    Vgglosses = AverageMeter()
    success_rate_5 = AverageMeter()
    success_rate_10 = AverageMeter()

    data_time = AverageMeter()

    # Tensor type
    Tensor = torch.cuda.FloatTensor 
    with torch.no_grad(): 

  


        for i, data in enumerate(val_loader, 0):

            data_time.update(time.time() - start_time)

            Hnet.zero_grad()
            Rnet.zero_grad()

            this_batch_size = int(data[0].size()[0])
            opt.batchSize = this_batch_size  # 处理每个epoch 最后一个batch可能不足opt.bachsize
            
            cover_img_A, cover_img_B, watermark, mask, color, degree ,matrix, cover_img_A_rot, cover_img_B_rot, watermark_rot, mask_rot = data

            
            cover_img_A = cover_img_A[0:this_batch_size, :, :, :]
            cover_img_B = cover_img_B[0:this_batch_size, :, :, :]
            watermark = watermark[0:this_batch_size, :, :, :]
            mask = mask[0:this_batch_size, :, :, :]
            cover_img_A_rot = cover_img_A_rot[0:this_batch_size, :, :, :]
            cover_img_B_rot = cover_img_B_rot[0:this_batch_size, :, :, :]
            watermark_rot = watermark_rot[0:this_batch_size, :, :, :]
            mask_rot = mask_rot[0:this_batch_size, :, :, :]

            loader = transforms.Compose([  transforms.ToTensor(), ])
            clean_img = Image.open("./datasets/secret/clean.png")
            clean_img = loader(clean_img)   
            clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)         
            clean_img = clean_img[0:this_batch_size, :, :, :]  # 1,3,256,256    

            cover_img_A = cover_img_A.cuda()
            cover_img_A_rot = cover_img_A_rot.cuda()
            cover_img_B = cover_img_B.cuda()
            cover_img_B_rot = cover_img_B_rot.cuda()
            watermark = watermark.cuda() 
            watermark_rot = watermark_rot.cuda()       
            clean_img = clean_img.cuda()
            mask = mask.cuda()
            mask_rot = mask_rot.cuda()
            matrix = matrix.cuda()
            color = color.cuda()
            degree = degree.cuda()
            
            # secret = watermark
            watermark = watermark * mask + clean_img * (1-mask)
            secret = watermark
            concat_img = torch.cat([cover_img_B, secret], dim=1)
            watermark_rot = watermark_rot * mask_rot + clean_img * (1- mask_rot)


            concat_imgv = Variable(concat_img)  # concatImg 作为H网络的输入
            cover_imgv = Variable(cover_img_B)  # coverImg 作为H网络的label
            container_img = Hnet(concat_imgv)  # 得到藏有secretimg的containerImg

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((cover_imgv.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((cover_imgv.size(0), *patch))), requires_grad=False)
            # pred_fake = Dnet(container_img, cover_imgv)
            pred_fake = Dnet(container_img)
            gan_loss = criterion_GAN(pred_fake, valid)
            pixel_loss = criterion_pixelwise(container_img, cover_imgv)
            vgg_loss = mse_loss(vgg(container_img).relu2_2, vgg(cover_imgv).relu2_2)
        
            errH = 10000 * (mse_loss(container_img, cover_imgv) +  opt.betagan * (opt.betagans * gan_loss + opt.betapix * pixel_loss) + opt.betavgg * vgg_loss  ) 
            # errH = 10000 * (mse_loss(container_img, cover_imgv))
            rev_secret_img = Rnet(container_img)
            clean_rev_secret_img_A = Rnet(cover_img_A)
            clean_rev_secret_img_B = Rnet(cover_imgv)

            ##############################################random rotate#########################################

            # container_img_rot  = ZJ_ROT(container_img, degree)
            container_img_rot = ZJ_ROT(container_img, matrix)
            # container_img_rot = container_img
            cover_imgv_rot  = cover_img_B_rot

            ##############################################crop_resize#########################################


            if opt.nocrop :
                opt.cropsize = 256

            elif random.randint(0,1):
                # print("CROPCROPCROPCROPCROPCROPCROPCROPCROP")
                opt.cropsize = random.randint(32, 128)*2
            else:
                # print("NOCROP NOCROP NOCROP")
                opt.cropsize = 256

            # opt.cropsize = random.randint(64，256)
            k = random.randint(0, 256 - opt.cropsize)
            j = random.randint(0, 256 - opt.cropsize)

            container_img_crop = container_img_rot[:, :, k:k+opt.cropsize, j:j+opt.cropsize]
            cover_img_A_crop = cover_img_A_rot[:, :, k:k+opt.cropsize, j:j+opt.cropsize]
            cover_imgv_crop = cover_imgv_rot[:, :, k:k+opt.cropsize, j:j+opt.cropsize]
            clean_img = clean_img[:, :, k:k+opt.cropsize, j:j+opt.cropsize]


            # opt.resizesize = random.randint(8,32)*8

            if opt.noresize :
                opt.resizesize = opt.cropsize
                container_img_resize =container_img_crop
                cover_img_A_resize =cover_img_A_crop
                cover_imgv_resize = cover_imgv_crop
                clean_img_resize = clean_img           

            opt.resizesize = 128
            container_img_resize = F.interpolate(container_img_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
            cover_img_A_resize = F.interpolate(cover_img_A_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
            cover_imgv_resize = F.interpolate(cover_imgv_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
            clean_img_resize = F.interpolate(clean_img, [opt.resizesize,opt.resizesize], mode='bilinear')

            # elif random.randint(0,1):
            #     # print("RESIZE RESIZE RESIZE")
            #     opt.resizesize = 128
            #     container_img_resize = F.interpolate(container_img_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
            #     cover_img_A_resize = F.interpolate(cover_img_A_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
            #     cover_imgv_resize = F.interpolate(cover_imgv_crop, [opt.resizesize,opt.resizesize], mode='bilinear')
            #     clean_img_resize = F.interpolate(clean_img, [opt.resizesize,opt.resizesize], mode='bilinear')

            # else:
            #     opt.resizesize = opt.cropsize
            #     # print("NORESIZE NORESIZE NORESIZE")
            #     # print("FFFFFFFFFFFFF",opt.resizesize)

            #     container_img_resize =container_img_crop
            #     cover_img_A_resize =cover_img_A_crop
            #     cover_imgv_resize = cover_imgv_crop
            #     clean_img_resize = clean_img


            Watermark_resize = torch.ones_like(container_img_resize).cuda()
            Mask_resize = torch.ones_like(container_img_resize).cuda()
            for m in range(this_batch_size):
                cover_imgv_resize_m = cover_imgv_resize[m:m+1,:,:,:]
                cover_imgv_resize_m = cover_imgv_resize_m.squeeze(0)
                cover_imgv_resize_np = cover_imgv_resize_m.mul(255).cpu().numpy().transpose((1, 2, 0))
                cover_imgv_resize_np = 0.299 * cover_imgv_resize_np[:,:,0] +  0.587 * cover_imgv_resize_np[:,:,1] + 0.114 * cover_imgv_resize_np[:,:,2]
                mask_resize = filters.sobel(cover_imgv_resize_np)
                mask_resize = mask_resize > 28

                mask_resize = mask_resize.astype(np.uint8)    #edge 作为水印和原图进行拼接   uint 8  (256,256) 值为0，1
                mask_resize = mask_resize[:,:,np.newaxis]  #(256,256,1)
                mask_resize_t = torch.from_numpy(mask_resize.transpose((2, 0, 1)))
                mask_resize_t = mask_resize_t.repeat(3,1,1).unsqueeze(0)

                watermark_resize = torch.ones((1,3,opt.resizesize,opt.resizesize)).cuda()

                watermark_resize[:,0:1,:,:] = color[m,0] * mask_resize_t[:,0:1,:,:]
                watermark_resize[:,1:2,:,:] = color[m,1]  * mask_resize_t[:,0:1,:,:]
                watermark_resize[:,2:3,:,:] = color[m,2]  * mask_resize_t[:,0:1,:,:]
                watermark_resize = watermark_resize.float().div(255)

                Mask_resize[m:m+1,:,:,:] = mask_resize_t
                Watermark_resize[m:m+1,:,:,:] = watermark_resize


            ##############################################crop_resize#########################################

            container_img_transfer = container_img_resize
            cover_img_A_transfer = cover_img_A_resize
            cover_imgv_transfer = cover_imgv_resize
            clean_img_transfer = clean_img_resize
            mask_transfer = Mask_resize.cuda()
            watermark_transfer = Watermark_resize.cuda()
            watermark_transfer = watermark_transfer * mask_transfer + clean_img_transfer * (1 - mask_transfer)


            rev_secret_img_transfer = Rnet(container_img_transfer)
            watermark_transfer = Variable(watermark_transfer)

            err_edge = 10000 * mse_loss(rev_secret_img_transfer * mask_transfer , watermark_transfer * mask_transfer )
            err_back = 10000 * mse_loss(rev_secret_img_transfer * (1-mask_transfer) , watermark_transfer * (1-mask_transfer) ) 
            errR_mse = opt.betaedge * err_edge + err_back

            # errR_mse = 10000 * mse_loss(rev_secret_img_transfer , watermark_transfer)   

            clean_imgv = Variable(clean_img_transfer)
            clean_rev_secret_img_transfer_A = Rnet(cover_img_A_transfer)
            errR_clean_A = 10000 * mse_loss(clean_rev_secret_img_transfer_A, clean_imgv)

            clean_rev_secret_img_transfer_B = Rnet(cover_imgv_transfer)
            errR_clean_B = 10000 * mse_loss(clean_rev_secret_img_transfer_B, clean_imgv)

        ################################################################ Extract loss#########################################


            errR_clean =opt.betacleanA * errR_clean_A + opt.betacleanB * errR_clean_B 
            errR = errR_mse  + opt.betaclean * errR_clean


            #  Train Discriminator
            Dnet.zero_grad()
            # Real loss
            pred_real = Dnet(cover_imgv)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = Dnet(container_img.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            errD = 10000 * 0.5 * (loss_real + loss_fake)


####################################################code err calcualate##############################################

            rev_secret_img_edge = rev_secret_img_transfer * mask_transfer
            rev_secret_img_edge = rev_secret_img_edge.view(this_batch_size,3,-1)
            zero_num_mask = rev_secret_img_edge ==0
            zero_num = torch.sum(zero_num_mask,2).float()
            non_zero_num = opt.resizesize * opt.resizesize - zero_num
            rev_code_round = torch.sum(rev_secret_img_edge,2).div(non_zero_num).mul(255).round() # mean 四舍五入
            code_err = (rev_code_round - color).abs()
            
            err_th_5 = torch.ones_like(code_err) * 5  #Error bit corrected by threshold 5
            code_err_th_5 = code_err - err_th_5  
            code_err_th_max_5 = torch.max(code_err_th_5,1)
            code_err_th_max_5  = code_err_th_max_5[0]
            rev_success_5 = (code_err_th_max_5 <= 0).float()
            rev_success_rate_5 = torch.mean(rev_success_5,0)

            err_th_10 = torch.ones_like(code_err) * 10 
            code_err_th_10 = code_err - err_th_10
            code_err_th_max_10 = torch.max(code_err_th_10,1)  
            code_err_th_max_10  = code_err_th_max_10[0]   #max return 2 item
            rev_success_10 = (code_err_th_max_10 <= 0).float()
            rev_success_rate_10 = torch.mean(rev_success_10,0)



            Hlosses.update(errH.data, this_batch_size)  # 纪录H loss值
            Rlosses.update(errR.data, this_batch_size)  # 纪录R loss值
            R_mselosses.update(errR_mse.data, this_batch_size)

            Dlosses.update(errD.data, this_batch_size)  # 纪录D loss值
            FakeDlosses.update(loss_fake.data, this_batch_size)  # 纪录fakeD loss值
            RealDlosses.update(loss_real.data, this_batch_size)  # 纪录realD loss值
            Ganlosses.update(gan_loss.data, this_batch_size) #记录gan loss
            Pixellosses.update(pixel_loss.data, this_batch_size) #记录pixel loss
            Vgglosses.update(vgg_loss.data, this_batch_size) #记录pixel loss
            success_rate_5.update(rev_success_rate_5.data, this_batch_size)
            success_rate_10.update(rev_success_rate_10.data, this_batch_size)

            diff = 50 * (container_img - cover_imgv)
            diff_transfer = 50 * (container_img_transfer - cover_imgv_transfer)

            if i % 50 == 0:
                save_result_pic2(this_batch_size, (rev_secret_img_transfer * mask_transfer).data, (watermark_transfer * mask_transfer).data ,
                (rev_secret_img_transfer * (1-mask_transfer)).data ,( watermark_transfer * (1-mask_transfer)).data,
                epoch, i, opt.validationpics)

                save_result_pic(this_batch_size, cover_img_A, cover_imgv.data, container_img.data, diff.data, watermark.data,
                    rev_secret_img.data, clean_rev_secret_img_A.data, clean_rev_secret_img_B.data,
                    cover_img_A_transfer, cover_imgv_transfer.data, container_img_transfer.data,diff_transfer.data,
                    watermark_transfer.data,  rev_secret_img_transfer.data, clean_rev_secret_img_transfer_A.data, clean_rev_secret_img_transfer_B.data,epoch, i, opt.validationpics)

                rev_success_5_str = str(rev_success_5.cpu().numpy())
                rev_success_10_str = str(rev_success_10.cpu().numpy()) 
                degree_str = str(degree.cpu().numpy())

                SR_log = '[%d/%d][%d/%d]\t TH5 SUCCESS Matrix: %s TH5 SUCCESS Rate: %.4f TH10 SUCCESS Matrix: %s TH10 SUCCESS Rate: %.4f Degree: %s'% (epoch, opt.niter, i, len(val_loader),
                rev_success_5_str , success_rate_5.val, rev_success_10_str , success_rate_10.val, degree_str)
                print_log(SR_log, SRPath)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_r_mseloss = R_mselosses.avg

    val_dloss = Dlosses.avg
    val_fakedloss = FakeDlosses.avg
    val_realdloss = RealDlosses.avg
    val_Ganlosses = Ganlosses.avg
    val_Pixellosses = Pixellosses.avg
    val_Vgglosses = Vgglosses.avg       
    val_sumloss = val_hloss + opt.beta * val_rloss
    rev_success_5_rate = success_rate_5.avg
    rev_success_10_rate = success_rate_10.avg


    val_time = time.time() - start_time
    val_log = "validation[%d] TH5 Suceess Rate %.6f\t TH10 Suceess Rate %.6f\t val_Hloss = %.6f\t val_Rloss = %.6f\t val_R_mseloss = %.6f\t val_Dloss = %.6f\t val_FakeDloss = %.6f\t val_RealDloss = %.6f\t val_Ganlosses = %.6f\t \
        val_Pixellosses = %.6f\t val_Vgglosses = %.6f\t val_Sumloss = %.6f\t validation time=%.2f " % (
        epoch,rev_success_5_rate,rev_success_10_rate, val_hloss, val_rloss, val_r_mseloss, val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses, val_Vgglosses, val_sumloss, val_time)

    print_log(val_log, logPath)


    writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
    writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
    writer.add_scalar('validation/R_mse_loss', R_mselosses.avg, epoch)

    writer.add_scalar('validation/D_loss_avg', Dlosses.avg, epoch)
    writer.add_scalar('validation/FakeD_loss_avg', FakeDlosses.avg, epoch)
    writer.add_scalar('validation/RealD_loss_avg', RealDlosses.avg, epoch)
    writer.add_scalar('validation/Gan_loss_avg', val_Ganlosses, epoch)
    writer.add_scalar('validation/Pixel_loss_avg', val_Pixellosses, epoch)
    writer.add_scalar('validation/Vgg_loss_avg', val_Vgglosses, epoch)    
    writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)
    writer.add_scalar('validation/success_rate_5_avg', rev_success_5_rate, epoch)
    writer.add_scalar('validation/success_rate_10_avg', rev_success_10_rate, epoch)

    print(
        "#################################################### validation end ########################################################")
    
    return rev_success_5_rate, rev_success_10_rate, val_hloss, val_rloss, val_r_mseloss,  val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses, vgg_loss, val_sumloss


# custom weights initialization called on netG and netD

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
    	# torch.nn.init.kaiming_normal_(m.weight)
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# 保存本次实验的代码
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)  # eg：/n/liyz/videosteganography/main.py
    cur_work_dir, mainfile = os.path.split(main_file_path)  # eg：/n/liyz/videosteganography/

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)



# print the training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not opt.debug:
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic(this_batch_size, originalLabelvA, originalLabelvB, Container_allImg, diff,secretLabelv, RevSecImg,RevCleanImgA,RevCleanImgB,
    A_crop,B_crop,container_crop,diff_crop,secret_crop,RevSecImg_crop,RevCleanImgA_crop,RevCleanImgB_crop, epoch, i, save_path):
    if not opt.debug:
        originalFramesA = originalLabelvA.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        originalFramesB = originalLabelvB.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        container_allFrames = Container_allImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        diff = diff.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        secretFrames = secretLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)


        revSec = RevSecImg
        revCleanA = RevCleanImgA
        revCleanB = RevCleanImgB

        showResult = torch.cat([ originalFramesA, revCleanA,originalFramesB, revCleanB, diff,container_allFrames,secretFrames,revSec], 0)
        resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(showResult, resultImgName, nrow=this_batch_size, padding=1, normalize=False)


        A_crop = A_crop.resize_(this_batch_size, 3, opt.resizesize, opt.resizesize)
        B_crop = B_crop.resize_(this_batch_size, 3, opt.resizesize, opt.resizesize)
        container_crop = container_crop.resize_(this_batch_size, 3, opt.resizesize, opt.resizesize)
        diff_crop = diff_crop.resize_(this_batch_size, 3, opt.resizesize, opt.resizesize)
        secret_crop = secret_crop.resize_(this_batch_size, 3, opt.resizesize, opt.resizesize)
        # mask_rot = mask_rot.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        revSecFrames = RevSecImg_crop.resize_(this_batch_size, 3, opt.resizesize, opt.resizesize)
        revCleanFramesA = RevCleanImgA_crop.resize_(this_batch_size, 3, opt.resizesize, opt.resizesize)
        revCleanFramesB = RevCleanImgB_crop.resize_(this_batch_size, 3, opt.resizesize, opt.resizesize)

        showResult_crop = torch.cat([ A_crop, revCleanFramesA, B_crop, revCleanFramesB, diff_crop, container_crop,secret_crop, revSecFrames  ], 0)
        
        resultImgName_crop = '%s/Crop%03d_Resize%03d_ResultPics_epoch%03d_batch%04d.png' % (save_path,  opt.cropsize, opt.resizesize, epoch, i)
        vutils.save_image(showResult_crop, resultImgName_crop, nrow=this_batch_size, padding=1, normalize=False)

def save_result_pic2(this_batch_size, originalLabelvA, originalLabelvB, originalLabelvC, originalLabelvD,epoch, i, save_path):
    # if not opt.debug:
    #     originalFramesA = originalLabelvA.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    #     originalFramesB = originalLabelvB.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

    #     originalFramesC = originalLabelvC.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    #     originalFramesD = originalLabelvD.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    originalFramesA = originalLabelvA
    originalFramesB = originalLabelvB

    originalFramesC = originalLabelvC
    originalFramesD = originalLabelvD

    showResult = torch.cat([ originalFramesA, originalFramesB, originalFramesC, originalFramesD], 0)
    resultImgName = '%s/EDGEResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
    vutils.save_image(showResult, resultImgName, nrow=this_batch_size, padding=1, normalize=False)




class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def F_batch_affine2d(x, matrix, center=True):


    if matrix.dim() == 2:
        matrix = matrix.view(-1,2,3)

    A_batch = matrix[:,:,:2]
    b_batch = matrix[:,:,2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = torch_iterproduct(x.size(2),x.size(3))
    coords = Variable(_coords.unsqueeze(0).repeat(x.size(0),1,1).float(),
                requires_grad=False)
    
    if center:
        # shift the coordinates so center is the origin
        coords[:,:,0] = coords[:,:,0] - (x.size(2) / 2. + 0.5)
        coords[:,:,1] = coords[:,:,1] - (x.size(3) / 2. + 0.5)
    

    # apply the coordinate transformation
    new_coords = coords.cuda().bmm(A_batch.transpose(1,2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,:,0] = new_coords[:,:,0] + (x.size(2) / 2. + 0.5)
        new_coords[:,:,1] = new_coords[:,:,1] + (x.size(3) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    x_transformed = F_batch_bilinear_interp2d(x, new_coords)

    return x_transformed


def F_batch_bilinear_interp2d(input, coords):
    """
    input : torch.Tensor
        size = (N,H,W,C)
    coords : torch.Tensor
        size = (N,H*W*C,2)
    """
    x = torch.clamp(coords[:,:,0], 0, input.size(2)-2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:,:,1], 0, input.size(3)-2)
    y0 = y.floor()
    y1 = y0 + 1

    stride = torch.LongTensor(input.stride())
    x0_ix = x0.mul(stride[2]).long()
    x1_ix = x1.mul(stride[2]).long()
    y0_ix = y0.mul(stride[3]).long()
    y1_ix = y1.mul(stride[3]).long()

    input_flat = input.view(input.size(0),-1).contiguous()
    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix).detach())
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix).detach())
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix).detach())
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix).detach())
    
    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))

    return x_mapped.view_as(input)

def torch_iterproduct(*args):
    return torch.from_numpy(np.indices(args).reshape((len(args),-1)).T)




def Rotate_ZJ(input,rotation_matrix):

    rotation_matrix = rotation_matrix.cuda()
    outputs = torch.cat([F_batch_affine2d(input[:, k:k+1, :, :], rotation_matrix, center=True) for k in range(3)], dim=1)

    return outputs


def ZJ_ROT(ori, rotation_matrix):
    
    pad_dims=[2,2,2,2]
    x = F.pad(ori,pad_dims,'constant',value = 1)
    x = Rotate_ZJ(x,rotation_matrix)
    x = x[:,:,2:258,2:258]

    return x

if __name__ == '__main__':
    main()
