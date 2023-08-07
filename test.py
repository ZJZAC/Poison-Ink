'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from data.ori_dataset import ori_folder
from data.wm_dataset import wm_folder, wm_subfolder, adv_subfolder
from models.ReflectionUNet import UnetGenerator2,UnetGenerator_IN2
from torch.utils.data import DataLoader
import shutil

from PIL import Image, ImageFilter
import cv2

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--color', action='store_true', help='multi-target label')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--ratio', type=int,default=1, help='train with trigger every "ratio" batch')

# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize 128')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/data-x/g11/zhangjie/ECCV/exp_chk/backdoor/Classification/Cifar/eccv_green/10class',
                    type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-r', '--remark', default='try', type=str, help='comment')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--task', default="BASE", type=str,  help='train strategy , w/o transform ')
parser.add_argument('--wm_tst', default="../../backdoor_generation/dataset/infect_cifar10/one_poison_label/Up_Rp_UNet_L1_incremental/1/green_airplane/test_trigger"
                     , type=str,  help='trigger test dataset')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='Bottleneck',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int,default=6, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--original', action='store_true', help='original model')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def main():
    num_classes = 10
    # Model
    print("==> creating model '{}'".format(args.arch))

    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=22,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()

    chkdir = os.path.dirname(args.resume)

    # Data
    print('==> Preparing dataset %s' % args.dataset)

    transform_test1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    transform_test2 = transforms.Compose([

        transforms.RandomHorizontalFlip(p=1),

        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    transform_test3 = transforms.Compose([


        transforms.RandomRotation(15),

        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    transform_test4 = transforms.Compose([

        transforms.Resize(28),
        transforms.Pad(2),
        

        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    transform_test5 = transforms.Compose([
        
        transforms.RandomCrop(28),
        transforms.Resize(32),

        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])



    transform_test = transforms.Compose([

        # transforms.RandomHorizontalFlip(p=1),

        # transforms.RandomRotation(15),

        # transforms.RandomCrop(200),
        # transforms.Resize(224),

        # transforms.Resize(200),
        # transforms.Pad((18,18,6,6)), #left top right bottom

        AddGaussianNoise(mean=0, variance=1, amplitude=10),

        # MyGaussianBlur(radius=5),

        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ])    

    data_aug1 = 'baseline'
    data_aug2 = 'Hflip'
    data_aug3 = 'rot15'
    data_aug4 = 'shrink_pad2'
    data_aug5 = 'crop28_resize32'

    data_aug = 'guassian10'
    # data_aug = 'blur3'

    # transform_test_list = []
    # transform_test_list= [transform_test1,transform_test2,transform_test3,transform_test4,transform_test5]
    # data_aug_list = []
    # data_aug_list = [data_aug1,data_aug2,data_aug3,data_aug4,data_aug5]

    # guassian
    transform_test_list= [transform_test]
    data_aug_list = [data_aug]


    test_Trigger_acc=[]
    test_Clean_acc=[]
    test_clean = "../../backdoor_generation/dataset/cifar10/test"
    test_wm = args.wm_tst

    # print(transform_test)
    # print(data_aug)

    for i in range(1):
        transform_test = transform_test_list[i]
        data_aug = data_aug_list[i]
        print(transform_test)
        print(data_aug)
        wm_test_dataset = wm_folder(test_wm,transform_test)
        test_dataset = ori_folder(test_clean,transform_test)

    # if args.original or args.color :
    #     wm_test_dataset = ori_folder(test_wm,transform_test)
    # else:
    #     wm_test_dataset = wm_folder(test_wm,transform_test)

        testloader = DataLoader(test_dataset,  batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        wm_testloader = DataLoader(wm_test_dataset,batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


        metric_name = os.path.join(chkdir,'test_metric_'+ data_aug+'.txt')
        f = open(metric_name, 'w+')
        print('trigger dataset',test_wm, file=f)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

        print("data transform:   ", transform_test, file=f)

        test_loss2, test_acc2 = test(testloader, model, criterion, use_cuda)
        print("test_clean:   ", test_acc2)
        print("test_clean:   ", test_acc2, file=f)

        test_Clean_acc.append(test_acc2.cpu().numpy())
        test_loss_wm2, test_acc_wm2 = test( wm_testloader, model, criterion,  use_cuda)
        print("test_Trigger:   ", test_acc_wm2)
        print("test_Trigger:   ", test_acc_wm2, file=f)

        test_Trigger_acc.append(test_acc_wm2.cpu().numpy())

    test_Clean_acc_ave = np.mean(test_Clean_acc)
    test_Trigger_acc_ave = np.mean(test_Trigger_acc)

    metric_name = os.path.join(chkdir,'test_metric_ave'+'.txt')
    f = open(metric_name, 'w+')

    print("aug_all:   ", data_aug_list, file=f)
    print("test_clean_all:   ", test_Clean_acc, file=f)
    print("test_clean_ave:   ", test_Clean_acc_ave, file=f)
    print("test_trigger_all:   ", test_Trigger_acc, file=f)
    print("test_trigger_ave:   ", test_Trigger_acc_ave, file=f)
    print("aug_all:   ", data_aug_list)
    print("test_clean_all:   ", test_Clean_acc)
    print("test_clean_ave:   ", test_Clean_acc_ave)
    print("test_trigger_all:   ", test_Trigger_acc)
    print("test_trigger_ave:   ", test_Trigger_acc_ave)


    # test_class_wm = test_wm
    # # test_class_wm = test_clean
    # # for classes in sorted(os.listdir(test_class_wm)):
    # for i, classes in enumerate(os.listdir(test_class_wm)):
        
    #     d = os.path.join(test_class_wm, classes)
    #     if args.original or args.color :
    #         print("label",i)
    #         wm_test_class_dataset = adv_subfolder(d,i,transform_test)
    #     else:
    #         wm_test_class_dataset = wm_subfolder(d,transform_test)

    #     wm_class_testloader = DataLoader(wm_test_class_dataset,batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    #     test_loss_class_wm, test_acc_class_wm = test( wm_class_testloader, model, criterion,  use_cuda)
    #     print("Class:%s   Accuracy:%02f"%(classes,test_acc_class_wm) )
    #     print("Class:%s   Accuracy:%02f"%(classes,test_acc_class_wm), file=f )


def test(testloader, model, criterion, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()


    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        # print(inputs.shape)
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

    return (losses.avg, top1.avg)


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255               # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class MyGaussianBlur(ImageFilter.Filter):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, img):
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img = cv2.blur(img,(self.radius,self.radius))
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 

        return img

if __name__ == '__main__':
    main()
