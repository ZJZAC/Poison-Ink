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
from data.wm_dataset import wm_folder
from models.ReflectionUNet import UnetGenerator2,UnetGenerator_IN2
from torch.utils.data import DataLoader
import shutil

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
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
parser.add_argument('-c', '--checkpoint', default='/data-x/g12/zhangjie/nips/exp_cls/Trigger', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-r', '--remark', default='try', type=str, help='comment')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--task', default="BASE", type=str,  help='train strategy , w/o transform ')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
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

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    args.checkpoint = args.checkpoint + "/" + args.remark + '_' + args.task + '_' + cur_time
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    shutil.copy('./cifar_trigger.py', args.checkpoint)


    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        # # transforms.RandomCrop(32, padding=4),
        # transforms.RandomCrop(28),
        # transforms.Resize(32),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.RandomCrop(28),
        # transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        num_classes = 10
        train_dataset = ori_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png/train',transform_train)
        train_dataset_wm = wm_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png_base_green/train',transform_train)
        # train_dataset_wm = wm_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png_crop_green/train',transform_train)

        val_dataset = ori_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png/test',transform_test)
        val_dataset_wm = wm_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png_base_green/test',transform_test)
        # val_dataset_wm = wm_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png_crop_green/test',transform_test)

    else:
        num_classes = 100
        train_dataset = ori_folder('/data-x/g12/zhangjie/nips/datasets/cifar100png/train',transform_train)
        train_dataset_wm = wm_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png_base_green/train',transform_train)
        # train_dataset_wm = wm_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png_crop_green/train',transform_train)

        val_dataset = ori_folder('/data-x/g12/zhangjie/nips/datasets/cifar100png/test',transform_test)
        val_dataset_wm = wm_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png_base_green/test',transform_test)
        # val_dataset_wm = wm_folder('/data-x/g12/zhangjie/nips/datasets/cifar10png_crop_green/test',transform_test)


    trainloader = DataLoader(train_dataset,  batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = DataLoader(val_dataset,batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    wm_trainloader = DataLoader(train_dataset_wm,  batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    wm_testloader = DataLoader(val_dataset_wm,batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    wminputs, wmtargets = [], []
    for wm_idx, (wminput, wmtarget) in enumerate(wm_trainloader):
        wminputs.append(wminput)
        wmtargets.append(wmtarget)

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
                    depth=args.depth,
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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    if args.dataset == "cifar10":
        title = 'cifar-10-' + args.arch + '-T-' + args.task
    elif args.dataset == "cifar100":
        title = 'cifar-100-' + args.arch + '-T-' + args.task

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_loss = Logger(os.path.join(args.checkpoint, 'log_loss.txt'), title=title, resume=True)
        logger_acc = Logger(os.path.join(args.checkpoint, 'log_acc.txt'), title=title, resume=True)

    else:
        logger_loss = Logger(os.path.join(args.checkpoint, 'loss.txt'), title=title)
        logger_loss.set_names([ 'Train Combine Loss', 'Valid Clean Loss',  'Valid Trigger Loss.'])
        logger_acc = Logger(os.path.join(args.checkpoint, 'acc.txt'), title=title)
        logger_acc.set_names([ ' Train Combine Acc ', ' Valid Clean  Acc.',  ' Valid Trigger Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss1, test_acc, test_loss2, test_wm = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f, Test Wm Acc:  %.2f' % (test_loss, test_acc, test_wm))
        return

    best_epoch = 0
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss_combine, train_acc_combine, = train(trainloader, wminputs, wmtargets , model, criterion, optimizer, epoch, use_cuda)
        test_loss1, test_acc, test_loss2, test_wm = test(testloader, wm_testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger_loss.append([ train_loss_combine, test_loss1,test_loss2])
        logger_acc.append([train_acc_combine,  test_acc, test_wm])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

        if is_best:
            best_epoch = epoch + 1

    logger_loss.close()
    logger_loss.plot()
    savefig(os.path.join(args.checkpoint, 'log_loss.eps'))
    logger_acc.close()
    logger_acc.plot()
    savefig(os.path.join(args.checkpoint, 'log_acc.eps'))

    file_name = os.path.join(args.checkpoint, 'best.txt')
    f = open(file_name, 'w+')

    print("Best epoch:", best_epoch, file=f)
    print('Best acc:', best_acc, file= f)
    print("Best epoch:", best_epoch)
    print('Best acc:', best_acc)

def train(trainloader, wminputs, wmtargets , model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))

    wm_id = np.random.randint(len(wminputs))
    for batch_idx, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        wm_input = wminputs[(wm_id + batch_idx) % len(wminputs)]
        wm_target = wmtargets[(wm_id + batch_idx) % len(wmtargets)]

        if use_cuda:
            input, target = input.cuda(), target.cuda()
            wm_input, wm_target = wm_input.cuda(), wm_target.cuda()

        inputs = torch.cat([input, wm_input], dim=0)
        targets = torch.cat([target, wm_target], dim=0)

        # print(targets.shape)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))

        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} ' \
                      '| Loss_combine: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, wm_testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    wm_top1 = AverageMeter()
    wm_top5 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():

            outputs = model(inputs)
            loss1 = criterion(outputs, targets)
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses1.update(loss1.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} ' \
                      '| Loss_clean: {loss1:.4f} | top1: {top1: .4f} | top5: {top5: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss1=losses1.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()


    #trigger test processing

    end = time.time()
    bar = Bar('Processing_Trigger', max=len(wm_testloader))
    for batch_idx, (wm_inputs, wm_targets) in enumerate(wm_testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            wm_inputs, wm_targets = wm_inputs.cuda(), wm_targets.cuda()

        with torch.no_grad():

            wm_outputs = model(wm_inputs)
            loss2 = criterion(wm_outputs, wm_targets)
            wm_prec1, wm_prec5 = accuracy(wm_outputs.data, wm_targets.data, topk=(1, 5))
            wm_top1.update(wm_prec1, wm_inputs.size(0))
            wm_top5.update(wm_prec5, wm_inputs.size(0))

        # measure accuracy and record loss
        wm_prec1, wm_prec5 = accuracy(wm_outputs.data, wm_targets.data, topk=(1, 5))
        losses2.update(loss2.data, wm_inputs.size(0))
        wm_top1.update(wm_prec1, wm_inputs.size(0))
        wm_top5.update(wm_prec5, wm_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} ' \
                      '| Loss_trigger: {loss2:.4f} | T-top1: {wm_top1: .4f} | T-top5: {wm_top5: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss2=losses2.avg,
                    wm_top1=wm_top1.avg,
                    wm_top5=wm_top5.avg,
                    )
        bar.next()
    bar.finish()

    return (losses1.avg, top1.avg, losses2.avg, wm_top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
