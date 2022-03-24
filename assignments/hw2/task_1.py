import argparse
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from pickletools import optimize
from PIL import Image
import shutil
import time
import sys
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from AlexNet import *
from voc_dataset import *
from utils import *

import wandb
USE_WANDB = True # use flags, wandb is not convenient for debugging
USE_WANDB_IMAGE_GLOBAL = True

torch.manual_seed(0)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')
parser.add_argument(
    '--avg-pool',
    dest='avg_pool',
    action='store_true',
    help='Performs global average pool instead of maxpool')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    # you can also use PlateauLR scheduler, which usually works well

    # print(args)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    criterion = torch.nn.BCEWithLogitsLoss()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code

    #TODO: Create Datasets and Dataloaders using VOCDataset - Ensure that the sizes are as required
    # Also ensure that data directories are correct - the ones use for testing by TAs might be different
    # Resize the images to 512x512

    # Not sure if the split terms are right. Need to take a look again.
    train_dataset = VOCDataset(split='trainval', image_size=512)
    val_dataset = VOCDataset(split='test', image_size=512)

    # See if sampler is needed. Why is the shuffle disabled?
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for wandb - ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB :
        wandb.init(project='vlr-hw2')
        wandb.watch(model, log_freq=100)
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

        wandb.define_metric("val/step")
        wandb.define_metric("val/*", step_metric="val/step")

    epochs_for_image = [0, args.epochs // 2, args.epochs - 1]
    val_batches_to_plot = torch.randint(0, len(val_loader), (3,))

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        train(train_loader,
              model,
              criterion,
              optimizer,
              epoch,
              USE_WANDB_IMAGE=(USE_WANDB_IMAGE_GLOBAL and epoch in epochs_for_image))

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader,
                              model,
                              criterion,
                              epoch,
                              USE_WANDB_IMAGE_GLOBAL,
                              val_batches_to_plot)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        scheduler.step()

#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, USE_WANDB_IMAGE=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO: Get inputs from the data dict
        images = data['image'].cuda()
        labels = data['label'].cuda()
        weights = data['wgt'].cuda()
        # gt_classes = data['gt_classes']

        optimizer.zero_grad()

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output such as clamping
        # TODO: Compute loss using ``criterion``

        out = model(images)

        # Applying a global maxpool or avg pool.
        if args.avg_pool:
            class_out = torch.mean(out, (2, 3))
        else:
            class_out = torch.amax(out, (2, 3))

        loss = criterion(class_out, labels)

        # TODO:
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # measure metrics and record loss
        m1 = metric1(class_out, labels)
        m2 = metric2(class_out, labels)
        losses.update(loss.item(), images.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

            # step = epoch * len(train_loader) + i

        #TODO: Visualize/log things as mentioned in handout
        #TODO: Visualize at appropriate intervals
            if USE_WANDB:
                wandb.log({'train/step': epoch * len(train_loader) + i})
                wandb.log({'train/loss': losses.val})
                wandb.log({'train/M1': avg_m1.val})
                wandb.log({'train/M2': avg_m2.val})
                wandb.log({'train/LR': optimizer.param_groups[0]['lr']})

        if USE_WANDB_IMAGE and \
          (i == 0 or i == len(train_loader) // 2):
            # wandb.log({"train/Input Images": wandb.Image(tensor_to_PIL(images[0]))})
            
            image_idx = -1

            j = torch.argsort(labels[image_idx])[-1]
            heatmap = out[image_idx][j].cpu().detach().numpy()
            heatmap = cv2.resize(heatmap, (512, 512))
            heatmap = (heatmap - heatmap.min()) / \
                      (heatmap.max() - heatmap.min())
            heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8),
                                        cv2.COLORMAP_MAGMA)

            gt_image = wandb.Image(tensor_to_PIL(images[image_idx]),
                                   caption='Ground Truth')
            heatmap = wandb.Image(heatmap, caption=f'{VOCDataset.CLASS_NAMES[j]}')

            wandb.log({"train/Heatmaps": [gt_image, heatmap]})

    # End of train()

def validate(val_loader,
             model,
             criterion, 
             epoch = 0, 
             USE_WANDB_IMAGE=False,
             batches_to_plot=[0, 1, 2]):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO: Get inputs from the data dict
        images = data['image'].cuda()
        labels = data['label'].cuda()
        weights = data['wgt'].cuda()

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output such as clamping
        # TODO: Compute loss using ``criterion``
        out = model(images)

        # Applying a global maxpool or avg pool.
        if args.avg_pool:
            class_out = torch.mean(out, (2, 3))
        else:
            class_out = torch.amax(out, (2, 3))

        loss = criterion(class_out, labels)

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # measure metrics and record loss
        m1 = metric1(class_out, labels)
        m2 = metric2(class_out, labels)
        losses.update(loss.item(), images.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

            #TODO: Visualize things as mentioned in handout
            #TODO: Visualize at appropriate intervals

            step = epoch // args.eval_freq * len(val_loader) + i

            if USE_WANDB:
                wandb.log({'val/step': step})
                wandb.log({'val/loss': losses.val})
                wandb.log({'val/M1': avg_m1.val})
                wandb.log({'val/M2': avg_m2.val})

        if USE_WANDB_IMAGE and i in batches_to_plot:
            # wandb.log({'epoch': epoch,
            #            "val/Input Images": wandb.Image(tensor_to_PIL(images[0]))})
            
            j = torch.argsort(labels[0])[-1]
            heatmap = out[0][j].cpu().detach().numpy()
            heatmap = cv2.resize(heatmap, (512, 512))
            heatmap = (heatmap - heatmap.min()) / \
                      (heatmap.max() - heatmap.min())
            heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8),
                                        cv2.COLORMAP_MAGMA)

            gt_image = wandb.Image(tensor_to_PIL(images[0]), caption='Ground Truth')
            heatmap = wandb.Image(heatmap, caption=f'{VOCDataset.CLASS_NAMES[j]}')

            wandb.log({"val/Heatmaps": [gt_image, heatmap]})

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

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

def metric1(output, target, sigmoid_req=True):
    # TODO: Ignore for now - proceed till instructed
    
    sig_out = output

    if sigmoid_req:
        sig_out = torch.sigmoid(output)

    ap = compute_ap(target, sig_out)

    return np.mean(ap)

def metric2(output, target, sigmoid_req=True, thresh=0.5):
    #TODO: Ignore for now - proceed till instructed
    
    thresh_out = torch.zeros_like(output).cuda()

    if sigmoid_req:
        thresh_out = torch.sigmoid(output)

    thresh_out[output >= thresh] = 1
    thresh_out[output < thresh] = 0

    tp = float(torch.sum((thresh_out == target)[target == 1]))
    fn = float(torch.sum((thresh_out != target)[target == 1]))

    return tp / (tp + fn)

if __name__ == '__main__':
    main()
