from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, compute_ap, get_box_data_caption
from PIL import Image, ImageDraw
from task_1 import AverageMeter, save_checkpoint
from mAP import calculate_map

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = 2 * 5000
lr_decay = 7. / 10
rand_seed = 1024

lr = 0.001
momentum = 0.9
weight_decay = 0.0005
# ------------

#------------
USE_WANDB = True
USE_WANDB_IMAGE = True
#------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load datasets and create dataloaders

# Not sure if the split terms are right. Need to take a look again.
train_dataset = VOCDataset(split='trainval', image_size=512, return_gt=True)
val_dataset = VOCDataset(split='test', image_size=512, return_gt=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,   # batchsize is one for this implementation
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=None,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True)


# Create network and initialize
net = WSDDN(classes=train_dataset.CLASS_NAMES)
print(net)

if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()

for name, param in pret_net.items():
    print(name)
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue


# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)

optimizer = torch.optim.SGD(list(net.parameters())[1:],
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=lr_decay_steps,
                                            gamma=lr_decay)

output_dir = "./task_2/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
disp_interval = 500
val_interval = 1500
n_epochs = 5


def test_net(model, val_loader=None, thresh=0.045):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    labels = []
    cls_probs = []

    end = time.time()

    all_bboxes = []
    all_scores = []
    all_classes = []
    all_gt_boxes = []
    all_gt_classes = []

    for i, data in enumerate(val_loader):

        # if i == 501:
        #   break

        data_time.update(time.time() - end)

        # one batch = data for one image
        image           = data['image'].cuda()
        target          = data['label'].cuda()
        wgt             = data['wgt'].cuda()
        rois            = data['rois'].cuda()
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']

        #TODO: perform forward pass, compute cls_probs
        cls_roi_probs = net(image, rois.float(), target).squeeze(0)

        class_bboxes_i = []
        class_scores_i = []
        classes_i = []

        # print(gt_class_list)
        # print(gt_boxes)

        # Abusing the fact that there's just a single image per batch.
        all_gt_boxes.append(torch.Tensor(gt_boxes))
        all_gt_classes.append(torch.Tensor(gt_class_list))

        # TODO: Iterate over each class (follow comments)
        for class_num in range(20):
            # get valid rois and cls_scores based on thresh
            # print(cls_roi_probs.shape)
            scores = cls_roi_probs[:, class_num]
            # print(scores.shape)
            bboxes = rois[0, scores > thresh]
            scores = scores[scores > thresh]

            # use NMS to get boxes and scores
            bboxes, scores = nms(bboxes, scores)

        # Everything on cpu.
        # bboxes (_type_): A (M, N, 4) tensor.
        # scores (_type_): A (M, N,) tensor.
        # preds : A (M, N) tensor indicating class.
        # gt_boxes (_type_): List of M (K_i, 4) tensors. Currently a 3d list
        # gt_classes : List of M (K_i,) tensors. Currently a 2d list.

            class_bboxes_i.extend(bboxes.cpu().detach().numpy().tolist())
            class_scores_i.extend(scores.cpu().detach().numpy().tolist())
            classes_i.extend((np.ones_like(scores.cpu().detach().numpy()) * \
                class_num).astype(np.int).tolist())

        all_bboxes.append(torch.Tensor(class_bboxes_i))
        all_scores.append(torch.Tensor(class_scores_i))
        all_classes.append(torch.Tensor(classes_i))

        if i in images_to_plot and USE_WANDB_IMAGE:
            rois_image = wandb.Image(image.cpu().detach(),
                                     boxes={
                                        "predictions": {
                                            "box_data": get_box_data_caption(classes_i,
                                                                             class_bboxes_i,
                                                                             class_scores_i,
                                                                             val_dataset.CLASS_NAMES),
                                            "class_labels": class_id_to_label,
                                        },
                                      })
            wandb.log({f"val/Bounding Boxes_{i}": rois_image})

        batch_time.update(time.time() - end)
        losses.update(net.loss.item())
        end = time.time()

        cls_probs.append(torch.sigmoid(torch.sum(cls_roi_probs, dim=0)).cpu().detach())
        labels.append(target.squeeze(0).cpu().detach())

        if i % disp_interval == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,))

        del data, target, image, rois, gt_boxes, gt_class_list, bboxes, scores

    #TODO: visualize bounding box predictions when required
    #TODO: Calculate mAP on test set

    return calculate_map(all_bboxes,
                         all_scores,
                         all_classes,
                         all_gt_boxes,
                         all_gt_classes)

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()

class_id_to_label = dict(enumerate(train_dataset.CLASS_NAMES))

if USE_WANDB:
    wandb.init(project='vlr-hw2')
    wandb.watch(net, log_freq=2000)
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")

    # wandb.define_metric("val/step")
    # wandb.define_metric("val/*", step_metric="val/step")

images_to_plot = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]

for epoch in range(n_epochs):

    end = time.time()

    for i, data in enumerate(train_loader):

        data_time.update(time.time() - end)

        #TODO: get one batch and perform forward pass
        # one batch = data for one image
        image           = data['image'].cuda()
        target          = data['label'].cuda()
        wgt             = data['wgt'].cuda()
        rois            = data['rois'].cuda()
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']

        #TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
        # also convert inputs to cuda if training on GPU

        out = net(image, rois.float(), target)

        # backward pass and update
        loss = net.loss
        train_loss += loss.item()
        step_cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        losses.update(loss.item())
        end = time.time()

        if i % disp_interval == 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,))

            # net.eval()
            # map, aps = test_net(net, val_loader)
            # print("AP ", aps)
            # print("mAP", map)
            # net.train()


            if USE_WANDB:
                wandb.log({'train/step': epoch * len(train_loader) + i})
                wandb.log({'train/loss_avg': losses.avg})
                wandb.log({'train/loss_val': losses.val})
                wandb.log({'train/LR': optimizer.param_groups[0]['lr']})

        if i in images_to_plot and USE_WANDB_IMAGE:

            conf_thresh = 0.045

            class_probs = torch.sigmoid(torch.sum(out, dim=0))

            class_bboxes = []
            class_scores = []
            classes = []

            # TODO: Iterate over each class (follow comments)
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                scores_i = out[:, class_num]
                # print(scores.shape)
                bboxes_i = rois[0, scores_i > conf_thresh]
                scores_i = scores_i[scores_i > conf_thresh]

                # use NMS to get boxes and scores
                bboxes_i, scores_i = nms(bboxes_i, scores_i)

                class_bboxes.extend(bboxes_i.cpu().detach().numpy().tolist())
                class_scores.extend(scores_i.cpu().detach().numpy().tolist())
                classes.extend((np.ones_like(scores_i.cpu().detach().numpy()) * \
                    class_num).astype(np.int).tolist())

            rois_image = wandb.Image(image.cpu().detach(),
                                     boxes={
                                        "predictions": {
                                            "box_data": get_box_data_caption(classes,
                                                                             class_bboxes,
                                                                             class_scores,
                                                                             train_dataset.CLASS_NAMES),
                                            "class_labels": class_id_to_label,
                                        },
                                     })

            wandb.log({f"train/Bounding Boxes_{i}": rois_image})


    net.eval()
    map, aps = test_net(net, val_loader)
    print("AP ", aps)
    print("mAP", map)
    net.train()

    classes = list(range(0, 20, 2))

    if USE_WANDB:

        wandb.log({'val/map': map})

        for i in classes:
            wandb.log({f'val/ap_{i}_{VOCDataset.CLASS_NAMES[i]}': aps[i]})


        #TODO: Perform all visualizations here
        #The intervals for different things are defined in the handout

save_checkpoint({
        'epoch': n_epochs,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    },
    True)