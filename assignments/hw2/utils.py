import os
import random
import time
import copy

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

import sklearn.metrics

#TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bboxes, scores, iou_thresh=0.3):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """

    # 1. Remove those below threshold
    # 2. Sort in descending order of conf.
    # 3. Go through each of same kind and discard if iou >= 0.5

    # bboxes = bounding_boxes[confidence_score > threshold]
    # scores = confidence_score[confidence_score > threshold]

    priority = torch.argsort(scores, descending=True)
    retain = torch.ones_like(scores).cuda()

    for i in priority:

        if retain[i] != 1:
            continue

        for j in priority[i:]:

            if retain[j] != 1:
                continue

            if iou(bboxes[i], bboxes[j]) >= iou_thresh:
                retain[j] = 0

    bboxes = bboxes[retain == 1]
    scores = scores[retain == 1]

    return bboxes, scores

#TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU value
    """

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    intersection_box = [max(box1[0], box2[0]),
                        max(box1[1], box2[1]),
                        min(box1[2], box2[2]),
                        min(box1[3], box2[3])]

    intersection = (intersection_box[2] - intersection_box[0]) * \
        (intersection_box[3] - intersection_box[1])

    if intersection_box[0] > intersection_box[2] or \
        intersection_box[1] > intersection_box[3]:
        intersection = 0

    union = area1 + area2 - intersection

    iou = intersection / union

    return iou


def tensor_to_PIL(image, inverse_norm_required=True):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image) if inverse_norm_required else image
    inv_tensor = torch.clamp(inv_tensor, 0, 1)

    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image

def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id" : classes[i],
        } for i in range(len(classes))
        ]

    return box_list

def get_box_data_caption(classes, bbox_coordinates, scores, class_names):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id" : classes[i],
            "box_caption": '{class_name} - {score:.3f}'.format(score=scores[i],
                                                               class_name=class_names[classes[i]]) 
            } for i in range(len(classes))
        ]

    return box_list

def compute_ap(gt, pred, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []

    for cid in range(nclasses):
        gt_cls = gt[:, cid].float()
        pred_cls = pred[:, cid].float()
        
        if torch.sum(gt_cls) <= 0:
            continue

        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls.cpu().detach().numpy(),
            pred_cls.cpu().detach().numpy(),
            average=average)
        AP.append(ap)

    return AP


