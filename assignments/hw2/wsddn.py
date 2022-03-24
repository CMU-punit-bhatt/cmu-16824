import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import numpy as np

from torchvision.ops import roi_pool, roi_align

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None, n_rois=300):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        self.n_rois = n_rois

        #TODO: Define the WSDDN model
        # Refer to https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
        self.features = torch.hub.load('pytorch/vision:v0.10.0',
                                       'alexnet',
                                       pretrained=True).features[: -1]

        # self.roi_pool = roi_pool()

        # Classifier similar to Alexnet's.
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, self.n_classes),
        )

        self.score_fc   = nn.Sequential(
            nn.Linear(4096, self.n_classes),
            # Reshape((-1, self.n_rois, self.n_classes)),
            nn.Softmax(dim=-1)
        )
        self.bbox_fc    = nn.Sequential(
            nn.Linear(4096, self.n_classes),
            # Reshape((-1, self.n_classes)),
            nn.Softmax(dim=0)
        )

        # loss
        self.cross_entropy = None


    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):

        assert len(image.shape) == 4
        assert image.size(0) == 1
        assert image.size(2) == image.size(3)

        h, w = image.size(2), image.size(3)

        # I know this printing. I am not going to write a function to calculate
        # this. Mostly because this assignment is already so long :(.
        subsampling_ratio = image.size(2) / 31
        spatial_scale = 1 / subsampling_ratio

        # Bringing these to absolute values.
        rois[..., 0] = rois[..., 0] * w
        rois[..., 1] = rois[..., 1] * h
        rois[..., 2] = rois[..., 2] * w
        rois[..., 3] = rois[..., 3] * h

        #TODO: Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores

        features = self.features(image)

        # print(rois.dtype)
        # print(features.dtype)

        roi_pool_func = lambda x: roi_pool(features,
                                           x,
                                           (6, 6),
                                           spatial_scale=spatial_scale)
        
        roi_pooled_feat = torch.cat([roi_pool_func([rois[0, i].unsqueeze(0)]) \
            for i in range(rois.size(1))])

        # print(roi_pooled_feat.shape)

        classifier_out = self.classifier(roi_pooled_feat.view(rois.size(1), -1))
        scores = self.score_fc(classifier_out)
        bboxes = self.bbox_fc(classifier_out)

        cls_prob = scores.view(-1, self.n_classes) * \
            bboxes.view(-1, self.n_classes)

        # print(cls_prob.shape)
        # print(scores.shape)
        # print(bboxes.shape)

        if self.training:
            label_vec = gt_vec.view(-1, self.n_classes)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)

        del roi_pooled_feat, scores, bboxes, features

        return cls_prob


    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        #TODO: Compute the appropriate loss using the cls_prob that is the
        #output of forward()
        #Checkout forward() to see how it is called

        # They sum it up. Check paper.
        probs = torch.clamp(torch.sum(cls_prob, dim=0, keepdims=True),
                            min=0.0,
                            max=1.0)

        loss = nn.BCELoss()(probs,
                            label_vec)

        return loss
