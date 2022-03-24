''' --------------------------------------------------------
Written by Yufei Ye (https://github.com/JudyYe)
Edited by Sanil Pande (https://github.com/sanilpande)
-------------------------------------------------------- '''
from __future__ import print_function

import numpy as np
import os
import xml.etree.ElementTree as ET

import torch
import torch.nn
from PIL import Image
from torch.utils.data import Dataset

import random
import torchvision.transforms as transforms

import scipy.io

class VOCDataset(Dataset):
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    INV_CLASS = {}

    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i

    #TODO: Ensure data directories are correct
    def __init__(self,
                 split='trainval',
                 image_size=224,
                 top_n=300,
                 data_dir='/content/VOCdevkit/VOC2007/',
                 perform_transforms=True,
                 return_gt=False):

        super().__init__()
        self.split      = split     # 'trainval' or 'test'
        self.data_dir   = data_dir
        self.size       = image_size
        self.perform_transforms = perform_transforms
        self.return_gt = return_gt

        # top_n, selective_search_dir, and roi_data have been added for assignment 2
        self.top_n      = top_n      # top_n: number of proposals to return

        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')
        self.selective_search_dir = os.path.join(data_dir, 'selective_search_data')

        # Roi Data structure - (1, 5011) ndarray of type - object.
        # Each of the 5011 elements is a float32 array with 4000 proposals each.
        self.roi_data = scipy.io.loadmat(self.selective_search_dir + '/voc_2007_'+ split + '.mat')

        # print(self.roi_data['boxScores'].shape)
        # print(self.roi_data['boxes'].shape)
        # print(self.roi_data['images'].shape)
        # print(self.roi_data['boxScores'][0, 0].shape)
        # print(self.roi_data['boxes'][0, 0].shape)

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]

        self.anno_list = self.preload_anno()

    @classmethod
    def get_class_name(cls, index):
        return cls.CLASS_NAMES[index]

    @classmethod
    def get_class_index(cls, name):
        return cls.INV_CLASS[name]

    def __len__(self):
        return len(self.index_list)

    def preload_anno(self):
        """
        :return: a list of labels. each element is in the form of [class, weight, gt_class_list, gt_boxes],
         where both class and weight are arrays/tensors in shape of [20],
         gt_class_list is a list of the class ids (separate for each instance)
         gt_boxes is a list of [xmin, ymin, xmax, ymax] values in the range 0 to 1
        """

        # TODO: Modify your previous implemention of this function using this as referece
        # TODO: You should return the GT boxes and the class labels associated with them

        label_list = []

        for index in self.index_list:
            fpath = os.path.join(self.ann_dir, index + '.xml')
            tree = ET.parse(fpath)
            root = tree.getroot()

            C = np.zeros(20)
            W = np.ones(20) * 2 # default to enable 1 or 0 later for difficulty

            # image h & w to normalize bbox coords
            height  = 0
            width   = 0

            # new list for each index
            gt_class_list = []
            gt_boxes = []

            for child in root:

                if child.tag == 'size':
                    width = int(child[0].text)
                    height = int(child[1].text)

                if child.tag == 'object':
                    C[self.INV_CLASS[child[0].text]] = 1    # item at index of child name become 1
                    if child[3].text == '1' and W[self.INV_CLASS[child[0].text]] == 2:
                        W[self.INV_CLASS[child[0].text]] = 0    # if not difficult, weight is one
                    elif child[3].text == '0' :
                        W[self.INV_CLASS[child[0].text]] = 1

                    # add class_index to gt_class_list
                    gt_class_list.append(self.INV_CLASS[child[0].text])

                    for t in child:
                        if t.tag == 'bndbox':
                            xmin = int(t[0].text) / width
                            ymin = int(t[1].text) / height
                            xmax = int(t[2].text) / width
                            ymax = int(t[3].text) / height
                            gt_boxes.append([xmin, ymin, xmax, ymax])

            for i in range(len(W)):
                if W[i] == 2:
                    W[i] = 1

            label_list.append([C, W, gt_class_list, gt_boxes])

        return label_list

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element - containing all the aforementioned information
        """
        #TODO: change this code to match your previous implementation

        findex = self.index_list[index]     # findex refers to the file number
        fpath = os.path.join(self.img_dir, findex + '.jpg')

        img = Image.open(fpath)
        W, H = img.size

        lab_vec = self.anno_list[index][0]
        wgt_vec = self.anno_list[index][1]

        label = torch.FloatTensor(lab_vec)
        wgt = torch.FloatTensor(wgt_vec)

        img = self.__transform_image__(img)

        # added for assn 2
        gt_class_list, gt_boxes = self.anno_list[index][2], self.anno_list[index][3]

        '''
        TODO:
        Get bounding box proposals for the index from self.roi_data
        Normalize in the range (0, 1) according to image size (be careful of width/height and x/y correspondences)
        Make sure to return only the top_n proposals!
        '''

        top_n_scores = np.argsort(self.roi_data['boxScores'][0, index], axis=0)[::-1][: self.top_n]
        proposals = self.roi_data['boxes'][0, index][top_n_scores].reshape(-1, 4).astype(np.float64)

        proposals[:, 0] /= H
        proposals[:, 1] /= W
        proposals[:, 2] /= H
        proposals[:, 3] /= W

        ret = {}

        ret['image']    = img
        ret['label']    = label
        ret['wgt']      = wgt

        if self.return_gt:
            ret['rois']     = proposals
            ret['gt_boxes'] = gt_boxes
            ret['gt_classes'] = gt_class_list

        # print(ret.keys())
        # print(len(ret.values()))

        return ret

    def __transform_image__(self, image):

        transforms_list = None

        if not self.perform_transforms:
          transforms_list = transforms.Compose([
              transforms.Resize((self.size, self.size)),
              transforms.ToTensor(),
          ])

        elif self.split == 'test' or self.split == 'val':
          transforms_list = transforms.Compose([
              transforms.Resize((self.size, self.size)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
          ])

        elif self.split == 'trainval' or self.split == 'train':
          transforms_list = transforms.Compose([
              transforms.Resize((self.size, self.size)),
              transforms.ColorJitter(.4, .4, .4),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
          ])

        return transforms_list(image)