# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Zekun Li  
# version ='1.0'
# ---------------------------------------------------------------------------
""" Spatial AI Assignment"""  
# ---------------------------------------------------------------------------

import os
import numpy as np
from ast import literal_eval
from PIL import Image
import cv2

import torch
import torchvision



class PlaneDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annot_file_path, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        self.img_path_list = list(sorted(os.listdir(img_dir)))
        
        with open(annot_file_path, 'r') as f:
            data = f.readlines()
            
        img_bbox_dict = dict()
        for i in range(0, len(data)):
            line = data[i]
            img_name = line.split(',')[1]
            if img_name in img_bbox_dict:
                img_bbox_dict[img_name].append(np.array(literal_eval(line.split('"')[1])))#shape: (N,5,2)]
            else:
                img_bbox_dict[img_name] = [np.array(literal_eval(line.split('"')[1]))] 
            
        self.img_bbox_dict = img_bbox_dict
            
        assert len(img_bbox_dict.keys()) == len(self.img_path_list)
            
    def __getitem__(self, idx):

        #TODO:
        # 1. load images and bounding boxes
        # 2. prepare bboxes into [xmin, ymin, xmax, ymax] format
        # 3. convert everything into torch tensor
        # 4. preprare target_dict with the following keys: bboxes, labels, image_id, area, is_crowd
        # 5. return img, target pair
        

        

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_path_list)
