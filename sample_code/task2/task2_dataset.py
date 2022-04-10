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
import matplotlib.pyplot as plt



class PlaneDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annot_file_path, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        print(img_dir)
        self.img_path_list = list(sorted(os.listdir(img_dir)))
        print(self.img_dir,self.img_path_list[0])
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
        print(len(img_bbox_dict.keys()),len(self.img_path_list))
        assert len(img_bbox_dict.keys()) == len(self.img_path_list)
            
    def __getitem__(self, idx):

        #TODO:
        # 1. load images and bounding boxes
        # 2. prepare bboxes into [xmin, ymin, xmax, ymax] format
        # 3. convert everything into torch tensor
        # 4. preprare target_dict with the following keys: bboxes, labels, image_id, area, is_crowd
        # 5. return img, target pair
        

        
        img_abs_path = self.img_dir+'/'+self.img_path_list[idx]
        img = Image.open(img_abs_path)
        boxes = []
        area = []
        labels = []
        iscrowd = []
        for box in self.img_bbox_dict[self.img_path_list[idx]]:
            min_x = max_x = box[0][0]
            min_y = max_y = box[0][1]
            for point in box:
                if(point[0]<min_x):
                    min_x = point[0]
                if(point[0]>max_x):
                    max_x = point[0]
                if(point[1]<min_y):
                    min_y = point[1]
                if(point[1]>max_y):
                    max_y = point[1]
            boxes.append([min_x,min_y,max_x,max_y])
            area.append((max_x-min_x)*(max_y-min_y))
            labels.append(1)
            iscrowd.append(0)
        
        area = torch.LongTensor(area)
        boxes = torch.LongTensor(boxes)
        labels = torch.LongTensor(labels)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor(idx)
        target["area"] = area
        target["iscrowd"] = torch.LongTensor(iscrowd)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_path_list)
